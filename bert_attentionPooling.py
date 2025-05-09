import pandas as pd, numpy as np, torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt, seaborn as sns

from datasets import Dataset
from transformers import (
    BertTokenizer, BertModel, BertPreTrainedModel,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from transformers.modeling_outputs import SequenceClassifierOutput

class AttnPool(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q = torch.nn.Linear(d_model, d_model)
        self.k = torch.nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5

    def forward(self, x, mask):             # x: (B,L,D)  mask: (B,L)
        q = self.q(x[:, :1])                # (B,1,D)
        k = self.k(x)                       # (B,L,D)
        scores = (q @ k.transpose(-2, -1)).squeeze(1) * self.scale  # (B,L)
        scores = scores.masked_fill(~mask.bool(), -1e4)
        attn = scores.softmax(-1).unsqueeze(-1)        # (B,L,1)
        return (attn * x).sum(1)                       # (B,D)

# Bert + Attn‑Pooling Head
class BertAttnPoolingHead(BertPreTrainedModel):
    def __init__(self, config, p_drop=0.1):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        d = config.hidden_size

        self.pool = AttnPool(d)
        self.mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(d * 3),     
            torch.nn.Linear(d * 3, d),
            torch.nn.GELU(),
            torch.nn.Dropout(p_drop),
            torch.nn.Linear(d, config.num_labels)
        )
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        out = self.bert(input_ids, attention_mask=attention_mask, **kwargs)
        x = out.last_hidden_state                      # (B,L,D)

        attn_vec = self.pool(x, attention_mask)        # (B,D)
        mean_vec = x.mean(1)
        max_vec  = x.max(1).values
        pooled   = torch.cat([attn_vec, mean_vec, max_vec], dim=-1)  # (B,3D)

        logits = self.mlp(pooled)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)

        return SequenceClassifierOutput(
            loss=loss, logits=logits,
            hidden_states=out.hidden_states,
            attentions=out.attentions
        )

# Data Loading and Pre‑processing
df = pd.read_csv(
    "/Users/huangzitong/.cache/kagglehub/datasets/snehaanbhawal/resume-dataset/versions/1/Resume/Resume.csv"
)
df = df.drop(columns=["Resume_html"])
df["label"] = df["Category"].astype("category").cat.codes
label_names = df["Category"].astype("category").cat.categories.tolist()

train_df, temp_df = train_test_split(df, test_size=0.4, stratify=df["label"], random_state=2)
val_df,   test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=2)

train_ds = Dataset.from_pandas(train_df[["Resume_str", "label"]])
val_ds   = Dataset.from_pandas(val_df[["Resume_str", "label"]])
test_ds  = Dataset.from_pandas(test_df[["Resume_str", "label"]])

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
def tokenize(batch): return tokenizer(batch["Resume_str"], padding="max_length", truncation=True, max_length=128)

for ds in (train_ds, val_ds, test_ds):
    ds.map(tokenize, batched=True, inplace=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"f1": f1_score(labels, preds, average='weighted')}

# Grid Search
learning_rates = [2e-5]
batch_sizes    = [16]
epochs         = [5]

best_f1, best_model, best_cfg = 0, None, {}

for lr in learning_rates:
    for bs in batch_sizes:
        for ep in epochs:
            print(f"\n Config: lr={lr}, bs={bs}, epochs={ep}")
            model = BertAttnPoolingHead.from_pretrained(
                "bert-base-uncased", num_labels=len(label_names)
            )

            args = TrainingArguments(
                output_dir="./finetuned_bert_resume",
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=lr,
                per_device_train_batch_size=bs,
                per_device_eval_batch_size=bs,
                num_train_epochs=ep,
                weight_decay=0.01,
                logging_dir="./logs", logging_steps=10,
                load_best_model_at_end=True, metric_for_best_model="f1"
            )

            trainer = Trainer(
                model=model, args=args,
                train_dataset=train_ds, eval_dataset=val_ds,
                tokenizer=tokenizer, compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
            )

            trainer.train()
            f1 = trainer.evaluate(val_ds)["eval_f1"]
            print(f"Validation F1 = {f1:.4f}")

            if f1 > best_f1:
                best_f1, best_model, best_cfg = f1, model, {"lr": lr, "bs": bs, "epochs": ep}

print(f"\n Best Config: {best_cfg} | Val F1: {best_f1:.4f}")

# === Test‑set Evaluation ===
trainer = Trainer(model=best_model, tokenizer=tokenizer)
test_out = trainer.predict(test_ds)
y_pred, y_true = np.argmax(test_out.predictions, 1), test_out.label_ids

print("\n Final Classification Report:")
print(classification_report(y_true, y_pred, target_names=label_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_names, yticklabels=label_names)
plt.title("Confusion Matrix")
plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
plt.tight_layout(); plt.savefig("combine.png", dpi=300); plt.show()
