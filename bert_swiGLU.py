# === Imports ===
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

# === Custom Model: Bert + Residual SwiGLU Head ===
class BertResSwiGLUHead(BertPreTrainedModel):
    def __init__(self, config, expansion=4, p_drop=0.1):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)

        d_in = config.hidden_size            # 768 for BERT‑base
        d_hidden = d_in * expansion

        self.ff = torch.nn.Sequential(
            torch.nn.LayerNorm(d_in),
            torch.nn.Linear(d_in, d_hidden * 2),  # [x ‖ gate]
            torch.nn.SiLU(),
            torch.nn.GLU(dim=-1),                 # x * σ(gate)
            torch.nn.Dropout(p_drop),
            torch.nn.Linear(d_hidden, d_in)
        )
        self.dropout = torch.nn.Dropout(p_drop)
        self.out_proj = torch.nn.Linear(d_in, config.num_labels)

        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        bert_out = self.bert(input_ids, attention_mask=attention_mask, **kwargs)
        pooled = bert_out.last_hidden_state[:, 0]          # [CLS]
        x = pooled + self.ff(pooled)                       # residual
        x = self.dropout(x)
        logits = self.out_proj(x)

        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)

        return SequenceClassifierOutput(
            loss=loss, logits=logits,
            hidden_states=bert_out.hidden_states,
            attentions=bert_out.attentions
        )

# === Data Loading & Pre‑processing ===
df = pd.read_csv(
    "/Users/xuwei/Desktop/spring2025/467_final/CSCI467_deepfake/data/1/Resume/Resume.csv"
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
def tokenize(batch):
    return tokenizer(batch["Resume_str"],
                     padding="max_length",
                     truncation=True,
                     max_length=128)

train_ds = train_ds.map(tokenize, batched=True)   #  ← no inplace
val_ds   = val_ds.map(tokenize,   batched=True)
test_ds  = test_ds.map(tokenize,  batched=True)

# keep the format settings exactly as before
for ds in (train_ds, val_ds, test_ds):
    ds.set_format(type="torch",
                  columns=["input_ids", "attention_mask", "label"])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"f1": f1_score(labels, preds, average='weighted')}

# === Hyper‑parameter Grid Search ===
learning_rates = [2e-3, 2e-5]
batch_sizes    = [16, 32]
epochs         = [3, 5]

best_f1, best_model, best_cfg = 0, None, {}

for lr in learning_rates:
    for bs in batch_sizes:
        for ep in epochs:
            print(f"\n🚀  Config: lr={lr}, bs={bs}, epochs={ep}")
            model = BertResSwiGLUHead.from_pretrained(
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

print(f"\n✅  Best Config: {best_cfg} | Val F1: {best_f1:.4f}")

# === Test‑set Evaluation ===
trainer = Trainer(model=best_model, tokenizer=tokenizer)
test_out = trainer.predict(test_ds)
y_pred, y_true = np.argmax(test_out.predictions, 1), test_out.label_ids

print("\nFinal Classification Report:")
print(classification_report(y_true, y_pred, target_names=label_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_names, yticklabels=label_names)
plt.title("Confusion Matrix – Residual SwiGLU Head")
plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
plt.tight_layout(); plt.savefig("cm_res_swiglu.png", dpi=300); plt.show()
