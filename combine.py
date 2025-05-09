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

import torch, torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

# Single‑head attention pool
class AttnPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5
    def forward(self, x, mask):
        q = self.q(x[:, :1])                       
        k = self.k(x)                                 
        scores = (q @ k.transpose(-2, -1)).squeeze(1) * self.scale
        scores = scores.masked_fill(~mask.bool(), -1e4)
        attn = scores.softmax(-1).unsqueeze(-1)           
        return (attn * x).sum(1)                        

# One Residual SwiGLU block
class ResSwiGLU(nn.Module):
    def __init__(self, d_in, expansion=4, p_drop=0.1):
        super().__init__()
        d_hid = d_in * expansion
        self.ff = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_hid * 2),  # [x ‖ gate]
            nn.SiLU(),
            nn.GLU(dim=-1),
            nn.Dropout(p_drop),
            nn.Linear(d_hid, d_in)
        )
    def forward(self, x):                        # residual
        return x + self.ff(x)

# Combine
class BertAttnPoolSwiGLUHead(BertPreTrainedModel):
    def __init__(self, config,
                 n_blocks: int = 2,  
                 expansion: int = 4,
                 p_drop: float = 0.1):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        d = config.hidden_size

        self.pool = AttnPool(d)

        self.in_ln = nn.LayerNorm(d * 3)

        self.swiglu_stack = nn.Sequential(
            *[ResSwiGLU(d * 3, expansion, p_drop) for _ in range(n_blocks)]
        )

        self.dropout = nn.Dropout(p_drop)
        self.out_proj = nn.Linear(d * 3, config.num_labels)

        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        out = self.bert(input_ids, attention_mask=attention_mask, **kwargs)
        x = out.last_hidden_state                                 # (B,L,D)

        attn_vec = self.pool(x, attention_mask)                   # (B,D)
        pooled   = torch.cat([attn_vec, x.mean(1), x.max(1).values], dim=-1)

        y = self.in_ln(pooled)
        y = self.swiglu_stack(y)
        y = self.dropout(y)
        logits = self.out_proj(y)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        return SequenceClassifierOutput(
            loss=loss, logits=logits,
            hidden_states=out.hidden_states, attentions=out.attentions
        )


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

train_ds = train_ds.map(tokenize, batched=True) 
val_ds   = val_ds.map(tokenize,   batched=True)
test_ds  = test_ds.map(tokenize,  batched=True)


for ds in (train_ds, val_ds, test_ds):
    ds.set_format(type="torch",
                  columns=["input_ids", "attention_mask", "label"])

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
            print(f"\n  Config: lr={lr}, bs={bs}, epochs={ep}")
            model = BertAttnPoolSwiGLUHead.from_pretrained(
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

print(f"\n DUANG Best Config: {best_cfg} | Val_F1: {best_f1:.4f}")

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
plt.title("Confusion Matrix  Residual SwiGLU Combine with Attention Pooling")
plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
plt.tight_layout(); plt.savefig("cm_combine.png", dpi=300); plt.show()
