import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

# === Load and preprocess ===
df = pd.read_csv("/Users/huangzitong/.cache/kagglehub/datasets/snehaanbhawal/resume-dataset/versions/1/Resume/Resume.csv")
df = df.drop(columns=["Resume_html"])
df["label"] = df["Category"].astype("category").cat.codes
label_names = df["Category"].astype("category").cat.categories.tolist()

# === Split Train/Val/Test ===
train_df, temp_df = train_test_split(df, test_size=0.4, stratify=df["label"], random_state=2)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=2)

# === Convert to HuggingFace Datasets ===
train_ds = Dataset.from_pandas(train_df[["Resume_str", "label"]])
val_ds = Dataset.from_pandas(val_df[["Resume_str", "label"]])
test_ds = Dataset.from_pandas(test_df[["Resume_str", "label"]])

# === Tokenizer and Tokenize Function ===
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
def tokenize(batch):
    return tokenizer(batch["Resume_str"], padding="max_length", truncation=True, max_length=128)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# === Metric Function ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    f1 = f1_score(labels, preds, average='weighted')
    return {"f1": f1}

# === Hyperparameter Search ===
learning_rates = [2e-3, 2e-5]
batch_sizes = [16, 32]
epochs = [3, 5]

best_f1 = 0
best_model = None
best_preds = None
best_config = {}

for lr in learning_rates:
    for bs in batch_sizes:
        for ep in epochs:
            print(f"\n🚀 Training config: lr={lr}, batch_size={bs}, epochs={ep}")
            model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_names))

            args = TrainingArguments(
                output_dir="./finetuned_bert_resume",
                evaluation_strategy="epoch",     # ✅ 每个 epoch 评估
                save_strategy="epoch",           # ✅ 每个 epoch 保存模型
                learning_rate=lr,
                per_device_train_batch_size=bs,
                per_device_eval_batch_size=bs,
                num_train_epochs=ep,
                weight_decay=0.01,
                logging_dir="./logs",
                logging_steps=10,
                load_best_model_at_end=True,     # ✅ 开启 best model 保存
                metric_for_best_model="f1"
            )

            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
            )

            trainer.train()
            val_result = trainer.predict(val_ds)
            val_preds = np.argmax(val_result.predictions, axis=1)
            f1 = f1_score(val_result.label_ids, val_preds, average='weighted')
            print(f"🔍 Validation F1 = {f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_config = {"lr": lr, "batch_size": bs, "epochs": ep}

# === Final Evaluation on Test Set ===
print(f"\n✅ Best Config: {best_config} | Best Validation F1: {best_f1:.4f}")

trainer = Trainer(
    model=best_model,
    tokenizer=tokenizer
)
test_result = trainer.predict(test_ds)
y_pred = np.argmax(test_result.predictions, axis=1)
y_true = test_result.label_ids

print("\n📋 Final Classification Report:")
print(classification_report(y_true, y_pred, target_names=label_names))

# === Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_names, yticklabels=label_names)
plt.xlabel("Predicted Label", fontsize=14)
plt.ylabel("True Label", fontsize=14)
plt.title("Confusion Matrix (Fine-tuned BERT+MLP)", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("confusion_matrix_finetuned_bert.png", dpi=300)
plt.show()