import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

resumeData=pd.read_csv("/Users/xuwei/Desktop/spring2025/CSCI467/CSCI467_deepfake/data/1/Resume/Resume.csv")

rData=resumeData.drop("Resume_html", axis=1)
rData["Category"]=rData["Category"].astype("category")
rData["Category"]=rData["Category"].cat.codes
x = rData["Resume_str"].tolist()
y = rData["Category"].astype(int).values
xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.4, random_state=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased").to(device)
bert.eval()

def encode_texts(texts, batch_size=16, max_len=128):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch,
                        padding=True,
                        truncation=True,
                        max_length=max_len,
                        return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            out = bert(input_ids, attention_mask=attention_mask)
        cls_emb = out.last_hidden_state[:, 0, :].cpu().numpy()
        all_embs.append(cls_emb)
    return np.vstack(all_embs)

max_perf = 0
best_b = 0
best_n = 0

b = 8
n = 100
print("Encoding training set...")
xT = encode_texts(xtr,b)
print("Encoding test set...")
xE = encode_texts(xte,b)


model = XGBClassifier(n_estimators=n, use_label_encoder=False, eval_metric="logloss")
model.fit(xT, ytr)


y_pred = model.predict(xE)
f1_weighted = f1_score(yte, y_pred, average='weighted')
# batch_sizes_list = [8, 16, 32,64]
# n_estimators_list = [100, 500, 1000, 5000]
# for b in batch_sizes_list:
#     for n in n_estimators_list:
#         print("Encoding training set...")
#         xT = encode_texts(xtr,b)
#         print("Encoding test set...")
#         xE = encode_texts(xte,b)

    
#         model = XGBClassifier(n_estimators=n, use_label_encoder=False, eval_metric="logloss")
#         model.fit(xT, ytr)

    
#         temp_pred = model.predict(xE)
#         f1_weighted = f1_score(yte, temp_pred, average='weighted')
#         if f1_weighted >max_perf:
#             max_perf = f1_weighted
#             best_b = b
#             best_n = n
#             best_model = model
# print(f"best batch size: {best_b}")
# print(f"best n_estimators: {best_n}")
# y_pred = best_model.predict(xE)

original_categories = resumeData["Category"].astype("category")
category_names = original_categories.cat.categories.tolist()

# Confusion Matrix
cm = confusion_matrix(yte, y_pred)

plt.figure(figsize=(14, 12))
sns.set(font_scale=1.0)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=category_names,
            yticklabels=category_names)

plt.xlabel("Predicted Label", fontsize=14)
plt.ylabel("True Label", fontsize=14)
plt.title("Confusion Matrix Bert+xgboost", fontsize=16)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig("confusion_matrix_tfidf_logreg.png", dpi=300)
plt.show()
