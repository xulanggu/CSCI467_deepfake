import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, f1_score

resumeData = pd.read_csv("/Users/xuwei/Desktop/spring2025/CSCI467/CSCI467_deepfake/data/1/Resume/Resume.csv")
rData = resumeData.drop("Resume_html", axis=1)
rData["Category"] = rData["Category"].astype("category").cat.codes
x = rData["Resume_str"].tolist()
y = rData["Category"].values

xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.4, random_state=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased").to(device)
bert.eval()

def encode_texts(texts, batch_size=16, max_len=128):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            out = bert(input_ids, attention_mask=attention_mask)
        cls_emb = out.last_hidden_state[:, 0, :].cpu().numpy()
        all_embs.append(cls_emb)
    return np.vstack(all_embs)

# batch_sizes_list = [8, 16, 32, 64]
# hidden_layer_sizes_list = [(50,), (100,), (100, 50), (100, 100)]

batch_sizes_list = [64]
hidden_layer_sizes_list = [(100,)]
max_perf = 0
best_bs = None
best_hls = None
best_model = None

for bs in batch_sizes_list:
    print(f"\nEncoding with batch size = {bs}")
    xT = encode_texts(xtr, batch_size=bs)
    xE = encode_texts(xte, batch_size=bs)

    for hls in hidden_layer_sizes_list:
        print(f"  Training MLP with hidden layers = {hls}")
        mlp = MLPClassifier(hidden_layer_sizes=hls,
                            batch_size=bs,
                            max_iter=200,
                            solver='adam',
                            random_state=42)
        mlp.fit(xT, ytr)

        y_pred = mlp.predict(xE)
        f1_w = f1_score(yte, y_pred, average='weighted')
        print(f"    Weighted F1 = {f1_w:.4f}")

        if f1_w > max_perf:
            max_perf = f1_w
            best_bs = bs
            best_hls = hls
            best_model = mlp

print(f"\nBest config: batch_size={best_bs}, hidden_layers={best_hls}, F1={max_perf:.4f}")

y_pred = best_model.predict(encode_texts(xte, batch_size=best_bs))
print("\nClassification Report for Best MLP:\n")
print(classification_report(yte, y_pred))