import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

resumeData=pd.read_csv("/Users/huangzitong/.cache/kagglehub/datasets/snehaanbhawal/resume-dataset/versions/1/Resume/Resume.csv")

rData=resumeData.drop("Resume_html", axis=1)
rData["Category"] = rData["Category"].astype("category").cat.codes
x = rData["Resume_str"].tolist()
y = rData["Category"].values

xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.4, random_state=2)



# Grid search parameters
max_features_list = [3000, 5000, 8000]
ngram_ranges = [(1, 1), (1, 2)]
C_values = [0.1, 1.0, 5.0]

best_f1 = 0
best_config = {}
best_model = None

for max_feat in max_features_list:
    for ngram in ngram_ranges:
        for C_val in C_values:
            print(f"\nTesting config: max_features={max_feat}, ngram_range={ngram}, C={C_val}")
            
            # TF-IDF Feature Extraction
            tfidf = TfidfVectorizer(max_features=max_feat, ngram_range=ngram, stop_words='english')
            xT = tfidf.fit_transform(xtr)
            xE = tfidf.transform(xte)

            # Train Logistic Regression
            clf = LogisticRegression(C=C_val, max_iter=500, solver='liblinear', random_state=42)
            clf.fit(xT, ytr)

            # Evaluate
            y_pred = clf.predict(xE)
            f1 = f1_score(yte, y_pred, average='weighted')
            print(f"Weighted F1 = {f1:.4f}")

            # Update best
            if f1 > best_f1:
                best_f1 = f1
                best_config = {
                    'max_features': max_feat,
                    'ngram_range': ngram,
                    'C': C_val
                }
                best_model = clf
                best_vectorizer = tfidf

# Final result
print("\n Best Configuration:")
print(best_config)
print(f" Best Weighted F1 Score: {best_f1:.4f}")

# Final report
xE_best = best_vectorizer.transform(xte)
y_pred_best = best_model.predict(xE_best)
print("\n Final Classification Report:")
print(classification_report(yte, y_pred_best))




# Confusion Matrix
original_categories = resumeData["Category"].astype("category")
category_names = original_categories.cat.categories.tolist()

# Confusion Matrix
cm = confusion_matrix(yte, y_pred_best)

plt.figure(figsize=(14, 12))
sns.set(font_scale=1.0)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=category_names,
            yticklabels=category_names)

plt.xlabel("Predicted Label", fontsize=14)
plt.ylabel("True Label", fontsize=14)
plt.title("Confusion Matrix (TF-IDF with Logistic Regression)", fontsize=16)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig("confusion_matrix_tfidf_logreg.png", dpi=300)
plt.show()