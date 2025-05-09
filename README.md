# Resume Classification with Enhanced BERT Pipeline

This repository contains our final project for CSCI467, which focuses on resume classification using a variety of models, including traditional non-transformer-based machine learning approaches and transformer-based methods.

## Project Structure

- `tfidf_logreg.py`: TF-IDF + Logistic Regression (baseline)
- `bert_model.py`: BERT + XGBoost classifier (baseline)
- `bert_mlp.py`: Original BERT + MLP head (not fine-tuned, baseline)
- `bert_mlp_finetune.py`: Fine-tuned BERT + MLP (baseline)
- `bert_swiglu.py`: BERT + Residual SwiGLU block
- `combine.py`: BERT + Attention Pooling + Residual SwiGLU block (final method)

## Dependencies

Install the required Python libraries:

```bash
pip install pandas numpy torch scikit-learn matplotlib seaborn datasets transformers
```

## Data Setup

Download the dataset from Kaggle:
https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset

Update the file path in each script accordingly, for example:
```python
df = pd.read_csv("/your/path/to/Resume.csv")
```

## Running the Code
To run a specific experiment, just execute the corresponding script:
```bash
python combine.py  # Final model: BERT + Attention Pooling + SwiGLU
python bert_mlp_finetune.py  # Finetuned BERT + MLP baseline
```

## Output
Each script will print classification metrics (Precision, Recall, F1) and save a confusion matrix figure.

## Notes
- The final model uses the best hyperparameters from earlier grid search (learning rate = `2e-5`, batch size = `16`, epochs = `5`).
- For evaluation, we use **weighted F1-score** due to class imbalance.