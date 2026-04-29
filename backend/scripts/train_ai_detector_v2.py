#!/usr/bin/env python
"""Train and benchmark AI-detection models (v2).

Two approaches are trained and cross-validated head-to-head on the same data:
1. TF-IDF + logistic regression (fast, tiny artifact)
2. Fine-tuned DistilBERT transformer (higher capacity)

The winner (best CV F1) is exported:
- LR winner -> models/ai_detector.joblib (+ tfidf vectorizer)
- Transformer winner -> models/ai_detector_transformer (HF format)

Usage:
  python scripts/train_ai_detector_v2.py            # benchmark both, ship winner
  python scripts/train_ai_detector_v2.py --engine lr
  python scripts/train_ai_detector_v2.py --engine transformer
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"


def load_data():
    df = pd.read_csv(DATA / "ai_training.csv")
    df = df.dropna(subset=["text"]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Dataset: {len(df)} rows (human={int(df.label.sum())}, ai={int((1 - df.label).sum())})")
    return df


# ---------------- Approach 1: TF-IDF + Logistic Regression ----------------

def train_lr(df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import classification_report
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=60000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")),
    ])
    scores = cross_val_score(pipeline, df["text"], df["label"], cv=5, scoring="f1")
    print(f"LR 5-fold CV F1: {scores.mean():.4f} (+/- {scores.std():.4f})")
    tr, te = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    pipeline.fit(tr["text"], tr["label"])
    pred = pipeline.predict(te["text"])
    print(classification_report(te["label"], pred, target_names=["ai", "human"]))
    return pipeline, float(scores.mean())


def save_lr(pipe, f1):
    MODELS.mkdir(exist_ok=True)
    import joblib
    joblib.dump(pipe, MODELS / "ai_detector.joblib")
    with open(MODELS / "ai_detector_metrics.json", "w") as fh:
        json.dump({"engine": "logistic_regression", "cv_f1": round(f1, 4),
                   "trained_on": "arxiv abstracts + controlled AI variants"}, fh, indent=2)
    print("Saved models/ai_detector.joblib")


# ---------------- Approach 2: DistilBERT fine-tune ----------------

def train_transformer(df, epochs=2):
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
    from datasets import Dataset
    from sklearn.model_selection import train_test_split
    import evaluate
    import numpy as np

    tr, te = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256, padding=False)

    train_ds = Dataset.from_pandas(tr).map(tokenize, batched=True)
    val_ds = Dataset.from_pandas(te).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=1)
        return {"f1": float(metric.compute(predictions=preds, references=eval_pred.label_ids)["f1"])}

    from transformers import DataCollatorWithPadding

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    args = TrainingArguments(
        output_dir=str(ROOT / "models" / "_hf_train"),
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=epochs,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=val_ds, compute_metrics=compute_metrics,
                      data_collator=collator)
    start = time.time()
    trainer.train()
    print(f"Fine-tuning took {time.time() - start:.0f}s")
    result = trainer.evaluate()
    f1 = result["eval_f1"]
    print(f"Transformer eval F1: {f1:.4f}")

    MODELS.mkdir(exist_ok=True)
    out = MODELS / "ai_detector_transformer"
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    with open(MODELS / "ai_detector_transformer_metrics.json", "w") as fh:
        json.dump({"engine": "distilbert_finetuned", "eval_f1": round(f1, 4)}, fh, indent=2)
    print("Saved models/ai_detector_transformer")
    return f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["lr", "transformer", "auto"], default="auto")
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    df = load_data()

    lr_f1 = None
    if args.engine in ("lr", "auto"):
        pipe, lr_f1 = train_lr(df)
        save_lr(pipe, lr_f1)
    if args.engine == "lr":
        return

    t_f1 = train_transformer(df, epochs=args.epochs)
    if args.engine == "transformer":
        return

    print(f"\n=== BENCHMARK: LR F1={lr_f1:.4f} vs Transformer F1={t_f1:.4f} ===")
    winner = "logistic_regression" if lr_f1 >= t_f1 else "transformer"
    with open(MODELS / "ai_detector_winner.json", "w") as fh:
        json.dump({"winner": winner, "lr_f1": round(lr_f1, 4), "transformer_f1": round(t_f1, 4)}, fh, indent=2)
    print(f"WINNER: {winner} -> service will auto-select at runtime")


if __name__ == "__main__":
    main()
