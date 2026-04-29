#!/usr/bin/env python
"""Improved LR detector: char n-grams + function-word ratios + stylometric features."""
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"


def make_features(df):
    """Build a richer stylometric feature matrix."""
    rows = []
    for text in df["text"]:
        lowered = text.lower()
        words = re.findall(r"\b[a-zA-Z]+\b", lowered)
        wc = max(1, len(words))
        sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        sent_lens = [len(s.split()) for s in sents] or [wc]
        mean_l = sum(sent_lens) / len(sent_lens)
        var_l = sum((l - mean_l) ** 2 for l in sent_lens) / len(sent_lens)
        paras = [p for p in text.split("\n\n") if p.strip()]
        para_lens = [len(p.split()) for p in paras] or [wc]
        p_mean = sum(para_lens) / len(para_lens)
        p_cv = (sum((l - p_mean) ** 2 for l in para_lens) / len(para_lens)) ** 0.5 / max(1e-6, p_mean)

        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        ttr = len(freq) / wc
        # hapax ratio: share of words appearing exactly once
        hapax = sum(1 for c in freq.values() if c == 1) / max(1, len(freq))
        # Zipf-ish: top-5 freq share
        top5 = sum(sorted(freq.values(), reverse=True)[:5]) / wc
        # function-word richness
        func = {"the", "a", "an", "of", "in", "to", "and", "or", "for", "is", "are",
                "was", "were", "with", "on", "by", "at", "as", "it", "this", "that",
                "which", "we", "our", "their", "its", "from", "into"}
        func_ratio = sum(1 for w in words if w in func) / wc
        # uppercase ratio & digit ratio
        cap_ratio = sum(1 for ch in text if ch.isupper()) / max(1, len(text))
        digit_ratio = sum(1 for ch in text if ch.isdigit()) / max(1, len(text))
        # avg word length
        avg_wlen = sum(len(w) for w in words) / wc
        # sentence length CV (burstiness)
        burst = (var_l ** 0.5) / max(1e-6, mean_l)
        # comma/semicolon density
        punct_density = (text.count(",") + text.count(";") + text.count(":")) / max(1, wc)
        # quote density
        quote_density = (text.count('"') + text.count("\u201c")) / max(1, wc)
        rows.append([ttr, hapax, top5, func_ratio, cap_ratio, digit_ratio,
                     avg_wlen, burst, punct_density, quote_density, p_cv])
    return np.array(rows, dtype=np.float32)


def main():
    df = pd.read_csv(DATA / "ai_training.csv").dropna(subset=["text"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"rows={len(df)} human={int(df.label.sum())}")

    X = make_features(df)
    y = df["label"].values

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Stylometric features + char TF-IDF stacked
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                            max_features=40000, sublinear_tf=True)
    X_tfidf = tfidf.fit_transform(df["text"])

    styl = Pipeline([("sc", StandardScaler())])
    X_styl = styl.fit_transform(X)

    from scipy.sparse import hstack

    X_full = hstack([X_styl, X_tfidf]).tocsr()

    clf = Pipeline([("lr", LogisticRegression(max_iter=2000, C=0.5, solver="lbfgs"))])
    cv = cross_val_score(clf, X_full, y, cv=5, scoring="f1")
    print(f"Combined 5-fold CV F1: {cv.mean():.4f} (+/- {cv.std():.4f})")

    tr_i, te_i = train_test_split(range(len(y)), test_size=0.2, random_state=42, stratify=y)
    clf.fit(X_full[tr_i], y[tr_i])
    pred = clf.predict(X_full[te_i])
    print(classification_report(y[te_i], pred, target_names=["ai", "human"]))

    MODELS.mkdir(exist_ok=True)
    import joblib
    joblib.dump((styl, tfidf, clf), MODELS / "ai_detector_v3.joblib")
    with open(MODELS / "ai_detector_v3_metrics.json", "w") as fh:
        json.dump({"engine": "stylometric_char_lr", "cv_f1": round(float(cv.mean()), 4)}, fh, indent=2)
    print("saved models/ai_detector_v3.joblib")


if __name__ == "__main__":
    main()
