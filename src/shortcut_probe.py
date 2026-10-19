"""
Shortcut probe: can a tiny model tell genuine from forged using only
intensity statistics (no stroke shape at all)?

Features per image: 32-bin histogram of the median-filtered channel over the
whole image, plus the fraction and mean intensity of non-white pixels.
Logistic regression, 5-fold cross-validation grouped by writer.

If this scores near 100%, the classes are separable by scanner/paper
artefacts and a CNN trained on that input would learn them instead of forgery cues.
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.join(os.path.dirname(__file__), "..", "data")


def features(X):
    ch = X[:, 0].reshape(len(X), -1)            # median-filtered stage
    hist = np.stack([np.histogram(r, bins=32, range=(0, 256))[0] for r in ch]) / ch.shape[1]
    nonwhite = ch < 250
    frac = nonwhite.mean(1, keepdims=True)
    mean_ink = np.array([r[m].mean() if m.any() else 255 for r, m in zip(ch, nonwhite)])[:, None]
    return np.hstack([hist, frac, mean_ink])


def main():
    meta = pd.read_csv(os.path.join(ROOT, "meta.csv"))
    y, groups = meta.label.values, meta.writer.values
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    for mode in ["raw", "canon", "binary"]:
        path = os.path.join(ROOT, f"cache_{mode}_X.npy")
        if not os.path.exists(path):
            continue
        F = features(np.load(path, mmap_mode="r"))
        acc = cross_val_score(clf, F, y, groups=groups, cv=GroupKFold(5))
        print(f"{mode:7s} intensity-only accuracy: {acc.mean():.3f} +/- {acc.std():.3f}")


if __name__ == "__main__":
    main()
