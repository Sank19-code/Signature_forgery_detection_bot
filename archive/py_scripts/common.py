"""Shared bits: data split, dataset, model, metrics."""
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from torchvision.models import EfficientNet_B2_Weights, efficientnet_b2

from preprocess import stages_to_channels

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)


# ------------------------------------------------------------------ data split

def writer_split(fold=0, n_folds=5, n_val=8, seed=42):
    """Writer-independent k-fold split: no writer is in more than one of train/val/test.

    Writers are shuffled once (seed), cut into n_folds chunks; chunk `fold` is
    the test set and n_val writers of the rest are held out for validation
    (model selection + decision threshold).
    """
    meta = pd.read_csv(os.path.join(DATA, "meta.csv"))
    writers = np.array(sorted(meta.writer.unique()))
    np.random.default_rng(seed).shuffle(writers)
    chunks = np.array_split(writers, n_folds)
    test_w = chunks[fold]
    rest = np.concatenate([c for i, c in enumerate(chunks) if i != fold])
    val_w, train_w = rest[:n_val], rest[n_val:]
    idx = lambda ws: np.where(meta.writer.isin(ws))[0]
    return meta, idx(train_w), idx(val_w), idx(test_w)


# --------------------------------------------------------------------- dataset

def augment(canon, rng):
    """Random geometry + stroke-width jitter on the canonical (white-bg) signature."""
    h, w = canon.shape
    ang, sc = rng.uniform(-6, 6), rng.uniform(0.82, 1.0)       # shrink only: wide signatures fill the canvas
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    M[0, 1] += np.tan(np.radians(rng.uniform(-8, 8)))        # shear
    M[:, 2] += (rng.uniform(-10, 10), rng.uniform(-8, 8))     # translate
    img = cv2.warpAffine(canon, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
    r = rng.random()
    if r < 0.25:                                              # thicker strokes
        img = cv2.erode(img, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
    elif r < 0.5:                                             # thinner strokes
        img = cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
    return np.where(img < 128, 0, 255).astype(np.uint8)       # back to binary


class SignatureSet(torch.utils.data.Dataset):
    """Canonical image -> (optional augmentation) -> 4-stage pipeline -> 3-ch tensor.

    Channels: 0 median-filtered, 1 high-boost, 2 Sobel.
    """

    def __init__(self, G, labels, train):
        self.G, self.y, self.train = G, labels.astype(np.float32), train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        canon = np.asarray(self.G[i])
        if self.train:
            canon = augment(canon, np.random.default_rng())
        x = (stages_to_channels(canon).astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(x), torch.tensor(self.y[i])


def load_gray(mode):
    return np.load(os.path.join(DATA, f"cache_{mode}_gray.npy"), mmap_mode="r")


# ----------------------------------------------------------------------- model

def build_model(pretrained=True, dropout=0.5):
    m = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None)
    m.classifier[0].p = dropout
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)   # logit: forged
    return m


@torch.no_grad()
def predict(model, loader, device):
    """Returns raw logits (not probabilities: sigmoid saturates to exactly 1.0
    for confident predictions, which breaks threshold selection)."""
    model.eval()
    zs, ys = [], []
    for x, y in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            zs.append(model(x.to(device)).float().squeeze(1).cpu())
        ys.append(y)
    return torch.cat(zs).numpy(), torch.cat(ys).numpy()


# --------------------------------------------------------------------- metrics

def eer_threshold(y, s):
    """Score threshold where FAR (forgeries accepted) == FRR (genuine rejected)."""
    fpr, tpr, thr = roc_curve(y, s)                  # positive class = forged
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float(thr[i]), float((fpr[i] + fnr[i]) / 2)


def metrics(y, s, thr):
    pred = s >= thr
    forged, genuine = y == 1, y == 0
    far = float((~pred[forged]).mean())              # forged called genuine  -> fraud gets through
    frr = float(pred[genuine].mean())                # genuine called forged  -> real customer blocked
    _, eer = eer_threshold(y, s)
    return {"acc": float((pred == y).mean()), "auc": float(roc_auc_score(y, s)),
            "far": far, "frr": frr, "eer": eer, "thr": thr}
