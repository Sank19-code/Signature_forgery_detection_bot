"""
Writer-dependent verification on top of learned features (SigNet-style protocol).

    python src/wd_svm.py --ckpt imagenet               # ImageNet features, no fine-tuning
    python src/wd_svm.py --ckpt imagenet --mode skel   # stroke-width-normalised input (DPI leak check)
    python src/wd_svm.py --ckpt binary                 # features of the single-image classifier (train.py)
    python src/wd_svm.py --ckpt writerid               # SigNet-style writer-ID features (writer_id.py)
    python src/wd_svm.py --ckpt writeridF              # + forgery head
Fine-tuned checkpoints are per fold (runs/<ckpt>_fold<k>/best.pt), so test writers are never seen.

For each unseen test writer:
  positives = their first n_ref genuine signatures (the "specimens on file")
  negatives = genuine signatures of the training writers ("random forgeries")
              (validation writers are kept out so they can pick the threshold)
  -> train an SVM on CNN embeddings, then score the writer's remaining genuine
     signatures and all 24 skilled forgeries. No skilled forgery is used for
     training, which matches a bank that only has genuine specimens.

Reports AUC / EER over all test writers, accuracy / FAR / FRR at one global
threshold chosen on the validation writers, and the mean per-writer EER (each
writer gets their own threshold, the usual writer-dependent figure).
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from torch.utils.data import DataLoader

from common import ROOT, build_model, eer_threshold, load_gray, metrics, writer_split
from siamese import SingleSet


@torch.no_grad()
def embed_all(model, G, device, bs=32):
    model.eval()
    feats = []
    for x in DataLoader(SingleSet(G, np.arange(len(G))), batch_size=bs):
        f = model.avgpool(model.features(x.to(device)))
        feats.append(torch.flatten(f, 1).float().cpu())
    return torch.cat(feats).numpy()


def fit_writer_svm(F, meta, w, neg, n_ref):
    """SVM for writer w: their first n_ref genuine (positives) vs other writers' genuine (negatives).
    Returns the fitted SVM and the meta indices of w's questioned signatures."""
    g = meta[meta.writer == w]
    is_ref = (g.label.values == 0) & (g["index"].values <= n_ref)
    pos = F[g.index.values[is_ref]]
    X = np.vstack([pos, neg])
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight={1: len(neg) / len(pos), 0: 1.0})
    return svm.fit(X, y), g.index.values[~is_ref]


def score_writers(F, meta, writers, neg, n_ref):
    ys, ss, per_writer = [], [], []
    for w in writers:
        svm, quest = fit_writer_svm(F, meta, w, neg, n_ref)
        s = -svm.decision_function(F[quest])            # high = looks NOT like this writer = forged
        yq = meta.label.values[quest]
        ys.append(yq)
        ss.append(s)
        per_writer.append(eer_threshold(yq, s)[1])
    return np.concatenate(ys), np.concatenate(ss), per_writer


def run_fold(F, meta, fold, n_ref):
    """Negatives = genuine signatures of the training writers only. The global threshold is
    the EER point on the validation writers, applied unchanged to the test writers."""
    _, tr, va, te = writer_split(fold=fold)
    neg = F[tr[meta.label.values[tr] == 0]]
    yv, sv, _ = score_writers(F, meta, sorted(set(meta.writer[va])), neg, n_ref)
    thr, _ = eer_threshold(yv, sv)
    yt, st, per_writer = score_writers(F, meta, sorted(set(meta.writer[te])), neg, n_ref)
    return {"fold": fold, **metrics(yt, st, thr), "mean_writer_eer": float(np.mean(per_writer))}


def features(tag, mode, fold, device):
    path = os.path.join(ROOT, "runs", "features", f"{tag}_{mode}" + ("" if tag == "imagenet" else f"_fold{fold}") + ".npy")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model = build_model(pretrained=(tag == "imagenet")).to(device)
        if tag != "imagenet":
            ck = torch.load(os.path.join(ROOT, "runs", f"{tag}_fold{fold}", "best.pt"), map_location=device)
            if "features" in ck:
                model.features.load_state_dict(ck["features"])
            else:
                model.load_state_dict(ck["model"])
        np.save(path, embed_all(model, load_gray(mode), device))
    F = np.load(path)
    return F / np.linalg.norm(F, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n_ref", type=int, default=5)
    ap.add_argument("--mode", default="binary", choices=["binary", "skel"])
    ap.add_argument("--ckpt", default="imagenet", help="imagenet | binary | writerid | writeridF")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    device = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    tag = a.ckpt
    meta, *_ = writer_split()

    rows = []
    for k in a.folds:
        r = run_fold(features(tag, a.mode, k, device), meta, k, a.n_ref)
        rows.append(r)
        print(f"  fold {k}: AUC {r['auc']:.4f}  EER {r['eer']:.4f}  acc {r['acc']:.4f}  FAR {r['far']:.4f}  "
              f"FRR {r['frr']:.4f}  per-writer EER {r['mean_writer_eer']:.4f}", flush=True)
    keys = ["auc", "eer", "acc", "far", "frr", "mean_writer_eer"]
    res = {"features": tag, "mode": a.mode, "n_ref": a.n_ref, "folds": rows,
           **{f"{k}_mean": float(np.mean([r[k] for r in rows])) for k in keys},
           **{f"{k}_std": float(np.std([r[k] for r in rows])) for k in keys}}
    os.makedirs(os.path.join(ROOT, "runs", "wdsvm"), exist_ok=True)
    with open(os.path.join(ROOT, "runs", "wdsvm", f"{tag}_{a.mode}_ref{a.n_ref}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"WD-SVM [{tag}, {a.mode}, {a.n_ref} refs]  " +
          "  ".join(f"{k} {res[k + '_mean']:.4f}+/-{res[k + '_std']:.4f}" for k in keys))


if __name__ == "__main__":
    main()
