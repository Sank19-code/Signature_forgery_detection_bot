"""
Reference-based signature verification with a Siamese EfficientNet-B2.

Question answered: "Given R genuine reference signatures of the claimed writer,
is this questioned signature a forgery?" - the way a bank compares a cheque
against the specimen on file. This generalises to writers never seen in
training, unlike judging a single signature on its own (train.py).

    python src/siamese.py --fold 0
    for k in 0 1 2 3 4; do python src/siamese.py --fold $k; done

Model:  shared backbone f(.) -> 1408-d pooled features -> 256-d embedding
        pair features [|e_r - e_q|, e_r * e_q] -> MLP -> logit(questioned is forged)
Train:  pairs drawn on the fly from training writers only:
        (genuine_w, other genuine_w) -> 0,   (genuine_w, skilled forgery of w) -> 1
Test:   per unseen writer, references = genuine #1..R; questioned = the other
        24-R genuine + all 24 forgeries; score = mean logit over the R references.
        Threshold chosen on validation writers (EER point), applied to test writers.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from common import (MEAN, ROOT, STD, augment, build_model, eer_threshold, load_gray,
                    metrics, writer_split)
from preprocess import stages_to_channels


def to_input(canon, rng=None):
    if rng is not None:
        canon = augment(canon, rng)
    return (stages_to_channels(canon).astype(np.float32) / 255.0 - MEAN) / STD


class PairTrainSet(Dataset):
    """Random balanced pairs from the given writers; regenerated every epoch."""

    def __init__(self, G, meta, idx, n_pairs):
        self.G, self.n = G, n_pairs
        m = meta.iloc[idx]
        self.gen = {w: g.index.values[g.label.values == 0] for w, g in m.groupby("writer")}
        self.forg = {w: g.index.values[g.label.values == 1] for w, g in m.groupby("writer")}
        self.writers = list(self.gen)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng()
        w = self.writers[rng.integers(len(self.writers))]
        ref, other = rng.choice(self.gen[w], 2, replace=False)
        label = int(rng.random() < 0.5)
        q = rng.choice(self.forg[w]) if label else other
        return (torch.from_numpy(to_input(np.asarray(self.G[ref]), rng)),
                torch.from_numpy(to_input(np.asarray(self.G[q]), rng)),
                torch.tensor(float(label)))


class SingleSet(Dataset):
    def __init__(self, G, idx):
        self.G, self.idx = G, idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return torch.from_numpy(to_input(np.asarray(self.G[self.idx[i]])))


class Siamese(nn.Module):
    def __init__(self, pretrained=True, emb=256):
        super().__init__()
        b = build_model(pretrained)
        self.features, self.pool = b.features, b.avgpool
        self.proj = nn.Sequential(nn.Dropout(0.3), nn.Linear(1408, emb))
        self.head = nn.Sequential(nn.Linear(2 * emb, 128), nn.ReLU(inplace=True),
                                  nn.Dropout(0.3), nn.Linear(128, 1))

    def embed(self, x):
        return self.proj(torch.flatten(self.pool(self.features(x)), 1))

    def compare(self, er, eq):
        return self.head(torch.cat([(er - eq).abs(), er * eq], 1)).squeeze(1)

    def forward(self, xr, xq):
        return self.compare(self.embed(xr), self.embed(xq))


@torch.no_grad()
def score_writers(model, G, meta, idx, n_ref, device):
    """Embed every image once, then score each questioned signature against its
    writer's first n_ref genuine signatures. Returns (meta indices, labels, logits)."""
    model.eval()
    dl = DataLoader(SingleSet(G, idx), batch_size=32)
    embs = []
    for x in dl:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            embs.append(model.embed(x.to(device)).float())
    E = dict(zip(idx, torch.cat(embs)))
    out_i, out_y, out_s = [], [], []
    m = meta.iloc[idx]
    for w, g in m.groupby("writer"):
        g = g.sort_values(["label", "index"])
        refs = g.index.values[(g.label.values == 0) & (g["index"].values <= n_ref)]
        quest = g.index.values[~((g.label.values == 0) & (g["index"].values <= n_ref))]
        er = torch.stack([E[r] for r in refs])                          # (R, d)
        for q in quest:
            eq = E[q].expand_as(er)
            out_s.append(model.compare(er, eq).float().mean().item())
            out_i.append(q)
            out_y.append(int(meta.label[q]))
    return np.array(out_i), np.array(out_y), np.array(out_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--pairs", type=int, default=3000, help="training pairs per epoch")
    ap.add_argument("--bs", type=int, default=12, help="pairs per batch")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--n_ref", type=int, default=5, help="reference genuine signatures per writer")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    name = f"siamese_fold{a.fold}"
    out = os.path.join(ROOT, "runs", name)
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta, tr, va, te = writer_split(fold=a.fold)
    G = load_gray("binary")
    print(f"[{name}] writers train {meta.writer[tr].nunique()} / val {meta.writer[va].nunique()} / "
          f"test {meta.writer[te].nunique()}  test writers {sorted(set(meta.writer[te]))}")

    dl_tr = DataLoader(PairTrainSet(G, meta, tr, a.pairs), batch_size=a.bs, shuffle=True,
                       drop_last=True, num_workers=a.workers, persistent_workers=a.workers > 0,
                       pin_memory=True)
    model = Siamese().to(device)
    body = list(model.features.parameters())
    rest = list(model.proj.parameters()) + list(model.head.parameters())
    opt = torch.optim.AdamW([{"params": body, "lr": a.lr}, {"params": rest, "lr": a.lr * 10}],
                            weight_decay=a.wd)
    steps, warm = a.epochs * len(dl_tr), len(dl_tr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (steps - warm))))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc, hist = -1.0, []
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, run = time.time(), 0.0
        for xr, xq, yb in dl_tr:
            xr, xq, yb = xr.to(device), xq.to(device), yb.to(device)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(xr, xq), yb * 0.9 + 0.05)          # label smoothing 0.1
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()

        _, yv, sv = score_writers(model, G, meta, va, a.n_ref, device)
        thr, _ = eer_threshold(yv, sv)
        m = metrics(yv, sv, thr)
        hist.append({"epoch": ep, "train_loss": run / len(dl_tr), **{f"val_{k}": v for k, v in m.items()}})
        flag = ""
        if m["auc"] > best_auc:
            best_auc, flag = m["auc"], "  *"
            torch.save({"model": model.state_dict(), "thr": thr, "n_ref": a.n_ref}, os.path.join(out, "best.pt"))
        print(f"ep {ep:2d}  loss {hist[-1]['train_loss']:.4f}  val acc {m['acc']:.3f}  auc {m['auc']:.4f}  "
              f"eer {m['eer']:.3f}  ({time.time() - t0:.0f}s){flag}", flush=True)

    pd.DataFrame(hist).to_csv(os.path.join(out, "history.csv"), index=False)
    ck = torch.load(os.path.join(out, "best.pt"), map_location=device)
    model.load_state_dict(ck["model"])
    ti, yt, st = score_writers(model, G, meta, te, a.n_ref, device)
    res = {"name": name, "mode": "siamese", "fold": a.fold, "n_ref": a.n_ref, "best_val_auc": best_auc,
           "test": metrics(yt, st, ck["thr"]), "test_writers": sorted(int(w) for w in set(meta.writer[te]))}
    np.save(os.path.join(out, "test_scores.npy"), np.stack([ti, yt, st]))
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    t = res["test"]
    print(f"TEST [{name}]  acc {t['acc']:.4f}  auc {t['auc']:.4f}  eer {t['eer']:.4f}  "
          f"FAR {t['far']:.4f}  FRR {t['frr']:.4f}  (logit thr {t['thr']:.3f} from val)")


if __name__ == "__main__":
    main()
