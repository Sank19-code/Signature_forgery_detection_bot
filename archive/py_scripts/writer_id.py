"""
Learn writer-discriminative features (SigNet-style), for the writer-dependent SVM.

    python src/writer_id.py --fold 0                # SigNet:   writer-ID head only
    python src/writer_id.py --fold 0 --lam 1.0      # SigNet-F: + genuine/forged head

Trains EfficientNet-B2 to recognise *which* of the 36 training writers wrote a
genuine signature. Validation writers are deliberately left out: they pick the
SVM decision threshold, and that only transfers to test writers if the feature
extractor has never seen them either. The resulting features describe how a
person signs, which is what comparing a questioned signature to specimens needs.
With --lam > 0 a second head also learns genuine vs skilled forgery on the same
writers (loss = CE_writer(genuine only) + lam * BCE_forged(all)).
Test writers are never seen. Saves runs/<name>/best.pt with the backbone weights
(last epoch; no model selection, so the test writers stay untouched).

Then:  python src/wd_svm.py --ckpt writerid      (or writeridF)
"""
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import ROOT, SignatureSet, build_model, load_gray, writer_split


class WriterNet(nn.Module):
    def __init__(self, n_writers):
        super().__init__()
        b = build_model()
        self.features, self.pool = b.features, b.avgpool
        self.drop = nn.Dropout(0.5)
        self.writer_head = nn.Linear(1408, n_writers)
        self.forg_head = nn.Linear(1408, 1)

    def forward(self, x):
        f = self.drop(torch.flatten(self.pool(self.features(x)), 1))
        return self.writer_head(f), self.forg_head(f).squeeze(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--lam", type=float, default=0.0, help="weight of the forgery head (0 = SigNet)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    a = ap.parse_args()
    name = f"writerid{'F' if a.lam > 0 else ''}_fold{a.fold}"
    out = os.path.join(ROOT, "runs", name)
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta, tr, va, te = writer_split(fold=a.fold)
    idx = tr
    if a.lam == 0:
        idx = idx[meta.label.values[idx] == 0]                 # writer ID uses genuine only
    writers = sorted(set(meta.writer.values[idx]))
    wmap = {w: i for i, w in enumerate(writers)}
    wid = np.array([wmap[w] for w in meta.writer.values[idx]])
    forged = meta.label.values[idx]
    print(f"[{name}] {len(idx)} images from {len(writers)} writers; test writers {sorted(set(meta.writer[te]))}")

    G = load_gray("binary")
    get = SignatureSet(np.asarray(G[idx]), forged, True).__getitem__
    dl = DataLoader(list(range(len(idx))), batch_size=a.bs, shuffle=True, drop_last=True,
                    collate_fn=lambda b: (torch.stack([get(i)[0] for i in b]),
                                          torch.tensor(wid[b]), torch.tensor(forged[b], dtype=torch.float32)))

    model = WriterNet(len(writers)).to(device)
    heads = list(model.writer_head.parameters()) + list(model.forg_head.parameters())
    opt = torch.optim.AdamW([{"params": model.features.parameters(), "lr": a.lr},
                             {"params": heads, "lr": a.lr * 10}], weight_decay=a.wd)
    steps, warm = a.epochs * len(dl), len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (steps - warm))))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    ce, bce = nn.CrossEntropyLoss(label_smoothing=0.1), nn.BCEWithLogitsLoss()

    for ep in range(1, a.epochs + 1):
        model.train()
        t0, run, correct, seen = time.time(), 0.0, 0, 0
        for xb, wb, fb in dl:
            xb, wb, fb = xb.to(device), wb.to(device), fb.to(device)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                wl, fl = model(xb)
                gen = fb == 0
                loss = ce(wl[gen], wb[gen]) if gen.any() else wl.sum() * 0
                if a.lam > 0:
                    loss = loss + a.lam * bce(fl, fb * 0.9 + 0.05)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
            correct += (wl[gen].argmax(1) == wb[gen]).sum().item()
            seen += gen.sum().item()
        print(f"ep {ep:2d}  loss {run / len(dl):.4f}  train writer-ID acc {correct / max(seen, 1):.3f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    torch.save({"features": model.features.state_dict()}, os.path.join(out, "best.pt"))
    print("saved", os.path.join(out, "best.pt"))


if __name__ == "__main__":
    main()
