"""
Fine-tune EfficientNet-B2 to classify genuine vs forged CEDAR signatures.

    python src/train.py --fold 0                 # one fold of 5-fold writer-independent CV
    for k in 0 1 2 3 4; do python src/train.py --fold $k; done

Saves runs/<name>/best.pt, history.csv and metrics.json.
The decision threshold (on the logit) is picked on the validation writers at
the EER point and then applied unchanged to the test writers.

Regularisation against memorising the ~36 training writers: augmentation
before the DIP pipeline (affine + stroke-width jitter), label smoothing,
dropout 0.5, weight decay 0.05, low backbone LR, cosine schedule.
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
from torch.utils.data import DataLoader

from common import (ROOT, SignatureSet, build_model, eer_threshold, load_gray,
                    metrics, predict, writer_split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="binary", choices=["binary", "canon", "raw"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--smooth", type=float, default=0.1, help="label smoothing")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    name = a.name or f"{a.mode}_fold{a.fold}"
    out = os.path.join(ROOT, "runs", name)
    os.makedirs(out, exist_ok=True)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta, tr, va, te = writer_split(fold=a.fold)
    G, y = load_gray(a.mode), meta.label.values
    print(f"[{name}] train {len(tr)} / val {len(va)} / test {len(te)} images, "
          f"test writers {sorted(set(meta.writer.values[te].tolist()))}")

    def loader(idx, train):
        ds = SignatureSet(np.asarray(G[idx]), y[idx], train)
        return DataLoader(ds, batch_size=a.bs, shuffle=train, drop_last=train,
                          num_workers=a.workers if train else 0,
                          persistent_workers=train and a.workers > 0, pin_memory=True)

    dl_tr, dl_va, dl_te = loader(tr, True), loader(va, False), loader(te, False)

    model = build_model().to(device)
    head = list(model.classifier.parameters())
    head_ids = {id(p) for p in head}
    body = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([{"params": body, "lr": a.lr},
                             {"params": head, "lr": a.lr * 10}], weight_decay=a.wd)
    steps = a.epochs * len(dl_tr)
    warm = len(dl_tr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (steps - warm))))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc, hist = -1.0, []
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, run = time.time(), 0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(xb).squeeze(1), yb * (1 - a.smooth) + a.smooth / 2)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item() * len(yb)

        s, yv = predict(model, dl_va, device)
        thr, _ = eer_threshold(yv, s)
        m = metrics(yv, s, thr)
        hist.append({"epoch": ep, "train_loss": run / (len(dl_tr) * a.bs), **{f"val_{k}": v for k, v in m.items()}})
        flag = ""
        if m["auc"] > best_auc:
            best_auc, flag = m["auc"], "  *"
            torch.save({"model": model.state_dict(), "thr": thr, "mode": a.mode}, os.path.join(out, "best.pt"))
        print(f"ep {ep:2d}  loss {hist[-1]['train_loss']:.4f}  val acc {m['acc']:.3f}  auc {m['auc']:.4f}  "
              f"eer {m['eer']:.3f}  ({time.time() - t0:.0f}s){flag}", flush=True)

    pd.DataFrame(hist).to_csv(os.path.join(out, "history.csv"), index=False)

    ck = torch.load(os.path.join(out, "best.pt"), map_location=device)
    model.load_state_dict(ck["model"])
    s, yt = predict(model, dl_te, device)
    res = {"name": name, "mode": a.mode, "fold": a.fold, "best_val_auc": best_auc,
           "test": metrics(yt, s, ck["thr"]),
           "test_writers": sorted(set(int(w) for w in meta.writer.values[te]))}
    np.save(os.path.join(out, "test_scores.npy"), np.stack([te, yt, s]))   # index, label, logit
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    t = res["test"]
    print(f"TEST [{name}]  acc {t['acc']:.4f}  auc {t['auc']:.4f}  eer {t['eer']:.4f}  "
          f"FAR {t['far']:.4f}  FRR {t['frr']:.4f}  (logit thr {t['thr']:.3f} from val)")


if __name__ == "__main__":
    main()
