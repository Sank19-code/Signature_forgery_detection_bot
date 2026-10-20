"""Collect every 5-fold result into one table.

    python src/summarize.py

Sources: runs/<model>_fold<k>/metrics.json (classifier, Siamese) and
runs/wdsvm/*.json (writer-dependent SVM verifiers). Writes runs/summary.csv.
"""
import glob
import json
import os

import pandas as pd

from common import ROOT

KEYS = ["auc", "eer", "acc", "far", "frr"]
rows = []

per_fold = {}
for f in sorted(glob.glob(os.path.join(ROOT, "runs", "*_fold*", "metrics.json"))):
    r = json.load(open(f))
    per_fold.setdefault(r["mode"], []).append(r["test"])
for mode, folds in per_fold.items():
    if len(folds) != 5:
        continue
    df = pd.DataFrame(folds)
    label = {"binary": "single-image classifier", "siamese": "Siamese (5 refs)"}.get(mode, mode)
    rows.append({"model": label, **{f"{k}": f"{df[k].mean():.3f} +/- {df[k].std(ddof=0):.3f}" for k in KEYS},
                 "per-writer eer": "-"})

for f in sorted(glob.glob(os.path.join(ROOT, "runs", "wdsvm", "*.json"))):
    r = json.load(open(f))
    rows.append({"model": f"WD-SVM {r['features']} feats, {r['mode']} input, {r['n_ref']} refs",
                 **{k: f"{r[k + '_mean']:.3f} +/- {r[k + '_std']:.3f}" for k in KEYS},
                 "per-writer eer": f"{r['mean_writer_eer_mean']:.3f} +/- {r['mean_writer_eer_std']:.3f}"})

out = pd.DataFrame(rows)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 60)
print("5-fold writer-independent CV (test writers never seen in training; threshold from validation writers)\n")
print(out.to_string(index=False))
out.to_csv(os.path.join(ROOT, "runs", "summary.csv"), index=False)
