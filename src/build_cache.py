"""
Preprocess every CEDAR image once and store the results as .npy files.

    python src/build_cache.py --mode canon     # default: canonicalise + 4-stage pipeline
    python src/build_cache.py --mode binary    # same, but ink fully binarised
    python src/build_cache.py --mode skel      # binary + stroke width normalised (skeleton redrawn 3 px)
    python src/build_cache.py --mode raw       # no canonicalise (shortcut demo only)

Writes data/cache_<mode>_gray.npy (N, H, W)     canonical signature images,
       data/cache_<mode>_X.npy    (N, 3, H, W)  their 4-stage pipeline channels,
       data/meta.csv  (path, label, writer, index)  with label 1 = forged, 0 = genuine.
"""
import argparse
import csv
import glob
import os
import re

import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from preprocess import canonical_image, stages_to_channels

ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
SIG_DIR = os.path.join(ROOT, "Dataset", "signatures")
NAME_RE = re.compile(r"(original|forgeries)_(\d+)_(\d+)\.png$")


def list_images():
    rows = []
    for p in glob.glob(os.path.join(SIG_DIR, "*", "*.png")):
        m = NAME_RE.search(os.path.basename(p))
        if m:
            rows.append((os.path.relpath(p, ROOT), int(m.group(1) == "forgeries"),
                         int(m.group(2)), int(m.group(3))))
    rows.sort(key=lambda r: (r[2], r[1], r[3]))
    return rows


def _work(args):
    rel, mode = args
    gray = cv2.imread(os.path.join(ROOT, rel), cv2.IMREAD_GRAYSCALE)
    canon = canonical_image(gray, binarize=(mode == "binary"), raw=(mode == "raw"), skel=(mode == "skel"))
    return canon, stages_to_channels(canon)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["canon", "binary", "skel", "raw"], default="canon")
    a = ap.parse_args()

    rows = list_images()
    print(f"{len(rows)} images, {len({r[2] for r in rows})} writers")
    with ProcessPoolExecutor() as ex:
        out = list(ex.map(_work, [(r[0], a.mode) for r in rows], chunksize=32))
    G = np.stack([o[0] for o in out])
    X = np.stack([o[1] for o in out])

    np.save(os.path.join(ROOT, f"cache_{a.mode}_gray.npy"), G)   # canonical image (train-time aug input)
    np.save(os.path.join(ROOT, f"cache_{a.mode}_X.npy"), X)      # precomputed 3-channel stages
    with open(os.path.join(ROOT, "meta.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "writer", "index"])
        w.writerows(rows)
    print("saved", X.shape)


if __name__ == "__main__":
    main()
