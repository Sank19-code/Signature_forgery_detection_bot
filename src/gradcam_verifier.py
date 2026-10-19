"""
Grad-CAM for the writer-dependent verifier (writer-ID features + per-writer RBF SVM).

    python src/gradcam_verifier.py --fold 0 --ckpt writerid

The forgery score of a questioned signature x for writer w is
    s(x) = -( sum_i a_i * exp(-gamma * ||sv_i - f(x)||^2) + b )
where f(x) is the L2-normalised CNN embedding and (sv_i, a_i, b, gamma) come from
w's fitted SVM. That is differentiable in f(x), so Grad-CAM can push d s / d A
through the SVM into the last conv block A. The heatmap shows the regions that
make the signature look *unlike this writer's specimens*.

Writes runs/<ckpt>_fold<k>/gradcam_verifier.png: per row one test writer with
a reference specimen, a genuine questioned signature and a skilled forgery.
"""
import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import ROOT, build_model, eer_threshold, load_gray, writer_split
from gradcam import overlay
from siamese import to_input
from wd_svm import features, fit_writer_svm, score_writers


class SVMGradCAM:
    def __init__(self, model, svm, device):
        self.model, self.acts, self.grads = model, None, None
        layer = model.features[-1]
        layer.register_forward_hook(lambda m, i, o: setattr(self, "acts", o))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "grads", go[0]))
        self.sv = torch.tensor(svm.support_vectors_, dtype=torch.float32, device=device)
        self.a = torch.tensor(svm.dual_coef_[0], dtype=torch.float32, device=device)
        self.b, self.gamma = float(svm.intercept_[0]), float(svm._gamma)

    def score(self, x):
        f = torch.flatten(self.model.avgpool(self.model.features(x)), 1)
        f = f / f.norm(dim=1, keepdim=True)
        k = torch.exp(-self.gamma * ((self.sv - f) ** 2).sum(1))
        return -((self.a * k).sum() + self.b)

    def __call__(self, x):
        self.model.zero_grad(set_to_none=True)
        s = self.score(x)
        s.backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self.acts).sum(1)).squeeze(0).detach().cpu().numpy()
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]), interpolation=cv2.INTER_CUBIC)
        cam = np.clip(cam, 0, None)
        return cam / (cam.max() + 1e-8), s.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--ckpt", default="writerid")
    ap.add_argument("--n_ref", type=int, default=5)
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    device = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")

    meta, tr, va, te = writer_split(fold=a.fold)
    G = load_gray("binary")
    F = features(a.ckpt, "binary", a.fold, device)
    neg = F[tr[meta.label.values[tr] == 0]]
    yv, sv, _ = score_writers(F, meta, sorted(set(meta.writer[va])), neg, a.n_ref)
    thr, _ = eer_threshold(yv, sv)

    model = build_model(pretrained=False).to(device).eval()
    ck = torch.load(os.path.join(ROOT, "runs", f"{a.ckpt}_fold{a.fold}", "best.pt"), map_location=device)
    model.features.load_state_dict(ck["features"])

    rng = np.random.default_rng(0)
    writers = rng.choice(sorted(set(meta.writer[te])), a.rows, replace=False)
    fig, axes = plt.subplots(a.rows, 3, figsize=(13, 2.5 * a.rows), squeeze=False)
    for r, w in enumerate(writers):
        svm, quest = fit_writer_svm(F, meta, w, neg, a.n_ref)
        cam = SVMGradCAM(model, svm, device)
        ref = meta[(meta.writer == w) & (meta.label == 0) & (meta["index"] == 1)].index[0]
        gen = rng.choice([q for q in quest if meta.label[q] == 0])
        forg = rng.choice([q for q in quest if meta.label[q] == 1])
        axes[r, 0].imshow(np.asarray(G[ref]), cmap="gray")
        axes[r, 0].set_title(f"writer {w}: reference specimen (1 of {a.n_ref})", fontsize=9)
        for c, k in ((1, gen), (2, forg)):
            x = torch.from_numpy(to_input(np.asarray(G[k])))[None].to(device)
            heat, s = cam(x)
            verdict = "FORGED" if s >= thr else "genuine"
            truth = "forgery" if meta.label[k] else "genuine"
            axes[r, c].imshow(overlay(np.asarray(G[k]), heat))
            axes[r, c].set_title(f"{os.path.basename(meta.path[k])} (truly {truth})\n"
                                 f"score {s:+.2f} vs thr {thr:+.2f} -> {verdict}", fontsize=9)
        for c in range(3):
            axes[r, c].axis("off")
    plt.suptitle("Grad-CAM through the per-writer SVM: red = regions that look unlike this writer's specimens",
                 fontsize=11)
    plt.tight_layout()
    path = os.path.join(ROOT, "runs", f"{a.ckpt}_fold{a.fold}", "gradcam_verifier.png")
    plt.savefig(path, dpi=110, bbox_inches="tight")
    print("saved", path)


if __name__ == "__main__":
    main()
