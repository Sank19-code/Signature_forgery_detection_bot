# Signature Forgery Detection (CEDAR)

Detecting skilled signature forgeries on CEDAR (55 writers × 24 genuine + 24 forged),
built on the four-stage DIP pipeline from `DIP_CaseStudy.ipynb` and EfficientNet-B2.
Every number below comes from **5-fold cross-validation over writers**: test writers are never seen in training.

**Headline:** a writer-dependent verifier (EfficientNet-B2 fine-tuned for writer identification, then a
per-writer SVM on 5 genuine specimens) reaches **AUC 0.931 ± 0.018, EER 13.8%, accuracy 84.4%** on unseen writers.
A single-image "is this forged?" classifier reaches only AUC 0.751 on the same splits.

## 1. The dataset has a shortcut, and the first job is removing it

CEDAR's two classes were scanned differently:

| | Genuine (`original_*`) | Forged (`forgeries_*`) |
|---|---|---|
| File type | 1-channel grayscale (4-channel RGBA for writers 29, 39, 55) | 3-channel RGB (4-channel RGBA for writers 29, 39, 55) |
| Paper (median pixel) | ~240, grey, noisy | ~253, clean white |
| Ink | light grey | dark |

A logistic regression that sees **only intensity statistics** (no stroke shape at all) separates the classes
almost perfectly (`src/shortcut_probe.ipynb`, 5-fold CV grouped by writer):

| Input given to the probe | Intensity-only accuracy |
|---|---|
| Raw scans | **100.0%**: paper colour alone gives the answer |
| Canonicalised, grey ink kept (ink levelled to a fixed range) | 83.3%: leak survives in the ink grey levels |
| Canonicalised, **binary ink** | **55.9%** (≈ chance) |

A CNN trained on raw scans would report near-perfect accuracy while learning "grey paper = genuine". So every image
is **canonicalised** first (`src/preprocess.ipynb`): Otsu ink segmentation → drop specks < 12 px → crop to the signature →
white background, binary ink → letterbox to 192×384.

The forgeries were also scanned at a different resolution, so stroke thickness could still leak. A **stroke-normalised**
input (skeletonise, redraw every stroke 3 px wide, leaving only the pen trajectory) is used as a leak check for the
final model in section 4.

## 2. Pipeline

```
scan ─► canonicalise (Otsu, speck removal, crop, binarise, letterbox 192×384)
     ─► [train only] augmentation: rotate ±6°, shear ±8°, shrink 0.82-1.0, shift, stroke width ±1 px
     ─► Stage 1 contrast stretching
     ─► Stage 2 median filter 3×3          ─┐
     ─► Stage 3 high-boost (k = 1.5)        ├─► 3-channel input
     ─► Stage 4 Sobel gradient magnitude   ─┘
     ─► EfficientNet-B2 (ImageNet-pretrained)
```

Augmentation runs *before* the DIP stages, so the Sobel and high-boost channels always match the augmented strokes.

## 3. Evaluation protocol

* **5-fold CV over writers** (`writer_split` in `src/common.ipynb`): per fold 11 test, 8 validation and 36 training
  writers; no writer is in two of these sets.
* Everything that is learned or tuned (network weights, best epoch, SVM negatives, **decision threshold**) uses
  training/validation writers only. The threshold is the EER point on the validation writers, applied unchanged
  to the test writers.
* Metrics: AUC; EER; and at that threshold, accuracy, **FAR** (forgeries accepted, so fraud gets through) and
  **FRR** (genuine rejected, so a real customer is blocked). "Per-writer EER" lets each writer have their own
  threshold, the figure usually quoted for writer-dependent systems.
* Means ± population std over the 5 folds.

## 4. Results

| Model | AUC | EER | Acc | FAR | FRR | Per-writer EER |
|---|---|---|---|---|---|---|
| A. Single-image classifier | 0.751 ± 0.046 | 32.0% | 66.8% | 26.2% | 40.2% | - |
| B. Siamese verifier, 5 refs | 0.702 ± 0.057 | 36.6% | 62.6% | 35.9% | 39.3% | - |
| C. Per-writer SVM, ImageNet features, 5 refs | 0.903 ± 0.024 | 17.6% | 82.1% | 20.1% | 15.1% | 13.2% |
| C. Per-writer SVM, classifier-A features, 5 refs | 0.860 ± 0.048 | 21.6% | 75.6% | 30.5% | 16.6% | 16.7% |
| **D. Per-writer SVM, writer-ID features, 5 refs** | **0.931 ± 0.018** | **13.8%** | **84.4%** | 20.8% | **9.1%** | **9.9%** |
| D, leak check: stroke-normalised input | 0.923 ± 0.018 | 15.3% | 83.0% | 21.7% | 11.0% | 12.3% |
| E. Per-writer SVM, writer-ID + forgery-head features (SigNet-F), 5 refs | 0.911 ± 0.024 | 17.1% | 80.7% | 24.1% | 13.2% | 11.1% |

Model D per fold (AUC): 0.955, 0.927, 0.914, 0.910, 0.948.

**Specimens on file vs. performance** (model D):

| Genuine references per writer | AUC | EER | Per-writer EER |
|---|---|---|---|
| 1 | 0.870 ± 0.043 | 21.3% | 14.0% |
| 3 | 0.922 ± 0.013 | 15.5% | 10.9% |
| 5 | 0.931 ± 0.018 | 13.8% | 9.9% |
| 10 | 0.944 ± 0.016 | 13.0% | 10.0% |

### A. Single-image classifier (`src/train.ipynb`)
Sees one signature and answers "forged?". 15 epochs, AdamW (backbone LR 1e-4, head 1e-3), weight decay 0.05,
label smoothing 0.1, dropout 0.5, cosine schedule, AMP. **Why it fails:** with no reference, a skilled forgery of an
unseen writer is just a plausible signature. The model learns *style* ("careful, slow-looking strokes = forged"),
so the validation threshold doesn't transfer (FAR/FRR swing ±20 points across folds). Grad-CAM
(`runs/binary_fold0/gradcam_grid.png`) shows every false rejection in fold 0 belongs to one writer (#29) whose
careful cursive looks "forgery-like", and missed forgeries put their attention on blank paper.

### B. Siamese verifier (`src/siamese.ipynb`)
Shared EfficientNet-B2 → 256-d embedding; `[|e_r − e_q|, e_r ⊙ e_q]` → MLP; trained on (genuine, genuine) vs
(genuine, skilled forgery) pairs. Worse than A: 36 training writers are too few to learn a comparison function
that generalises, and validation AUC swung ±0.1 between epochs.

### C/D. Writer-dependent verification (`src/wd_svm.ipynb`, `src/writer_id.ipynb`)
How banks actually verify: compare against the specimens on file. For each test writer, an RBF SVM is trained on
the CNN embeddings of their **5 genuine specimens** (positives) against genuine signatures of the training writers
(negatives). **No skilled forgery is used to build the verifier**, which matches a bank with genuine specimens only.
The SVM then scores the writer's other 19 genuine signatures and all 24 forgeries.

The feature extractor matters (SigNet idea, Hafemann et al.): fine-tuning EfficientNet-B2 to **identify which of the
36 training writers** signed (cross-entropy, 20 epochs) gives features that describe *how a person signs*, and beats
ImageNet features on 4 of 5 folds by +0.008 to +0.047 AUC (the fifth is a tie, 0.909 vs 0.910). Fine-tuning for "forged vs genuine" (classifier A's features) is *worse* than
ImageNet: those features learn forgery style, not writer identity. The same effect shows in **E (SigNet-F)**, which
adds a genuine/forged head to the writer-ID training: it is below D on all 5 folds (0.911 vs 0.931 AUC). With only
36 training writers, the forgery objective pulls the features toward forgery style instead of writer identity.

**Leak check:** with every stroke redrawn at the same width, model D keeps AUC 0.923, so it reads the shape of
the signature, not scanner resolution.

### Explainability
`src/gradcam_verifier.ipynb` backpropagates the per-writer SVM score
`s(x) = −(Σ aᵢ·exp(−γ‖svᵢ − f(x)‖²) + b)` through the CNN (the torch score matches scikit-learn's
`decision_function` to 5 decimals). Red marks regions that look *unlike this writer's specimens*, e.g. a forged
closed loop where the writer draws a flat underline, or a differently formed capital
(`runs/writerid_fold0/gradcam_verifier.png`).

## 5. Reproduce

**To present the project, open `Final_Signature_Forgery_Detection.ipynb`** (project folder): one notebook that walks
through the dataset, the shortcut, every DIP preprocessing step, all models, results, Grad-CAM and a verification
demo, with all outputs saved. It loads the trained models and results (no retraining, ~1 minute to re-run).

The notebooks below in `src/` produce those models and results.

All code is in Jupyter notebooks in `src/`. **Open them from the `src/` folder**
with the Python 3.11 kernel, and run them in this order:

| # | Notebook | What it does | Run time with existing results |
|---|---|---|---|
| 1 | `build_cache.ipynb` | extracts the dataset if needed, preprocesses all 2,640 scans into `data/cache_*.npy`, shows the pipeline | ~20 s (skips existing caches) |
| 2 | `shortcut_probe.ipynb` | the intensity-only probe from section 1 | ~40 s |
| 3 | `train.ipynb` | A: single-image classifier, 5 folds | loads results |
| 4 | `siamese.ipynb` | B: Siamese verifier, 5 folds | loads results |
| 5 | `writer_id.ipynb` | D/E feature extractor (`LAM = 0` SigNet, `LAM = 1.0` SigNet-F) | loads checkpoints |
| 6 | `wd_svm.ipynb` | C/D/E: per-writer SVM verification, leak check, reference-count sweep | loads results |
| 7 | `gradcam.ipynb` | Grad-CAM for model A | ~30 s |
| 8 | `gradcam_verifier.ipynb` | Grad-CAM through the per-writer SVM for model D | ~30 s |
| 9 | `summarize.ipynb` | every result in one table → `runs/summary.csv` | ~20 s |

`preprocess.ipynb` and `common.ipynb` are library notebooks: they only define functions and are loaded by the
others with `%run ./common.ipynb`.

Notebooks 3–6 **load existing results instead of retraining**. Set `RETRAIN = True` (`RERUN = True` in
`wd_svm.ipynb`, `REBUILD = True` in `build_cache.ipynb`) in the settings cell at the top to recompute. Training
times per fold on an RTX 3050 Laptop (4 GB): classifier ≈ 11 min, Siamese ≈ 40 min, writer-ID ≈ 7 min,
SigNet-F ≈ 13 min; the SVM stage takes seconds.

Setup:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install opencv-python scikit-learn scikit-image pandas matplotlib ipykernel
```

## Files

| Path | What |
|---|---|
| `src/preprocess.ipynb` | canonicalisation, stroke normalisation, the four DIP stages (library) |
| `src/common.ipynb` | writer-independent split, dataset + augmentation, threaded batch loader, model, metrics, SVM helpers (library) |
| `src/build_cache.ipynb` | preprocesses all 2,640 scans once into `data/cache_*.npy` |
| `src/shortcut_probe.ipynb` | intensity-only probe that exposes the scanner shortcut |
| `src/train.ipynb` | A: single-image classifier |
| `src/siamese.ipynb` | B: Siamese verifier |
| `src/writer_id.ipynb` | D/E: writer-identification fine-tuning (feature extractor), optional forgery head |
| `src/wd_svm.ipynb` | C/D/E: writer-dependent SVM verification |
| `src/gradcam.ipynb`, `src/gradcam_verifier.ipynb` | Grad-CAM for A, and through the SVM for D |
| `src/summarize.ipynb` | all results → `runs/summary.csv` |
| `Final_Signature_Forgery_Detection.ipynb` | **the final notebook to present**: the whole project with all outputs |
| `DIP_CaseStudy.ipynb` | the original four-stage preprocessing notebook (runs on Colab or locally) |
| `archive/py_scripts/` | the earlier `.py` versions of the notebooks, which produced the results above |
| `runs/_v1_overfit/` | first attempt (no regularisation; threshold saturated at probability 1.0), kept for the record |
| `runs/_v2_writerid_trainval/` | writer-ID trained on train+val writers, discarded because it miscalibrated the validation threshold |
