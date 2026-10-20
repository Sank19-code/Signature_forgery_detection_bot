"""
Preprocessing for CEDAR signature forgery detection.

Two parts:
  1. canonicalise()  - removes scanner differences between the two classes.
     In CEDAR, genuine signatures are grayscale scans on grey paper (~240) with
     light ink, forgeries are RGB scans on white paper (~253) with dark ink.
     Left alone, a CNN learns "grey paper = genuine" instead of anything about
     the strokes. So we segment the ink with Otsu, drop specks, crop to the
     signature, paint the background pure white and level the ink intensities.
  2. The four-stage DIP pipeline from DIP_CaseStudy.ipynb
     (contrast stretch -> median -> high-boost -> Sobel), applied to the
     canonical image. Stages 2, 3 and 4 are stacked as the 3 input channels
     of the CNN.
"""
import cv2
import numpy as np

OUT_H, OUT_W = 192, 384          # signatures are ~2:1 wide


# ---------------------------------------------------------------- canonicalise

def ink_mask(gray, min_area=12):
    """Otsu threshold (ink = dark), then drop connected components smaller than min_area px."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def canonicalise(gray, binarize=False, margin=8):
    """Return a white-background, cropped, ink-levelled grayscale image (not yet resized)."""
    mask = ink_mask(gray)
    if not mask.any():
        return np.full_like(gray, 255)

    ys, xs = np.where(mask)
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin + 1, gray.shape[0])
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin + 1, gray.shape[1])
    g, m = gray[y0:y1, x0:x1], mask[y0:y1, x0:x1]

    out = np.full(g.shape, 255, np.uint8)
    if binarize:
        out[m] = 0
    else:
        # Ink levelling: map this image's ink range [p2, p98] onto a fixed [0, 160],
        # so "how dark the pen/scanner was" is the same for every image.
        ink = g[m].astype(np.float64)
        lo, hi = np.percentile(ink, 2), np.percentile(ink, 98)
        scaled = (ink - lo) / max(hi - lo, 1.0) * 160.0
        out[m] = np.clip(scaled, 0, 160).astype(np.uint8)
    return out


def letterbox(img, h=OUT_H, w=OUT_W, fill=255):
    """Resize keeping aspect ratio, pad to (h, w) with white."""
    s = min(h / img.shape[0], w / img.shape[1])
    nh, nw = max(int(round(img.shape[0] * s)), 1), max(int(round(img.shape[1] * s)), 1)
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.full((h, w), fill, np.uint8)
    y, x = (h - nh) // 2, (w - nw) // 2
    out[y:y + nh, x:x + nw] = r
    return out


# ------------------------------------------------- four-stage DIP pipeline
# (same functions as the notebook)

def contrast_stretching(img):
    """Linear stretch: maps [min, max] -> [0, 255]"""
    r_min, r_max = float(np.min(img)), float(np.max(img))
    if r_max == r_min:
        return img.copy()
    stretched = (img.astype(np.float64) - r_min) * (255.0 / (r_max - r_min))
    return np.uint8(np.clip(stretched, 0, 255))


def median_filter(img, ksize=3):
    return cv2.medianBlur(img, ksize)


def high_boost_filter(img, ksize=3, k=1.5):
    """High-boost = original + k * (original - blurred),  k >= 1"""
    blurred = cv2.blur(img, (ksize, ksize)).astype(np.float64)
    mask = img.astype(np.float64) - blurred
    boosted = img.astype(np.float64) + k * mask
    return np.uint8(np.clip(boosted, 0, 255))


def sobel_edge_detection(img, ksize=3):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    return np.uint8(np.clip(magnitude, 0, 255))


def pipeline_stages(canon):
    s1 = contrast_stretching(canon)
    s2 = median_filter(s1, ksize=3)
    s3 = high_boost_filter(s2, ksize=3, k=1.5)
    s4 = sobel_edge_detection(s3, ksize=3)
    return s1, s2, s3, s4


def normalise_strokes(canon):
    """Skeletonise the ink and redraw it ~3 px wide: removes pen width / scan DPI,
    keeps only the trajectory (shape) of the signature."""
    from skimage.morphology import skeletonize
    skel = skeletonize(canon < 128).astype(np.uint8)
    ink = cv2.dilate(skel, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
    return np.where(ink > 0, 0, 255).astype(np.uint8)


def canonical_image(gray, binarize=False, raw=False, skel=False):
    """Scan -> (OUT_H, OUT_W) uint8 canonical signature (white background).

    raw=True skips canonicalise (used only to demonstrate the scanner shortcut).
    skel=True additionally normalises stroke width (implies binarize).
    """
    if raw:
        return letterbox(gray)
    canon = letterbox(canonicalise(gray, binarize or skel))
    return normalise_strokes(canon) if skel else canon


def stages_to_channels(canon):
    """Canonical image -> (3, H, W) uint8 CNN input: [median, high-boost, Sobel]."""
    _, s2, s3, s4 = pipeline_stages(canon)
    return np.stack([s2, s3, s4])


def to_channels(gray, binarize=False, raw=False):
    """Full preprocessing for one grayscale image -> (3, OUT_H, OUT_W) uint8."""
    return stages_to_channels(canonical_image(gray, binarize, raw))
