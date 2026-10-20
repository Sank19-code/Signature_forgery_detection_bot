"""
Inference and Pipeline Module for EfficientNet-B2 Signature Classification.
Used by Streamlit and standalone prediction scripts.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b2

OUT_H, OUT_W = 192, 384
MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)


# ---------------------------------------------------------------- Preprocessing
def ink_mask(gray: np.ndarray, min_area: int = 12) -> np.ndarray:
    """Otsu threshold followed by removing small speckles."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def canonicalise(gray: np.ndarray, binarize: bool = True, margin: int = 8) -> np.ndarray:
    """Crop to signature bounding box, make background white and normalize ink."""
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
        ink = g[m].astype(np.float64)
        lo, hi = np.percentile(ink, 2), np.percentile(ink, 98)
        scaled = (ink - lo) / max(hi - lo, 1.0) * 160.0
        out[m] = np.clip(scaled, 0, 160).astype(np.uint8)
    return out


def letterbox(img: np.ndarray, h: int = OUT_H, w: int = OUT_W, fill: int = 255) -> np.ndarray:
    """Preserve aspect ratio and pad to (h, w)."""
    s = min(h / img.shape[0], w / img.shape[1])
    nh, nw = max(int(round(img.shape[0] * s)), 1), max(int(round(img.shape[1] * s)), 1)
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.full((h, w), fill, np.uint8)
    y, x = (h - nh) // 2, (w - nw) // 2
    out[y:y + nh, x:x + nw] = r
    return out


# ---------------------------------------------------- Digital Image Processing
def contrast_stretching(img: np.ndarray) -> np.ndarray:
    """Stage 1: Linear contrast stretch mapping [min, max] -> [0, 255]."""
    r_min, r_max = float(np.min(img)), float(np.max(img))
    if r_max == r_min:
        return img.copy()
    stretched = (img.astype(np.float64) - r_min) * (255.0 / (r_max - r_min))
    return np.uint8(np.clip(stretched, 0, 255))


def median_filter(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Stage 2: Median filter noise reduction."""
    return cv2.medianBlur(img, ksize)


def high_boost_filter(img: np.ndarray, ksize: int = 3, k: float = 1.5) -> np.ndarray:
    """Stage 3: High-boost edge sharpening."""
    blurred = cv2.blur(img, (ksize, ksize)).astype(np.float64)
    mask = img.astype(np.float64) - blurred
    boosted = img.astype(np.float64) + k * mask
    return np.uint8(np.clip(boosted, 0, 255))


def sobel_edge_detection(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Stage 4: Sobel gradient magnitude edge detection."""
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    return np.uint8(np.clip(magnitude, 0, 255))


def stages_to_channels(canon: np.ndarray) -> np.ndarray:
    """Stack Stage 2 (Median), Stage 3 (High-Boost), Stage 4 (Sobel) as 3 channels."""
    s1 = contrast_stretching(canon)
    s2 = median_filter(s1, ksize=3)
    s3 = high_boost_filter(s2, ksize=3, k=1.5)
    s4 = sobel_edge_detection(s3, ksize=3)
    return np.stack([s2, s3, s4], axis=0)


def preprocess_image(image_input) -> tuple[np.ndarray, np.ndarray]:
    """
    Takes an image (filepath, bytes, or numpy array), applies canonicalization,
    and returns:
      1) canonical grayscale image (for visualization)
      2) normalized 3-channel PyTorch tensor (1, 3, 192, 384)
    """
    if isinstance(image_input, str):
        gray = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
    elif isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    elif isinstance(image_input, np.ndarray):
        gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY) if image_input.ndim == 3 else image_input
    else:
        raise ValueError("Unsupported input type for image_input")

    if gray is None:
        raise ValueError("Failed to load or decode image")

    canon = letterbox(canonicalise(gray, binarize=True))
    channels = stages_to_channels(canon)
    normalized = (channels.astype(np.float32) / 255.0 - MEAN) / STD
    tensor = torch.from_numpy(normalized).unsqueeze(0)
    return canon, tensor


# ------------------------------------------------------------- EfficientNet-B2
def build_model(pretrained: bool = False, dropout: float = 0.5) -> nn.Module:
    """Builds EfficientNet-B2 with single logit binary classification head."""
    model = efficientnet_b2(weights=None)
    model.classifier[0].p = dropout
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    return model


def load_classifier(model_path: str = "models/efficientnet_b2_signature.pt", device: str = "cpu"):
    """Loads checkpoint weights and decision threshold."""
    model = build_model()
    thr = 0.5
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            raw_thr = ckpt.get("thr", 0.0)
            # convert logit threshold to sigmoid probability threshold
            thr = float(1.0 / (1.0 + np.exp(-raw_thr)))
        elif isinstance(ckpt, dict):
            model.load_state_dict(ckpt)
    else:
        print(f"Warning: Checkpoint not found at {model_path}. Using uninitialized model.")
    
    model.to(device)
    model.eval()
    return model, thr


@torch.no_grad()
def predict_signature(model: nn.Module, tensor: torch.Tensor, device: str = "cpu") -> float:
    """Returns probability of the signature being forged (0.0 to 1.0)."""
    model.eval()
    tensor = tensor.to(device)
    logit = model(tensor).squeeze().item()
    prob_forged = 1.0 / (1.0 + np.exp(-logit))
    return prob_forged
