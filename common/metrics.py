import numpy as np
from .config import EPS

def calculate_metrics(org_bin: np.ndarray, rec_bin: np.ndarray):
    org_bin = org_bin.astype(np.float32).reshape(-1)
    rec_bin = rec_bin.astype(np.float32).reshape(-1)
    num = float(np.sum(org_bin * rec_bin))
    den = float(np.sqrt(np.sum(org_bin**2) * np.sum(rec_bin**2)) + EPS)
    nc = float(num / den)
    ber = float(np.mean(np.bitwise_xor(org_bin.astype(np.uint8), rec_bin.astype(np.uint8))))
    return nc, ber

def nc_only(a: np.ndarray, b: np.ndarray) -> float:
    return float(calculate_metrics(a, b)[0])

def hamming(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.uint8).ravel()
    b = b.astype(np.uint8).ravel()
    return float(np.mean(a ^ b))

def psnr_gray(a: np.ndarray, b: np.ndarray) -> float:
    # a,b: uint8 grayscale 512x512
    a_f = a.astype(np.float32)
    b_f = b.astype(np.float32)
    mse = float(np.mean((a_f - b_f) ** 2))
    if mse <= 1e-12:
        return 99.0  # effectively identical
    return float(10.0 * np.log10((255.0 * 255.0) / mse))

def psnr_color(a: np.ndarray, b: np.ndarray) -> float:
    """
    PSNR for color images.
    a,b: uint8 images with shape (H,W,3) in BGR/RGB (same format for both)
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"Expected color images HxWx3, got {a.shape}")

    a_f = a.astype(np.float32)
    b_f = b.astype(np.float32)

    mse = float(np.mean((a_f - b_f) ** 2))  # averaged over H*W*3
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10((255.0 * 255.0) / mse))