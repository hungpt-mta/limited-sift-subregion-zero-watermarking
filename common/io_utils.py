import os
import glob
import cv2
import numpy as np
from .config import IMG_SIZE

def list_images(folder: str):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No images found in {folder}. Put images into folder '{folder}'.")
    return files

def safe_name(path: str):
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name

def imread_bgr_resized(path: str, size: int = IMG_SIZE):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    img = cv2.resize(img, (int(size), int(size)), interpolation=cv2.INTER_AREA)
    return img  # uint8 BGR

def bgr_to_gray(bgr: np.ndarray):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if gray.shape != (IMG_SIZE, IMG_SIZE):
        gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return gray

def bgr_to_y(bgr: np.ndarray):
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
    y = yuv[:, :, 0].astype(np.float32)
    return y

def bits_to_wm_image(bits_1024: np.ndarray):
    bits_1024 = bits_1024.astype(np.uint8).reshape(32, 32)
    return (bits_1024 * 255).astype(np.uint8)

def load_watermark_bits_1024(path: str):
    wm = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if wm is None:
        raise FileNotFoundError(f"Cannot read watermark: {path}")
    wm = cv2.resize(wm, (32, 32), interpolation=cv2.INTER_NEAREST)
    bits = (wm >= 128).astype(np.uint8).reshape(-1)  # 1024
    return bits, wm
