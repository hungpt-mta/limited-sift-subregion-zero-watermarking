"""
Size-consistent reimplementation of Yuan et al. (2024) DWT-DAISY-DCT
zero-watermarking feature extraction for 32x32 watermark comparison.

This plaintext-domain version preserves Yuan et al.'s 8x4 DAISY-DCT binary
block and repeats that published block extraction over a central 4x8 sampling
grid on LL3. The 32 binary blocks are tiled into a 32x32 feature image.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any

import cv2
import numpy as np

from common.config import IMG_SIZE, EPS

try:
    from skimage.feature import daisy as _skimage_daisy
    HAS_SKIMAGE_DAISY = True
except Exception:  # pragma: no cover
    _skimage_daisy = None
    HAS_SKIMAGE_DAISY = False

METHOD_NAME = "Yuan et al. DWT-DAISY-DCT"
WM_SIZE = 32
GRID_ROWS = 4
GRID_COLS = 8
LOW_ROWS = 8
LOW_COLS = 4
CENTER_FRACTION = 0.70
BINARIZE_MODE = "local_mean"  # local_mean performed per 8x4 block.

# Fallback DAISY-like descriptor parameters.
DAISY_RINGS = 3
DAISY_RING_POINTS = 8
DAISY_ORIENTATIONS = 8
DAISY_PATCH_RADIUS = 3
DAISY_RING_STEP = 4.0


@dataclass
class YuanDaisyDctExtractor:
    grid_rows: int = GRID_ROWS
    grid_cols: int = GRID_COLS
    center_fraction: float = CENTER_FRACTION
    binarize_mode: str = BINARIZE_MODE
    patch_radius: int = DAISY_PATCH_RADIUS
    ring_step: float = DAISY_RING_STEP
    use_skimage: bool = True

    def extract_bits(self, gray512: np.ndarray) -> np.ndarray:
        gray = _to_gray512(gray512)
        ll3 = _haar_dwt_levels(gray, levels=3)
        coeff_map = self._feature_coefficients_32x32(ll3)
        bits = _binarize_coeff_map(coeff_map, mode=self.binarize_mode)
        return bits.reshape(-1).astype(np.uint8)

    def _feature_coefficients_32x32(self, ll3: np.ndarray) -> np.ndarray:
        h, w = ll3.shape[:2]
        points = _central_grid_points(h, w, self.grid_rows, self.grid_cols, self.center_fraction)
        out = np.zeros((self.grid_rows * LOW_ROWS, self.grid_cols * LOW_COLS), dtype=np.float32)

        if self.use_skimage and HAS_SKIMAGE_DAISY:
            # Compute DAISY descriptors densely once, then take descriptors closest to sample points.
            # step=1 provides a dense descriptor grid; rings/histograms/orientations match the paper's
            # 25x8 layout: center + 3 rings x 8 histograms, 8 orientations.
            desc_img = _skimage_daisy(
                ll3.astype(np.float32),
                step=1,
                radius=15,
                rings=3,
                histograms=8,
                orientations=8,
                normalization='l2',
                visualize=False,
            )
            dh, dw = desc_img.shape[:2]
            # skimage descriptor coordinates are offset by radius from the original image boundary.
            y_offset = max(0, (h - dh) // 2)
            x_offset = max(0, (w - dw) // 2)
            for idx, (y, x) in enumerate(points):
                r = idx // self.grid_cols
                c = idx % self.grid_cols
                yy = int(np.clip(round(y) - y_offset, 0, dh - 1))
                xx = int(np.clip(round(x) - x_offset, 0, dw - 1))
                desc = desc_img[yy, xx].astype(np.float32).reshape(25, 8)
                dct_desc = cv2.dct(desc)
                block = dct_desc[:LOW_ROWS, :LOW_COLS]
                out[r*LOW_ROWS:(r+1)*LOW_ROWS, c*LOW_COLS:(c+1)*LOW_COLS] = block
            return out

        # Fallback: lightweight DAISY-like 25x8 gradient histogram descriptor.
        mag, ori = _gradient_mag_ori(ll3)
        for idx, (y, x) in enumerate(points):
            r = idx // self.grid_cols
            c = idx % self.grid_cols
            desc = _daisy_like_descriptor(
                mag, ori, y=float(y), x=float(x),
                patch_radius=int(self.patch_radius), ring_step=float(self.ring_step),
                rings=DAISY_RINGS, ring_points=DAISY_RING_POINTS,
                orientations=DAISY_ORIENTATIONS,
            )
            dct_desc = cv2.dct(desc.astype(np.float32))
            block = dct_desc[:LOW_ROWS, :LOW_COLS]
            out[r*LOW_ROWS:(r+1)*LOW_ROWS, c*LOW_COLS:(c+1)*LOW_COLS] = block
        return out


def _to_gray512(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    gray = gray.astype(np.float32)
    if gray.shape[:2] != (IMG_SIZE, IMG_SIZE):
        gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
    return gray


def _haar_dwt_ll(gray: np.ndarray) -> np.ndarray:
    a = gray.astype(np.float32)
    h, w = a.shape[:2]
    if h % 2:
        a = a[:-1, :]
    if w % 2:
        a = a[:, :-1]
    return ((a[0::2, 0::2] + a[0::2, 1::2] + a[1::2, 0::2] + a[1::2, 1::2]) * 0.5).astype(np.float32)


def _haar_dwt_levels(gray: np.ndarray, levels: int = 3) -> np.ndarray:
    out = gray.astype(np.float32)
    for _ in range(int(levels)):
        out = _haar_dwt_ll(out)
    return out


def _central_grid_points(h: int, w: int, rows: int, cols: int, frac: float) -> np.ndarray:
    frac = float(np.clip(frac, 0.1, 1.0))
    span_y = max(rows, int(round(h * frac)))
    span_x = max(cols, int(round(w * frac)))
    y0 = (h - span_y) / 2.0
    x0 = (w - span_x) / 2.0
    ys = np.linspace(y0, y0 + span_y - 1, int(rows))
    xs = np.linspace(x0, x0 + span_x - 1, int(cols))
    return np.asarray([(float(y), float(x)) for y in ys for x in xs], dtype=np.float32)


def _gradient_mag_ori(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    img = img.astype(np.float32)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    ori = (cv2.phase(gx, gy, angleInDegrees=False) % (2.0 * np.pi)).astype(np.float32)
    return mag.astype(np.float32), ori.astype(np.float32)


def _weighted_histogram_at(mag: np.ndarray, ori: np.ndarray, y: float, x: float,
                           patch_radius: int, orientations: int) -> np.ndarray:
    h, w = mag.shape[:2]
    yc = int(round(y)); xc = int(round(x))
    r = int(max(1, patch_radius))
    y1, y2 = max(0, yc-r), min(h, yc+r+1)
    x1, x2 = max(0, xc-r), min(w, xc+r+1)
    patch_mag = mag[y1:y2, x1:x2]
    patch_ori = ori[y1:y2, x1:x2]
    if patch_mag.size == 0:
        return np.zeros((orientations,), dtype=np.float32)
    yy, xx = np.mgrid[y1:y2, x1:x2]
    sigma = max(1.0, r / 2.0)
    weights = np.exp(-((yy - y)**2 + (xx - x)**2) / (2.0 * sigma * sigma)).astype(np.float32)
    vals = patch_mag * weights
    bins = np.floor(patch_ori / (2.0*np.pi) * orientations).astype(np.int32)
    bins = np.clip(bins, 0, orientations-1)
    hist = np.zeros((orientations,), dtype=np.float32)
    for b in range(orientations):
        hist[b] = float(vals[bins == b].sum())
    s = float(hist.sum())
    if s > EPS:
        hist /= s
    return hist


def _daisy_like_descriptor(mag: np.ndarray, ori: np.ndarray, y: float, x: float,
                           patch_radius: int = DAISY_PATCH_RADIUS,
                           ring_step: float = DAISY_RING_STEP,
                           rings: int = DAISY_RINGS,
                           ring_points: int = DAISY_RING_POINTS,
                           orientations: int = DAISY_ORIENTATIONS) -> np.ndarray:
    samples = [(y, x)]
    for rr in range(1, rings + 1):
        rad = rr * float(ring_step)
        for k in range(ring_points):
            theta = 2.0 * np.pi * k / float(ring_points)
            samples.append((y + rad * np.sin(theta), x + rad * np.cos(theta)))
    desc = np.zeros((len(samples), orientations), dtype=np.float32)
    for i, (sy, sx) in enumerate(samples):
        desc[i, :] = _weighted_histogram_at(mag, ori, sy, sx, patch_radius, orientations)
    norm = float(np.linalg.norm(desc))
    if norm > EPS:
        desc /= norm
    return desc.astype(np.float32)


def _binarize_coeff_map(coeff_map: np.ndarray, mode: str = BINARIZE_MODE) -> np.ndarray:
    coeff_map = coeff_map.astype(np.float32)
    mode = str(mode).lower()
    if mode == 'sign':
        return (coeff_map >= 0.0).astype(np.uint8)
    if mode == 'global_mean':
        return (coeff_map >= float(coeff_map.mean())).astype(np.uint8)
    if mode == 'local_mean':
        out = np.zeros_like(coeff_map, dtype=np.uint8)
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                block = coeff_map[r*LOW_ROWS:(r+1)*LOW_ROWS, c*LOW_COLS:(c+1)*LOW_COLS]
                out[r*LOW_ROWS:(r+1)*LOW_ROWS, c*LOW_COLS:(c+1)*LOW_COLS] = (block >= float(block.mean())).astype(np.uint8)
        return out
    raise ValueError(f'Unknown binarize mode: {mode}')


EXTRACTOR = YuanDaisyDctExtractor()


def extract_bits(gray512: np.ndarray) -> np.ndarray:
    return EXTRACTOR.extract_bits(gray512)


def meta_info() -> Dict[str, Any]:
    return {
        'method': METHOD_NAME,
        'domain': 'plaintext only',
        'dwt': '3-level orthonormal Haar LL',
        'sampling': f'central {GRID_ROWS}x{GRID_COLS} grid on LL3',
        'descriptor': 'skimage DAISY 25x8 if available; otherwise DAISY-like 25x8 gradient histogram',
        'uses_skimage_daisy': bool(HAS_SKIMAGE_DAISY and EXTRACTOR.use_skimage),
        'dct_low_frequency_block': f'{LOW_ROWS}x{LOW_COLS} per sampling point',
        'feature_size': '32x32 bits',
        'binarization': BINARIZE_MODE,
        'note': 'size-consistent reimplementation: 32 DAISY-DCT blocks, each 8x4, tiled to 32x32',
    }
