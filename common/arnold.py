import numpy as np
from .config import ARNOLD_A, ARNOLD_B, ARNOLD_ITERS

def arnold_scramble_bin(img_bin_2d: np.ndarray, a: int = ARNOLD_A, b: int = ARNOLD_B, iters: int = ARNOLD_ITERS) -> np.ndarray:
    assert img_bin_2d.ndim == 2 and img_bin_2d.shape[0] == img_bin_2d.shape[1]
    N = img_bin_2d.shape[0]
    out = img_bin_2d.copy().astype(np.uint8)
    for _ in range(iters):
        tmp = np.zeros_like(out)
        for x in range(N):
            for y in range(N):
                xp = (x + a * y) % N
                yp = (b * x + (a * b + 1) * y) % N
                tmp[xp, yp] = out[x, y]
        out = tmp
    return out

def arnold_unscramble_bin(img_bin_2d: np.ndarray, a: int = ARNOLD_A, b: int = ARNOLD_B, iters: int = ARNOLD_ITERS) -> np.ndarray:
    assert img_bin_2d.ndim == 2 and img_bin_2d.shape[0] == img_bin_2d.shape[1]
    N = img_bin_2d.shape[0]
    out = img_bin_2d.copy().astype(np.uint8)
    for _ in range(iters):
        tmp = np.zeros_like(out)
        for x in range(N):
            for y in range(N):
                xo = ((a * b + 1) * x - a * y) % N
                yo = (-b * x + y) % N
                tmp[xo, yo] = out[x, y]
        out = tmp
    return out

def scramble_bits_1024(bits_1024: np.ndarray):
    wm = bits_1024.reshape(32, 32).astype(np.uint8)
    scr = arnold_scramble_bin(wm)
    return scr.reshape(-1).astype(np.uint8)

def unscramble_bits_1024(bits_1024: np.ndarray):
    wm = bits_1024.reshape(32, 32).astype(np.uint8)
    rec = arnold_unscramble_bin(wm)
    return rec.reshape(-1).astype(np.uint8)
