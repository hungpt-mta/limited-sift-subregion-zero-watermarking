import numpy as np
import cv2
from common.io_utils import bgr_to_y

def _dct_matrix(N: int = 8) -> np.ndarray:
    D = np.zeros((N, N), dtype=np.float32)
    for u in range(N):
        cu = np.sqrt(1.0 / N) if u == 0 else np.sqrt(2.0 / N)
        for x in range(N):
            D[u, x] = cu * np.cos(((2 * x + 1) * u * np.pi) / (2 * N))
    return D

_DCT8 = _dct_matrix(8)

def _vmf_feature_map_from_y(y: np.ndarray) -> np.ndarray:
    y = y.astype(np.float32)
    H, W = y.shape
    bs = 8
    if (H % bs) != 0 or (W % bs) != 0:
        raise ValueError(f"VMF expects H,W divisible by 8, got {H}x{W}.")

    nb_h = H // bs
    nb_w = W // bs

    M = np.zeros((nb_h, nb_w), dtype=np.uint8)

    for by in range(nb_h):
        for bx in range(nb_w):
            block = y[by*bs:(by+1)*bs, bx*bs:(bx+1)*bs].astype(np.float64)

            Q, R = np.linalg.qr(block)

            d = np.sign(np.diag(R))
            d[d == 0] = 1.0
            R = (d[:, None] * R)

            R0 = R[0, :].astype(np.float32)
            F = (_DCT8 @ R0.reshape(-1, 1)).reshape(-1)
            M[by, bx] = 1 if (F[2] > F[1]) else 0

    return M

def extract_bits(bgr512: np.ndarray) -> np.ndarray:
    bgr256 = cv2.resize(bgr512, (256, 256), interpolation=cv2.INTER_AREA)
    y256 = bgr_to_y(bgr256)
    M32 = _vmf_feature_map_from_y(y256)
    if M32.shape != (32, 32):
        raise ValueError(f"VMF32 expects 32x32 feature map, got {M32.shape}.")
    return M32.reshape(-1).astype(np.uint8)
