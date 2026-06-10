import numpy as np
import cv2
import pywt
from common.config import IMG_SIZE

class SchemeIISVD1024ZW_GRAY:
    def __init__(self,
                 wavelet: str = "haar",
                 ll_block: int = 4,
                 ll_stride: int = 2,
                 k1: int = 2025,
                 k2: int = 2026,
                 k3: int = 2027,
                 k4: int = 2028,
                 central_portion: bool = True,
                 central_frac: float = 0.5):
        self.wavelet = str(wavelet)
        self.ll_block = int(ll_block)
        self.ll_stride = int(ll_stride)

        self.k1 = int(k1)
        self.k2 = int(k2)
        self.k3 = int(k3)
        self.k4 = int(k4)

        self.central_portion = bool(central_portion)
        self.central_frac = float(central_frac)

        if self.ll_block <= 0 or self.ll_stride <= 0:
            raise ValueError("ll_block and ll_stride must be positive.")
        if not (0.0 <= self.central_frac <= 1.0):
            raise ValueError("central_frac must be in [0,1].")

    @staticmethod
    def _central_bounds(length: int, frac: float):
        frac = float(np.clip(frac, 0.0, 1.0))
        if frac <= 0.0:
            return 0, length
        span = max(1, int(round(length * frac)))
        start = (length - span) // 2
        end = start + span
        return start, end

    def _compute_S_matrix(self, gray: np.ndarray) -> np.ndarray:
        if gray.shape != (IMG_SIZE, IMG_SIZE):
            gray = cv2.resize(gray.astype(np.float32), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)

        LL, _ = pywt.dwt2(gray.astype(np.float32), self.wavelet)
        LL = np.asarray(LL, dtype=np.float32)

        h2, w2 = LL.shape
        b = self.ll_block
        s = self.ll_stride

        rows = (h2 - b) // s + 1
        cols = (w2 - b) // s + 1
        if rows <= 0 or cols <= 0:
            raise ValueError(f"Invalid S size from LL={LL.shape}, block={b}, stride={s}")

        S = np.zeros((rows, cols), dtype=np.float32)
        for r in range(rows):
            y = r * s
            for c in range(cols):
                x = c * s
                blk = LL[y:y+b, x:x+b]
                S[r, c] = float(np.linalg.svd(blk, compute_uv=False)[0])
        return S

    def _master_share_32x32(self, S: np.ndarray) -> np.ndarray:
        rows, cols = S.shape
        rs1 = np.random.RandomState(self.k1)
        rs2 = np.random.RandomState(self.k2)
        rs3 = np.random.RandomState(self.k3)
        rs4 = np.random.RandomState(self.k4)

        if self.central_portion:
            r0, r1 = self._central_bounds(rows, self.central_frac)
            c0, c1 = self._central_bounds(cols, self.central_frac)
        else:
            r0, r1 = 0, rows
            c0, c1 = 0, cols

        M = np.zeros((32, 32), dtype=np.uint8)
        for i in range(32):
            for j in range(32):
                ii = rs1.randint(r0, r1)
                jj = rs2.randint(c0, c1)
                pp = rs3.randint(r0, r1)
                qq = rs4.randint(c0, c1)
                M[i, j] = 1 if (S[ii, jj] - S[pp, qq]) > 0.0 else 0
        return M

    def extract_bits(self, gray512: np.ndarray) -> np.ndarray:
        S = self._compute_S_matrix(gray512)
        M = self._master_share_32x32(S)
        return M.reshape(-1).astype(np.uint8)

SVD_EXTRACTOR = SchemeIISVD1024ZW_GRAY(
    wavelet="haar",
    ll_block=4,
    ll_stride=2,
    k1=2025, k2=2026, k3=2027, k4=2028,
    central_portion=False,
    central_frac=0.5
)
