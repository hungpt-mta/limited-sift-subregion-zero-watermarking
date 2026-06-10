"""
Our SIFT strongest-keypoint sub-region zero-watermarking method.

Registration:
  - Detect SIFT keypoints on the original image.
  - Keep the top-P strongest keypoints ranked by response, subject to a valid
    64x64 sub-region being fully inside the image.
  - For every selected keypoint, build a 64x64 sub-region centered at the keypoint.
  - Extract a local master share from Y -> Haar DWT LL1 -> DCT -> binarization.
  - Store the tuple (keypoint, descriptor, local zero-watermark) in state.

Verification:
  - Optionally align attacked image to the registered image using SIFT+homography.
  - Detect top-P valid SIFT keypoints on the corrected image.
  - Match detected descriptors to registered descriptors by normalized
    Euclidean similarity using one-to-one greedy assignment.
  - For one-to-one matches above threshold T, extract local watermarks and reconstruct the
    global scrambled watermark by majority voting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import time

import cv2
import numpy as np

from common.config import IMG_SIZE, EPS


# --------------------- public method name / defaults ---------------------
METHOD_NAME = "Our SIFT-Subregion ZW"
WM_SIZE = 32
SUBREGION_SIZE = 64
DEFAULT_P = 50
DEFAULT_T = 0.65
USE_VOTING = True
MATCHING_MODE = "one_to_one_greedy"  # prevents many-to-one descriptor matches  # Ablation switch: True=majority voting over Q local watermarks; False=best matched sub-region only

# SIFT + alignment defaults. They are intentionally local to this method so the
# file can run even when the benchmark notebook does not define global constants.
SIFT_NFEATURES = 2000
USE_GEOM_ALIGN = True
RATIO_TEST = 0.75
MIN_GOOD_MATCHES = 8
RANSAC_REPROJ = 5.0
MIN_INLIERS = 6
MIN_INLIER_RATIO = 0.25
MAX_BLACK_FRAC = None  # disabled for medical images with naturally black backgrounds

# DCT binarization. The paper text only says "binarize"; mean threshold is a
# stable default for DCT coefficient matrices and can be changed if needed.
BINARIZE_MODE = "mean"  # "mean" or "zero"


@dataclass
class AlignResult:
    aligned_bgr: np.ndarray
    ok: bool
    good: int
    inliers: int
    inlier_ratio: float
    reason: str


@dataclass
class MethodState:
    ref_shape: Tuple[int, int]
    wm_size: int
    subregion_size: int
    P: int
    T: float
    ref_kps: List[cv2.KeyPoint]
    ref_desc: Optional[np.ndarray]
    tuples: List[Dict[str, Any]]
    wm_scrambled_bits: np.ndarray
    align_meta: Dict[str, Any]
    use_voting: bool = True
    use_alignment: bool = True
    transform_mode: str = "dwt_dct"
    keypoint_selection: str = "strongest"
    region_mode: str = "multi"
    random_seed: int = 0


def to_gray(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim == 2:
        return bgr.astype(np.uint8)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def sift_create():
    return cv2.SIFT_create(nfeatures=int(SIFT_NFEATURES))


def black_fraction(bgr: np.ndarray, threshold: int = 5) -> float:
    gray = to_gray(bgr)
    return float(np.mean(gray <= threshold))


def _kp_to_tuple(kp: cv2.KeyPoint) -> Tuple[float, float, float, float, float, int, int]:
    return (float(kp.pt[0]), float(kp.pt[1]), float(kp.size), float(kp.angle),
            float(kp.response), int(kp.octave), int(kp.class_id))


def _tuple_to_kp(t: Tuple[float, float, float, float, float, int, int]) -> cv2.KeyPoint:
    x, y, size, angle, response, octave, class_id = t
    kp = cv2.KeyPoint(float(x), float(y), float(size), float(angle), float(response), int(octave), int(class_id))
    return kp


def _valid_center(kp: cv2.KeyPoint, h: int, w: int, sub_size: int = SUBREGION_SIZE) -> bool:
    half = sub_size // 2
    x = int(round(kp.pt[0]))
    y = int(round(kp.pt[1]))
    return (x - half >= 0) and (y - half >= 0) and (x + half <= w) and (y + half <= h)


def pick_topk_by_response(kps, desc, k: int, image_shape=None, sub_size: int = SUBREGION_SIZE):
    if kps is None or desc is None or len(kps) == 0:
        return [], None
    idx_all = np.argsort([-kp.response for kp in kps])
    chosen = []
    chosen_idx = []
    h = w = None
    if image_shape is not None:
        h, w = image_shape[:2]
    for idx in idx_all:
        kp = kps[int(idx)]
        if h is not None and not _valid_center(kp, h, w, sub_size=sub_size):
            continue
        chosen.append(kp)
        chosen_idx.append(int(idx))
        if len(chosen) >= int(k):
            break
    if not chosen_idx:
        return [], None
    return chosen, desc[np.array(chosen_idx, dtype=np.int32), :].astype(np.float32)



def pick_keypoints(kps, desc, k: int, image_shape=None, sub_size: int = SUBREGION_SIZE, mode: str = "strongest", seed: int = 0):
    """Select valid keypoints for registration/verification.

    mode='strongest': top-k by SIFT response.
    mode='random': random k valid keypoints using a deterministic seed.
    mode='all': all valid keypoints (no top-k limit).
    """
    if kps is None or desc is None or len(kps) == 0:
        return [], None
    h = w = None
    if image_shape is not None:
        h, w = image_shape[:2]
    valid_idx = []
    for idx, kp in enumerate(kps):
        if h is not None and not _valid_center(kp, h, w, sub_size=sub_size):
            continue
        valid_idx.append(int(idx))
    if not valid_idx:
        return [], None
    mode = str(mode).lower().strip()
    if mode in {"all", "full", "full_sift"} or int(k) < 0:
        chosen_idx = valid_idx
    elif mode == "random":
        rng = np.random.default_rng(int(seed))
        chosen_idx = list(rng.choice(np.array(valid_idx, dtype=np.int32), size=min(int(k), len(valid_idx)), replace=False))
    else:
        chosen_idx = sorted(valid_idx, key=lambda i: -kps[int(i)].response)[:min(int(k), len(valid_idx))]
    chosen = [kps[int(i)] for i in chosen_idx]
    return chosen, desc[np.array(chosen_idx, dtype=np.int32), :].astype(np.float32)

def crop_subregion_centered(bgr: np.ndarray, kp: cv2.KeyPoint, sub_size: int = SUBREGION_SIZE) -> Optional[np.ndarray]:
    h, w = bgr.shape[:2]
    half = sub_size // 2
    x = int(round(kp.pt[0]))
    y = int(round(kp.pt[1]))
    x1, x2 = x - half, x + half
    y1, y2 = y - half, y + half
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return None
    reg = bgr[y1:y2, x1:x2].copy()
    if reg.shape[0] != sub_size or reg.shape[1] != sub_size:
        return None
    return reg


def haar_dwt_ll1(gray: np.ndarray) -> np.ndarray:
    """One-level orthonormal Haar DWT LL sub-band."""
    a = gray.astype(np.float32)
    h, w = a.shape[:2]
    if h % 2 != 0:
        a = a[:-1, :]
    if w % 2 != 0:
        a = a[:, :-1]
    # LL = (a00 + a01 + a10 + a11) / 2 for orthonormal Haar.
    ll = (a[0::2, 0::2] + a[0::2, 1::2] + a[1::2, 0::2] + a[1::2, 1::2]) * 0.5
    return ll.astype(np.float32)


def _binarize(mat: np.ndarray) -> np.ndarray:
    mat = mat.astype(np.float32)
    if BINARIZE_MODE == "zero":
        return (mat >= 0.0).astype(np.uint8)
    return (mat >= float(np.mean(mat))).astype(np.uint8)


def subregion_master_share(region_bgr: np.ndarray, wm_size: int = WM_SIZE, transform_mode: str = "dwt_dct") -> np.ndarray:
    """Extract a binary master share from a local region.

    Modes:
      - dwt_dct: region size = 2*wm_size, Y -> Haar LL1 -> DCT -> mean threshold
      - dwt_only: region size = 2*wm_size, Y -> Haar LL1 -> mean threshold
      - dct_only: region size = wm_size, Y -> DCT -> mean threshold

    This matches the ablation protocol used in the revision experiments.
    """
    mode = str(transform_mode).lower().strip()
    if mode not in {"dwt_dct", "dwt_only", "dct_only"}:
        raise ValueError(f"Unknown transform_mode: {transform_mode}")

    target = wm_size if mode == "dct_only" else 2 * wm_size
    if region_bgr.shape[0] != target or region_bgr.shape[1] != target:
        region_bgr = cv2.resize(region_bgr, (target, target), interpolation=cv2.INTER_AREA)
    y = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)

    if mode == "dwt_only":
        ll1 = haar_dwt_ll1(y)
        if ll1.shape != (wm_size, wm_size):
            ll1 = cv2.resize(ll1, (wm_size, wm_size), interpolation=cv2.INTER_AREA)
        return _binarize(ll1).reshape(-1).astype(np.uint8)

    if mode == "dct_only":
        dct = cv2.dct(y.astype(np.float32))
        if dct.shape != (wm_size, wm_size):
            dct = cv2.resize(dct, (wm_size, wm_size), interpolation=cv2.INTER_AREA)
        return _binarize(dct).reshape(-1).astype(np.uint8)

    # dwt_dct
    ll1 = haar_dwt_ll1(y)
    if ll1.shape != (wm_size, wm_size):
        ll1 = cv2.resize(ll1, (wm_size, wm_size), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(ll1.astype(np.float32))
    return _binarize(dct).reshape(-1).astype(np.uint8)


def dwt_level_ll(gray: np.ndarray, level: int) -> np.ndarray:
    ll = gray.astype(np.float32)
    for _ in range(int(level)):
        ll = haar_dwt_ll1(ll)
    return ll.astype(np.float32)


def global_master_share(bgr512: np.ndarray, wm_size: int = WM_SIZE) -> np.ndarray:
    """Global-region baseline: 512x512 -> LL level -> DCT -> binary watermark-size feature.

    64x64 watermark uses LL3, 32x32 uses LL4, 16x16 uses LL5.
    """
    bgr = cv2.resize(bgr512, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA) if bgr512.shape[:2] != (IMG_SIZE, IMG_SIZE) else bgr512
    y = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    level_map = {64: 3, 32: 4, 16: 5}
    level = level_map.get(int(wm_size))
    if level is None:
        level = int(round(np.log2(IMG_SIZE / int(wm_size))))
    ll = dwt_level_ll(y, level)
    if ll.shape != (wm_size, wm_size):
        ll = cv2.resize(ll, (wm_size, wm_size), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(ll.astype(np.float32))
    return _binarize(dct).reshape(-1).astype(np.uint8)

def _l2_normalize_desc(desc: np.ndarray) -> np.ndarray:
    d = desc.astype(np.float32)
    norm = np.linalg.norm(d, axis=1, keepdims=True) + EPS
    return d / norm


def descriptor_similarity(desc1: np.ndarray, desc2: np.ndarray) -> float:
    """Normalized Euclidean similarity in [0, 1].

    SIFT descriptors are L2-normalized first. The maximum Euclidean distance
    between two unit vectors is sqrt(2), so sim = 1 - dist/sqrt(2).
    """
    a = desc1.astype(np.float32).reshape(1, -1)
    b = desc2.astype(np.float32).reshape(1, -1)
    a = _l2_normalize_desc(a)[0]
    b = _l2_normalize_desc(b)[0]
    dist = float(np.linalg.norm(a - b))
    sim = 1.0 - dist / float(np.sqrt(2.0))
    return float(max(0.0, min(1.0, sim)))


def align_test_to_ref_homography(ref_bgr: np.ndarray, test_bgr: np.ndarray, ref_kps, ref_desc) -> AlignResult:
    h, w = ref_bgr.shape[:2]
    test_rs = cv2.resize(test_bgr, (w, h), interpolation=cv2.INTER_LINEAR) if test_bgr.shape[:2] != (h, w) else test_bgr

    if (not USE_GEOM_ALIGN) or ref_desc is None or ref_kps is None or len(ref_kps) < 5:
        return AlignResult(test_rs, False, 0, 0, 0.0, "no_ref_sift")

    sift = sift_create()
    kps2, desc2 = sift.detectAndCompute(to_gray(test_rs), None)
    if desc2 is None or kps2 is None or len(kps2) < 5:
        return AlignResult(test_rs, False, 0, 0, 0.0, "no_test_sift")

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw = bf.knnMatch(ref_desc.astype(np.float32), desc2.astype(np.float32), k=2)

    good = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < RATIO_TEST * n.distance:
            good.append(m)
    if len(good) < MIN_GOOD_MATCHES:
        return AlignResult(test_rs, False, len(good), 0, 0.0, "few_good_matches")

    src = np.float32([ref_kps[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)  # ref
    dst = np.float32([kps2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)     # test

    H, mask = cv2.findHomography(dst, src, cv2.RANSAC, ransacReprojThreshold=float(RANSAC_REPROJ))
    if H is None or mask is None:
        return AlignResult(test_rs, False, len(good), 0, 0.0, "H_failed")

    inliers = int(mask.sum())
    inlier_ratio = float(inliers) / float(max(1, len(good)))

    if inliers < MIN_INLIERS or inlier_ratio < MIN_INLIER_RATIO:
        return AlignResult(test_rs, False, len(good), inliers, inlier_ratio, "few_inliers_H")

    aligned = cv2.warpPerspective(
        test_rs, H, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    # Do not reject the homography by black-pixel ratio. Medical images often
    # contain large natural black backgrounds, so black_fraction may reject valid
    # alignments. RANSAC inlier count and inlier ratio are used instead.
    return AlignResult(aligned, True, len(good), inliers, inlier_ratio, "ok_H")


def register(bgr512: np.ndarray, wm_scrambled_bits: np.ndarray, P: int = DEFAULT_P, T: float = DEFAULT_T, use_voting: bool = USE_VOTING,
             use_alignment: bool = True, transform_mode: str = "dwt_dct", keypoint_selection: str = "strongest",
             region_mode: str = "multi", random_seed: int = 0, wm_size: int = WM_SIZE) -> Tuple[np.ndarray, MethodState]:
    """Register the original image.

    Returns host_bits_for_benchmark = wm_scrambled_bits. This lets the existing
    benchmark framework use a zero global key for this direct-voting method,
    while verification returns the reconstructed scrambled watermark.

    Timing metadata is stored in state.align_meta so the benchmark can report
    registration cost without re-running the method.
    """
    t_reg0 = time.perf_counter()
    if bgr512 is None:
        raise ValueError("bgr512 is required for Our SIFT-Subregion ZW.")
    t_resize0 = time.perf_counter()
    bgr = cv2.resize(bgr512, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA) if bgr512.shape[:2] != (IMG_SIZE, IMG_SIZE) else bgr512.copy()
    registration_resize_time_sec = time.perf_counter() - t_resize0

    wm_scrambled_bits = wm_scrambled_bits.astype(np.uint8).reshape(-1)
    wm_size = int(wm_size)
    if wm_scrambled_bits.size != wm_size * wm_size:
        raise ValueError(f"Expected {wm_size*wm_size} scrambled watermark bits, got {wm_scrambled_bits.size}.")

    t_sift0 = time.perf_counter()
    sift = sift_create()
    all_kps, all_desc = sift.detectAndCompute(to_gray(bgr), None)
    registration_sift_detect_time_sec = time.perf_counter() - t_sift0

    t_select0 = time.perf_counter()
    top_kps, top_desc = pick_keypoints(all_kps, all_desc, int(P), image_shape=bgr.shape, sub_size=(wm_size if str(transform_mode).lower()=="dct_only" else 2*wm_size), mode=keypoint_selection, seed=random_seed)
    registration_topP_selection_time_sec = time.perf_counter() - t_select0

    tuples: List[Dict[str, Any]] = []
    t_zw0 = time.perf_counter()
    registration_subregion_crop_time_sec = 0.0
    registration_ms_extract_time_sec = 0.0
    registration_xor_store_time_sec = 0.0
    if str(region_mode).lower().strip() in {"global", "global_region"}:
        t_ms0 = time.perf_counter()
        ms = global_master_share(bgr, wm_size=wm_size)
        registration_ms_extract_time_sec += time.perf_counter() - t_ms0
        t_xor0 = time.perf_counter()
        zw = (ms ^ wm_scrambled_bits).astype(np.uint8)
        tuples.append({"kp_tuple": (IMG_SIZE/2, IMG_SIZE/2, float(IMG_SIZE), 0.0, 1.0, 0, 0),
                       "desc": np.zeros((128,), dtype=np.float32), "zw": zw, "ms": ms})
        registration_xor_store_time_sec += time.perf_counter() - t_xor0
    else:
        for idx, kp in enumerate(top_kps):
            t_crop0 = time.perf_counter()
            reg = crop_subregion_centered(bgr, kp, (wm_size if str(transform_mode).lower()=="dct_only" else 2*wm_size))
            registration_subregion_crop_time_sec += time.perf_counter() - t_crop0
            if reg is None:
                continue
            t_ms0 = time.perf_counter()
            ms = subregion_master_share(reg, wm_size=wm_size, transform_mode=transform_mode)
            registration_ms_extract_time_sec += time.perf_counter() - t_ms0
            t_xor0 = time.perf_counter()
            zw = (ms ^ wm_scrambled_bits).astype(np.uint8)
            tuples.append({
                "kp_tuple": _kp_to_tuple(kp),
                "desc": top_desc[idx].astype(np.float32).copy(),
                "zw": zw,
                "ms": ms,
            })
            registration_xor_store_time_sec += time.perf_counter() - t_xor0
    registration_zw_generation_time_sec = time.perf_counter() - t_zw0

    # Use the selected registration keypoints/descriptors also for homography.
    # If very few valid keypoints were selected, fall back to all detected SIFT features.
    if len(top_kps) >= 5 and top_desc is not None:
        ref_kps = top_kps
        ref_desc = top_desc.astype(np.float32)
    else:
        ref_kps = list(all_kps) if all_kps is not None else []
        ref_desc = all_desc.astype(np.float32) if all_desc is not None else None

    registration_total_time_sec_internal = time.perf_counter() - t_reg0
    state = MethodState(
        ref_shape=bgr.shape[:2], wm_size=wm_size, subregion_size=(wm_size if str(transform_mode).lower()=="dct_only" else 2*wm_size),
        P=int(P), T=float(T), ref_kps=ref_kps, ref_desc=ref_desc, tuples=tuples,
        wm_scrambled_bits=wm_scrambled_bits.copy(),
        align_meta={
            "n_registered_tuples": len(tuples),
            "n_ref_kps_for_align": len(ref_kps),
            "n_all_ref_keypoints": 0 if all_kps is None else len(all_kps),
            "n_valid_topP_keypoints": len(top_kps),
            "registration_resize_time_sec": registration_resize_time_sec,
            "registration_sift_detect_time_sec": registration_sift_detect_time_sec,
            "registration_topP_selection_time_sec": registration_topP_selection_time_sec,
            "registration_subregion_crop_time_sec": registration_subregion_crop_time_sec,
            "registration_ms_extract_time_sec": registration_ms_extract_time_sec,
            "registration_xor_store_time_sec": registration_xor_store_time_sec,
            "registration_zw_generation_time_sec": registration_zw_generation_time_sec,
            "registration_total_time_sec_internal": registration_total_time_sec_internal,
            "transform_mode": str(transform_mode),
            "keypoint_selection": str(keypoint_selection),
            "region_mode": str(region_mode),
            "random_seed": int(random_seed),
        },
        use_voting=bool(use_voting), use_alignment=bool(use_alignment), transform_mode=str(transform_mode), keypoint_selection=str(keypoint_selection), region_mode=str(region_mode), random_seed=int(random_seed),
    )
    return wm_scrambled_bits.copy(), state

def verify(bgr512: np.ndarray, state: MethodState) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Verify an attacked/suspicious image and return reconstructed scrambled watermark bits.

    The returned metadata includes detailed timing so the benchmark can report
    verification complexity by phase: alignment, test-SIFT detection, top-P
    selection, descriptor matching, local watermark extraction, and voting.
    """
    t_verify0 = time.perf_counter()
    if bgr512 is None:
        raise ValueError("bgr512 is required for Our SIFT-Subregion ZW.")
    if state is None or not isinstance(state, MethodState):
        raise ValueError("A MethodState from register() is required.")
    h, w = state.ref_shape
    t_resize0 = time.perf_counter()
    bgr = cv2.resize(bgr512, (w, h), interpolation=cv2.INTER_AREA) if bgr512.shape[:2] != (h, w) else bgr512.copy()
    verification_resize_time_sec = time.perf_counter() - t_resize0

    # Reconstruct the reference image is not stored in state to save memory. Since homography
    # needs only shape + registered keypoints/descriptors, use test image shape and ref features.
    # For warping target dimensions, a black dummy image with the reference shape is sufficient.
    dummy_ref = np.zeros((h, w, 3), dtype=np.uint8)
    t_align0 = time.perf_counter()
    align = align_test_to_ref_homography(dummy_ref, bgr, state.ref_kps, state.ref_desc) if getattr(state, "use_alignment", True) else AlignResult(bgr, False, 0, 0, 0.0, "alignment_disabled")
    verification_alignment_time_sec = time.perf_counter() - t_align0
    corr_bgr = align.aligned_bgr if align.ok else bgr

    if getattr(state, "region_mode", "multi") in {"global", "global_region"}:
        t_ext0 = time.perf_counter()
        ms_prime = global_master_share(corr_bgr, wm_size=state.wm_size)
        zw0 = state.tuples[0]["zw"] if len(state.tuples) else np.zeros((state.wm_size*state.wm_size,), dtype=np.uint8)
        rec_scr = (ms_prime ^ zw0).astype(np.uint8)
        verification_local_extract_time_sec = time.perf_counter() - t_ext0
        verification_total_time_sec_internal = time.perf_counter() - t_verify0
        return rec_scr, {
            "Q": 1, "Q_one_to_one": 1, "Q_unique_registered": 1, "Q_unique_test": 1,
            "Q_best_per_test_above_T": 1, "Q_candidate_pairs_above_T": 1, "unique_best_registered_above_T": 1,
            "max_duplicate_count_best_per_test": 1, "matching_mode": "global_region",
            "matched_test_indices": [0], "matched_indices": [0], "similarities": [1.0],
            "candidate_best_similarities": [1.0], "candidate_best_indices": [0],
            "mean_similarity": 1.0, "std_similarity": 0.0, "min_similarity": 1.0, "max_similarity": 1.0,
            "mean_best_similarity_all_candidates": 1.0, "std_best_similarity_all_candidates": 0.0,
            "min_best_similarity_all_candidates": 1.0, "max_best_similarity_all_candidates": 1.0,
            "n_test_detected_keypoints": 0, "n_test_selected_keypoints": 0,
            "n_registered_tuples": len(state.tuples), "n_ref_keypoints_for_align": len(state.ref_kps),
            "n_all_ref_keypoints": state.align_meta.get("n_all_ref_keypoints"),
            "n_valid_topP_keypoints": state.align_meta.get("n_valid_topP_keypoints"),
            "verification_resize_time_sec": verification_resize_time_sec,
            "verification_alignment_time_sec": verification_alignment_time_sec,
            "verification_sift_detect_time_sec": 0.0, "verification_topP_selection_time_sec": 0.0,
            "verification_similarity_matrix_time_sec": 0.0, "verification_matching_assignment_time_sec": 0.0,
            "verification_local_extract_time_sec": verification_local_extract_time_sec, "verification_voting_time_sec": 0.0,
            "verification_total_time_sec_internal": verification_total_time_sec_internal,
            "align_ok": align.ok, "align_reason": align.reason, "good": align.good, "inliers": align.inliers, "inlier_ratio": align.inlier_ratio,
            "voting_enabled": False, "voting_mode": "global_region"
        }

    t_sift0 = time.perf_counter()
    sift = sift_create()
    kps, desc = sift.detectAndCompute(to_gray(corr_bgr), None)
    verification_sift_detect_time_sec = time.perf_counter() - t_sift0
    n_test_detected = 0 if kps is None else len(kps)
    t_select0 = time.perf_counter()
    test_kps, test_desc = pick_keypoints(kps, desc, state.P, image_shape=corr_bgr.shape, sub_size=state.subregion_size, mode=getattr(state, "keypoint_selection", "strongest"), seed=getattr(state, "random_seed", 0)+12345)
    verification_topP_selection_time_sec = time.perf_counter() - t_select0
    n_test_selected = 0 if test_kps is None else len(test_kps)

    if test_desc is None or len(test_kps) == 0 or len(state.tuples) == 0:
        return np.zeros((state.wm_size * state.wm_size,), dtype=np.uint8), {
            "Q": 0, "align_ok": align.ok, "align_reason": align.reason,
            "good": align.good, "inliers": align.inliers, "inlier_ratio": align.inlier_ratio,
            "n_test_detected_keypoints": n_test_detected,
            "n_test_selected_keypoints": n_test_selected,
            "n_registered_tuples": len(state.tuples),
            "n_ref_keypoints_for_align": len(state.ref_kps),
            "n_all_ref_keypoints": state.align_meta.get("n_all_ref_keypoints"),
            "candidate_best_similarities": [],
            "mean_best_similarity_all_candidates": None,
            "max_best_similarity_all_candidates": None,
            "min_best_similarity_all_candidates": None,
        }

    t_sim0 = time.perf_counter()
    reg_desc = np.stack([t["desc"] for t in state.tuples], axis=0).astype(np.float32)
    reg_desc_n = _l2_normalize_desc(reg_desc)
    test_desc_n = _l2_normalize_desc(test_desc.astype(np.float32))

    extracted = []
    matched_indices = []
    matched_test_indices = []
    similarities = []
    candidate_best_similarities = []
    candidate_best_indices = []

    # Similarity matrix between selected test descriptors and registered descriptors.
    # sim_matrix[i, j] is the normalized Euclidean similarity between test keypoint i
    # and registered keypoint j.
    sim_matrix = 1.0 - (np.linalg.norm(test_desc_n[:, None, :] - reg_desc_n[None, :, :], axis=2) / float(np.sqrt(2.0)))
    sim_matrix = np.clip(sim_matrix, 0.0, 1.0)
    verification_similarity_matrix_time_sec = time.perf_counter() - t_sim0

    # Diagnostic: best registered descriptor for each test descriptor before enforcing one-to-one matching.
    best_j_per_test = np.argmax(sim_matrix, axis=1).astype(int)
    best_sim_per_test = sim_matrix[np.arange(sim_matrix.shape[0]), best_j_per_test].astype(float)
    candidate_best_similarities = [float(x) for x in best_sim_per_test.tolist()]
    candidate_best_indices = [int(x) for x in best_j_per_test.tolist()]
    Q_best_per_test_above_T = int(np.sum(best_sim_per_test >= float(state.T)))
    if Q_best_per_test_above_T > 0:
        vals, counts = np.unique(best_j_per_test[best_sim_per_test >= float(state.T)], return_counts=True)
        max_duplicate_count_best_per_test = int(np.max(counts))
        unique_best_registered_above_T = int(len(vals))
    else:
        max_duplicate_count_best_per_test = 0
        unique_best_registered_above_T = 0

    # One-to-one greedy assignment: collect all candidate pairs above T, sort by
    # similarity, and accept a pair only if neither its test keypoint nor its
    # registered keypoint has already been used. This prevents many-to-one
    # matches from inflating Q on different images.
    t_match0 = time.perf_counter()
    cand_i, cand_j = np.where(sim_matrix >= float(state.T))
    candidates = [(float(sim_matrix[i, j]), int(i), int(j)) for i, j in zip(cand_i, cand_j)]
    candidates.sort(key=lambda x: x[0], reverse=True)
    Q_candidate_pairs_above_T = int(len(candidates))
    used_test = set()
    used_reg = set()
    accepted_pairs = []
    for sim, i, j in candidates:
        if i in used_test or j in used_reg:
            continue
        reg = crop_subregion_centered(corr_bgr, test_kps[i], state.subregion_size)
        if reg is None:
            continue
        used_test.add(i)
        used_reg.add(j)
        accepted_pairs.append((sim, i, j, reg))

    verification_matching_assignment_time_sec = time.perf_counter() - t_match0

    t_extract0 = time.perf_counter()
    for sim, i, j, reg in accepted_pairs:
        ms_prime = subregion_master_share(reg, wm_size=state.wm_size, transform_mode=getattr(state, "transform_mode", "dwt_dct"))
        # Algorithm 2: W'_i,k = MS'_i XOR ZW_j. This is the encrypted/scrambled local watermark.
        local_w_scr = (ms_prime ^ state.tuples[j]["zw"]).astype(np.uint8)
        extracted.append(local_w_scr)
        matched_test_indices.append(i)
        matched_indices.append(j)
        similarities.append(float(sim))
    verification_local_extract_time_sec = time.perf_counter() - t_extract0

    Q = len(extracted)
    Q_unique_registered = len(set(matched_indices))
    Q_unique_test = len(set(matched_test_indices))
    voting_mode = "majority_voting" if state.use_voting else "best_match_only"
    best_match_similarity = None
    best_match_registered_index = None

    t_vote0 = time.perf_counter()
    if Q == 0:
        # No detected test keypoint has descriptor similarity >= T against any
        # registered descriptor. In this case, no local watermark is trustworthy,
        # so return an all-zero watermark and expose the failure reason in meta.
        rec_scr = np.zeros((state.wm_size * state.wm_size,), dtype=np.uint8)
        voting_mode = "no_valid_matches"
    elif state.use_voting:
        stack = np.stack(extracted, axis=0).astype(np.uint8)
        # Voting Eq. (3.3): bit=1 if at least half of matched local watermarks vote 1.
        rec_scr = (np.sum(stack, axis=0) >= (Q / 2.0)).astype(np.uint8)
    else:
        # Ablation without voting: keep the sub-region mechanism and descriptor
        # matching, but reconstruct the global watermark from only the single
        # strongest matched local sub-region. This directly measures how much
        # majority voting improves over one local verification result.
        best_idx = int(np.argmax(np.asarray(similarities, dtype=np.float32)))
        rec_scr = extracted[best_idx].astype(np.uint8)
        best_match_similarity = float(similarities[best_idx])
        best_match_registered_index = int(matched_indices[best_idx])

    verification_voting_time_sec = time.perf_counter() - t_vote0
    verification_total_time_sec_internal = time.perf_counter() - t_verify0

    meta = {
        "Q": Q,
        "Q_one_to_one": Q,
        "Q_unique_registered": Q_unique_registered,
        "Q_unique_test": Q_unique_test,
        "Q_best_per_test_above_T": Q_best_per_test_above_T,
        "Q_candidate_pairs_above_T": Q_candidate_pairs_above_T,
        "unique_best_registered_above_T": unique_best_registered_above_T,
        "max_duplicate_count_best_per_test": max_duplicate_count_best_per_test,
        "matching_mode": MATCHING_MODE,
        "supports_ablation_options": True,
        "matched_test_indices": matched_test_indices,
        "voting_enabled": bool(state.use_voting),
        "voting_mode": voting_mode,
        "best_match_similarity": best_match_similarity,
        "best_match_registered_index": best_match_registered_index,
        "matched_indices": matched_indices,
        "similarities": similarities,
        "candidate_best_similarities": candidate_best_similarities,
        "candidate_best_indices": candidate_best_indices,
        "mean_similarity": None if len(similarities) == 0 else float(np.mean(similarities)),
        "std_similarity": None if len(similarities) == 0 else float(np.std(similarities)),
        "min_similarity": None if len(similarities) == 0 else float(np.min(similarities)),
        "max_similarity": None if len(similarities) == 0 else float(np.max(similarities)),
        "mean_best_similarity_all_candidates": None if len(candidate_best_similarities) == 0 else float(np.mean(candidate_best_similarities)),
        "std_best_similarity_all_candidates": None if len(candidate_best_similarities) == 0 else float(np.std(candidate_best_similarities)),
        "min_best_similarity_all_candidates": None if len(candidate_best_similarities) == 0 else float(np.min(candidate_best_similarities)),
        "max_best_similarity_all_candidates": None if len(candidate_best_similarities) == 0 else float(np.max(candidate_best_similarities)),
        "n_test_detected_keypoints": n_test_detected,
        "n_test_selected_keypoints": n_test_selected,
        "n_registered_tuples": len(state.tuples),
        "n_ref_keypoints_for_align": len(state.ref_kps),
        "n_all_ref_keypoints": state.align_meta.get("n_all_ref_keypoints"),
        "n_valid_topP_keypoints": state.align_meta.get("n_valid_topP_keypoints"),
        "verification_resize_time_sec": verification_resize_time_sec,
        "verification_alignment_time_sec": verification_alignment_time_sec,
        "verification_sift_detect_time_sec": verification_sift_detect_time_sec,
        "verification_topP_selection_time_sec": verification_topP_selection_time_sec,
        "verification_similarity_matrix_time_sec": verification_similarity_matrix_time_sec,
        "verification_matching_assignment_time_sec": verification_matching_assignment_time_sec,
        "verification_local_extract_time_sec": verification_local_extract_time_sec,
        "verification_voting_time_sec": verification_voting_time_sec,
        "verification_total_time_sec_internal": verification_total_time_sec_internal,
        "align_ok": align.ok,
        "align_reason": align.reason,
        "good": align.good,
        "inliers": align.inliers,
        "inlier_ratio": align.inlier_ratio,
    }
    return rec_scr.astype(np.uint8), meta


def meta_info() -> Dict[str, Any]:
    return {
        "method": METHOD_NAME,
        "P": DEFAULT_P,
        "T": DEFAULT_T,
        "use_voting": USE_VOTING,
        "subregion_size": SUBREGION_SIZE,
        "wm_size": WM_SIZE,
        "sift_nfeatures": SIFT_NFEATURES,
        "ratio_test": RATIO_TEST,
        "binarize_mode": BINARIZE_MODE,
        "matching_mode": MATCHING_MODE,
        "supports_ablation_options": True,
    }
