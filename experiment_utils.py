from __future__ import annotations

import json, math, time, hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
import pandas as pd

from common.io_utils import list_images, safe_name, imread_bgr_resized
from common.arnold import arnold_scramble_bin, arnold_unscramble_bin
from common.metrics import calculate_metrics, psnr_color
from common.attacks import make_attack_suite_bgr
from methods import our_sift_subregion_zw as our

IMG_SIZE = 512
DESC_DIM = 128
FLOAT_BYTES = 4
INT_BYTES = 4
KP_META_BYTES = 5 * FLOAT_BYTES + 2 * INT_BYTES
DESC_BYTES = DESC_DIM * FLOAT_BYTES


def stable_seed(*parts) -> int:
    h = hashlib.sha256('::'.join(map(str, parts)).encode('utf-8')).hexdigest()
    return int(h[:8], 16)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_images_optional(folder: Path) -> List[Tuple[str, Path]]:
    if not folder.exists():
        return []
    try:
        return [(safe_name(p), Path(p)) for p in list_images(str(folder))]
    except FileNotFoundError:
        return []


def load_datasets(root: Path, folders: List[str]) -> List[Tuple[str, Path, str]]:
    out=[]
    for folder in folders:
        f = root / folder
        for name, path in list_images_optional(f):
            out.append((f'{folder.replace("/","_")}_{name}', path, folder))
    if not out:
        raise FileNotFoundError('No input images found in: ' + ', '.join(folders))
    return out


def load_watermark_bits(path: Path, size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    wm = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if wm is None:
        raise FileNotFoundError(f'Cannot read watermark: {path}')
    wm = cv2.resize(wm, (int(size), int(size)), interpolation=cv2.INTER_NEAREST)
    bits = (wm >= 128).astype(np.uint8)
    return bits.reshape(-1), wm


def scramble_bits(bits: np.ndarray, size: int) -> np.ndarray:
    return arnold_scramble_bin(bits.reshape(size, size).astype(np.uint8)).reshape(-1).astype(np.uint8)


def unscramble_bits(bits_scr: np.ndarray, size: int) -> np.ndarray:
    return arnold_unscramble_bin(bits_scr.reshape(size, size).astype(np.uint8)).reshape(-1).astype(np.uint8)


def bits_to_image(bits: np.ndarray, size: int) -> np.ndarray:
    return bits.reshape(size, size).astype(np.uint8) * 255


def save_bits_image(bits: np.ndarray, size: int, path: Path) -> float:
    ensure_dir(path.parent)
    t0=time.perf_counter()
    cv2.imwrite(str(path), bits_to_image(bits, size))
    return time.perf_counter()-t0


def read_bgr_timed(path: Path) -> Tuple[np.ndarray, float]:
    t0=time.perf_counter()
    bgr = imread_bgr_resized(str(path), size=IMG_SIZE)
    return bgr, time.perf_counter()-t0


def write_attacked_image(path: Path, bgr: np.ndarray) -> float:
    ensure_dir(path.parent)
    t0=time.perf_counter()
    ok=cv2.imwrite(str(path), bgr)
    if not ok:
        raise IOError(f'Failed to write {path}')
    return time.perf_counter()-t0


def storage_estimate(st: our.MethodState) -> Dict[str, Any]:
    n_all = int(st.align_meta.get('n_all_ref_keypoints') or 0)
    n_valid = int(st.align_meta.get('n_valid_topP_keypoints') or 0)
    n_store = int(len(st.tuples))
    wm_size = int(st.wm_size)
    zw_packed = math.ceil((wm_size*wm_size)/8)
    zw_uint8 = wm_size*wm_size
    per = KP_META_BYTES + DESC_BYTES
    full = n_all * per
    top_sift = n_store * per
    total_packed = top_sift + n_store * zw_packed
    total_uint8 = top_sift + n_store * zw_uint8
    return dict(
        n_all_sift_keypoints=n_all,
        n_valid_keypoints=n_valid,
        n_registered_tuples=n_store,
        wm_size=wm_size,
        full_sift_storage_bytes=full,
        topP_sift_storage_bytes=top_sift,
        topP_ZW_storage_packed_bytes=n_store*zw_packed,
        topP_ZW_storage_uint8_bytes=n_store*zw_uint8,
        topP_total_storage_packed_bytes=total_packed,
        topP_total_storage_uint8_bytes=total_uint8,
        sift_storage_reduction_vs_all=(None if full<=0 else 1-top_sift/full),
        topP_total_packed_vs_full_sift_ratio=(None if full<=0 else total_packed/full),
    )


def flatten_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        'Q','Q_one_to_one','Q_unique_registered','Q_unique_test','Q_best_per_test_above_T',
        'Q_candidate_pairs_above_T','unique_best_registered_above_T','max_duplicate_count_best_per_test',
        'matching_mode','align_ok','align_reason','good','inliers','inlier_ratio',
        'n_all_ref_keypoints','n_valid_topP_keypoints','n_registered_tuples','n_ref_keypoints_for_align',
        'n_test_detected_keypoints','n_test_selected_keypoints','mean_similarity','std_similarity','min_similarity','max_similarity',
        'mean_best_similarity_all_candidates','std_best_similarity_all_candidates','min_best_similarity_all_candidates','max_best_similarity_all_candidates',
        'voting_mode','voting_enabled','verification_resize_time_sec','verification_alignment_time_sec','verification_sift_detect_time_sec',
        'verification_topP_selection_time_sec','verification_similarity_matrix_time_sec','verification_matching_assignment_time_sec',
        'verification_local_extract_time_sec','verification_voting_time_sec','verification_total_time_sec_internal'
    ]
    d={k:meta.get(k) for k in keys}
    for lk in ['similarities','matched_indices','matched_test_indices','candidate_best_similarities','candidate_best_indices']:
        vals=meta.get(lk, []) or []
        if lk.endswith('indices') or lk in ['matched_indices','matched_test_indices','candidate_best_indices']:
            d[lk]=';'.join(str(int(x)) for x in vals[:200])
        else:
            d[lk]=';'.join(f'{float(x):.4f}' for x in vals[:200])
    return d


def add_effective_cols(df: pd.DataFrame, qmins: List[int], nc_col='nc', q_col='Q', thresholds=(0.85,0.90)) -> pd.DataFrame:
    out=df.copy()
    q=out[q_col].fillna(0)
    for qmin in qmins:
        eff=f'nc_effective_Qmin{qmin}'
        out[eff]=out[nc_col].where(q>=qmin, 0.0)
        for th in thresholds:
            out[f'accepted_Qmin{qmin}_NC{int(th*100):03d}']=(out[eff]>=th).astype(int)
    return out


def run_register(image_path: Path, wm_scr_bits: np.ndarray, *, P=50, T=0.65, wm_size=32, use_voting=True, use_alignment=True, transform_mode='dwt_dct', keypoint_selection='strongest', region_mode='multi', random_seed=0) -> Tuple[our.MethodState, Dict[str,Any]]:
    t0=time.perf_counter()
    host, read_host = read_bgr_timed(image_path)
    t_alg=time.perf_counter()
    _, st = our.register(host, wm_scr_bits, P=P, T=T, use_voting=use_voting, use_alignment=use_alignment, transform_mode=transform_mode, keypoint_selection=keypoint_selection, region_mode=region_mode, random_seed=random_seed, wm_size=wm_size)
    alg=time.perf_counter()-t_alg
    e2e=time.perf_counter()-t0
    row={'registration_end_to_end_time_sec': e2e, 'registration_algorithm_time_sec': alg, 'registration_read_host_image_time_sec': read_host}
    row.update({k:st.align_meta.get(k) for k in [
        'registration_resize_time_sec','registration_sift_detect_time_sec','registration_topP_selection_time_sec','registration_subregion_crop_time_sec','registration_ms_extract_time_sec','registration_xor_store_time_sec','registration_zw_generation_time_sec','registration_total_time_sec_internal','transform_mode','keypoint_selection','region_mode','random_seed']})
    row.update(storage_estimate(st))
    return st,row


def run_verify_from_image(bgr: np.ndarray, st: our.MethodState, wm_clear_bits: np.ndarray, wm_size:int) -> Tuple[Dict[str,Any], np.ndarray]:
    t0=time.perf_counter()
    rec_scr, meta = our.verify(bgr, st)
    alg=time.perf_counter()-t0
    rec_clear = unscramble_bits(rec_scr, wm_size)
    nc, ber = calculate_metrics(wm_clear_bits, rec_clear)
    row={'nc':nc,'ber':ber,'verification_algorithm_time_sec':alg}
    row.update(flatten_meta(meta))
    return row, rec_clear


def summarize_robustness(df: pd.DataFrame, group_cols: List[str], qmins: List[int]) -> pd.DataFrame:
    aggs = dict(
        mean_nc_raw=('nc','mean'), std_nc_raw=('nc','std'), min_nc_raw=('nc','min'), max_nc_raw=('nc','max'),
        mean_Q=('Q','mean'), std_Q=('Q','std'), min_Q=('Q','min'), max_Q=('Q','max'),
        mean_align_ok=('align_ok','mean'), n=('nc','count')
    )
    # PSNR is used only to characterize attack severity: PSNR(original host, attacked image).
    # It is not an imperceptibility metric for zero-watermarking.
    psnr_col = None
    if 'attack_psnr_db' in df.columns:
        psnr_col = 'attack_psnr_db'
    elif 'psnr_actual_db' in df.columns:
        psnr_col = 'psnr_actual_db'
    if psnr_col is not None:
        aggs.update({
            'mean_attack_psnr_db': (psnr_col, 'mean'),
            'std_attack_psnr_db': (psnr_col, 'std'),
            'min_attack_psnr_db': (psnr_col, 'min'),
            'max_attack_psnr_db': (psnr_col, 'max'),
        })
    base=df.groupby(group_cols, dropna=False).agg(**aggs).reset_index()
    for qmin in qmins:
        eff=f'nc_effective_Qmin{qmin}'
        if eff in df.columns:
            e=df.groupby(group_cols, dropna=False).agg(**{
                f'mean_nc_effective_Qmin{qmin}':(eff,'mean'),
                f'std_nc_effective_Qmin{qmin}':(eff,'std'),
                f'min_nc_effective_Qmin{qmin}':(eff,'min')
            }).reset_index()
            base=base.merge(e,on=group_cols,how='left')
    return base


def decision_summary(df: pd.DataFrame, qmins: List[int], thresholds=(0.85,0.90), group_cols: Optional[List[str]]=None) -> pd.DataFrame:
    rows=[]
    if group_cols is None:
        groups=[((), df)]
    else:
        groups=list(df.groupby(group_cols, dropna=False))
    for key, sub in groups:
        if group_cols is None:
            base={}
        else:
            if not isinstance(key, tuple): key=(key,)
            base=dict(zip(group_cols,key))
        for qmin in qmins:
            eff=f'nc_effective_Qmin{qmin}'
            for th in thresholds:
                accepted=(sub[eff]>=th)
                row=dict(base)
                row.update(Qmin=qmin, NC_threshold=th, trials=len(sub), accepted=int(accepted.sum()), success_rate=float(accepted.mean()), mean_nc_raw=float(sub['nc'].mean()), std_nc_raw=float(sub['nc'].std() if len(sub)>1 else 0), mean_nc_effective=float(sub[eff].mean()), std_nc_effective=float(sub[eff].std() if len(sub)>1 else 0), mean_Q=float(sub['Q'].mean()), std_Q=float(sub['Q'].std() if len(sub)>1 else 0))
                rows.append(row)
    return pd.DataFrame(rows)
