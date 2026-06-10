from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from experiment_utils import *
from common.attacks import make_attack_suite_bgr
from common.progress import log, fmt_eta, now_str
from methods.yuan_daisy_dct import YuanDaisyDctExtractor

# Thanh VMF32 implementation can be placed either at:
#   1) methods/thanh_vmf32.py
#   2) thanh_vmf32.py in the same folder as this script
try:
    from methods.thanh_vmf32 import extract_bits as thanh_extract_bits
except ImportError:
    from thanh_vmf32 import extract_bits as thanh_extract_bits


ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()

METHOD_PROPOSED = "Proposed"
METHOD_YUAN = "Yuan DWT-DAISY-DCT"
METHOD_THANH_ALIGN = "Thanh VMF32 + full-SIFT alignment"


# ---------------------------------------------------------------------
# Yuan baseline
# ---------------------------------------------------------------------
def yuan_register(bgr: np.ndarray, wm_scr: np.ndarray) -> dict:
    ext = YuanDaisyDctExtractor()
    feat = ext.extract_bits(bgr)
    zw = (feat ^ wm_scr).astype(np.uint8)
    return {"extractor": ext, "zw": zw}


def yuan_verify(bgr: np.ndarray, state: dict, wm_clear: np.ndarray) -> tuple[dict, np.ndarray]:
    feat = state["extractor"].extract_bits(bgr)
    rec_scr = (feat ^ state["zw"]).astype(np.uint8)
    rec = unscramble_bits(rec_scr, 32)
    nc, ber = calculate_metrics(wm_clear, rec)
    return dict(nc=nc, ber=ber, Q=np.nan, align_ok=np.nan), rec


# ---------------------------------------------------------------------
# Thanh VMF32 baseline with full-SIFT geometric alignment
# ---------------------------------------------------------------------
def _make_sift():
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("OpenCV SIFT is unavailable. Please install opencv-contrib-python.")
    return cv2.SIFT_create()


def _gray(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim == 2:
        return bgr
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _kp_to_array(kps) -> np.ndarray:
    """Store enough keypoint metadata for full-SIFT storage accounting."""
    arr = np.zeros((len(kps), 7), dtype=np.float32)
    for i, kp in enumerate(kps):
        arr[i] = [
            kp.pt[0],
            kp.pt[1],
            kp.size,
            kp.angle,
            kp.response,
            kp.octave,
            kp.class_id,
        ]
    return arr


def _full_sift_storage_bytes(n_keypoints: int, descriptor_dtype_bytes: int = 4) -> int:
    """
    Storage accounting consistent with storing full SIFT:
    - descriptor: 128 float values
    - keypoint metadata: 7 float-like values
    """
    return int(n_keypoints * (128 * descriptor_dtype_bytes + 7 * 4))


def _detect_full_sift(bgr: np.ndarray) -> tuple[list, np.ndarray | None]:
    sift = _make_sift()
    kps, desc = sift.detectAndCompute(_gray(bgr), None)
    if desc is not None:
        desc = desc.astype(np.float32)
    return kps, desc


def _align_by_full_sift(
    attacked_bgr: np.ndarray,
    ref_kps: list,
    ref_desc: np.ndarray | None,
    ref_shape: tuple[int, int, int],
    ratio: float = 0.75,
    min_matches: int = 8,
    ransac_reproj_threshold: float = 5.0,
) -> tuple[np.ndarray, dict]:
    """
    Align attacked image to the registered image coordinate system using full SIFT
    keypoints/descriptors stored at registration.

    Homography direction:
        attacked/test keypoints -> registered/reference keypoints
    """
    t0 = time.perf_counter()
    test_kps, test_desc = _detect_full_sift(attacked_bgr)
    detect_time = time.perf_counter() - t0

    info = {
        "thanh_align_ok": False,
        "thanh_align_reason": "",
        "thanh_n_ref_keypoints": len(ref_kps) if ref_kps is not None else 0,
        "thanh_n_test_keypoints": len(test_kps) if test_kps is not None else 0,
        "thanh_good_matches": 0,
        "thanh_inliers": 0,
        "thanh_inlier_ratio": 0.0,
        "thanh_sift_detect_time_sec": detect_time,
        "thanh_matching_time_sec": 0.0,
        "thanh_warp_time_sec": 0.0,
    }

    if ref_desc is None or test_desc is None or len(ref_kps) < min_matches or len(test_kps) < min_matches:
        info["thanh_align_reason"] = "not_enough_keypoints"
        return attacked_bgr, info

    t0 = time.perf_counter()
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(test_desc, ref_desc, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    info["thanh_matching_time_sec"] = time.perf_counter() - t0
    info["thanh_good_matches"] = len(good)

    if len(good) < min_matches:
        info["thanh_align_reason"] = "not_enough_good_matches"
        return attacked_bgr, info

    src_pts = np.float32([test_kps[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([ref_kps[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_reproj_threshold)
    if H is None or mask is None:
        info["thanh_align_reason"] = "homography_failed"
        return attacked_bgr, info

    inliers = int(mask.ravel().sum())
    info["thanh_inliers"] = inliers
    info["thanh_inlier_ratio"] = inliers / max(len(good), 1)

    if inliers < min_matches:
        info["thanh_align_reason"] = "not_enough_inliers"
        return attacked_bgr, info

    h, w = ref_shape[:2]
    t0 = time.perf_counter()
    aligned = cv2.warpPerspective(
        attacked_bgr,
        H,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    info["thanh_warp_time_sec"] = time.perf_counter() - t0
    info["thanh_align_ok"] = True
    info["thanh_align_reason"] = "ok_H"
    return aligned, info


def thanh_aligned_register(bgr: np.ndarray, wm_scr: np.ndarray) -> tuple[dict, dict]:
    t0 = time.perf_counter()

    t_sift = time.perf_counter()
    kps, desc = _detect_full_sift(bgr)
    sift_time = time.perf_counter() - t_sift

    t_feat = time.perf_counter()
    feat = thanh_extract_bits(bgr)
    zw = (feat ^ wm_scr).astype(np.uint8)
    feature_time = time.perf_counter() - t_feat

    n_kp = len(kps)
    full_sift_bytes = _full_sift_storage_bytes(n_kp)
    zw_bytes = int(np.ceil(zw.size / 8.0))
    total_storage_bytes = full_sift_bytes + zw_bytes

    state = {
        "zw": zw,
        "ref_shape": bgr.shape,
        "ref_kps": kps,
        "ref_desc": desc,
        "n_ref_keypoints": n_kp,
    }

    elapsed = time.perf_counter() - t0
    reg = {
        "registration_algorithm_time_sec": elapsed,
        "registration_end_to_end_time_sec": elapsed,
        "registration_sift_detect_time_sec": sift_time,
        "registration_zw_generation_time_sec": feature_time,
        "n_all_sift_keypoints": n_kp,
        "n_valid_keypoints": n_kp,
        "n_registered_tuples": n_kp,
        "wm_size": 32,
        "full_sift_storage_bytes": full_sift_bytes,
        "topP_sift_storage_bytes": full_sift_bytes,
        "topP_ZW_storage_packed_bytes": zw_bytes,
        "topP_ZW_storage_uint8_bytes": int(zw.size),
        "topP_total_storage_packed_bytes": total_storage_bytes,
        "topP_total_storage_uint8_bytes": full_sift_bytes + int(zw.size),
        "sift_storage_reduction_vs_all": 0.0,
        "topP_total_packed_vs_full_sift_ratio": total_storage_bytes / max(full_sift_bytes, 1),
    }
    return state, reg


def thanh_aligned_verify(
    attacked_bgr: np.ndarray,
    state: dict,
    wm_clear: np.ndarray,
) -> tuple[dict, np.ndarray]:
    t0 = time.perf_counter()

    aligned, align_info = _align_by_full_sift(
        attacked_bgr,
        ref_kps=state["ref_kps"],
        ref_desc=state["ref_desc"],
        ref_shape=state["ref_shape"],
    )

    t_feat = time.perf_counter()
    feat = thanh_extract_bits(aligned)
    feature_time = time.perf_counter() - t_feat

    rec_scr = (feat ^ state["zw"]).astype(np.uint8)
    rec = unscramble_bits(rec_scr, 32)
    nc, ber = calculate_metrics(wm_clear, rec)

    total = time.perf_counter() - t0
    row = dict(
        nc=nc,
        ber=ber,
        Q=np.nan,
        align_ok=align_info["thanh_align_ok"],
        verification_algorithm_time_sec=total,
        verification_total_time_sec_internal=total,
        verification_alignment_time_sec=(
            align_info["thanh_sift_detect_time_sec"]
            + align_info["thanh_matching_time_sec"]
            + align_info["thanh_warp_time_sec"]
        ),
        verification_local_extract_time_sec=feature_time,
    )
    row.update(align_info)
    return row, rec


def baseline_reg_row(
    *,
    method: str,
    image: str,
    dataset: str,
    elapsed_sec: float,
    storage_bytes: int = 128,
) -> dict:
    return dict(
        method=method,
        image=image,
        dataset=dataset,
        P=np.nan,
        T=np.nan,
        registration_algorithm_time_sec=elapsed_sec,
        registration_end_to_end_time_sec=elapsed_sec,
        n_registered_tuples=1,
        topP_total_storage_packed_bytes=storage_bytes,
    )


def update_common_attack_fields(
    row: dict,
    *,
    method: str,
    image: str,
    dataset: str,
    attack: dict,
    psnr: float,
    P: float | int | None,
    T: float | None,
) -> dict:
    row.update(
        method=method,
        image=image,
        dataset=dataset,
        suite=attack.get("suite"),
        attack_name=attack["name"],
        attack_group=attack["group"],
        attack_intensity=attack["intensity"],
        attack_psnr_db=psnr,
        psnr_actual_db=psnr,
        P=P if P is not None else np.nan,
        T=T if T is not None else np.nan,
    )
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--medical-dir", default="data/dataset2_medical_12")
    ap.add_argument("--out-dir", default="out_baseline_suite1")
    ap.add_argument("--watermark", default="data/watermark/watermark.png")
    ap.add_argument("--P", type=int, default=50)
    ap.add_argument("--T", type=float, default=0.65)
    ap.add_argument("--Qmin", type=int, default=10)
    ap.add_argument("--NC-threshold", type=float, default=0.70)
    ap.add_argument("--progress-every", type=int, default=100)
    ap.add_argument("--partial-every", type=int, default=1000)
    args = ap.parse_args()

    t_all = time.perf_counter()
    out = ensure_dir(ROOT / args.out_dir)
    log(f"Output directory: {out}")

    images = load_datasets(ROOT, [args.medical_dir])
    attacks = make_attack_suite_bgr("suite1")
    if not images:
        raise RuntimeError("No medical images found.")

    methods = [METHOD_PROPOSED, METHOD_YUAN, METHOD_THANH_ALIGN]
    log(f"Loaded {len(images)} medical images; Suite 1 has {len(attacks)} attacks.")
    log(f"Methods: {', '.join(methods)}")

    wm_clear, _ = load_watermark_bits(ROOT / args.watermark, 32)
    wm_scr = scramble_bits(wm_clear, 32)

    manifest = vars(args) | {
        "methods": methods,
        "thanh_alignment": "full_sift_homography",
        "thanh_storage": "full_sift_keypoints_descriptors_plus_32x32_zero_watermark_bits",
        "n_images": len(images),
        "n_attacks": len(attacks),
        "started_at": now_str(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    rows: list[dict] = []
    regrows: list[dict] = []
    proposed_states: dict[str, dict] = {}
    yuan_states: dict[str, dict] = {}
    thanh_states: dict[str, dict] = {}

    log("Stage 1/3: Registering Proposed, Yuan, and Thanh aligned states.")
    t_stage = time.perf_counter()

    for idx, (name, path, ds) in enumerate(images, start=1):
        # Proposed method
        st, reg = run_register(path, wm_scr, P=args.P, T=args.T, wm_size=32)
        proposed_states[name] = st
        reg.update(method=METHOD_PROPOSED, image=name, dataset=ds, P=args.P, T=args.T)
        regrows.append(reg)

        host, _ = read_bgr_timed(path)

        # Yuan baseline
        t0 = time.perf_counter()
        ys = yuan_register(host, wm_scr)
        rt = time.perf_counter() - t0
        yuan_states[name] = ys
        regrows.append(
            baseline_reg_row(
                method=METHOD_YUAN,
                image=name,
                dataset=ds,
                elapsed_sec=rt,
                storage_bytes=128,
            )
        )

        # Thanh VMF32 baseline with full-SIFT alignment
        ts, treg = thanh_aligned_register(host, wm_scr)
        thanh_states[name] = ts
        treg.update(method=METHOD_THANH_ALIGN, image=name, dataset=ds, P=np.nan, T=np.nan)
        regrows.append(treg)

        log(f"  Registered {idx}/{len(images)} images for all methods ({fmt_eta(t_stage, idx, len(images))})")

    pd.DataFrame(regrows).to_csv(out / "baseline_suite1_registration_metadata.csv", index=False)

    log("Stage 2/3: Running Suite 1 attacks for Proposed, Yuan, and Thanh aligned.")
    total = len(images) * len(attacks) * len(methods)
    done = 0
    t_stage = time.perf_counter()

    for img_idx, (name, path, ds) in enumerate(images, start=1):
        host, _ = read_bgr_timed(path)

        for attack in attacks:
            attacked = attack["fn"](host.copy())
            psnr = psnr_color(host, attacked)

            # Proposed method
            row, _ = run_verify_from_image(attacked, proposed_states[name], wm_clear, 32)
            update_common_attack_fields(
                row,
                method=METHOD_PROPOSED,
                image=name,
                dataset=ds,
                attack=attack,
                psnr=psnr,
                P=args.P,
                T=args.T,
            )
            rows.append(row)
            done += 1

            # Yuan baseline
            t0 = time.perf_counter()
            yr, _ = yuan_verify(attacked, yuan_states[name], wm_clear)
            vt = time.perf_counter() - t0
            yr.update(
                verification_algorithm_time_sec=vt,
                verification_total_time_sec_internal=vt,
            )
            update_common_attack_fields(
                yr,
                method=METHOD_YUAN,
                image=name,
                dataset=ds,
                attack=attack,
                psnr=psnr,
                P=None,
                T=None,
            )
            rows.append(yr)
            done += 1

            # Thanh baseline with full-SIFT alignment
            tr, _ = thanh_aligned_verify(attacked, thanh_states[name], wm_clear)
            update_common_attack_fields(
                tr,
                method=METHOD_THANH_ALIGN,
                image=name,
                dataset=ds,
                attack=attack,
                psnr=psnr,
                P=None,
                T=None,
            )
            rows.append(tr)
            done += 1

            if args.progress_every and done % args.progress_every == 0:
                log(f"  Baseline comparison progress {done}/{total} ({fmt_eta(t_stage, done, total)})")

            if args.partial_every and done % args.partial_every == 0:
                pd.DataFrame(rows).to_csv(out / "baseline_suite1_detail_partial.csv", index=False)

        log(f"  Finished {img_idx}/{len(images)} images")

    df = add_effective_cols(pd.DataFrame(rows), [args.Qmin], thresholds=[args.NC_threshold])
    df.to_csv(out / "baseline_suite1_detail.csv", index=False)

    summarize_robustness(df, ["method", "attack_group"], [args.Qmin]).to_csv(
        out / "baseline_suite1_summary_by_group.csv", index=False
    )
    summarize_robustness(df, ["method", "attack_group", "attack_name", "attack_intensity"], [args.Qmin]).to_csv(
        out / "baseline_suite1_summary_by_attack.csv", index=False
    )

    manifest = vars(args) | {
        "methods": methods,
        "thanh_alignment": "full_sift_homography",
        "thanh_storage": "full_sift_keypoints_descriptors_plus_32x32_zero_watermark_bits",
        "n_images": len(images),
        "n_attacks": len(attacks),
        "finished_at": now_str(),
        "total_elapsed_sec": time.perf_counter() - t_all,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"DONE. Total elapsed {(time.perf_counter() - t_all) / 60:.1f} min. Output: {out}")


if __name__ == "__main__":
    main()
