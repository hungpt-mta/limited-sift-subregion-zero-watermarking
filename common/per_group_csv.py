import os
import re
import csv
import numpy as np

def _sanitize(s: str) -> str:
    keep = []
    for ch in str(s):
        if ch.isalnum() or ch in ("_", "-", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)

def attack_family(attack_name: str) -> str:
    a = str(attack_name)
    if a == "baseline":
        return "baseline"
    if a.startswith("gauss_noise_sigma"):
        return "gauss_noise"
    if a.startswith("saltpepper_d"):
        return "saltpepper"
    if a.startswith("speckle_var"):
        return "speckle"
    if a.startswith("gauss_blur_k"):
        return "gauss_blur"
    if a.startswith("median_blur_k"):
        return "median_blur"
    if a.startswith("jpeg_q"):
        return "jpeg"
    if a.startswith("resize_x"):
        return "resize"
    if a.startswith("rotate_") and a.endswith("deg"):
        return "rotate"
    if a.startswith("crop_center_"):
        return "crop_center"
    if a.startswith("brightness_beta"):
        return "brightness"
    if a.startswith("contrast_alpha"):
        return "contrast"
    if a.startswith("translate_x"):
        return "translate_x"
    if "_" in a:
        return a.rsplit("_", 1)[0]
    return a

def attack_param_value(attack_name: str) -> str:
    a = str(attack_name)
    if a == "baseline":
        return "baseline"

    patterns = [
        (r"^gauss_noise_sigma([0-9]*\.?[0-9]+)$", 1),
        (r"^saltpepper_d([0-9]*\.?[0-9]+)$", 1),
        (r"^speckle_var([0-9]*\.?[0-9]+)$", 1),
        (r"^gauss_blur_k([0-9]+)$", 1),
        (r"^median_blur_k([0-9]+)$", 1),
        (r"^jpeg_q([0-9]+)$", 1),
        (r"^resize_x([0-9]*\.?[0-9]+)$", 1),
        (r"^rotate_(-?[0-9]*\.?[0-9]+)deg$", 1),
        (r"^crop_center_([0-9]*\.?[0-9]+)$", 1),
        (r"^brightness_beta(-?[0-9]*\.?[0-9]+)$", 1),
        (r"^contrast_alpha([0-9]*\.?[0-9]+)$", 1),
        (r"^translate_x([0-9]*\.?[0-9]+)$", 1),
    ]
    for pat, gi in patterns:
        m = re.match(pat, a)
        if m:
            return m.group(gi)

    m = re.findall(r"-?[0-9]*\.?[0-9]+", a)
    if m:
        return m[-1]
    return a

def _param_sort_key(param_str: str):
    if param_str == "baseline":
        return (0, 0.0, param_str)
    try:
        return (1, float(param_str), param_str)
    except Exception:
        return (2, 0.0, param_str)

def write_per_attackgroup_matrix_csv(rows, metric: str, out_dir: str):
    assert metric in ("nc", "ber")
    os.makedirs(out_dir, exist_ok=True)

    methods = sorted({str(r["method"]) for r in rows})

    buckets = {}
    for r in rows:
        attack = str(r["attack"])
        fam = attack_family(attack)
        param = attack_param_value(attack)
        m = str(r["method"])
        buckets.setdefault((fam, param, m), []).append(float(r[metric]))

    families = sorted({attack_family(str(r["attack"])) for r in rows})

    for fam in families:
        params = sorted({param for (ff, param, m) in buckets.keys() if ff == fam}, key=_param_sort_key)

        mat = {}
        for param in params:
            row = {}
            for m in methods:
                vals = buckets.get((fam, param, m), [])
                row[m] = float(np.mean(np.asarray(vals, dtype=np.float32))) if vals else np.nan
            mat[param] = row

        out_path = os.path.join(out_dir, f"per_group_{metric}_{_sanitize(fam)}.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["param"] + methods)
            for param in params:
                w.writerow([param] + [("" if np.isnan(mat[param][m]) else f"{mat[param][m]:.4f}") for m in methods])

        print(f"[OK] Saved per-group MATRIX {metric.upper()} CSV: {out_path}")

def write_runtime_mean_csv(rows, out_csv: str):
    reg_by_im = {}
    ver_by_m = {}

    for r in rows:
        img = str(r["image"])
        m = str(r["method"])
        reg_by_im[(img, m)] = float(r["t_reg_ms"])
        ver_by_m.setdefault(m, []).append(float(r["t_ver_ms"]))

    methods = sorted({str(r["method"]) for r in rows})

    reg_mean = {}
    for m in methods:
        vals = [v for (img, mm), v in reg_by_im.items() if mm == m]
        reg_mean[m] = float(np.mean(np.array(vals, dtype=np.float32))) if vals else 0.0

    ver_mean = {}
    for m in methods:
        vals = ver_by_m.get(m, [])
        ver_mean[m] = float(np.mean(np.array(vals, dtype=np.float32))) if vals else 0.0

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "t_reg_ms_mean", "t_ver_ms_mean"])
        for m in methods:
            w.writerow([m, f"{reg_mean[m]:.4f}", f"{ver_mean[m]:.4f}"])

    print(f"[OK] Saved runtime mean CSV: {out_csv}")
