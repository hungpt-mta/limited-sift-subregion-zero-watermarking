from __future__ import annotations
import argparse, json, time, re
from pathlib import Path
import cv2
import pandas as pd
import numpy as np
from experiment_utils import *
from common.progress import log, fmt_eta, now_str

ROOT = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd().resolve()


def safe_name(s: str) -> str:
    s = str(s).strip().replace('\\', '/').split('/')[-1]
    s = re.sub(r'[^A-Za-z0-9_.-]+', '_', s)
    return s or 'dataset'


def phash_like(path: Path):
    """A lightweight pHash-like descriptor used only as an auxiliary similarity indicator."""
    bgr, _ = read_bgr_timed(path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)
    low = dct[:8, :8].copy()
    vals = low.flatten()[1:]
    return (vals >= np.median(vals)).astype(np.uint8), bgr


def ham(a, b) -> float:
    return float(np.mean(a ^ b))


def summarize_pair_type(df: pd.DataFrame, eff_col: str, threshold: float, dataset_value: str | None = None):
    rows = []
    group_cols = ['pair_type'] if dataset_value is None else ['dataset', 'pair_type']
    for keys, sub in df.groupby(group_cols, dropna=False):
        if dataset_value is None:
            pair_type = keys
            dataset = 'ALL'
        else:
            dataset, pair_type = keys
        acc = sub[eff_col] >= threshold
        rows.append(dict(
            dataset=dataset,
            pair_type=pair_type,
            trials=int(len(sub)),
            accepted=int(acc.sum()),
            acceptance_rate=float(acc.mean()) if len(sub) else 0.0,
            mean_nc_raw=float(sub['nc'].mean()) if len(sub) else np.nan,
            std_nc_raw=float(sub['nc'].std()) if len(sub) > 1 else 0.0,
            min_nc_raw=float(sub['nc'].min()) if len(sub) else np.nan,
            max_nc_raw=float(sub['nc'].max()) if len(sub) else np.nan,
            mean_nc_effective=float(sub[eff_col].mean()) if len(sub) else np.nan,
            std_nc_effective=float(sub[eff_col].std()) if len(sub) > 1 else 0.0,
            min_nc_effective=float(sub[eff_col].min()) if len(sub) else np.nan,
            max_nc_effective=float(sub[eff_col].max()) if len(sub) else np.nan,
            mean_Q=float(sub['Q'].mean()) if len(sub) else np.nan,
            std_Q=float(sub['Q'].std()) if len(sub) > 1 else 0.0,
            min_Q=float(sub['Q'].min()) if len(sub) else np.nan,
            max_Q=float(sub['Q'].max()) if len(sub) else np.nan,
            mean_phash_distance=float(sub['phash_distance'].mean()) if 'phash_distance' in sub.columns and len(sub) else np.nan,
        ))
    return rows


def make_decision_summary(df: pd.DataFrame, eff_col: str, threshold: float):
    rows = []
    for dataset, sub in df.groupby('dataset', dropna=False):
        gen = sub[sub['pair_type'] == 'genuine']
        imp = sub[sub['pair_type'] == 'impostor']
        gen_acc = int((gen[eff_col] >= threshold).sum()) if len(gen) else 0
        imp_acc = int((imp[eff_col] >= threshold).sum()) if len(imp) else 0
        rows.append(dict(
            dataset=dataset,
            genuine_trials=int(len(gen)),
            genuine_accepted=gen_acc,
            FRR=float(1.0 - gen_acc / len(gen)) if len(gen) else np.nan,
            impostor_trials=int(len(imp)),
            false_accepts=imp_acc,
            FAR=float(imp_acc / len(imp)) if len(imp) else np.nan,
            mean_nc_effective_genuine=float(gen[eff_col].mean()) if len(gen) else np.nan,
            mean_nc_effective_impostor=float(imp[eff_col].mean()) if len(imp) else np.nan,
            mean_Q_genuine=float(gen['Q'].mean()) if len(gen) else np.nan,
            mean_Q_impostor=float(imp['Q'].mean()) if len(imp) else np.nan,
            max_Q_impostor=float(imp['Q'].max()) if len(imp) else np.nan,
            max_nc_effective_impostor=float(imp[eff_col].max()) if len(imp) else np.nan,
        ))
    # Overall row
    gen = df[df['pair_type'] == 'genuine']
    imp = df[df['pair_type'] == 'impostor']
    gen_acc = int((gen[eff_col] >= threshold).sum()) if len(gen) else 0
    imp_acc = int((imp[eff_col] >= threshold).sum()) if len(imp) else 0
    rows.append(dict(
        dataset='ALL',
        genuine_trials=int(len(gen)),
        genuine_accepted=gen_acc,
        FRR=float(1.0 - gen_acc / len(gen)) if len(gen) else np.nan,
        impostor_trials=int(len(imp)),
        false_accepts=imp_acc,
        FAR=float(imp_acc / len(imp)) if len(imp) else np.nan,
        mean_nc_effective_genuine=float(gen[eff_col].mean()) if len(gen) else np.nan,
        mean_nc_effective_impostor=float(imp[eff_col].mean()) if len(imp) else np.nan,
        mean_Q_genuine=float(gen['Q'].mean()) if len(gen) else np.nan,
        mean_Q_impostor=float(imp['Q'].mean()) if len(imp) else np.nan,
        max_Q_impostor=float(imp['Q'].max()) if len(imp) else np.nan,
        max_nc_effective_impostor=float(imp[eff_col].max()) if len(imp) else np.nan,
    ))
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=(
            'Within-directory discriminability test. Each input directory is evaluated independently. '
            'No cross-directory comparisons are performed. By default, impostor pairs are directed: '
            '(registered=a,test=b) and (registered=b,test=a) are counted as two verification trials.'
        )
    )
    ap.add_argument('--dirs', nargs='+', default=['hard_negative_bonexray', 'hard_negative_chestxray', 'hard_negative_ct', 'hard_negative_mri', 'hard_negative_ultrasound'],
                    help='List of directories. Each directory should contain images from the same modality/category.')
    ap.add_argument('--out-dir', default='out_hard_negative')
    ap.add_argument('--watermark', default='data/watermark/watermark.png')
    ap.add_argument('--P', type=int, default=50)
    ap.add_argument('--T', type=float, default=0.65)
    ap.add_argument('--Qmin', type=int, default=10)
    ap.add_argument('--NC-threshold', type=float, default=0.85)
    ap.add_argument('--wm-size', type=int, default=32)
    ap.add_argument('--undirected-impostors', action='store_true',
                    help='If set, each unordered impostor pair is evaluated once only. Default is directed all-pairs.')
    ap.add_argument('--progress-every', type=int, default=100)
    ap.add_argument('--partial-every', type=int, default=1000)
    args = ap.parse_args()

    t_all = time.perf_counter()
    out = ensure_dir(ROOT / args.out_dir)

    images = load_datasets(ROOT, args.dirs)
    if not images:
        raise RuntimeError('No images found for within-directory discriminability test.')

    # Create robust unique IDs and group by directory/dataset.
    records = []
    seen = set()
    for name, path, ds in images:
        ds_safe = safe_name(ds)
        base = str(name)
        uid = f'{ds_safe}__{base}'
        k = 2
        while uid in seen:
            uid = f'{ds_safe}__{base}_{k}'
            k += 1
        seen.add(uid)
        records.append(dict(uid=uid, image=base, path=Path(path), dataset=ds_safe))

    groups = {ds: recs for ds, recs in pd.DataFrame(records).groupby('dataset', sort=False)}
    groups = {ds: [dict(r) for _, r in sub.iterrows()] for ds, sub in groups.items()}

    log(f'Output directory: {out}')
    log(f'Loaded {len(records)} images from {len(groups)} directories: {list(groups.keys())}')
    log('Comparison mode: ' + ('undirected impostor pairs' if args.undirected_impostors else 'directed impostor pairs'))

    wm_clear, _ = load_watermark_bits(ROOT / args.watermark, args.wm_size)
    wm_scr = scramble_bits(wm_clear, args.wm_size)

    manifest = vars(args) | {
        'n_images': len(records),
        'datasets': {ds: len(recs) for ds, recs in groups.items()},
        'comparison_scope': 'within_directory_only',
        'cross_directory_pairs': False,
        'directed_impostor_pairs': not args.undirected_impostors,
        'started_at': now_str(),
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    log('Stage 1/4: Computing pHash-like features and loading images into RAM.')
    feats = {}
    bgr_cache = {}
    t_stage = time.perf_counter()
    for idx, rec in enumerate(records, start=1):
        feats[rec['uid']], bgr_cache[rec['uid']] = phash_like(rec['path'])
        if idx % max(1, min(args.progress_every, 50)) == 0 or idx == len(records):
            log(f'  pHash/images {idx}/{len(records)} ({fmt_eta(t_stage, idx, len(records))})')

    log('Stage 2/4: Registering all images.')
    states = {}
    regrows = []
    t_stage = time.perf_counter()
    for idx, rec in enumerate(records, start=1):
        st, reg = run_register(rec['path'], wm_scr, P=args.P, T=args.T, wm_size=args.wm_size)
        states[rec['uid']] = st
        reg.update(uid=rec['uid'], image=rec['image'], dataset=rec['dataset'])
        regrows.append(reg)
        if idx % max(1, min(args.progress_every, 50)) == 0 or idx == len(records):
            log(f'  Registered {idx}/{len(records)} ({fmt_eta(t_stage, idx, len(records))})')
    pd.DataFrame(regrows).to_csv(out / 'within_dir_registration_metadata.csv', index=False)

    # Total expected trials.
    total = 0
    for ds, recs in groups.items():
        n = len(recs)
        total += n  # genuine
        total += n * (n - 1) // 2 if args.undirected_impostors else n * (n - 1)

    log('Stage 3/4: Verifying genuine and within-directory all impostor pairs.')
    log(f'  Expected total verification trials: {total}')
    rows = []
    done = 0
    t_stage = time.perf_counter()

    for ds, recs in groups.items():
        n = len(recs)
        log(f'  Dataset {ds}: {n} images; genuine={n}; impostor=' +
            (f'{n*(n-1)//2} unordered' if args.undirected_impostors else f'{n*(n-1)} directed'))

        # Genuine trials
        for rec in recs:
            row, _ = run_verify_from_image(bgr_cache[rec['uid']], states[rec['uid']], wm_clear, args.wm_size)
            row.update(
                dataset=ds,
                registered_uid=rec['uid'], test_uid=rec['uid'],
                registered_image=rec['image'], test_image=rec['image'],
                registered_dataset=ds, test_dataset=ds,
                pair_type='genuine', phash_distance=0.0,
                directed=not args.undirected_impostors,
            )
            rows.append(row)
            done += 1
            if args.progress_every and done % args.progress_every == 0:
                log(f'  Progress {done}/{total} ({fmt_eta(t_stage, done, total)})')
            if args.partial_every and done % args.partial_every == 0:
                pd.DataFrame(rows).to_csv(out / 'within_dir_discriminability_detail_partial.csv', index=False)

        # Impostor trials within the same directory only.
        if args.undirected_impostors:
            pair_iter = ((i, j) for i in range(n) for j in range(i + 1, n))
        else:
            pair_iter = ((i, j) for i in range(n) for j in range(n) if i != j)

        for i, j in pair_iter:
            reg_rec = recs[i]
            test_rec = recs[j]
            d = ham(feats[reg_rec['uid']], feats[test_rec['uid']])
            row, _ = run_verify_from_image(bgr_cache[test_rec['uid']], states[reg_rec['uid']], wm_clear, args.wm_size)
            row.update(
                dataset=ds,
                registered_uid=reg_rec['uid'], test_uid=test_rec['uid'],
                registered_image=reg_rec['image'], test_image=test_rec['image'],
                registered_dataset=ds, test_dataset=ds,
                pair_type='impostor', phash_distance=d,
                directed=not args.undirected_impostors,
            )
            rows.append(row)
            done += 1
            if args.progress_every and done % args.progress_every == 0:
                log(f'  Progress {done}/{total} ({fmt_eta(t_stage, done, total)})')
            if args.partial_every and done % args.partial_every == 0:
                pd.DataFrame(rows).to_csv(out / 'within_dir_discriminability_detail_partial.csv', index=False)

    log('Stage 4/4: Applying Qmin rule and writing summaries.')
    df = add_effective_cols(pd.DataFrame(rows), [args.Qmin], thresholds=[args.NC_threshold])
    eff = f'nc_effective_Qmin{args.Qmin}'
    accepted_col = f'accepted_Qmin{args.Qmin}_NC{str(args.NC_threshold).replace(".", "p")}'
    df[accepted_col] = df[eff] >= args.NC_threshold

    df.to_csv(out / 'within_dir_discriminability_detail.csv', index=False)
    # Legacy-compatible name for convenience.
    df.to_csv(out / 'hard_negative_detail.csv', index=False)

    # Per-directory detail files and summaries.
    all_summary_rows = []
    decision_rows = make_decision_summary(df, eff, args.NC_threshold)
    for ds, sub in df.groupby('dataset', sort=False):
        ds_safe = safe_name(ds)
        ds_dir = ensure_dir(out / ds_safe)
        sub.to_csv(ds_dir / f'{ds_safe}_discriminability_detail.csv', index=False)
        srows = summarize_pair_type(sub, eff, args.NC_threshold, dataset_value='dataset')
        pd.DataFrame(srows).to_csv(ds_dir / f'{ds_safe}_discriminability_summary_by_pair_type.csv', index=False)
        drows = [r for r in decision_rows if r['dataset'] == ds]
        pd.DataFrame(drows).to_csv(ds_dir / f'{ds_safe}_decision_summary.csv', index=False)
        all_summary_rows.extend(srows)

    # Combined summaries.
    by_pair = pd.DataFrame(all_summary_rows)
    by_pair.to_csv(out / 'within_dir_summary_by_dataset_pair_type.csv', index=False)

    overall_pair_rows = summarize_pair_type(df, eff, args.NC_threshold, dataset_value=None)
    pd.DataFrame(overall_pair_rows).to_csv(out / 'within_dir_summary_overall_by_pair_type.csv', index=False)

    decision_df = pd.DataFrame(decision_rows)
    decision_df.to_csv(out / 'within_dir_decision_summary.csv', index=False)
    # Legacy-compatible summary name.
    decision_df.to_csv(out / 'hard_negative_summary.csv', index=False)

    manifest = manifest | {
        'finished_at': now_str(),
        'total_verification_trials': int(len(df)),
        'total_elapsed_sec': time.perf_counter() - t_all,
        'outputs': [
            'within_dir_discriminability_detail.csv',
            'within_dir_summary_by_dataset_pair_type.csv',
            'within_dir_summary_overall_by_pair_type.csv',
            'within_dir_decision_summary.csv',
            '<dataset>/<dataset>_discriminability_detail.csv',
            '<dataset>/<dataset>_discriminability_summary_by_pair_type.csv',
            '<dataset>/<dataset>_decision_summary.csv',
        ]
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    log(f'DONE. Total elapsed {(time.perf_counter() - t_all) / 60:.1f} min. Output: {out}')


if __name__ == '__main__':
    main()
