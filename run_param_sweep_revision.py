from __future__ import annotations
import argparse, json, time, sys, os
from pathlib import Path
from datetime import datetime
import cv2, pandas as pd, numpy as np
from experiment_utils import *
from common.attacks import make_attack_suite_bgr

# Works both when executed as a .py file and when pasted into a notebook/IPython.
if "__file__" in globals():
    ROOT = Path(__file__).resolve().parent
else:
    ROOT = Path.cwd().resolve()

P_LIST=[25,50,75,100]
T_LIST=[0.60, 0.65,0.70,0.75,0.80,0.85,0.90,0.95]
#T_LIST=[0.60]
QMIN_LIST=[1,5,10,15,20]
NC_THRESHOLDS=[0.70, 0.75, 0.80, 0.85,0.90]
#NC_THRESHOLDS=[0.85,0.90]

def _now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def log(msg: str):
    print(f"[{_now()}] {msg}", flush=True)


def fmt_eta(start_time: float, done: int, total: int) -> str:
    if done <= 0:
        return "ETA unknown"
    elapsed = time.perf_counter() - start_time
    rate = done / max(elapsed, 1e-9)
    remaining = max(0, total - done) / max(rate, 1e-9)
    return f"elapsed={elapsed/60:.1f} min, ETA={remaining/60:.1f} min, rate={rate:.2f} it/s"


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--medical-dir', default='data/dataset2_medical_12')
    ap.add_argument('--color-dir', default='data/dataset2_natural_12')
    ap.add_argument('--suite', default='suite2', choices=['suite1','suite2','both'])
    ap.add_argument('--out-dir', default='out_param_sweep')
    ap.add_argument('--watermark', default='data/watermark/watermark.png')
    ap.add_argument('--registration-repeats', type=int, default=1)
    ap.add_argument('--progress-every', type=int, default=100, help='Print progress every N verification trials.')
    ap.add_argument('--partial-every', type=int, default=1000, help='Write partial CSV every N verification trials. Set 0 to disable.')
    args=ap.parse_args()

    t_all0 = time.perf_counter()
    out=ensure_dir(ROOT/args.out_dir)
    log(f"Output directory: {out}")
    log(f"Loading datasets: {args.medical_dir}, {args.color_dir}")
    images=load_datasets(ROOT, [args.medical_dir,args.color_dir])
    if not images:
        raise RuntimeError('No input images found. Please check medicalhosts/ and colorhosts/.')
    log(f"Loaded {len(images)} images.")

    log(f"Loading watermark: {args.watermark}")
    wm_clear,_=load_watermark_bits(ROOT/args.watermark,32)
    wm_scr=scramble_bits(wm_clear,32)

    log(f"Building attack suite: {args.suite}")
    attacks=make_attack_suite_bgr(args.suite)
    log(f"Attack count: {len(attacks)}")

    manifest = {
        'script':'run_param_sweep_revision.py',
        'P_LIST':P_LIST,
        'T_LIST':T_LIST,
        'QMIN_LIST':QMIN_LIST,
        'NC_THRESHOLDS':NC_THRESHOLDS,
        'images':len(images),
        'suite':args.suite,
        'attacks':len(attacks),
        'registration_repeats':args.registration_repeats,
        'started_at':_now(),
        'root':str(ROOT),
    }
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    # ===================== Registration =====================
    log('Stage 1/5: Registration started.')
    reg_rows=[]; states={}
    total_reg = len(P_LIST) * len(images) * max(1,args.registration_repeats)
    done_reg = 0
    t_stage = time.perf_counter()
    for P in P_LIST:
        log(f"Registration for P={P} started.")
        t_p = time.perf_counter()
        for img_idx,(img_name,img_path,dataset) in enumerate(images, start=1):
            rows=[]; st_final=None
            for rep in range(max(1,args.registration_repeats)):
                st,row=run_register(img_path, wm_scr, P=P, T=T_LIST[0], wm_size=32)
                st_final=st; row.update(P=P,image=img_name,dataset=dataset,repeat=rep)
                rows.append(row)
                done_reg += 1
            # use median timing, last state
            med=pd.DataFrame(rows).median(numeric_only=True).to_dict(); base=rows[-1].copy(); base.update({k:med.get(k,base.get(k)) for k in med})
            reg_rows.append(base); states[(P,img_name)]=st_final
            if img_idx == len(images) or img_idx % max(1, min(5, len(images))) == 0:
                log(f"  P={P}: registered {img_idx}/{len(images)} images ({fmt_eta(t_stage, done_reg, total_reg)})")
        log(f"Registration for P={P} finished in {(time.perf_counter()-t_p):.2f}s.")
        pd.DataFrame(reg_rows).to_csv(out/'registration_metadata_partial.csv', index=False)
    reg=pd.DataFrame(reg_rows); reg.to_csv(out/'registration_metadata.csv', index=False)
    reg.groupby('P').mean(numeric_only=True).reset_index().to_csv(out/'registration_storage_summary.csv', index=False)
    log('Stage 1/5: Registration finished. Files written: registration_metadata.csv, registration_storage_summary.csv')

    # ===================== Crossval discriminability =====================
    log('Stage 2/5: Cross-validation discriminability started.')
    cv_rows=[]
    total_cv = len(P_LIST) * len(T_LIST) * len(images) * len(images)
    done_cv = 0
    t_stage = time.perf_counter()
    for P in P_LIST:
      for T in T_LIST:
        log(f"Crossval P={P}, T={T:.2f} started.")
        t_pt = time.perf_counter()
        for reg_i,(reg_name,_,reg_ds) in enumerate(images, start=1):
          st=states[(P,reg_name)]; st.T=T
          for test_name,test_path,test_ds in images:
            bgr,_=read_bgr_timed(test_path)
            row,_rec=run_verify_from_image(bgr,st,wm_clear,32)
            row.update(P=P,T=T,registered_image=reg_name,test_image=test_name,registered_dataset=reg_ds,test_dataset=test_ds,same_image=int(reg_name==test_name))
            cv_rows.append(row)
            done_cv += 1
            if args.progress_every and done_cv % args.progress_every == 0:
                log(f"  Crossval progress {done_cv}/{total_cv} ({fmt_eta(t_stage, done_cv, total_cv)})")
            if args.partial_every and done_cv % args.partial_every == 0:
                pd.DataFrame(cv_rows).to_csv(out/'crossval_detail_partial.csv', index=False)
        log(f"Crossval P={P}, T={T:.2f} finished in {(time.perf_counter()-t_pt):.2f}s.")
    cv=add_effective_cols(pd.DataFrame(cv_rows), QMIN_LIST, thresholds=NC_THRESHOLDS)
    cv.to_csv(out/'crossval_detail.csv', index=False)
    log('Stage 2/5: Cross-validation finished. File written: crossval_detail.csv')

    log('Stage 3/5: Cross-validation decision summary started.')
    dec=[]
    for (P,T),sub in cv.groupby(['P','T']):
      genuine=sub['same_image']==1; imp=~genuine
      for qmin in QMIN_LIST:
        eff=f'nc_effective_Qmin{qmin}'
        for th in NC_THRESHOLDS:
          acc=sub[eff]>=th
          dec.append(dict(P=P,T=T,Qmin=qmin,NC_threshold=th,genuine_trials=int(genuine.sum()),impostor_trials=int(imp.sum()),accepted_genuine=int(acc[genuine].sum()),accepted_impostor=int(acc[imp].sum()),FRR=float(1-acc[genuine].mean()),FAR=float(acc[imp].mean()),mean_nc_raw_genuine=float(sub.loc[genuine,'nc'].mean()),mean_nc_raw_impostor=float(sub.loc[imp,'nc'].mean()),mean_nc_effective_genuine=float(sub.loc[genuine,eff].mean()),mean_nc_effective_impostor=float(sub.loc[imp,eff].mean()),mean_Q_genuine=float(sub.loc[genuine,'Q'].mean()),mean_Q_impostor=float(sub.loc[imp,'Q'].mean()),max_Q_impostor=float(sub.loc[imp,'Q'].max()),max_nc_effective_impostor=float(sub.loc[imp,eff].max())))
    pd.DataFrame(dec).to_csv(out/'crossval_decision.csv', index=False)
    log('Stage 3/5: Cross-validation decision summary finished. File written: crossval_decision.csv')

    # ===================== Robustness =====================
    log('Stage 4/5: Robustness evaluation started.')
    rb_rows=[]
    total_rb = len(P_LIST) * len(T_LIST) * len(images) * len(attacks)
    done_rb = 0
    t_stage = time.perf_counter()
    for P in P_LIST:
      for T in T_LIST:
        log(f"Robustness P={P}, T={T:.2f} started.")
        t_pt = time.perf_counter()
        for img_idx,(img_name,img_path,dataset) in enumerate(images, start=1):
          host,_=read_bgr_timed(img_path)
          st=states[(P,img_name)]; st.T=T
          for a in attacks:
            attacked=a['fn'](host.copy())
            psnr=psnr_color(host, attacked)
            row,_rec=run_verify_from_image(attacked,st,wm_clear,32)
            row.update(P=P,T=T,image=img_name,dataset=dataset,suite=a.get('suite'),attack_name=a['name'],attack_group=a['group'],attack_intensity=a['intensity'],attack_psnr_db=psnr,psnr_actual_db=psnr,psnr_reported=a.get('psnr_reported'))
            rb_rows.append(row)
            done_rb += 1
            if args.progress_every and done_rb % args.progress_every == 0:
                log(f"  Robustness progress {done_rb}/{total_rb} ({fmt_eta(t_stage, done_rb, total_rb)})")
            if args.partial_every and done_rb % args.partial_every == 0:
                pd.DataFrame(rb_rows).to_csv(out/'robustness_detail_partial.csv', index=False)
        log(f"Robustness P={P}, T={T:.2f} finished in {(time.perf_counter()-t_pt):.2f}s.")
    rb=add_effective_cols(pd.DataFrame(rb_rows), QMIN_LIST, thresholds=NC_THRESHOLDS)
    rb.to_csv(out/'robustness_detail.csv', index=False)
    log('Stage 4/5: Robustness detail finished. File written: robustness_detail.csv')

    # ===================== Summaries =====================
    log('Stage 5/5: Summary generation started.')
    summarize_robustness(rb,['P','T','suite','attack_group'],QMIN_LIST).to_csv(out/'robustness_summary_by_group.csv', index=False)
    summarize_robustness(rb,['P','T','suite','attack_group','attack_name','attack_intensity'],QMIN_LIST).to_csv(out/'robustness_summary_by_attack.csv', index=False)
    decision_summary(rb,QMIN_LIST,NC_THRESHOLDS,group_cols=['P','T']).to_csv(out/'robustness_decision.csv', index=False)

    # best candidates
    cvd=pd.read_csv(out/'crossval_decision.csv'); rbd=pd.read_csv(out/'robustness_decision.csv')
    best=cvd.merge(rbd,on=['P','T','Qmin','NC_threshold'],suffixes=('_cv','_robust'))
    best=best.merge(pd.read_csv(out/'registration_storage_summary.csv'),on='P',how='left')
    # rbd success_rate is verification success under attacks. FRR_on_attacks = 1 - success_rate.
    best['FRR_on_attacks'] = 1.0 - best['success_rate'].fillna(0.0)
    storage_col = 'topP_total_storage_packed_bytes' if 'topP_total_storage_packed_bytes' in best.columns else None
    storage_penalty = (best[storage_col].fillna(0)/1024.0) if storage_col else 0
    best['score']=1000*((best['FAR']==0)&(best['FRR']==0)).astype(int)-100*best['FRR_on_attacks'].fillna(1)+10*best['mean_nc_effective'].fillna(0)-0.01*storage_penalty
    best.sort_values('score',ascending=False).to_csv(out/'best_candidates.csv', index=False)
    manifest['finished_at'] = _now()
    manifest['total_elapsed_sec'] = time.perf_counter() - t_all0
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    log('Stage 5/5: Summary generation finished.')
    log(f"DONE. Total elapsed: {(time.perf_counter()-t_all0)/60:.1f} min. Output: {out}")

if __name__=='__main__':
    main()
