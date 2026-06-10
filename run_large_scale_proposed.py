from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import pandas as pd
from experiment_utils import *
from common.attacks import make_attack_suite_bgr
from common.progress import log, fmt_eta, now_str
ROOT=Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd().resolve()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--natural-dir', default='data/dataset1_natural_600')
    ap.add_argument('--medical-dir', default='data/dataset1_medical_400')
    ap.add_argument('--suite', default='suite2', choices=['suite1','suite2','both'])
    ap.add_argument('--out-dir', default='out_large_scale_1K')
    ap.add_argument('--watermark', default='data/watermark/watermark.png')
    ap.add_argument('--P', type=int, default=50); ap.add_argument('--T', type=float, default=0.65)
    ap.add_argument('--Qmin', type=int, default=10); ap.add_argument('--NC-threshold', type=float, default=0.70)
    ap.add_argument('--max-images-per-dataset', type=int, default=0, help='0 means all')
    ap.add_argument('--impostors-per-image', type=int, default=10)
    ap.add_argument('--progress-every', type=int, default=100, help='Print progress every N verification/discriminability trials')
    ap.add_argument('--partial-every', type=int, default=1000, help='Write partial CSV every N rows; 0 disables')
    args=ap.parse_args(); t_all=time.perf_counter(); out=ensure_dir(ROOT/args.out_dir)
    log(f"Output directory: {out}")
    log(f"Loading large datasets: {args.natural_dir}, {args.medical_dir}")
    images=load_datasets(ROOT, [args.natural_dir,args.medical_dir])
    if args.max_images_per_dataset>0:
        kept=[]
        for ds in sorted(set(x[2] for x in images)):
            dsimgs=[x for x in images if x[2]==ds][:args.max_images_per_dataset]
            kept.extend(dsimgs)
        images=kept
    if not images: raise RuntimeError('No images found for large-scale evaluation.')
    log(f"Loaded {len(images)} images.")
    wm_clear,_=load_watermark_bits(ROOT/args.watermark,32); wm_scr=scramble_bits(wm_clear,32)
    attacks=make_attack_suite_bgr(args.suite)
    log(f"Attack suite {args.suite}: {len(attacks)} attacks.")
    (out/'manifest.json').write_text(json.dumps(vars(args)|{'n_images':len(images),'n_attacks':len(attacks),'started_at':now_str()}, indent=2), encoding='utf-8')

    log('Stage 1/5: Registration started.')
    states={}; reg_rows=[]; t_stage=time.perf_counter()
    for idx,(name,path,ds) in enumerate(images, start=1):
        st,row=run_register(path,wm_scr,P=args.P,T=args.T,wm_size=32)
        row.update(image=name,dataset=ds,P=args.P,T=args.T)
        states[name]=st; reg_rows.append(row)
        if idx % max(1, min(args.progress_every, 50)) == 0 or idx==len(images):
            log(f"  Registered {idx}/{len(images)} images ({fmt_eta(t_stage,idx,len(images))})")
    reg=pd.DataFrame(reg_rows); reg.to_csv(out/'registration_metadata.csv', index=False)
    reg.groupby('dataset').mean(numeric_only=True).reset_index().to_csv(out/'registration_summary_by_dataset.csv', index=False)
    log('Stage 1/5: Registration finished.')

    log('Stage 2/5: Robustness evaluation started.')
    rows=[]; total=len(images)*len(attacks); done=0; t_stage=time.perf_counter()
    for img_idx,(name,path,ds) in enumerate(images, start=1):
        host,_=read_bgr_timed(path); st=states[name]; st.T=args.T
        for a in attacks:
            attacked=a['fn'](host.copy())
            psnr=psnr_color(host,attacked)
            row,_=run_verify_from_image(attacked,st,wm_clear,32)
            row.update(image=name,dataset=ds,P=args.P,T=args.T,Qmin=args.Qmin,suite=a.get('suite'),attack_name=a['name'],attack_group=a['group'],attack_intensity=a['intensity'],attack_psnr_db=psnr,psnr_actual_db=psnr)
            rows.append(row); done+=1
            if args.progress_every and done % args.progress_every == 0:
                log(f"  Robustness progress {done}/{total} ({fmt_eta(t_stage,done,total)})")
            if args.partial_every and done % args.partial_every == 0:
                pd.DataFrame(rows).to_csv(out/'large_scale_detail_partial.csv', index=False)
        if img_idx % max(1, min(25, len(images))) == 0 or img_idx==len(images):
            log(f"  Finished attacks for {img_idx}/{len(images)} images")
    df=add_effective_cols(pd.DataFrame(rows), [args.Qmin], thresholds=[args.NC_threshold])
    df.to_csv(out/'large_scale_detail.csv', index=False)
    summarize_robustness(df,['dataset','attack_group'],[args.Qmin]).to_csv(out/'large_scale_group_summary.csv', index=False)
    summarize_robustness(df,['dataset','attack_group','attack_name','attack_intensity'],[args.Qmin]).to_csv(out/'large_scale_attack_summary.csv', index=False)
    decision_summary(df,[args.Qmin],[args.NC_threshold],group_cols=['dataset','attack_group']).to_csv(out/'large_scale_success_by_group.csv', index=False)
    log('Stage 2/5: Robustness finished.')

    log('Stage 3/5: Discriminability with sampled impostors started.')
    rng=random.Random(123); disc=[]; total_disc=len(images)*(1+min(args.impostors_per_image,max(0,len(images)-1))); done=0; t_stage=time.perf_counter()
    for idx,(name,path,ds) in enumerate(images, start=1):
        st=states[name]; st.T=args.T
        candidates=[x for x in images if x[0]!=name]
        sample=rng.sample(candidates, min(args.impostors_per_image, len(candidates))) if candidates else []
        for test_name,test_path,test_ds,label in [(name,path,ds,'genuine')]+[(x[0],x[1],x[2],'impostor') for x in sample]:
            bgr,_=read_bgr_timed(test_path)
            row,_=run_verify_from_image(bgr,st,wm_clear,32)
            row.update(registered_image=name,test_image=test_name,registered_dataset=ds,test_dataset=test_ds,pair_type=label,P=args.P,T=args.T)
            disc.append(row); done+=1
            if args.progress_every and done % args.progress_every == 0:
                log(f"  Discriminability progress {done}/{total_disc} ({fmt_eta(t_stage,done,total_disc)})")
            if args.partial_every and done % args.partial_every == 0:
                pd.DataFrame(disc).to_csv(out/'large_scale_discriminability_detail_partial.csv', index=False)
    disc=add_effective_cols(pd.DataFrame(disc), [args.Qmin], thresholds=[args.NC_threshold])
    disc.to_csv(out/'large_scale_discriminability_detail.csv', index=False)
    dec=[]
    for pair_type,sub in disc.groupby('pair_type'):
        eff=f'nc_effective_Qmin{args.Qmin}'; acc=sub[eff]>=args.NC_threshold
        dec.append(dict(pair_type=pair_type,trials=len(sub),accepted=int(acc.sum()),acceptance_rate=float(acc.mean()),mean_nc_raw=float(sub['nc'].mean()),std_nc_raw=float(sub['nc'].std()),mean_nc_effective=float(sub[eff].mean()),mean_Q=float(sub['Q'].mean()),std_Q=float(sub['Q'].std())))
    pd.DataFrame(dec).to_csv(out/'large_scale_discriminability_summary.csv', index=False)
    log('Stage 3/5: Discriminability finished.')

    log('Stage 4/5: Low/high feature stratification started.')
    reg2=reg.copy()
    reg2['feature_bin']=pd.qcut(reg2['n_all_sift_keypoints'].rank(method='first'), q=4, labels=['low','mid_low','mid_high','high'])
    merged=df.merge(reg2[['image','feature_bin','n_all_sift_keypoints']],on='image',how='left')
    summarize_robustness(merged,['feature_bin','attack_group'],[args.Qmin]).to_csv(out/'large_scale_low_high_feature_summary.csv', index=False)
    log('Stage 4/5: Low/high feature stratification finished.')

    manifest=vars(args)|{'n_images':len(images),'n_attacks':len(attacks),'finished_at':now_str(),'total_elapsed_sec':time.perf_counter()-t_all}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    log(f"DONE. Total elapsed {(time.perf_counter()-t_all)/60:.1f} min. Output: {out}")
if __name__=='__main__': main()
