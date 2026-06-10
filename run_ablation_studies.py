from __future__ import annotations
import argparse, json, time
from pathlib import Path
import pandas as pd
from experiment_utils import *
from common.attacks import make_attack_suite_bgr
from common.progress import log, fmt_eta, now_str
ROOT=Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd().resolve()

def build_variants(P:int,T:float):
    return [
        dict(variant='Full proposed', P=P,T=T,use_alignment=True,use_voting=True,transform_mode='dwt_dct',keypoint_selection='strongest',region_mode='multi',wm_size=32,random_seed=0),
        dict(variant='w/o image correction', P=P,T=T,use_alignment=False,use_voting=True,transform_mode='dwt_dct',keypoint_selection='strongest',region_mode='multi',wm_size=32,random_seed=0),
        dict(variant='w/o voting', P=P,T=T,use_alignment=True,use_voting=False,transform_mode='dwt_dct',keypoint_selection='strongest',region_mode='multi',wm_size=32,random_seed=0),
        dict(variant='DWT only', P=P,T=T,use_alignment=True,use_voting=True,transform_mode='dwt_only',keypoint_selection='strongest',region_mode='multi',wm_size=32,random_seed=0),
        dict(variant='DCT only', P=P,T=T,use_alignment=True,use_voting=True,transform_mode='dct_only',keypoint_selection='strongest',region_mode='multi',wm_size=32,random_seed=0),
        dict(variant='DWT-DCT', P=P,T=T,use_alignment=True,use_voting=True,transform_mode='dwt_dct',keypoint_selection='strongest',region_mode='multi',wm_size=32,random_seed=0),
        dict(variant='Random P keypoints seed1', P=P,T=T,use_alignment=True,use_voting=True,transform_mode='dwt_dct',keypoint_selection='random',region_mode='multi',wm_size=32,random_seed=1),
        dict(variant='Random P keypoints seed2', P=P,T=T,use_alignment=True,use_voting=True,transform_mode='dwt_dct',keypoint_selection='random',region_mode='multi',wm_size=32,random_seed=2),
        dict(variant='Random P keypoints seed3', P=P,T=T,use_alignment=True,use_voting=True,transform_mode='dwt_dct',keypoint_selection='random',region_mode='multi',wm_size=32,random_seed=3),
        dict(variant='Full SIFT keypoints', P=-1,T=T,use_alignment=True,use_voting=True,transform_mode='dwt_dct',keypoint_selection='all',region_mode='multi',wm_size=32,random_seed=0),
        dict(variant='Global region', P=1,T=T,use_alignment=True,use_voting=False,transform_mode='dwt_dct',keypoint_selection='strongest',region_mode='global',wm_size=32,random_seed=0),
        dict(variant='Watermark 16x16', P=P,T=T,use_alignment=True,use_voting=True,transform_mode='dwt_dct',keypoint_selection='strongest',region_mode='multi',wm_size=16,random_seed=0),
        dict(variant='Watermark 32x32', P=P,T=T,use_alignment=True,use_voting=True,transform_mode='dwt_dct',keypoint_selection='strongest',region_mode='multi',wm_size=32,random_seed=0),
        dict(variant='Watermark 64x64', P=P,T=T,use_alignment=True,use_voting=True,transform_mode='dwt_dct',keypoint_selection='strongest',region_mode='multi',wm_size=64,random_seed=0),
        dict(variant='Global 16x16', P=1,T=T,use_alignment=True,use_voting=False,transform_mode='dwt_dct',keypoint_selection='strongest',region_mode='global',wm_size=16,random_seed=0),
        dict(variant='Global 64x64', P=1,T=T,use_alignment=True,use_voting=False,transform_mode='dwt_dct',keypoint_selection='strongest',region_mode='global',wm_size=64,random_seed=0),
    ]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--medical-dir', default='data/dataset2_medical_12')
    ap.add_argument('--color-dir', default='data/dataset2_natural_12')
    ap.add_argument('--suite', default='suite2', choices=['suite1','suite2','both'])
    ap.add_argument('--out-dir', default='out_ablation')
    ap.add_argument('--watermark', default='data/watermark/watermark.png')
    ap.add_argument('--P', type=int, default=50); ap.add_argument('--T', type=float, default=0.65)
    ap.add_argument('--Qmin', type=int, default=10); ap.add_argument('--NC-threshold', type=float, default=0.85)
    ap.add_argument('--progress-every', type=int, default=100)
    ap.add_argument('--partial-every', type=int, default=1000)
    args=ap.parse_args(); t_all=time.perf_counter(); out=ensure_dir(ROOT/args.out_dir)
    log(f"Output directory: {out}")
    images=load_datasets(ROOT,[args.medical_dir,args.color_dir]); attacks=make_attack_suite_bgr(args.suite)
    variants=build_variants(args.P,args.T)
    log(f"Loaded {len(images)} images; {len(attacks)} attacks; {len(variants)} variants.")
    (out/'manifest.json').write_text(json.dumps({'variants':[v['variant'] for v in variants],'suite':args.suite,'images':len(images),'Qmin':args.Qmin,'started_at':now_str()}, indent=2), encoding='utf-8')
    reg_rows=[]; rows=[]; cv_rows=[]
    total_reg=len(variants)*len(images)
    total_cv=len(variants)*len(images)*len(images)
    total_rb=len(variants)*len(images)*len(attacks)
    done_reg=done_cv=done_rb=0
    t_reg=t_cv=t_rb=time.perf_counter()
    for vi,v in enumerate(variants, start=1):
        log(f"Variant {vi}/{len(variants)} started: {v['variant']} (wm={v['wm_size']}, transform={v['transform_mode']}, P={v['P']})")
        wm_clear,_=load_watermark_bits(ROOT/args.watermark, v['wm_size']); wm_scr=scramble_bits(wm_clear,v['wm_size'])
        states={}
        # Registration per variant
        for idx,(name,path,ds) in enumerate(images, start=1):
            st,reg=run_register(path,wm_scr,P=v['P'],T=v['T'],wm_size=v['wm_size'],use_voting=v['use_voting'],use_alignment=v['use_alignment'],transform_mode=v['transform_mode'],keypoint_selection=v['keypoint_selection'],region_mode=v['region_mode'],random_seed=stable_seed(v['variant'],name))
            states[name]=st; reg.update(variant=v['variant'],image=name,dataset=ds,**{k:v[k] for k in ['P','T','wm_size','use_alignment','use_voting','transform_mode','keypoint_selection','region_mode','random_seed']})
            reg_rows.append(reg); done_reg+=1
            if args.progress_every and done_reg % max(1,args.progress_every) == 0:
                log(f"  Registration progress {done_reg}/{total_reg} ({fmt_eta(t_reg,done_reg,total_reg)})")
        pd.DataFrame(reg_rows).to_csv(out/'ablation_registration_storage_partial.csv', index=False)
        # Crossval
        for reg_name,_,reg_ds in images:
            st=states[reg_name]
            for test_name,test_path,test_ds in images:
                bgr,_=read_bgr_timed(test_path)
                row,_=run_verify_from_image(bgr,st,wm_clear,v['wm_size'])
                row.update(variant=v['variant'],registered_image=reg_name,test_image=test_name,same_image=int(reg_name==test_name),registered_dataset=reg_ds,test_dataset=test_ds,wm_size=v['wm_size'])
                cv_rows.append(row); done_cv+=1
                if args.progress_every and done_cv % args.progress_every == 0:
                    log(f"  Crossval progress {done_cv}/{total_cv} ({fmt_eta(t_cv,done_cv,total_cv)})")
                if args.partial_every and done_cv % args.partial_every == 0:
                    pd.DataFrame(cv_rows).to_csv(out/'ablation_crossval_detail_partial.csv', index=False)
        # Attacks
        for img_idx,(name,path,ds) in enumerate(images, start=1):
            host,_=read_bgr_timed(path); st=states[name]
            for a in attacks:
                attacked=a['fn'](host.copy()); psnr=psnr_color(host,attacked)
                row,_=run_verify_from_image(attacked,st,wm_clear,v['wm_size'])
                row.update(variant=v['variant'],image=name,dataset=ds,wm_size=v['wm_size'],suite=a.get('suite'),attack_name=a['name'],attack_group=a['group'],attack_intensity=a['intensity'],attack_psnr_db=psnr,psnr_actual_db=psnr)
                rows.append(row); done_rb+=1
                if args.progress_every and done_rb % args.progress_every == 0:
                    log(f"  Robustness progress {done_rb}/{total_rb} ({fmt_eta(t_rb,done_rb,total_rb)})")
                if args.partial_every and done_rb % args.partial_every == 0:
                    pd.DataFrame(rows).to_csv(out/'ablation_detail_partial.csv', index=False)
        log(f"Variant {vi}/{len(variants)} finished: {v['variant']}")
    reg=pd.DataFrame(reg_rows); reg.to_csv(out/'ablation_registration_storage.csv', index=False)
    detail=add_effective_cols(pd.DataFrame(rows),[args.Qmin],thresholds=[args.NC_threshold]); detail.to_csv(out/'ablation_detail.csv', index=False)
    summarize_robustness(detail,['variant','attack_group'],[args.Qmin]).to_csv(out/'ablation_summary_by_attack_group.csv', index=False)
    decision_summary(detail,[args.Qmin],[args.NC_threshold],group_cols=['variant']).to_csv(out/'ablation_summary_overall.csv', index=False)
    cv=add_effective_cols(pd.DataFrame(cv_rows),[args.Qmin],thresholds=[args.NC_threshold]); cv.to_csv(out/'ablation_crossval_detail.csv', index=False)
    dec=[]
    for var,sub in cv.groupby('variant'):
        genuine=sub['same_image']==1; imp=~genuine; eff=f'nc_effective_Qmin{args.Qmin}'; acc=sub[eff]>=args.NC_threshold
        dec.append(dict(variant=var,FRR=float(1-acc[genuine].mean()),FAR=float(acc[imp].mean()),mean_nc_effective_genuine=float(sub.loc[genuine,eff].mean()),mean_nc_effective_impostor=float(sub.loc[imp,eff].mean()),mean_Q_genuine=float(sub.loc[genuine,'Q'].mean()),mean_Q_impostor=float(sub.loc[imp,'Q'].mean()),max_Q_impostor=float(sub.loc[imp,'Q'].max())))
    pd.DataFrame(dec).to_csv(out/'ablation_crossval_summary.csv', index=False)
    log(f"DONE. Total elapsed {(time.perf_counter()-t_all)/60:.1f} min. Output: {out}")
if __name__=='__main__': main()
