"""Typeset saved evidence only: no model loading, training or inference."""
from pathlib import Path
import csv
import hashlib
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
MAIN = ROOT / "experiments/2026-10-15_clean_only_result_analysis"
LINK = ROOT / "experiments/2026-10-14_error_feature_linkage"
AUDIT = ROOT / "docs/thesis/2026-10-19_manuscript_audit"
NOISES = ["clean", "0", "-4", "-8", "-12", "-16", "-20"]
NAMES = {
    "randomforest": "RF (PCA+XGBRF)", "conformer": "CNN+Transformer",
    "alexnet": "AlexNet", "hgb": "HGB", "extra_trees": "ET",
    "ensemble__original3_performance": "3-perf",
    "ensemble__original3_mse": "3-MSE",
    "ensemble__original3_hgb": "3+HGB",
    "ensemble__original3_extra_trees": "3+ET",
    "ensemble__original3_hgb_extra_trees": "5-model",
    "ensemble__simple_equal": "5-equal",
}
CORE = "ensemble__original3_mse"
FIVE = "ensemble__original3_hgb_extra_trees"
ET4 = "ensemble__original3_extra_trees"
TRACKED = {}


def disk(path):
    path = Path(path).resolve()
    return Path("\\\\?\\"+str(path)) if os.name == "nt" else path


def track(path):
    path = Path(path)
    TRACKED[path.relative_to(ROOT).as_posix()] = hashlib.sha256(disk(path).read_bytes()).hexdigest()
    return disk(path)


def rows(path):
    with track(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def table(name, header, body, spec):
    s = ["% Generated from saved evidence; do not hand-edit.",
         r"\begin{tabular}{"+spec+"}", r"\toprule",
         " & ".join(header)+r" \\", r"\midrule"]
    s.extend(" & ".join(str(v) for v in line)+r" \\" for line in body)
    s += [r"\bottomrule", r"\end{tabular}", ""]
    (HERE/"tables"/name).write_text("\n".join(s), encoding="utf-8")


def savefig(name, fig):
    fig.savefig(HERE/"figures"/(name+".pdf"), bbox_inches="tight")
    fig.savefig(HERE/"figures"/(name+".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    for folder in [HERE/"tables", HERE/"figures", HERE/"data", AUDIT]:
        folder.mkdir(parents=True, exist_ok=True)
    pooled = rows(MAIN/"pooled_metrics.csv")
    assert len(pooled) == 77
    look = {(r["noise"], r["model_key"]): r for r in pooled}
    assert all(float(r["n"]) == 540 and float(r["n_onb"]) == 30 for r in pooled)
    keys = list(NAMES)
    def f(noise, key, field):
        return float(look[noise, key][field])
    def num(noise, key, field):
        return f"{f(noise,key,field):.2f}"
    def integer(noise, key, field):
        return str(int(f(noise,key,field)))

    for noise, name in [("clean", "clean_models.tex"), ("-20", "noisy_models.tex")]:
        table(name, ["方式", "RMSE", "$R^2$", "近傍RMSE", "FN", "FP"],
              [[NAMES[k], num(noise,k,"rmse_all_kW_m2"), f"{f(noise,k,'r2'):.4f}",
                num(noise,k,"rmse_onb_kW_m2"),integer(noise,k,"fn"),integer(noise,k,"fp")]
               for k in keys], "lrrrrr")
    selection = ["ensemble__original3_performance",CORE,ET4,FIVE,"hgb","extra_trees"]
    table("noise_rmse.tex", ["基準SNR", *[NAMES[k] for k in selection]],
          [["無雑音" if n=="clean" else "$"+n+"$", *[num(n,k,"rmse_all_kW_m2") for k in selection]]
           for n in NOISES], "lrrrrrr")
    table("degradation.tex", ["方式", "無雑音", "$-20$ dB", "増加量"],
          [[NAMES[k],num("clean",k,"rmse_all_kW_m2"),num("-20",k,"rmse_all_kW_m2"),
            f"{f('-20',k,'rmse_all_kW_m2')-f('clean',k,'rmse_all_kW_m2'):.2f}"]
           for k in selection], "lrrr")
    table("detection.tex", ["基準SNR", "3-MSE FN/FP", "3+ET FN/FP", "5-model FN/FP", "5 Recall [\%]"],
          [["無雑音" if n=="clean" else "$"+n+"$",
            *[integer(n,k,"fn")+" / "+integer(n,k,"fp") for k in [CORE,ET4,FIVE]],
            f"{100*f(n,FIVE,'recall'):.2f}"] for n in NOISES], "lrrrr")
    onb=rows(MAIN/"recorded_onb_point_detection.csv")
    table("onset.tex", ["基準SNR", "記録上ONB段階", "検知", "見逃し", "検知割合 [\%]"],
          [["無雑音" if r["noise"]=="clean" else "$"+r["noise"]+"$",
            r["n_at_recorded_onb"],r["detected_onb_chunks"],r["missed_onb_chunks"],
            f"{100*float(r['detection_fraction_at_recorded_onb']):.2f}"]
           for r in onb if r["model_key"]==FIVE], "lrrrr")
    qrows=rows(MAIN/"q100_by_source_day.csv")
    qlook={(r["noise"],r["source_day"],r["model_key"]):r for r in qrows}
    days=["2025.06.11_0.3_2","2025.06.18_0.3_3"]
    table("q100.tex", ["日・条件", "ONB", "3-MSE", "3+ET", "5-model", "$g_{100}$ (5)"],
          [[day[5:10]+(" 無雑音" if n=="clean" else " $-20$"),
            f"{float(qlook[n,day,FIVE]['onb_threshold_kW_m2']):.2f}",
            *[f"{float(qlook[n,day,k]['q100_kW_m2']):.2f}" for k in [CORE,ET4,FIVE]],
            f"{float(qlook[n,day,FIVE]['g100_kW_m2']):.2f}"] for day in days for n in ["clean","-20"]], "lrrrrr")
    weights=rows(MAIN/"weights.csv")
    table("weights.tex", ["方式", "RF", "CNN+Tf", "AlexNet", "HGB", "ET"],
          [[NAMES["ensemble__"+s], *[f"{100*next(float(r['weight']) for r in weights if r['strategy_name']==s and r['model_key']==k):.2f}"
            for k in ["randomforest","conformer","alexnet","hgb","extra_trees"]]]
           for s in ["original3_performance","original3_mse","original3_hgb","original3_extra_trees","original3_hgb_extra_trees","simple_equal"]], "lrrrrr")
    direction=rows(MAIN/"error_direction_contributions.csv")
    table("directions.tex", ["条件", "誤差群", "$n$", "改善数", "MSE減少寄与", "割合 [\%]"],
          [["無雑音" if r["noise"]=="clean" else "$-20$ dB",
            "逆符号を含む" if r["group"]=="opposite_signs" else "同方向",
            r["n"],r["improved_chunks"],f"{float(r['mse_improvement_contribution']):.2f}",
            f"{float(r['share_of_total_mse_improvement_percent']):.2f}"] for r in direction if r["noise"] in ["clean","-20"]], "llrrrr")
    feature=rows(LINK/"existing_feature_dependence.csv")
    changes=sorted(set(r["intervention"] for r in feature))
    shuffle="time_permutation"
    table("time_shuffle.tex", ["方式", "元入力", "時間行並べ替え後", "増加量"],
          [[NAMES[r["model_key"]],f"{float(r['rmse_all_kW_m2__original']):.2f}",
            f"{float(r['rmse_all_kW_m2']):.2f}",f"{float(r['rmse_change']):.2f}"]
           for r in feature if r["noise"]=="clean" and r["intervention"]==shuffle], "lrrr")

    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    x=np.arange(7)
    fig,ax=plt.subplots(figsize=(7.1,3.9), constrained_layout=True)
    for k in selection:
        ax.plot(x,[f(n,k,"rmse_all_kW_m2") for n in NOISES],marker="o",label=NAMES[k],lw=2.3 if k==FIVE else 1.25)
    ax.set(xticks=x,xticklabels=["Clean","0","-4","-8","-12","-16","-20"],xlabel="Reference SNR [dB]",ylabel=r"Overall RMSE [kW/m$^2$]")
    ax.grid(alpha=.2);ax.legend(ncol=3,fontsize=9)
    savefig("noise_rmse",fig)
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.5),constrained_layout=True)
    for ax,col,title in zip(axes,["fn","fp"],["False negatives (315 positives)","False positives (225 negatives)"]):
        for k in [CORE,ET4,FIVE]:ax.plot(x,[f(n,k,col) for n in NOISES],marker="o",label=NAMES[k])
        ax.set(xticks=x,xticklabels=["C","0","-4","-8","-12","-16","-20"],xlabel="Reference SNR [dB]",ylabel="Segments",title=title)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=9);savefig("detection_errors",fig)
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.5),constrained_layout=True)
    for ax,day in zip(axes,days):
        for k in [CORE,ET4,FIVE]:ax.plot(x,[float(qlook[n,day,k]["q100_kW_m2"]) for n in NOISES],marker="o",label=NAMES[k])
        ax.axhline(float(qlook['clean',day,FIVE]['onb_threshold_kW_m2']),ls='--',color='gray',label="ONB")
        ax.set(xticks=x,xticklabels=["C","0","-4","-8","-12","-16","-20"],xlabel="Reference SNR [dB]",ylabel=r"$q_{100}$ [kW/m$^2$]",title=day[:10])
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8,ncol=2);savefig("q100",fig)

    conditions=json.loads(track(MAIN/"conditions.json").read_text(encoding='utf-8'))
    run=ROOT/conditions['source_run']
    distribution=[]
    fig,axes=plt.subplots(2,2,figsize=(7.1,6.0),constrained_layout=True)
    for j,(noise,suffix) in enumerate([('clean','no_noise'),('-20','-20')]):
        folder='heatflux_no_noise' if noise=='clean' else 'heatflux_reference_SNR=-20'
        pred=rows(run/folder/'fold_pred'/f'pred_f1_{suffix}.csv')
        for i,(day,dayid) in enumerate(zip(days,['20250611','20250618'])):
            subset=[r for r in pred if r['source_wav_id'].startswith(dayid+'::')]
            assert len(subset)==270
            ax=axes[i,j]
            target=np.array([float(r['y_true'])/1000 for r in subset])
            for k,c,offset in [(CORE,'#737373',-.8),(FIVE,'#176c8c',.8)]:
                yp=np.array([float(r[k])/1000 for r in subset])
                ax.scatter(target+offset,yp,s=8,alpha=.5,color=c,label=NAMES[k])
            th=float(qlook[noise,day,FIVE]['onb_threshold_kW_m2'])
            ax.plot([0,900],[0,900],c='gray',ls=':',lw=1)
            ax.axhline(th,c='#a33131',ls='--',lw=.8);ax.axvline(th,c='#a33131',ls='--',lw=.8)
            ax.set(xlim=(-20,920),ylim=(-50,950),title=day[:10]+' / '+('Clean' if noise=='clean' else '-20 dB'),xlabel=r'Recorded $q$ [kW/m$^2$]',ylabel=r'Predicted $q$ [kW/m$^2$]')
            if i==0 and j==0:ax.legend(fontsize=8)
            distribution.append({'day':day,'noise':noise,'n':len(subset),'q_min_kW_m2':float(target.min()),'q_max_kW_m2':float(target.max())})
    savefig('prediction_scatter',fig)
    track(ROOT/'configs/datasets/waterflow_20260817_1s.yaml')
    for rel in ['code/utils/dataloading/acoustic_summary_features.py','code/utils/models/regression/base_regression.py','code/utils/models/regression/acoustic_regression.py','code/utils/ensemble/fixed_core_stacking.py','code/utils/dataloading/waterflow_preprocessing.py','docs/thesis/2026-10-19_chapter_logic_and_evidence_review.md','docs/research_plan/2026-10-16_unified_six_chapter_thesis_outline.md','docs/research_plan/2026-10-16_after_presentation_professor_discussion.md']:
        track(ROOT/rel)
    for rel in ['experiments/2026-10-15_clean_only_result_analysis/verification.json',
                'experiments/2026-10-15_clean_only_result_analysis/README.md',
                'experiments/2026-10-14_error_feature_linkage/regional_loss_decomposition.csv',
                'experiments/2026-10-14_error_feature_linkage/existing_input_removal_summary.csv',
                'experiments/2026-10-07_onb_input_and_integration/README.md',
                'Pool_boiling/Subcooling_20_degrees/0.3/2025.06.11_0.3_2/data/npy/waterflow_20260817_1s/global_reference_manifest.json',
                'Pool_boiling/Subcooling_20_degrees/0.3/2025.06.11_0.3_2/data/npy/waterflow_20260817_1s/maxfreq=3kHz/preprocess_manifest.json']:
        track(ROOT/rel)
    # Freeze the small numeric source tables alongside the manuscript.
    for source in [MAIN/'pooled_metrics.csv',MAIN/'weights.csv',MAIN/'q100_by_source_day.csv',MAIN/'recorded_onb_point_detection.csv',MAIN/'error_direction_contributions.csv',LINK/'existing_feature_dependence.csv']:
        (HERE/'data'/source.name).write_bytes(source.read_bytes())
    manifest={'date':'2026-10-19','main_run':'143609','run_hash':'da540f2d','fit_id':'7560205f8feb','unit':'kW/m2 unless explicitly stated','model_fit_or_inference_performed':False,'source_sha256':TRACKED,'prediction_figure_scope':distribution,'intervention_names':changes}
    (HERE/'data/source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (AUDIT/'numeric_sources.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'metric_rows':len(pooled),'tables':len(list((HERE/'tables').glob('*.tex'))),'figures':len(list((HERE/'figures').glob('*.pdf'))),'sources':len(TRACKED),'prediction_scope':distribution},ensure_ascii=False))


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
