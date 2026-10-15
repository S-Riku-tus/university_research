"""Create presentation figures from previously saved diagnostic metrics."""
from pathlib import Path
import hashlib
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
PATHS=[ROOT/"experiments/2026-07-24_noise_shortcut_diagnostic/metrics_summary.csv",
       ROOT/"experiments/2026-09-19_noise_recovery_review/r2_from_saved_predictions.csv"]


def main():
    before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in PATHS]
    diagnostic=pd.read_csv(PATHS[0])
    values=[]
    for variant,noise in [("relative_mixture","heatflux_no_noise"),
                          ("relative_mixture","heatflux_SNR=-20"),
                          ("relative_noise_only","heatflux_SNR=-20"),
                          ("fixed_mixture","heatflux_SNR=-20")]:
        selected=diagnostic.loc[diagnostic.experiment_name.eq("2025.06.18_0.3_3")
                                &diagnostic.variant.eq(variant)&diagnostic.noise_dir.eq(noise)
                                &diagnostic.split_strategy.eq("kfold")
                                &diagnostic.feature_set.eq("full_spectrogram_pca")]
        assert len(selected)==1
        values.append(float(selected.r2_mean.iloc[0]))
    plt.rcParams.update({"font.size":16,"axes.spines.top":False,"axes.spines.right":False})
    fig,ax=plt.subplots(figsize=(9,4.8))
    bars=ax.bar(range(4),values,color=["#98cedb","#1686e7","#dfa13d","#41a694"],width=.65)
    ax.bar_label(bars,fmt="%.4f",padding=4)
    ax.set_xticks(range(4),["Clean","Relative mixture\n-20 dB","Relative noise only\n-20 dB","Fixed mixture\n-20 dB"],fontsize=13)
    ax.set_ylabel("Diagnostic PCA model: mean R²");ax.set_ylim(0,1.06)
    fig.tight_layout()
    for ext in ["png","svg"]:fig.savefig(OUT/("slide_old_noise_shortcut."+ext),dpi=190,bbox_inches="tight")
    plt.close(fig)
    recovered=pd.read_csv(PATHS[1],dtype={"reference_snr":str})
    subset=recovered.loc[recovered.model.eq("ensemble__inner_holdout")].set_index("reference_snr")
    order=["no_noise","0","-4","-8","-12","-16","-20"]
    assert len(subset)==7 and set(subset.index)==set(order)
    fig,ax=plt.subplots(figsize=(8.5,4.4))
    ax.plot(range(7),subset.loc[order,"r2"],"o-",color="#1686e7",linewidth=2.7)
    ax.set_xticks(range(7),["Clean","0","-4","-8","-12","-16","-20"])
    ax.set_xlabel("Reference SNR [dB]");ax.set_ylabel("R²: separately trained models")
    ax.annotate("Recovery remained",xy=(5,subset.loc["-16","r2"]),xytext=(2.8,.92),
                arrowprops={"arrowstyle":"->","color":"#526b86"},fontsize=14)
    ax.set_ylim(.79,.95);fig.tight_layout()
    for ext in ["png","svg"]:fig.savefig(OUT/("slide_after_generation_fix_matched."+ext),dpi=190,bbox_inches="tight")
    plt.close(fig)
    assert before==[hashlib.sha256(p.read_bytes()).hexdigest() for p in PATHS]
    record={"sources":[{"path":str(p.relative_to(ROOT)),"sha256":s} for p,s in zip(PATHS,before)],
            "sources_unchanged":True,"new_fits":0,"new_inference":0,
            "old_diagnostic_scope":"June18/22kHz/PCA/KFold; not current five-model performance",
            "matched_scope":"September18/run151715, June11->June18, 22kHz, matched, 1080 chunks",
            "causal_comparison_with_latest_current_run":False}
    (OUT/"history_figure_sources.json").write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"old_diagnostic_r2":values,"sources_unchanged":True}))


if __name__=="__main__":main()
