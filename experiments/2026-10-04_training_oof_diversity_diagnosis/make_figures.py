"""Standalone figures for the training-side diagnosis and SVR pilot."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import OUTPUT, THRESHOLDS, read_csv


def main():
    wav_features = read_csv(OUTPUT / "training_wav_features.csv")
    profiles = read_csv(OUTPUT / "oof_wav_profiles.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for day, color in [("20250611", "#2864a0"), ("20250618", "#b65030")]:
        for noise, linestyle in [("clean", "-"), ("-20", "--")]:
            rows = sorted((r for r in wav_features if r["noise"] == noise and r["day"] == day),
                          key=lambda r: float(r["heat_flux_kW_m2"]))
            axes[0].plot([float(r["heat_flux_kW_m2"])/(THRESHOLDS[day]/1000) for r in rows],
                         [float(r["mean_log_power_2100_2500"]) for r in rows],
                         color=color, linestyle=linestyle, marker="o", markersize=3,
                         label=f"{day[4:]} / {noise}")
    axes[0].axvline(1, color="black", linestyle=":", linewidth=1)
    axes[0].set(xlabel="Measured heat flux / day-specific ONB", ylabel="Mean log10 band power in model input",
                title="2.1-2.5 kHz band: global trend and local reversal")
    axes[0].legend(fontsize=8)
    clean = sorted((r for r in profiles if r["noise"] == "clean" and r["region"] == "ONB_to_1.5ONB"),
                   key=lambda r:(r["day"], float(r["heat_flux_kW_m2"])))
    x = np.arange(len(clean))
    matched = {r["source_wav_id"]:r for r in profiles if r["noise"] == "-20"}
    axes[1].bar(x-.18,[int(r["common_fn"]) for r in clean],width=.36,label="Clean training OOF",color="#2864a0")
    axes[1].bar(x+.18,[int(matched[r["source_wav_id"]]["common_fn"]) for r in clean],width=.36,label="Matched -20 training OOF",color="#b65030")
    labels=[f"{r['day'][4:]}\nq={float(r['heat_flux_kW_m2']):.1f}\nfold {r['fold']}" for r in clean]
    axes[1].set_xticks(x,labels,fontsize=8)
    axes[1].set(ylabel="Common false negatives / 45 training chunks",ylim=(0,49),
                title="Common failure depends on recording and fold")
    axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(axis="y",alpha=.2)
    for extension in ["png","pdf"]:
        fig.savefig(OUTPUT/f"training_diagnosis.{extension}",dpi=180)
    plt.close(fig)

    all_results=[]
    for prefix,label in [("svr","Band-feature SVR"),("svr_shape","Shape-only SVR")]:
        rows=read_csv(OUTPUT/f"{prefix}_training_oof_metrics.csv")
        all_results.append((label,next(r for r in rows if r['condition']=='clean' and r['scope']=='all' and r['method']=='SVR')))
        if prefix=='svr':
            baseline=next(r for r in rows if r['condition']=='clean' and r['scope']=='all' and r['method']=='existing3_performance_oof_diagnostic')
            blend=next(r for r in rows if r['condition']=='clean' and r['scope']=='all' and r['method']=='fixed_old75_SVR25')
    all_results=[("Existing 3\nOOF diagnostic",baseline),all_results[0],("Old 75% +\nSVR 25%",blend),all_results[1]]
    fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    for ax,key,title in zip(axes,["rmse_kW_m2","fn","fp"],["RMSE (kW/m2)","False negatives / 945","False positives / 675"]):
        values=[float(r[key]) for _,r in all_results]
        bars=ax.bar(range(4),values,color=["#556677","#2864a0","#668e58","#b65030"])
        ax.set_xticks(range(4),[label for label,_ in all_results],fontsize=8,rotation=18)
        ax.set_title(title)
        ax.set_ylim(0,max(values)*1.2)
        for b,v in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,v+max(values)*.025,f"{v:.2f}" if key=='rmse_kW_m2' else f"{v:.0f}",ha="center",fontsize=9)
        ax.grid(axis="y",alpha=.2)
    fig.suptitle("Exploratory clean training OOF: useful corrections and added errors",fontsize=12)
    for extension in ["png","pdf"]:
        fig.savefig(OUTPUT/f"svr_pilot_tradeoffs.{extension}",dpi=180)
    plt.close(fig)
    print("Two diagnostic figures saved as PNG and PDF.")


if __name__ == "__main__":main()
