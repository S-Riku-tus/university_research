"""Additional checks and scientific figures from the frozen snapshot and saved arrays.

No model training or gradient recomputation. Output must be a new directory.
"""
import argparse
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from analyze_results import ROOT, MODELS, NOISES, LABELS, longpath, relative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.snapshot = args.snapshot.resolve()
    out = args.output
    if out.exists() and any(out.iterdir()):
        raise SystemExit("Use a new empty directory.")
    out.mkdir(parents=True, exist_ok=True)
    sources = {}

    def read(p):
        data = p.read_bytes()
        sources[relative(p)] = hashlib.sha256(data).hexdigest()
        return data

    def table(name):
        return pd.read_csv(io.BytesIO(read(args.snapshot/(name+".csv"))), keep_default_na=False, dtype={"snr":str})

    def savefig(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(out/f"{name}.{ext}", bbox_inches="tight", dpi=180)
        plt.close(fig)

    samples = table("sample_correspondence")
    expl = table("explainability_summary")
    masks = table("group_mask_performance")
    inv = table("run_inventory")
    predictions = table("wav_predictions")
    validation = {"mask_baselines_checked": 0, "signed_map_sums_checked": 0}
    for _, row in inv.iterrows():
        d = longpath(ROOT/row.run_manifest).parent
        for f in range(1,4):
            df = pd.read_csv(io.BytesIO(read(d/f"fold_pred/pred_f{f}_{row.snr}.csv")))
            group = df.groupby("source_wav_id")[["y_true",*MODELS]].median()
            for key in MODELS[:3]:
                score = r2_score(group.y_true, group[key])
                match = masks[(masks.snr==row.snr)&(masks.fold==f)&(masks.model==key)]
                assert np.allclose(match.base_r2, score, rtol=0, atol=1e-7)
                validation["mask_baselines_checked"] += 1
    ig = expl[expl.method=="integrated_gradients"]
    merged = ig.merge(samples[["snr","fold","model","sample_id","sample_dir"]], on=["snr","fold","model","sample_id"], validate="one_to_one")
    for _, row in merged.iterrows():
        a = np.load(io.BytesIO(read(longpath(ROOT/row.sample_dir)/"integrated_gradients_signed.npy")))
        assert np.isclose(float(np.sum(a)), float(row.attribution_sum), rtol=2e-6, atol=.1)
        validation["signed_map_sums_checked"] += 1

    error_rows = []
    for snr in NOISES:
        g = predictions[predictions.snr==snr]
        for key in MODELS:
            errors = g[key+"_pred_median"]-g.y_true
            total = np.square(errors).sum()
            for idx, row in g.iterrows():
                error_rows.append({"snr":snr,"model":key,"source_wav_id":row.source_wav_id,"y_true":row.y_true,
                    "prediction":row[key+"_pred_median"],"error":errors.loc[idx],"fraction_total_squared_error":errors.loc[idx]**2/total})
    pd.DataFrame(error_rows).to_csv(out/"wav_error_contributions.csv",index=False,encoding="utf-8-sig")
    previous_path = ROOT/"experiments/2026-09-15_research_status_snapshot/wav_median_metrics.csv"
    previous = pd.read_csv(io.BytesIO(read(previous_path)),keep_default_na=False,dtype={"snr":str})
    previous = previous[previous.maxfreq=="maxfreq=22kHz"]
    now = table("wav_median_metrics")
    comp = now.merge(previous[["snr","model_key","r2","recall"]],on=["snr","model_key"],suffixes=("_sep15","_sep14"))
    comp["r2_change"] = comp.r2_sep15-comp.r2_sep14
    comp[["snr","model_key","r2_sep14","r2_sep15","r2_change","recall_sep14","recall_sep15"]].to_csv(out/"previous_run_comparison.csv",index=False,encoding="utf-8-sig")

    # Matched representative: first chunk of the same ONB WAV, for all three models.
    chosen = samples[(samples.snr.isin(["no_noise","0","-20"])) & (samples.sample_id=="near_onb_above_val0180")]
    assert len(chosen)==9 and chosen.source_wav_id.nunique()==1 and chosen.chunk_index.nunique()==1
    chosen.to_csv(out/"representative_samples.csv",index=False,encoding="utf-8-sig")
    raw_inputs, maps, diagnostic = {}, {}, []
    for snr in ["no_noise","0","-20"]:
        row = chosen[(chosen.snr==snr)&(chosen.model=="cnntf_v2_gap")].iloc[0]
        d = longpath(ROOT/row.sample_dir)
        manifest_path = longpath(ROOT/inv[inv.snr==snr].iloc[0].run_manifest)
        m = json.loads(read(manifest_path))
        source = longpath(Path(m["dataset"]["data_path"]))/row.sample_filename
        arr = np.load(io.BytesIO(read(source)),allow_pickle=False)
        raw_inputs[snr] = arr
        # Probe only the known scalar log front end, not the trained neural network.
        value = float(np.median(arr[arr>0]))
        ratio = value/1e-12
        alpha = np.linspace(0,1,65)
        approx = np.trapz(ratio/(1+alpha*ratio), alpha)
        exact = np.log1p(ratio)
        diagnostic.append({"snr":snr,"positive_input_power_median":value,"input_power_to_log_scale_ratio":ratio,
            "scalar_log_exact_delta":exact,"scalar_log_trapezoid_64":approx,"scalar_log_relative_error":abs(approx-exact)/abs(exact),
            "log_change_fraction_at_first_alpha":np.log1p(ratio/64)/exact,
            "scope":"scalar_front_end_only; not a trained-model completeness test"})
        for model in MODELS[1:3]:
            row = chosen[(chosen.snr==snr)&(chosen.model==model)].iloc[0]
            path = longpath(ROOT/row.sample_dir)
            maps[snr,model] = np.load(io.BytesIO(read(path/"integrated_gradients_magnitude.npy")),allow_pickle=False)
            # Preserve original plots for tracing the redrawn array figure.
            original = path/"integrated_gradients_signed.png"
            (out/f"original_{snr}_{model}_ig_signed.png").write_bytes(read(original))
        (out/f"original_{snr}_input.png").write_bytes(read(d/"input_spectrogram.png"))
    pd.DataFrame(diagnostic).to_csv(out/"log_frontend_quadrature_probe.csv",index=False,encoding="utf-8-sig")

    fig, axes = plt.subplots(3,3,figsize=(13,9),sharex=True,sharey=True,layout="constrained")
    logs = {s:np.log1p(raw_inputs[s]/1e-12) for s in raw_inputs}
    vmin = min(a.min() for a in logs.values()); vmax = max(a.max() for a in logs.values())
    for i,snr in enumerate(["no_noise","0","-20"]):
        img=axes[i,0].imshow(logs[snr].T,origin="lower",aspect="auto",extent=[0,1,0,22],vmin=vmin,vmax=vmax,cmap="magma")
        for j,model in enumerate(MODELS[1:3],1):
            attr=axes[i,j].imshow(maps[snr,model].T,origin="lower",aspect="auto",extent=[0,1,0,22],vmin=0,vmax=1,cmap="viridis")
            s=chosen[(chosen.snr==snr)&(chosen.model==model)].iloc[0]
            er=float(ig[(ig.snr==snr)&(ig.model==model)&(ig.sample_id==s.sample_id)].iloc[0].completeness_relative_error)
            axes[i,j].set_title(f"{LABELS[j]}: prediction {s.y_pred/1000:.1f} kW/m2\nIG relative error {er:.2g}",fontsize=10)
        axes[i,0].set_title("Input log power: "+("Clean" if snr=="no_noise" else "reference SNR "+snr+" dB"))
        axes[i,0].set_ylabel("Frequency (kHz)")
    for ax in axes[-1]: ax.set_xlabel("Time within chunk (s)")
    fig.colorbar(img,ax=axes[:,0].tolist(),shrink=.7,label="log1p(power / 1e-12); shared input scale")
    fig.colorbar(attr,ax=axes[:,1:].ravel().tolist(),shrink=.7,label="Saved normalized IG magnitude (within-map visualization)")
    fig.suptitle("Same ONB WAV / chunk 0 / fold 1 / true heat flux 368.978 kW/m2\nIG numerical consistency is unresolved; maps are exploratory",fontsize=13)
    savefig(fig,"matched_onb_input_and_ig")
    fig, axes=plt.subplots(1,3,figsize=(13,4),sharex=True,sharey=True)
    for ax,snr in zip(axes,["no_noise","0","-20"]):
        g=predictions[predictions.snr==snr].sort_values("y_true")
        for key,label in zip(MODELS,LABELS):
            ax.plot(g.y_true/1000,g[key+"_pred_median"]/1000,marker="o",markersize=3,label=label)
        ax.plot([0,860],[0,860],color="black",ls="--",lw=1)
        ax.axhline(368.978105,color="gray",ls=":");ax.axvline(368.978105,color="gray",ls=":")
        ax.set_title("Clean" if snr=="no_noise" else "Reference SNR "+snr+" dB")
        ax.set_xlabel("True heat flux (kW/m2)");ax.grid(alpha=.2)
    axes[0].set_ylabel("Predicted WAV median (kW/m2)"); axes[0].legend(fontsize=7)
    fig.tight_layout();savefig(fig,"operating_point_predictions")
    for path in ["code/utils/models/regression/base_regression.py","code/utils/explainability/spectrogram_explainers.py","code/utils/explainability/training_integration.py"]:
        read(ROOT/path)
    (out/"verification.json").write_text(json.dumps({"created_at":datetime.now().astimezone().isoformat(),**validation,"sources_sha256":sources,
        "note":"No learned model was loaded. Scalar log probe illustrates a numerical mechanism, not its full-network causal confirmation."},ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(validation));print(pd.DataFrame(diagnostic).to_string(index=False))


if __name__=="__main__":
    main()
