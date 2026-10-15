"""Connect existing residual and feature-intervention evidence; no inference."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
from utils.experiment.run_helpers import open_text

CORE = "ensemble__original3_mse"
FIVE = "ensemble__original3_hgb_extra_trees"
ET4 = "ensemble__original3_extra_trees"
HGB4 = "ensemble__original3_hgb"
MEMBERS = ["randomforest", "conformer", "alexnet", "hgb", "extra_trees"]
MODELS = [CORE, HGB4, ET4, FIVE]
NOISE = ["clean", "0", "-4", "-8", "-12", "-16", "-20"]
POOLED = "pooled_source_day_thresholds"
SOURCE = ROOT / "experiments/2026-10-07_onb_200epoch_result_analysis"
INPUT = ROOT / "experiments/2026-10-07_onb_input_and_integration"


def read(path):
    with open_text(path, "r", encoding="utf-8-sig") as handle:
        return pd.read_csv(handle, dtype={"noise": str})


def digest(path):
    with open_text(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def save(name, rows):
    pd.DataFrame(rows).to_csv(OUT/name, index=False, encoding="utf-8-sig")


def main():
    comparison = read(SOURCE/"comparison_across_runs.csv")
    report = Path(comparison.loc[comparison.seed.eq(42)].iloc[0].source_result).parent
    metric_path = report/"onb_comparison/main_metrics.csv"
    fit_path = report/"heatflux_no_noise/ensemble_training_oof_fit_f1.json"
    with open_text(fit_path, "r", encoding="utf-8") as handle:
        fit = json.load(handle)
    weights = fit["weights"][FIVE.removeprefix("ensemble__")]
    metrics = read(metric_path)
    interventions = read(INPUT/"stage1/metrics.csv")
    removal = read(INPUT/"stage3/metrics.csv")
    source_paths = [metric_path, fit_path, INPUT/"stage1/metrics.csv", INPUT/"stage3/metrics.csv"]
    frames, region_counts, decompositions, directions, correlations = {}, [], [], [], []
    original_hashes = {str(path): digest(path) for path in source_paths}
    for noise in NOISE:
        directory = "heatflux_no_noise" if noise == "clean" else "heatflux_reference_SNR="+noise
        suffix = "no_noise" if noise == "clean" else noise
        path = report/directory/f"fold_pred/pred_f1_{suffix}.csv"
        source_paths.append(path)
        original_hashes[str(path)] = digest(path)
        frame = read(path)
        assert len(frame) == 540 and not frame.duplicated(["source_wav_id", "chunk_index"]).any()
        y = frame.y_true.to_numpy(float)/1000
        days = frame.source_wav_id.str[:8]
        thresholds = days.map({"20250611": 221.5051102, "20250618": 271.6776816}).to_numpy(float)
        assert np.isfinite(thresholds).all()
        positive = y >= thresholds
        near = abs(y-thresholds) <= .1*abs(thresholds)
        masks = {"all": np.ones(len(y), bool), "pre_onb": ~positive,
                 "near_onb_10pct": near, "high_above_onb_band": positive & ~near}
        # In this evaluation, the near band contains only the two first ONB stages.
        assert not np.any(near & ~positive)
        assert sum(np.sum(masks[k]) for k in list(masks)[1:]) == 540
        assert np.allclose(sum(frame[k].to_numpy(float)*weights[k] for k in MEMBERS), frame[FIVE], rtol=1e-8, atol=1e-4)
        frames[noise] = frame
        for region, mask in masks.items():
            region_counts.append({"noise": noise, "region": region, "chunks": int(mask.sum()),
                                  "source_wavs": int(frame.loc[mask, "source_wav_id"].nunique())})
            truth = y[mask]
            prediction = frame[FIVE].to_numpy(float)[mask]/1000
            block_predictions = np.column_stack([frame[k].to_numpy(float)[mask]/1000 for k in [CORE,"hgb","extra_trees"]])
            block_errors = block_predictions-truth[:, None]
            mixed = (block_errors.min(axis=1)<0) & (block_errors.max(axis=1)>0)
            for baseline in [CORE, ET4, HGB4, "extra_trees"]:
                before = frame[baseline].to_numpy(float)[mask]/1000
                error, adjustment = before-truth, prediction-before
                old_mse, new_mse = float(np.mean(error**2)), float(np.mean((prediction-truth)**2))
                alignment, cost = float(2*np.mean(error*adjustment)), float(np.mean(adjustment**2))
                assert np.isclose(new_mse-old_mse, alignment+cost, atol=1e-9, rtol=1e-10)
                decompositions.append({"noise": noise, "region": region, "baseline": baseline,
                                       "chunks": len(truth), "baseline_rmse": np.sqrt(old_mse), "five_rmse": np.sqrt(new_mse),
                                       "five_minus_baseline_mse": new_mse-old_mse,
                                       "error_adjustment_alignment": alignment, "adjustment_squared_cost": cost,
                                       "improved_chunks": int(np.sum((prediction-truth)**2<error**2)),
                                       "worsened_chunks": int(np.sum((prediction-truth)**2>error**2))})
                if baseline == CORE:
                    for label, selected in [("mixed_sign_blocks",mixed),("same_sign_blocks",~mixed)]:
                        total_gain = np.sum(error[selected]**2-(prediction[selected]-truth[selected])**2)
                        directions.append({"noise":noise,"region":region,"block_error_pattern":label,
                                           "chunks":int(selected.sum()),"improved_chunks":int(np.sum((prediction[selected]-truth[selected])**2<error[selected]**2)),
                                           "net_sse_reduction_kW_m2_squared":float(total_gain),
                                           "contribution_to_total_mse_reduction":float(total_gain/len(truth))})
            for i, left in enumerate(MEMBERS):
                for right in MEMBERS[i+1:]:
                    e1 = frame[left].to_numpy(float)[mask]/1000-truth
                    e2 = frame[right].to_numpy(float)[mask]/1000-truth
                    rho = float(np.corrcoef(e1,e2)[0,1]) if np.std(e1)>0 and np.std(e2)>0 else np.nan
                    correlations.append({"noise":noise,"region":region,"left":left,"right":right,
                                         "residual_correlation":rho,"residual_product_mean":float(np.mean(e1*e2)),
                                         "opposite_error_fraction":float(np.mean(e1*e2<0))})
        # Verify full metrics against the saved current main table.
        for model in MODELS:
            row = metrics.loc[metrics.noise.eq(noise)&metrics.source_day.eq(POOLED)&metrics.model_key.eq(model)].iloc[0]
            pred = frame[model].to_numpy(float)/1000
            assert np.isclose(np.sqrt(np.mean((pred-y)**2)),row.rmse_all_kW_m2,atol=1e-9)
            assert int(np.sum(positive & (pred<thresholds))) == int(row.fn)
            assert int(np.sum(~positive & (pred>=thresholds))) == int(row.fp)
    save("regional_loss_decomposition.csv",decompositions)
    save("error_direction_contributions.csv",directions)
    save("regional_residual_relationships.csv",correlations)
    save("evaluation_regions.csv",region_counts)
    selected = interventions.loc[interventions.source_day.eq(POOLED)&interventions.model_key.isin(MEMBERS+[CORE,FIVE])].copy()
    baseline = selected.loc[selected.intervention.eq("unchanged"),["noise","model_key","rmse_all_kW_m2","fn","fp"]]
    paired = selected.merge(baseline,on=["noise","model_key"],suffixes=("","__original"),validate="many_to_one")
    paired["rmse_change"] = paired.rmse_all_kW_m2-paired.rmse_all_kW_m2__original
    paired["fn_change"] = paired.fn-paired.fn__original
    save("existing_feature_dependence.csv",paired[["noise","intervention","model_key","rmse_all_kW_m2__original","rmse_all_kW_m2","rmse_change","fn_change"]])
    removal_keys=["hgb","extra_trees", "hgb__absolute_power5","hgb__relative_shape29",
                  "extra_trees__absolute_power5","extra_trees__relative_shape29"]
    reused=removal.loc[removal.scope.eq("outer_development")&removal.source_day.eq(POOLED)
                      &removal.noise.isin(["clean","-20"])&removal.model_key.isin(removal_keys)]
    assert len(reused)==12 and reused.n.eq(540).all()
    save("existing_input_removal_summary.csv",reused[["noise","model_key","rmse_all_kW_m2","rmse_onb_kW_m2","fn","fp"]])

    # Native CSV for slide tables. All values come from the saved current run.
    table=metrics.loc[metrics.source_day.eq(POOLED)&metrics.noise.isin(["clean","-20"])&metrics.model_key.isin(MODELS+["hgb","extra_trees"])]
    save("slide_endpoint_table.csv",table[["noise","model_key","rmse_all_kW_m2","rmse_onb_kW_m2","fn","fp"]])
    qtable=metrics.loc[~metrics.source_day.eq(POOLED)&metrics.noise.eq("clean")&metrics.model_key.isin(MODELS)]
    save("slide_q100_table.csv",qtable[["source_day","model_key","q100_kW_m2","g100_kW_m2"]])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":15,"axes.spines.top":False,"axes.spines.right":False})
    labels={CORE:"Original 3",HGB4:"3 + HGB",ET4:"3 + ET",FIVE:"3 + HGB + ET",
            "conformer":"CNN+Transformer","alexnet":"AlexNet","hgb":"HGB","extra_trees":"ExtraTrees"}
    colors={CORE:"#526b86",HGB4:"#41a694",ET4:"#dfa13d",FIVE:"#1686e7"}
    def finish(fig,name):
        fig.tight_layout()
        fig.savefig(OUT/(name+".png"),dpi=190,bbox_inches="tight")
        fig.savefig(OUT/(name+".svg"),bbox_inches="tight")
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(9.5,4.7))
    for model in MODELS:
        values=[metrics.loc[metrics.noise.eq(n)&metrics.source_day.eq(POOLED)&metrics.model_key.eq(model),"rmse_all_kW_m2"].iloc[0] for n in NOISE]
        ax.plot(range(7),values,"o-",label=labels[model],color=colors[model],linewidth=2.7)
    ax.set_xticks(range(7),["Clean","0","-4","-8","-12","-16","-20"])
    ax.set_xlabel("Reference SNR [dB]");ax.set_ylabel("RMSE [kW/m²]")
    ax.legend(ncol=2,fontsize=12);ax.grid(axis="y",alpha=.18)
    finish(fig,"slide_regression_noise_trend")
    fig,ax=plt.subplots(figsize=(8.7,4.7))
    for i,noise in enumerate(["clean","-20"]):
        vals=[int(table.loc[table.noise.eq(noise)&table.model_key.eq(k),"fn"].iloc[0]) for k in MODELS]
        bars=ax.bar(np.arange(4)+(i-.5)*.34,vals,width=.34,label="Clean" if noise=="clean" else "-20 dB",color=["#98cedb","#1686e7"][i])
        ax.bar_label(bars,fontsize=15,padding=3)
    ax.set_xticks(range(4),["Original 3","3 + HGB","3 + ET","5 models"])
    ax.set_ylabel("False negatives / 315 positive chunks");ax.legend();ax.set_ylim(0,80)
    finish(fig,"slide_onb_false_negatives")
    d=pd.DataFrame(decompositions)
    fig,axs=plt.subplots(1,2,figsize=(11.7,4.3))
    regions=["pre_onb","near_onb_10pct","high_above_onb_band"]
    for i,noise in enumerate(["clean","-20"]):
        vals=[-d.loc[d.noise.eq(noise)&d.baseline.eq(CORE)&d.region.eq(r),"five_minus_baseline_mse"].iloc[0] for r in regions]
        bars=axs[0].bar(np.arange(3)+(i-.5)*.34,vals,width=.34,label="Clean" if noise=="clean" else "-20 dB",color=["#98cedb","#1686e7"][i])
    axs[0].axhline(0,color="#555555",linewidth=.9)
    axs[0].set_xticks(range(3),["Pre-ONB","Near ONB","High q"])
    axs[0].set_ylabel("MSE reduction: original 3 to 5\n[(kW/m²)²]")
    axs[0].legend(fontsize=11)
    for i,variant in enumerate(["unchanged","time_permutation"]):
        keys=["conformer","alexnet","hgb","extra_trees"]
        vals=[paired.loc[paired.noise.eq("clean")&paired.intervention.eq(variant)&paired.model_key.eq(k),"rmse_all_kW_m2"].iloc[0] for k in keys]
        axs[1].bar(np.arange(4)+(i-.5)*.34,vals,width=.34,label="Original" if i==0 else "Time permutation",color=["#98cedb","#1686e7"][i])
    axs[1].set_xticks(range(4),["CNN+Tf","AlexNet","HGB","ET"])
    axs[1].set_ylabel("Clean RMSE [kW/m²]");axs[1].legend(fontsize=11)
    finish(fig,"slide_error_and_feature_linkage")

    # Compare gains from mixed-sign and same-sign blocks without claiming causality.
    pattern=pd.DataFrame(directions)
    fig,ax=plt.subplots(figsize=(8.7,4.6))
    for i,kind in enumerate(["mixed_sign_blocks","same_sign_blocks"]):
        vals=[pattern.loc[pattern.noise.eq(n)&pattern.region.eq("all")&pattern.block_error_pattern.eq(kind),"contribution_to_total_mse_reduction"].iloc[0] for n in ["clean","-20"]]
        bars=ax.bar(np.arange(2)+(i-.5)*.34,vals,width=.34,label="Mixed signs" if i==0 else "Same sign",color=["#526b86","#1686e7"][i])
        ax.bar_label(bars,fmt="%.1f",padding=3)
    ax.set_xticks(range(2),["Clean","-20 dB"])
    ax.set_ylabel("Contribution to total MSE reduction\n[(kW/m²)²]");ax.legend();ax.axhline(0,color="#555555",linewidth=.9)
    finish(fig,"error_direction_contributions")
    verification={"scope":"2025 June11+18, seed42, 200 epochs, 1s/3kHz, within_wav_chunk, clean_only",
                  "evaluation_chunks_per_noise":540,"noise_conditions":NOISE,
                  "regional_decompositions":len(decompositions),"direction_groups":len(directions),
                  "correlation_rows":len(correlations),"existing_feature_rows":len(paired),
                  "reused_input_removal_rows":len(reused),
                  "identity_and_saved_metrics_checks":"passed",
                  "new_model_fits":0,"new_model_inference":0,"new_input_interventions":0,
                  "sources":[{"path":str(p.relative_to(ROOT)),"sha256":original_hashes[str(p)]} for p in source_paths],
                  "sources_unchanged":all(digest(p)==original_hashes[str(p)] for p in source_paths),
                  "limitations":["Fixed-fit input intervention and retraining input removal answer different questions.",
                                 "Residual decomposition is an identity, not proof of the physical acoustic mechanism.",
                                 "Feature dependence and integration benefit are compatible evidence, not isolated causal contribution.",
                                 "Only existing target runs; no 2026 data or new July9 evaluation."]}
    assert verification["sources_unchanged"]
    (OUT/"verification.json").write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding="utf-8")
    print(d.loc[d.noise.isin(["clean","-20"])&d.baseline.eq(CORE)].round(3).to_string(index=False))
    print(pattern.loc[pattern.noise.isin(["clean","-20"])&pattern.region.eq("all")].round(3).to_string(index=False))
    print(d.loc[d.noise.isin(["clean","-20"])&d.baseline.eq(ET4)&d.region.eq("all")].round(3).to_string(index=False))
    print(json.dumps({"sources_unchanged":True,"regional_rows":len(d),"fits":0,"inference":0}))


if __name__ == "__main__":
    main()
