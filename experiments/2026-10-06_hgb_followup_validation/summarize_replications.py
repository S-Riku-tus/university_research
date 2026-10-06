"""Compare both predefined seeds without selecting the better seed or method."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"code"))
import run_hgb_complementarity_validation as run


def save_csv(name,rows):
    assert rows
    with (OUT/name).open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def read_csv(name):
    with (OUT/name).open(encoding="utf-8-sig",newline="") as stream:return list(csv.DictReader(stream))


def main():
    rows=read_csv("metrics.csv")
    outer=[r for r in rows if r["unit"]=="outer_unused_chunk"]
    lookup={(int(r["seed"]),r["method"],r["noise"],r["day"]):r for r in outer}
    seeds=run.CONFIG["replication_seeds"]
    methods=sorted({r["method"] for r in outer})
    summary=[]
    for seed in seeds:
        for method in methods:
            clean=lookup[(seed,method,"clean","two_day")];last=lookup[(seed,method,"-20","two_day")]
            noisy=[lookup[(seed,method,noise,"two_day")] for noise in run.NOISES[1:]]
            summary.append({"seed":seed,"method":method,"training_exposure":"noise_exposed_combiner" if "noisy" in method else "clean_only",
                "clean_rmse":float(clean["rmse"]),"clean_mae":float(clean["mae"]),"clean_fn":int(clean["fn"]),"clean_fp":int(clean["fp"]),"clean_recall":float(clean["recall"]),
                "noise_mean_rmse":float(np.mean([float(r["rmse"]) for r in noisy])),"noise_mean_fn":float(np.mean([int(r["fn"]) for r in noisy])),"noise_fp_sum":sum(int(r["fp"]) for r in noisy),
                "minus20_rmse":float(last["rmse"]),"minus20_fn":int(last["fn"]),"minus20_fp":int(last["fp"]),"minus20_near_rmse":float(last["rmse_onb"]),
                "clean_q100_0611":float(lookup[(seed,method,"clean","20250611")]["q100"]),"clean_q100_0618":float(lookup[(seed,method,"clean","20250618")]["q100"]),
                "minus20_q100_0611":float(lookup[(seed,method,"-20","20250611")]["q100"]),"minus20_q100_0618":float(lookup[(seed,method,"-20","20250618")]["q100"])})
    save_csv("replication_summary.csv",summary)
    contrasts=[("hgb4_clean_mse","existing3_clean_mse"),("hgb4_clean_mse","hgb"),("hgb4_noisy_mse","hgb4_clean_mse"),
        ("hgb4_noisy_mse","existing3_noisy_mse"),("hgb4_diversity","hgb4_clean_mse"),("hgb4_noisy_ridge","hgb4_noisy_mse"),("conformer_hgb_clean_mse","hgb4_clean_mse"),
        ("hgb4_clean_ridge","existing3_clean_ridge"),("hgb4_noisy_ridge","existing3_noisy_ridge")]
    deltas=[]
    for seed in seeds:
        for candidate,reference in contrasts:
            for noise in run.NOISES:
                c=lookup[(seed,candidate,noise,"two_day")];r=lookup[(seed,reference,noise,"two_day")]
                deltas.append({"seed":seed,"candidate":candidate,"reference":reference,"noise":noise,
                    **{f"delta_{key}":float(c[key])-float(r[key]) for key in ["rmse","mae","recall","fp","fn","rmse_onb"]}})
    save_csv("paired_method_deltas.csv",deltas)
    aggregate=[]
    for method in methods:
        cells=[r for r in summary if r["method"]==method]
        aggregate.append({"method":method,**{f"mean_{key}":float(np.mean([r[key] for r in cells])) for key in ["clean_rmse","clean_mae","clean_fn","clean_fp","noise_mean_rmse","noise_mean_fn","noise_fp_sum","minus20_rmse","minus20_fn","minus20_fp"]},
            "seeds":str(seeds),"independent_experiment_days":False})
    save_csv("seed_aggregates.csv",aggregate)
    # Check both IDs and endpoint suffixes directly from the saved final predictions.
    groups=defaultdict(list)
    for row in read_csv("outer_predictions.csv"):groups[(int(row["seed"]),row["method"],row["noise"])].append(row)
    wav_changes=[];region_changes=[];complementarity=[]
    for seed in seeds:
        for noise in run.NOISES:
            old=groups[(seed,"existing3_clean_mse",noise)];new=groups[(seed,"hgb4_clean_mse",noise)]
            assert [(r["source_wav_id"],r["chunk_index"]) for r in old]==[(r["source_wav_id"],r["chunk_index"]) for r in new]
            y=np.asarray([float(r["y_kW_m2"]) for r in old]);t=np.asarray([float(r["onb_kW_m2"]) for r in old])
            p=np.asarray([float(r["prediction_kW_m2"]) for r in old]);q=np.asarray([float(r["prediction_kW_m2"]) for r in new])
            wav=np.asarray([r["source_wav_id"] for r in old]);old_sse=(p-y)**2;new_sse=(q-y)**2
            for name in sorted(set(wav)):
                mask=wav==name
                wav_changes.append({"seed":seed,"noise":noise,"source_wav_id":name,"chunks":int(mask.sum()),
                    "existing3_mse":float(old_sse[mask].mean()),"hgb4_mse":float(new_sse[mask].mean()),"mse_reduction":float((old_sse-new_sse)[mask].mean())})
            regions={"below60":y<60,"pre_onb_60_to_threshold":(y>=60)&(y<t),"onb_to_1p2_threshold":(y>=t)&(y<1.2*t),
                "above_1p2_threshold":y>=1.2*t,"near_onb_overlapping":np.abs(y-t)<=.1*t}
            for name,mask in regions.items():
                region_changes.append({"seed":seed,"noise":noise,"region":name,"chunks":int(mask.sum()),
                    "existing3_rmse":float(np.sqrt(old_sse[mask].mean())),"hgb4_rmse":float(np.sqrt(new_sse[mask].mean())),
                    "mse_reduction":float((old_sse-new_sse)[mask].mean()),"existing3_bias":float((p-y)[mask].mean()),"hgb4_bias":float((q-y)[mask].mean())})
            base=np.column_stack([[float(r["prediction_kW_m2"]) for r in groups[(seed,key,noise)]] for key in run.KEYS])
            common_fn=(y>=t)&np.all(base[:,:3]<t[:,None],axis=1)
            complementarity.append({"seed":seed,"noise":noise,"three_base_common_fn":int(common_fn.sum()),"hgb_rescues_base_common_fn":int((common_fn&(base[:,3]>=t)).sum()),
                "existing3_fn_corrected_by_hgb4":int(((y>=t)&(p<t)&(q>=t)).sum()),"existing3_tp_lost_by_hgb4":int(((y>=t)&(p>=t)&(q<t)).sum()),
                "existing3_fp_corrected_by_hgb4":int(((y<t)&(p>=t)&(q<t)).sum()),"existing3_tn_lost_by_hgb4":int(((y<t)&(p<t)&(q>=t)).sum()),
                "hgb_vs_conformer_residual_correlation":float(np.corrcoef(base[:,3]-y,base[:,1]-y)[0,1])})
    save_csv("paired_wav_errors.csv",wav_changes);save_csv("paired_region_errors.csv",region_changes);save_csv("error_complementarity.csv",complementarity)
    endpoint_checks=0;metric_checks=0
    for key,cells in groups.items():
        assert len(cells)==len({(r["source_wav_id"],int(r["chunk_index"])) for r in cells})==540
        y=np.asarray([float(r["y_kW_m2"]) for r in cells]);p=np.asarray([float(r["prediction_kW_m2"]) for r in cells]);t=np.asarray([float(r["onb_kW_m2"]) for r in cells])
        expected=lookup[(*key,"two_day")]
        np.testing.assert_allclose(np.linalg.norm(p-y)/np.sqrt(540),float(expected["rmse"]),rtol=1e-12)
        assert int(((y>=t)&(p<t)).sum())==int(expected["fn"])
        assert int(((y<t)&(p>=t)).sum())==int(expected["fp"])
        metric_checks+=1
        for day in run.THRESHOLDS:
            mask=np.asarray([r["source_wav_id"].startswith(day) for r in cells]);levels=np.unique(y[mask]);failed=y[mask & (p<t)]
            possible=levels[levels>failed.max()] if len(failed) else levels
            q=float(possible[0]) if len(possible) else float("nan")
            np.testing.assert_allclose(q,float(lookup[(*key,day)]["q100"]),rtol=1e-12,equal_nan=True);endpoint_checks+=1
    artifacts=[]
    for seed in seeds:
        folder=OUT/f"seed{seed}";manifest=run.read_json(folder/"manifest.json")
        train=set(manifest["train_indices"]);test=set(manifest["test_indices"]);assert not train&test and len(train)==1620 and len(test)==540
        wavs={r["source_wav_id"] for r in manifest["samples"]}
        for indices,expected_count in [(train,45),(test,15)]:
            counts=defaultdict(int)
            for i in indices:counts[manifest["samples"][i]["source_wav_id"]]+=1
            assert set(counts)==wavs and set(counts.values())=={expected_count}
        inner_held=set()
        for fold in manifest["folds"]:
            fit=set(fold["fit_indices"]);held=set(fold["held_indices"])
            assert not fit&held and fit|held==train and not inner_held&held
            assert {manifest["samples"][i]["source_wav_id"] for i in fit}==wavs
            inner_held.update(held)
        assert inner_held==train
        for stage in ["fold1","fold2","fold3","final"]:
            for model in run.KEYS:
                audit=run.read_json(folder/stage/f"{model}_complete.json")
                assert audit["config_sha256"]==run.CONFIG_HASH and audit["reload_verified"] and audit["clean_fit"] and not audit["noise_refit"]
                artifact=folder/stage/(f"{model}.weights.h5" if model in ["conformer","alexnet"] else f"{model}.joblib")
                assert hashlib.sha256(artifact.read_bytes()).hexdigest()==audit["artifact_sha256"]
                if model in ["conformer","alexnet"]:
                    assert audit["training_info"]["epochs_completed"]==150
                    assert audit["training_info"]["actual_batch_size"]==(12 if model=="conformer" else 8)
                artifacts.append(audit)
    run.save_json(OUT/"verification.json",{"status":"passed","base_models_verified":len(artifacts),"full_150_epoch_keras_fits":sum(r["model"] in ["conformer","alexnet"] for r in artifacts),
        "outer_metric_sets":metric_checks,"independent_day_endpoint_checks":endpoint_checks,"shared_outer_training_chunks":0,
        "all_seeds_reported":seeds,"noisy_oof_base_models_all_clean_fitted":True,"combiner_noise_exposure_separated":True,"saved_base_model_hashes_verified":True,
        "every_wav_outer_fit_test_chunks":[45,15],"all_36_wavs_in_every_internal_fit":True})
    # Show individual seeds and false alarms so averaging does not hide instability.
    fig,axes=plt.subplots(1,3,figsize=(15,4.7))
    shown=[("existing3_clean_mse","Existing3, clean weights"),("hgb","HGB single"),("hgb4_clean_mse","HGB added, clean weights"),("hgb4_noisy_mse","HGB added, noisy weights")]
    for method,label in shown:
        rmse=[];fn=[];fp=[]
        for noise in run.NOISES:
            rmse.append(np.mean([float(lookup[(seed,method,noise,"two_day")]["rmse"]) for seed in seeds]))
            fn.append(np.mean([int(lookup[(seed,method,noise,"two_day")]["fn"]) for seed in seeds]))
            fp.append(np.mean([int(lookup[(seed,method,noise,"two_day")]["fp"]) for seed in seeds]))
        line,=axes[0].plot(range(7),rmse,marker="o",label=label)
        axes[1].plot(range(7),fn,marker="o",color=line.get_color(),label=label)
        axes[2].plot(range(7),fp,marker="o",color=line.get_color(),label=label)
        for seed in seeds:
            for ax,key in zip(axes,["rmse","fn","fp"]):
                ax.plot(range(7),[float(lookup[(seed,method,noise,"two_day")][key]) for noise in run.NOISES],color=line.get_color(),alpha=.3,linestyle="--",linewidth=1)
    for ax in axes:ax.set_xticks(range(7),run.NOISES);ax.set_xlabel("Evaluation noise (dB)");ax.grid(alpha=.25)
    axes[0].set_ylabel("RMSE (kW/m2)");axes[1].set_ylabel("False negatives / 315");axes[2].set_ylabel("False positives / 225")
    axes[0].set_title("Paired seeds43 and44; dashed = each seed");axes[1].set_title("Detection misses");axes[2].set_title("False alarms, including split sensitivity")
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc="lower center",ncol=2,fontsize=9)
    fig.tight_layout(rect=(0,.17,1,1));fig.savefig(OUT/"replication_comparison.png",dpi=170);fig.savefig(OUT/"replication_comparison.pdf");plt.close(fig)
    print("Summary complete;",len(artifacts),"base models and",endpoint_checks,"day endpoints verified")
    for row in summary:
        if row["method"] in [r[0] for r in shown]:print(row)


if __name__=="__main__":main()
