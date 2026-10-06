"""HGB feature-group dependence and common input interventions on saved models.

Feature permutation runs on CPU; neural intervention checks use the recorded GPU path.
"""
from __future__ import annotations

import os
import sys
if "--device" in sys.argv and sys.argv[sys.argv.index("--device")+1] == "cpu":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
import csv
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PRIOR = ROOT/"experiments/2026-10-06_chunk_complementary_model_pilots"
sys.path.insert(0,str(ROOT/"code"));sys.path.insert(0,str(PRIOR))
import pilot as p
from utils.dataloading.acoustic_summary_features import AcousticFrequency34


def save_csv(name,rows):
    assert rows
    with (OUT/name).open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def save_json(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def permutation_dependence():
    rows,folds,data_root,outer,base,y,days=p.setup()
    names=p.load_features(rows,data_root,"clean","train")[1][:34]
    feature_groups={"power_and_band_ratios":list(range(7)),"frequency_shape": [7,8,*range(10,34)],
                    "temporal_cv":[9],"fine_relative_power":list(range(10,34)),"all_features":list(range(34))}
    registry=next(r for r in p.candidates() if r["method"]=="frequency34_HGB_rmse")
    audits=p.read_json(PRIOR/"frequency34_HGB_audit.json")["nested_fits"]
    models={r["fold"]:joblib.load(PRIOR/r["artifact"]) for r in audits if r["policy"]=="rmse"}
    results=[]
    for noise in ["clean","-20"]:
        features=p.load_features(rows,data_root,noise,"train")[0][:,:34]
        baseline=np.load(PRIOR/"frequency34_HGB_rmse_oof.npz")["clean" if noise=="clean" else "minus20"]
        t=np.asarray([p.THRESHOLDS[d] for d in days])
        for group,indices in feature_groups.items():
            for repeat in range(10):
                permuted=np.full(len(rows),np.nan)
                for fold,(fit,held) in enumerate(folds,1):
                    changed=features[held].copy()
                    donor=np.random.default_rng(4300+repeat*17+fold).permutation(len(held))
                    changed[:,indices]=changed[donor][:,indices]
                    permuted[held]=models[fold].predict(changed)
                results.append({"noise":noise,"group":group,"repeat":repeat,"baseline_rmse":float(np.sqrt(np.mean((baseline-y)**2))),
                    "rmse_increase":float(np.sqrt(np.mean((permuted-y)**2))-np.sqrt(np.mean((baseline-y)**2))),
                    "mse_increase":float(np.mean((permuted-y)**2)-np.mean((baseline-y)**2)),
                    "fn_change":int(((y>=t)&(permuted<t)).sum()-((y>=t)&(baseline<t)).sum()),
                    "fp_change":int(((y<t)&(permuted>=t)).sum()-((y<t)&(baseline>=t)).sum())})
    save_csv("hgb_group_permutation.csv",results)
    summary=[]
    for noise in ["clean","-20"]:
        for group in feature_groups:
            cells=[r for r in results if r["noise"]==noise and r["group"]==group]
            summary.append({"noise":noise,"group":group,"rmse_increase_mean":float(np.mean([r["rmse_increase"] for r in cells])),
                "rmse_increase_std":float(np.std([r["rmse_increase"] for r in cells])),
                "mse_increase_mean":float(np.mean([r["mse_increase"] for r in cells])),
                "fn_change_mean":float(np.mean([r["fn_change"] for r in cells])),"fp_change_mean":float(np.mean([r["fp_change"] for r in cells]))})
    save_csv("hgb_group_permutation_summary.csv",summary)
    save_json("hgb_group_permutation_definition.json",{"groups":{group:[names[i] for i in idx] for group,idx in feature_groups.items()},
        "evaluation":"Corresponding held chunk fold models, clean and clean-fitted minus20 OOF; prior nested-selected HGB values 31/15/15",
        "repeats":10,"overlapping_groups_not_additive":True,"intervention_may_break_feature_relationships":True,"physical_causality":False})
    print("[XAI] held-fold feature-group permutation complete",flush=True)


def select_representatives():
    lookup={}
    wanted={"baseline_performance","frequency34_HGB_rmse","frequency34_HGB_rmse__mse_simplex",*p.KEYS}
    with (PRIOR/"outer_predictions.csv").open(encoding="utf-8-sig",newline="") as stream:
        for r in csv.DictReader(stream):
            if r["noise"] in ["clean","-4","-20"] and r["method"] in wanted:
                lookup.setdefault((r["noise"],p.identity(r)),{})[r["method"]]=r
    result=[]
    for scope,noise in [("clean_ONB_and_above","clean"),("minus4_below60","-4"),("minus20_below60","-20"),("minus20_near_ONB","-20")]:
        for day in p.THRESHOLDS:
            options=[]
            for (condition,idx),methods in lookup.items():
                if condition!=noise or not idx[0].startswith(day):continue
                r=methods["baseline_performance"];y=float(r["y_kW_m2"]);t=float(r["onb_kW_m2"])
                eligible= y>=t if scope=="clean_ONB_and_above" else (y<60 if "below60" in scope else abs(y-t)<=.1*t)
                if not eligible:continue
                baseline=float(r["prediction_kW_m2"]);enhanced=float(methods["frequency34_HGB_rmse__mse_simplex"]["prediction_kW_m2"])
                gain=(baseline-y)**2-(enhanced-y)**2
                options.append((gain,idx,methods))
            options.sort(key=lambda x:(x[0],x[1]))
            chosen=[("minimum_gain",r) for r in options[:2]]+[("maximum_gain",r) for r in options[-2:]]
            for rank,(gain,idx,methods) in chosen:
                selection=("largest_deterioration" if gain<0 else "smallest_improvement") if rank=="minimum_gain" else ("largest_improvement" if gain>=0 else "smallest_deterioration")
                r=methods["baseline_performance"]
                result.append({"scope":scope,"selection":selection,"gain_rank":rank,"noise":noise,"source_wav_id":idx[0],"chunk_index":idx[1],
                    "y_kW_m2":float(r["y_kW_m2"]),"onb_kW_m2":float(r["onb_kW_m2"]),"squared_error_reduction":gain,
                    **{key:float(methods[key]["prediction_kW_m2"]) for key in p.KEYS},"hgb":float(methods["frequency34_HGB_rmse"]["prediction_kW_m2"])})
    assert len(result)==32
    save_csv("representative_chunks.csv",result)
    return result


def common_interventions(device="gpu"):
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(2)
    if device=="gpu":
        if not tf.config.list_physical_devices("GPU"):raise RuntimeError("Use the recorded GPU path for strict endpoint matching; CPU feature permutation remains available separately.")
        for gpu in tf.config.list_physical_devices("GPU"):tf.config.experimental.set_memory_growth(gpu,True)
        tf.keras.utils.set_random_seed(43);tf.config.experimental.enable_op_determinism()
    from utils.models.regression.base_regression import RegressionModelMaker
    from utils.training.model_training import ModelTrainer
    train_rows,folds,data_root,outer,base,y,days=p.setup()
    selected=select_representatives()
    metadata={noise:{p.identity(r):r for r in p.read_csv(data_root/p.NOISE_DIRS[noise]/"chunk_manifest.csv")} for noise in ["clean","-4","-20"]}
    clean_spectra=[]
    for row in train_rows:
        raw=np.load(p.lp(data_root/p.NOISE_DIRS["clean"]/metadata["clean"][p.identity(row)]["sample_filename"]))
        clean_spectra.append(raw.mean(axis=0))
    baseline_spectrum=np.median(np.asarray(clean_spectra),axis=0).astype(np.float32)
    np.save(OUT/"training_median_spectrum.npy",baseline_spectrum)
    cases=[];original=[]
    for row in selected:
        raw=np.load(p.lp(data_root/p.NOISE_DIRS[row["noise"]]/metadata[row["noise"]][p.identity(row)]["sample_filename"])).astype(np.float32)
        cases.append(row);original.append(raw)
    original=np.asarray(original)[...,None]
    masks={"unchanged":original}
    bands=[(0,500),(500,1000),(1000,1500),(1500,2000),(2000,2500),(2500,3000),(1700,2100),(2100,2500),(2500,2900)]
    for low,high in bands:
        changed=original.copy();a,b=round(224*low/3000),round(224*high/3000)
        changed[:,:,a:b,0]=baseline_spectrum[None,None,a:b]
        masks[f"freq_{low}_{high}_training_median"]=changed
    # Median replacement can inject power into a quiet sample. A fixed attenuation
    # contrast makes that reference effect distinguishable from band dependence.
    for low,high in [(1700,2100),(2100,2500),(2500,2900)]:
        changed=original.copy();a,b=round(224*low/3000),round(224*high/3000)
        changed[:,:,a:b,0]*=.1
        masks[f"freq_{low}_{high}_power_x0.1"]=changed
    masks["global_power_x0.1"]=original*.1
    masks["temporal_shuffle"]=original[:,np.random.default_rng(4306).permutation(224)]
    masks["temporal_average"]=np.repeat(original.mean(axis=1,keepdims=True),224,axis=1)
    artifact_root=p.RUN/"maxfreq=3kHz/heatflux_no_noise/fitted_state/fold1"
    artifact_manifest=p.read_json(artifact_root/"artifact_manifest.json")
    scaler=joblib.load(p.lp(artifact_root/"target_scaler.joblib"));pca=joblib.load(p.lp(artifact_root/"pca.joblib"))
    trainer=ModelTrainer(42);transformer=AcousticFrequency34()
    hgb_entry=next(r for r in p.candidates() if r["method"]=="frequency34_HGB_rmse")
    dependency=[];audits=[];numeric={}
    # Preserve the original RF PCA GEMM batch geometry. Six-decimal quantization
    # can still cross a tree split when changing the float32 projection batch size.
    rf_context={}
    outer_positions={p.identity(r):i for i,r in enumerate(outer)}
    for noise in ["clean","-4","-20"]:
        rf_context[noise]=np.asarray([np.load(p.lp(data_root/p.NOISE_DIRS[noise]/metadata[noise][p.identity(r)]["sample_filename"])) for r in outer],dtype=np.float32)[...,None]
    for key in [*p.KEYS,"hgb"]:
        if key=="randomforest":
            model=joblib.load(p.lp(artifact_root/"randomforest.joblib"))
            def predict(raw):
                result=np.empty(len(cases))
                # The original MKL setting was 20. Diagnostic 1/2/8/16 thread
                # projections can cross the six-decimal boundary in PCA87/98.
                with threadpool_limits(limits=20,user_api="blas"):
                    for noise,context in rf_context.items():
                        positions=[i for i,row in enumerate(cases) if row["noise"]==noise]
                        changed=context.copy()
                        for i in positions:changed[outer_positions[p.identity(cases[i])]]=raw[i]
                        values=scaler.inverse_transform(model.predict(trainer.transform_pca(pca,changed)).reshape(-1,1)).ravel()/1000
                        for i in positions:result[i]=values[outer_positions[p.identity(cases[i])]]
                return result
        elif key=="hgb":
            model=joblib.load(PRIOR/hgb_entry["final_artifact"])
            def predict(raw):return model.predict(transformer.transform(raw))
        else:
            maker=RegressionModelMaker((224,224,1));model=maker.cnn_transformer_v2() if key=="conformer" else maker.alexnet()
            model.load_weights(p.lp(artifact_root/f"{key}.weights.h5"))
            def predict(raw):return scaler.inverse_transform(model.predict(raw,batch_size=32,verbose=0)).ravel()/1000
        numeric[key]={}
        for intervention,raw in masks.items():numeric[key][intervention]=predict(raw)
        expected=np.asarray([row[key] for row in cases]);restored=numeric[key]["unchanged"]
        np.testing.assert_allclose(restored,expected,rtol=1e-5,atol=.002)
        if key=="hgb":
            np.testing.assert_allclose(numeric[key]["temporal_shuffle"],restored,rtol=1e-10,atol=1e-8)
        for intervention,values in numeric[key].items():
            for i,row in enumerate(cases):
                dependency.append({"case_index":i,"scope":row["scope"],"selection":row["selection"],"noise":row["noise"],"source_wav_id":row["source_wav_id"],"chunk_index":row["chunk_index"],
                    "model":key,"intervention":intervention,"y_kW_m2":row["y_kW_m2"],"onb_kW_m2":row["onb_kW_m2"],
                    "original_prediction":float(restored[i]),"changed_prediction":float(values[i]),"prediction_change":float(values[i]-restored[i]),
                    "absolute_error_change":float(abs(values[i]-row["y_kW_m2"])-abs(restored[i]-row["y_kW_m2"])),
                    "positive_before":bool(restored[i]>=row["onb_kW_m2"]),"positive_after":bool(values[i]>=row["onb_kW_m2"])})
        audits.append({"model":key,"baseline_reload_max_abs_kW_m2":float(np.max(np.abs(restored-expected))),"saved_model_used":True,"prediction_device":device,"number_of_cases":len(cases),"interventions":list(masks),"rf_projection_context":"original540, MKL20 threads" if key=="randomforest" else None})
        print(f"[XAI] common interventions {key}: {len(cases)} cases x {len(masks)}",flush=True)
        del model;tf.keras.backend.clear_session()
    save_csv("common_input_dependency.csv",dependency)
    aggregate=[]
    for scope in sorted({row["scope"] for row in cases}):
        for key in numeric:
            for intervention in masks:
                if intervention=="unchanged":continue
                values=[r for r in dependency if r["scope"]==scope and r["model"]==key and r["intervention"]==intervention]
                aggregate.append({"scope":scope,"model":key,"intervention":intervention,"n_selected":len(values),
                    "mean_prediction_change":float(np.mean([r["prediction_change"] for r in values])),"mean_abs_prediction_change":float(np.mean([abs(r["prediction_change"]) for r in values])),
                    "mean_absolute_error_change":float(np.mean([r["absolute_error_change"] for r in values])),"positive_to_negative":sum(r["positive_before"] and not r["positive_after"] for r in values),
                    "negative_to_positive":sum(not r["positive_before"] and r["positive_after"] for r in values)})
    save_csv("common_input_dependency_summary.csv",aggregate)
    save_json("xai_audit.json",{"baseline_models":str(artifact_root.relative_to(ROOT)),"models":audits,"reference_spectrum_fit_scope":"seed42 training 1620 clean inputs only",
        "representative_selection":"32 post hoc extremes in four prespecified scopes; not representative population sampling","frequency_coordinates":"approximate post-resize coordinates",
        "physical_causal_claim":False,"models_retrained":False,"hgb_temporal_shuffle_invariance_verified":True,
        "rf_small_batch_limitation":"One of 32 selected baseline predictions differed by 52.0069 kW/m2 under two BLAS threads. Original540 and MKL20 restore saved predictions. Thread counts1/2/8/16 also changed a few PCA87/98 quantizations in full540. Main PCA version2 remains unchanged; general thread/batch independence is not established."})
    # Display one improved and one deteriorated case per scope in comparable spectra.
    chosen=[next(i for i,row in enumerate(cases) if row["scope"]==scope and row["gain_rank"]==rank) for scope in sorted({r["scope"] for r in cases}) for rank in ["maximum_gain","minimum_gain"]]
    fig,axes=plt.subplots(4,2,figsize=(10,11))
    lo=float(np.quantile(np.log10(np.maximum(original,1e-20)),.02));hi=float(np.quantile(np.log10(np.maximum(original,1e-20)),.98))
    for ax,i in zip(axes.flat,chosen):
        row=cases[i];ax.imshow(np.log10(np.maximum(original[i,:,:,0],1e-20)).T,origin="lower",aspect="auto",extent=[0,1,0,3],vmin=lo,vmax=hi,cmap="magma")
        ax.set_title(f"{row['scope']}\n{row['source_wav_id'][:8]} chunk {row['chunk_index']}, q={row['y_kW_m2']:.1f}",fontsize=9)
        ax.set_xlabel("Within-chunk time (s)");ax.set_ylabel("Frequency (kHz)")
    fig.tight_layout();fig.savefig(OUT/"representative_spectrograms.png",dpi=160);fig.savefig(OUT/"representative_spectrograms.pdf");plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--phase",choices=["all","permutation","interventions"],default="all");parser.add_argument("--device",choices=["cpu","gpu"],default="gpu");args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=2):
        if args.phase in ["all","permutation"]:permutation_dependence()
        if args.phase in ["all","interventions"]:common_interventions(args.device)
