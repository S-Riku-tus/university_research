"""Read only the two authorized clean-only runs; compare completed outputs."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import copy
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"code"))
from utils.experiment.run_helpers import open_text,windows_long_path

NOISE=["clean","0","-4","-8","-12","-16","-20"]
MODELS=["randomforest","conformer","alexnet","hgb","extra_trees"]
JST=timezone(timedelta(hours=9))


def read_json(path):
    with open_text(path,"r",encoding="utf-8") as h:return json.load(h)


def csv(path):
    with open_text(path,"r",encoding="utf-8-sig") as h:return pd.read_csv(h)


def sha(path):
    with open_text(path,"rb") as h:return hashlib.sha256(h.read()).hexdigest()


def main():
    base=ROOT/"Pool_boiling/Subcooling_20_degrees/0.3/2025.06.11_0.3_2_6.18_0.3_3/regression_result/npy/ensemble/202610/15"
    runs=[next(p for p in base.iterdir() if p.name.endswith(stamp)) for stamp in ["143609","194258"]]
    freqs=[Path(windows_long_path(p/"maxfreq=3kHz")) for p in runs]
    manifests=[read_json(p/"heatflux_no_noise/run_manifest.json") for p in freqs]
    ignored={"run_instance_id","execution_id"}
    configs=[copy.deepcopy(j["validation_config"]) for j in manifests]
    for config in configs:
        for field in ignored:config["output"].pop(field,None)
    comparisons={key:manifests[0].get(key)==manifests[1].get(key)
                 for key in ["run_hash","dataset","model_params","run_specs","learning_context","execution_config_hash","parameter_set_tag"]}
    comparisons["validation_config_excluding_execution_ids"]=configs[0]==configs[1]
    assert all(comparisons.values()),comparisons
    assert manifests[1]["learning_context"]["training_noise"]=="clean_only"
    conditions=[]
    for noise in NOISE:
        name="heatflux_no_noise" if noise=="clean" else "heatflux_reference_SNR="+noise
        rows=[]
        for p in freqs:
            marker=p/name/"completed.json"
            rows.append({"complete":marker.exists(),"completion":read_json(marker) if marker.exists() else None})
        assert rows[0]["complete"]
        comparison={"noise":noise,"first_complete":True,"rerun_complete":rows[1]["complete"],
                    "prediction_comparison":"pending"}
        if rows[1]["complete"]:
            suffix="no_noise" if noise=="clean" else noise
            paths=[p/name/f"fold_pred/pred_f1_{suffix}.csv" for p in freqs]
            a,b=[csv(p) for p in paths]
            keys=["source_wav_id","chunk_index"]
            a,b=[d.sort_values(keys).reset_index(drop=True) for d in [a,b]]
            assert len(a)==len(b)==540 and a[keys].equals(b[keys])
            assert np.array_equal(a.y_true.to_numpy(),b.y_true.to_numpy())
            cols=["y_true"]+[c for c in a if c in MODELS or c.startswith("ensemble__")]
            assert all(c in b for c in cols)
            assert np.isfinite(a[cols].to_numpy(float)).all() and np.isfinite(b[cols].to_numpy(float)).all()
            delta=float(np.max(abs(a[cols].to_numpy(float)-b[cols].to_numpy(float))))
            equal=bool(np.allclose(a[cols],b[cols],rtol=0,atol=1e-6))
            comparison.update(prediction_comparison="identical" if delta==0 else "equivalent_within_tolerance" if equal else "different",
                              max_absolute_prediction_difference_W_m2=delta,
                              first_csv_sha256=sha(paths[0]),rerun_csv_sha256=sha(paths[1]))
        conditions.append(comparison)
    done=sum(r["rerun_complete"] for r in conditions)
    numerical_equal=all(r["prediction_comparison"] in ["identical","equivalent_within_tolerance"] for r in conditions) if done==7 else None
    metric_comparison=None
    if done==7:
        reports=[p/"onb_comparison" for p in freqs]
        if all((p/"verification.json").exists() for p in reports):
            audits=[read_json(p/"verification.json") for p in reports]
            assert all(j["status"]=="passed" and j["epochs_complete"] and j["same_evaluation_chunks_and_targets"] for j in audits)
            a,b=[csv(p/"main_metrics.csv") for p in reports]
            keys=["seed","fold","noise","model_key","source_day"]
            a,b=[d.sort_values(keys).reset_index(drop=True) for d in [a,b]]
            assert len(a)==len(b)==231 and a[keys].equals(b[keys])
            numeric=a.select_dtypes(include=[np.number]).columns.tolist()
            aa,bb=a[numeric].to_numpy(float),b[numeric].to_numpy(float)
            assert np.array_equal(np.isnan(aa),np.isnan(bb))
            max_difference=float(np.nanmax(abs(aa-bb)))
            assert np.allclose(aa,bb,rtol=0,atol=1e-9,equal_nan=True)
            for col in ["fn","fp","tp","tn","q100_kW_m2"]:
                assert np.array_equal(a[col],b[col],equal_nan=True)
            first_weights=csv(reports[0]/"ensemble_weights.csv")
            second_weights=csv(reports[1]/"ensemble_weights.csv")
            weights_identical=first_weights.equals(second_weights)
            weight_keys=["seed","fold","strategy_name","model_key","fit_scope"]
            assert first_weights[weight_keys].equals(second_weights[weight_keys])
            max_weight_difference=float(abs(first_weights.weight-second_weights.weight).max())
            assert np.allclose(first_weights.weight,second_weights.weight,rtol=0,atol=1e-12)
            metric_comparison={"rows":231,"numeric_fields":len(numeric),"numeric_cells_compared":int(aa.size),
                               "max_absolute_difference":max_difference,"equal_within_1e_minus9":True,
                               "fn_fp_tp_tn_q100_exactly_equal":True,"ensemble_weights_tables_identical":weights_identical,
                               "max_ensemble_weight_difference":max_weight_difference,"ensemble_weights_equivalent_within_1e_minus12":True,
                               "both_completion_audits_passed":True}
    record={"checked_at_jst":datetime.now(JST).isoformat(),
            "first_run":str(runs[0].relative_to(ROOT)),"rerun":str(runs[1].relative_to(ROOT)),
            "run_hash":manifests[0]["run_hash"],"scientific_conditions_equal":comparisons,
            "allowed_config_difference":"output.execution_id and output.run_instance_id only",
            "rerun_completed_conditions":done,"planned_conditions":7,"conditions":conditions,
            "all_results_identical":all(r["prediction_comparison"]=="identical" for r in conditions) if done==7 else None,
            "all_results_numerically_equal":numerical_equal,
            "prediction_absolute_tolerance_W_m2":1e-6,"main_metric_comparison":metric_comparison,
            "canonical_presentation_result":"143609 remains the documented source; 194258 figures can be used after equivalent metrics verification" if numerical_equal and metric_comparison else "143609 completed clean_only until complete rerun comparison",
            "matched_read":False,"new_training":0,"new_inference":0}
    (OUT/"run_consistency.json").write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:record[k] for k in ["checked_at_jst","rerun_completed_conditions","all_results_identical","all_results_numerically_equal","main_metric_comparison","matched_read"]}))
    print(json.dumps(comparisons))


if __name__=="__main__":main()
