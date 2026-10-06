"""Separate the HGB training RNG from which chunks enter its training set.

This is a follow-up diagnostic, not a hyperparameter or outer-score selection.
"""
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"code"))
import run_hgb_complementarity_validation as run


def main():
    data,metadata,y,splits=run.dataset()
    features={n:np.load(OUT/"cache"/f"features_{n}.npz")["X"] for n in run.NOISES}
    rows=[];parameters={k:v for k,v in run.CONFIG["candidate"].items() if k not in ["family","representation"]}
    predictions={};common_predictions={};reference_indices=splits[42][1]
    for split_seed in [42,43,44]:
        fit,held=splits[split_seed]
        scaler=MinMaxScaler().fit((y[fit]*1000).reshape(-1,1))
        target=scaler.transform((y[fit]*1000).reshape(-1,1)).ravel()
        days=np.asarray([metadata[int(i)]["source_wav_id"][:8] for i in held]);t=np.asarray([run.THRESHOLDS[d] for d in days])
        all_p=[]
        for model_seed in [42,43,44,45]:
            model=HistGradientBoostingRegressor(**parameters,random_state=model_seed)
            model.fit(features["clean"][fit],target)
            for noise in ["clean","-20"]:
                p=scaler.inverse_transform(model.predict(features[noise][held]).reshape(-1,1)).ravel()/1000
                predictions[(split_seed,model_seed,noise)]=p
                if model_seed==42:
                    common_predictions[(split_seed,noise)]=scaler.inverse_transform(model.predict(features[noise][reference_indices]).reshape(-1,1)).ravel()/1000
                rows.append({"outer_split_seed":split_seed,"hgb_random_state":model_seed,"noise":noise,"fit_chunks":len(fit),"test_chunks":len(held),
                    "rmse":float(np.sqrt(np.mean((p-y[held])**2))),"fp":int(((y[held]<t)&(p>=t)).sum()),"fn":int(((y[held]>=t)&(p<t)).sum()),
                    "parameters_fixed":True,"outer_used_for_selection":False})
        print(f"HGB seed separation split{split_seed} complete",flush=True)
    run.save_csv(OUT/"hgb_split_vs_model_seed.csv",rows)
    comparisons=[]
    for split_seed in [42,43,44]:
        for noise in ["clean","-20"]:
            original=predictions[(split_seed,42,noise)]
            comparisons.append({"outer_split_seed":split_seed,"noise":noise,"max_prediction_difference_between_hgb_seeds":max(float(np.max(np.abs(predictions[(split_seed,s,noise)]-original))) for s in [43,44,45])})
    run.save_csv(OUT/"hgb_rng_invariance.csv",comparisons)
    common=[]
    thresholds=np.asarray([run.THRESHOLDS[metadata[int(i)]["source_wav_id"][:8]] for i in reference_indices])
    for split_seed in [42,43,44]:
        for noise in ["clean","-20"]:
            pred=common_predictions[(split_seed,noise)];original=common_predictions[(42,noise)]
            common.append({"training_split_seed":split_seed,"fixed_input_set":"seed42 outer540, diagnostic only; may overlap other seeds' clean training",
                "noise":noise,"mean_abs_prediction_change_vs_seed42":float(np.mean(np.abs(pred-original))),"max_abs_prediction_change_vs_seed42":float(np.max(np.abs(pred-original))),
                "pre_ONB_positive_count":int(((y[reference_indices]<thresholds)&(pred>=thresholds)).sum()),"holdout_claim_for_other_training_splits":False})
    run.save_csv(OUT/"hgb_common_input_diagnostic.csv",common)
    run.save_json(OUT/"hgb_sensitivity_audit.json",{"fit_operations":12,"parameters_fixed":parameters,"factorial_diagnostic":{"outer_splits":[42,43,44],"hgb_random_states":[42,43,44,45]},
        "outer_used_for_selection":False,"timing":"Exploratory diagnostic prompted by seed43 strong-noise false alarms; not independent confirmation","feature_definition":"Same frequency34; all1620 baseline clean feature values checked identical"})


if __name__=="__main__":
    with threadpool_limits(limits=2):main()
