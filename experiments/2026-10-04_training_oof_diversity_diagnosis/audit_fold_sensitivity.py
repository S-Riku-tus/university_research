"""Prepare, but do not train, one controlled alternative internal partition."""

import json
import numpy as np

from analyze import ROOT, OUTPUT, SCOPE, NOISE_DIRS, THRESHOLDS, read_json, save_csv


def main():
    path=ROOT/SCOPE["runs"]["clean_only"]/"maxfreq=3kHz"/NOISE_DIRS["clean"]/"internal_validation_fold1.json"
    audit=read_json(path)
    samples=audit["samples"]
    groups=np.asarray([r["source_wav_id"] for r in samples])
    original=np.zeros(len(samples),dtype=int)
    for f in audit["folds"]:original[f["validation_indices"]]=f["fold"]
    alternatives=original.copy()
    swap={"20250618::index=9.271677":2,"20250618::index=10.322114":1}
    for group,new_fold in swap.items():
        assert np.sum(groups==group)==45
        alternatives[groups==group]=new_fold
    y=np.asarray([r["heat_flux"]/1000 for r in samples])
    t=np.asarray([THRESHOLDS[g.split('::')[0]]/1000 for g in groups])
    records,summary=[],[]
    for fold in [1,2,3]:
        fit,held=np.flatnonzero(alternatives!=fold),np.flatnonzero(alternatives==fold)
        assert len(fit)==1080 and len(held)==540 and set(groups[fit]).isdisjoint(groups[held])
        counts={}
        for name,assignment in [('original',original),('alternative',alternatives)]:
            fit_mask=assignment!=fold
            near=fit_mask & (np.abs(y-t)<=.1*t)
            counts[f'{name}_fit_onb_neighborhood_wavs']=len(set(groups[near]))
        summary.append({'fold':fold,'fit_chunks':len(fit),'held_chunks':len(held),**counts})
        records.append({'fold':fold,'fit_chunks':len(fit),'held_chunks':len(held),
                        'shared_source_wavs':0,'fit_wavs':sorted(set(groups[fit])),'held_wavs':sorted(set(groups[held]))})
    config={"created_date":"2026-10-04","status":"Partition audited; model training not performed; not a main-run override",
        "purpose":"Test sensitivity to both ONB recordings being held out in the same internal fold",
        "source_run":SCOPE['runs']['clean_only'],"fixed_conditions":"3kHz, 1s, no selection, adopted parameters, same outer 1620/540 split",
        "changed_factor":"Swap the held-out folds of 20250618 ONB and its next measured heat-flux stage",
        "swap":swap,"folds":records,"summary":summary,
        "outer_test_used":False,"main_config_modified":False,
        "required_followup":"Retrain existing inner-fold models under this controlled partition before attributing errors to fold layout",
        "limitations":["Availability of an ONB recording in inner-fit does not prove performance will improve", "One split pair does not estimate general partition or seed uncertainty"]}
    dest=ROOT/'configs/experiments/2026-10-04_internal_onb_fold_sensitivity.json'
    dest.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    save_csv('internal_fold_sensitivity_audit.csv',summary)
    print('Alternative internal partition verified; fit ONB-neighborhood WAVs:',summary)


if __name__=='__main__':main()
