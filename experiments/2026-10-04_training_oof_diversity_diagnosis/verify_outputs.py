"""Verify aligned outputs and derive full metrics from already saved predictions."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score

from analyze import OUTPUT, metrics, read_csv, save_csv


def identity(row):
    return row["source_wav_id"],int(row["chunk_index"])


def main():
    diagnostic=read_csv(OUTPUT/'oof_sample_diagnostics.csv')
    clean=[r for r in diagnostic if r['noise']=='clean']
    ref_ids=[identity(r) for r in clean]
    assert len(ref_ids)==len(set(ref_ids))==1620
    features=read_csv(OUTPUT/'training_input_features.csv')
    assert len(features)==3240
    for noise in ['clean','0','-4','-8','-12','-16','-20']:
        rows=[r for r in diagnostic if r['noise']==noise]
        assert [identity(r) for r in rows]==ref_ids
    for noise in ['clean','-20']:
        assert {identity(r) for r in features if r['noise']==noise}==set(ref_ids)
    assert len(read_csv(OUTPUT/'training_wav_features.csv'))==72
    for prefix in ['svr','svr_shape']:
        rows=read_csv(OUTPUT/f'{prefix}_training_oof_predictions.csv')
        assert [identity(r) for r in rows]==ref_ids
        y=np.asarray([float(r['heat_flux_kW_m2']) for r in rows])
        t=np.asarray([float(r['onb_kW_m2']) for r in rows])
        assert np.allclose(y,[float(r['y_kW_m2']) for r in clean])
        scopes={'all':np.ones(len(y),bool),
                **{region:np.asarray([r['region']==region for r in rows]) for region in ['below_60','60_to_ONB','ONB_to_1.5ONB','above_1.5ONB']},
                **{f'day_{day}':np.asarray([r['source_wav_id'].startswith(day) for r in rows]) for day in ['20250611','20250618']}}
        methods={'SVR':np.asarray([float(r['SVR_clean']) for r in rows]),
                 'existing3_performance_oof_diagnostic':np.asarray([float(r['existing3_performance_oof_diagnostic']) for r in rows]),
                 'fixed_4_equal':np.asarray([float(r['fixed_4_equal']) for r in rows]),
                 'fixed_old75_SVR25':np.asarray([float(r['fixed_old75_SVR25']) for r in rows])}
        stress=np.asarray([float(r['SVR_minus20_clean_fitted_transfer']) for r in rows])
        output=[]
        saved=read_csv(OUTPUT/f'{prefix}_training_oof_metrics.csv')
        for condition,predictions in [('clean',methods),('-20_clean_fitted_transfer',{'SVR':stress})]:
            for scope,mask in scopes.items():
                for name,p in predictions.items():
                    assert np.isfinite(p).all()
                    ym,tm,pm=y[mask],t[mask],p[mask]
                    result=metrics(ym,pm,tm)
                    prior=next(r for r in saved if r['condition']==condition and r['scope']==scope and r['method']==name)
                    assert abs(result['rmse_kW_m2']-float(prior['rmse_kW_m2']))<1e-9
                    assert result['fp']==int(prior['fp']) and result['fn']==int(prior['fn'])
                    positive=ym>=tm
                    pred_positive=pm>=tm
                    tp=int((positive&pred_positive).sum()); fp=result['fp']; fn=result['fn']
                    near=np.abs(ym-tm)<=.1*tm
                    score=pm-tm
                    extra={'r2':float(r2_score(ym,pm)),
                           'precision':tp/(tp+fp) if tp+fp else 0.,
                           'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                           'onb_neighborhood_n':int(near.sum()),
                           'onb_neighborhood_rmse':float(np.sqrt(np.mean((pm[near]-ym[near])**2))) if near.any() else '',
                           'roc_auc_margin':float(roc_auc_score(positive,score)) if len(set(positive))==2 else '',
                           'pr_auc_margin':float(average_precision_score(positive,score)) if len(set(positive))==2 else ''}
                    output.append({'condition':condition,'scope':scope,'method':name,**result,**extra})
        save_csv(f'{prefix}_full_metrics.csv',output)
        audit=json.loads((OUTPUT/f'{prefix}_pilot_audit.json').read_text(encoding='utf-8'))
        assert audit['outer_test_used'] is False and audit['noisy_data_used_in_hyperparameter_selection'] is False
        for f in audit['source_wav_disjoint_folds']:
            assert not set(f['fit_wavs'])&set(f['held_wavs'])
            assert (OUTPUT/f['artifact']).is_file() and f['reload_prediction_verified']
    for filename in ['analyze.py','pilot_svr.py','make_figures.py','audit_fold_sensitivity.py','verify_outputs.py']:
        path=OUTPUT/filename
        compile(path.read_text(encoding='utf-8'),str(path),'exec')
    print('Verified: seven aligned 1620-row OOF sets, 3240 training-only feature records, both nested SVR pilots, six reloadable models, metric recomputation, and script syntax.')


if __name__=='__main__':main()
