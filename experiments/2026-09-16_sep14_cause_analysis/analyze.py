"""Read only the six September 14 runs; no training or later-run discovery."""
from pathlib import Path
import hashlib
import json
import itertools
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SNAP = ROOT / 'experiments/2026-09-15_research_status_snapshot'
MODELS = ['rf', 'cnntf_v2_gap', 'alexnet']
ENSEMBLES = ['ensemble__simple_equal', 'ensemble__inner_holdout']
sources = {}

def read(path):
    path = Path(path)
    if not str(path).startswith('\\\\?\\'):
        path = Path('\\\\?\\' + str(path.absolute()))
    data = path.read_bytes()
    sources[str(path)] = hashlib.sha256(data).hexdigest()
    return pd.read_csv(path)

def save(name, rows):
    pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding='utf-8-sig')

inv = read(SNAP / 'run_inventory.csv')
inv = inv[inv.run.eq('20260914_selected_log_architecture')]
assert len(inv) == 6 and inv.experiment.eq('2025.06.11_0.3_2').all()
metrics, residuals, pairs, decompositions, weights, wav_rows, folds, subset = [], [], [], [], [], [], [], []
reference = None
for run in inv.itertuples():
    p = Path('\\\\?\\' + str((ROOT / run.run_manifest).absolute())).parent
    manifest_path = p / 'run_manifest.json'
    sources[str(manifest_path)] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['validation_config']['data']['experiment_names'] == ['2025.06.11_0.3_2']
    c = {'frequency': run.maxfreq, 'snr': run.snr}
    chunks = pd.concat([read(f) for f in sorted((p/'fold_pred').glob('*.csv'))], ignore_index=True)
    chunks = chunks.sort_values(['source_wav_id', 'chunk_index']).reset_index(drop=True)
    keys = chunks[['source_wav_id','chunk_index','fold','y_true']]
    if reference is None:
        reference = keys
    else:
        pd.testing.assert_frame_equal(keys, reference)
    assert chunks.source_wav_id.nunique() == 18 and len(chunks) == 1080
    w = read(p / f'ensemble_weights_{run.snr}.csv')
    weights.extend([{**c, **r} for r in w.to_dict('records')])
    saved = read(p / 'wav_eval' / f'wav_predictions_{run.snr}.csv').set_index('source_wav_id')
    wav = chunks.groupby('source_wav_id')[['y_true', *MODELS, *ENSEMBLES]].median()
    wav['fold'] = chunks.groupby('source_wav_id')['fold'].first()
    for model in MODELS + ENSEMBLES:
        assert np.allclose(wav[model], saved.loc[wav.index, f'{model}_pred_median'])
    for unit, df in [('chunk', chunks), ('wav_median', wav)]:
        y = df.y_true.to_numpy()
        err = df[MODELS].to_numpy() - y[:,None]
        variance = np.mean((y-y.mean())**2)
        for model in MODELS + ENSEMBLES:
            e = df[model].to_numpy()-y
            metrics.append({**c, 'unit':unit, 'model':model, 'r2':1-np.mean(e**2)/variance,
                            'rmse_kW':np.sqrt(np.mean(e**2))/1000, 'bias_kW':e.mean()/1000,
                            'slope_prediction_vs_truth':np.polyfit(y,df[model],1)[0]})
        for a,b in itertools.combinations(range(3),2):
            pairs.append({**c,'unit':unit,'left':MODELS[a],'right':MODELS[b],
                          'residual_pearson':np.corrcoef(err[:,a],err[:,b])[0,1],
                          'same_sign_fraction':np.mean(err[:,a]*err[:,b]>0)})
        for ens in ENSEMBLES:
            # Exact change relative to CNN, using the actual aggregated prediction.
            e = df.cnntf_v2_gap.to_numpy()-y
            delta = df[ens].to_numpy()-df.cnntf_v2_gap.to_numpy()
            cross = 2*np.mean(e*delta)/variance
            penalty = np.mean(delta**2)/variance
            loss = np.mean(((df[ens]-y)**2-e**2))/variance
            assert abs(cross+penalty-loss)<1e-12
            decompositions.append({**c,'unit':unit,'ensemble':ens,
                                   'cross_term_div_var':cross,'shift_square_div_var':penalty,
                                   'r2_loss_vs_cnn':loss})
        if unit == 'chunk':
            avg_mse=np.mean(err**2); diversity=np.mean(np.var(df[MODELS].to_numpy(),axis=1))
            assert np.allclose(df['ensemble__simple_equal'],df[MODELS].mean(axis=1))
            assert abs((avg_mse-diversity)-np.mean((df['ensemble__simple_equal']-y)**2))<1e-3
        for f,g in df.groupby('fold'):
            for model in MODELS+ENSEMBLES:
                folds.append({**c,'unit':unit,'fold':f,'model':model,
                              'r2':1-np.mean((g[model]-g.y_true)**2)/np.var(g.y_true)})
    for ids in itertools.combinations(MODELS,2):
        mixed = chunks.assign(mixed=chunks[list(ids)].mean(axis=1)).groupby('source_wav_id').mixed.median()
        subset.append({**c,'models':'+'.join(ids),'posthoc_wav_r2':1-np.mean((mixed-wav.y_true)**2)/np.var(wav.y_true)})
    for idx,row in wav.iterrows():
        rec={**c,'source_wav_id':idx,**row.to_dict()}
        for model in MODELS+ENSEMBLES:
            rec[model+'_sq_error']=(row[model]-row.y_true)**2
        wav_rows.append(rec)

for name,rows in [('metrics.csv',metrics),('residual_pairs.csv',pairs),('ensemble_decomposition.csv',decompositions),
                  ('weights.csv',weights),('wav_predictions.csv',wav_rows),('fold_metrics.csv',folds),('posthoc_pair_diagnostics.csv',subset)]:
    save(name,rows)

mask=read(SNAP/'group_mask_performance.csv')
groups=['maxfreq','snr','model','axis','group']
means=mask.groupby(groups,as_index=False).agg(mean_r2_drop=('r2_drop','mean'),min_r2_drop=('r2_drop','min'),max_r2_drop=('r2_drop','max'))
means.to_csv(OUT/'mask_means.csv',index=False,encoding='utf-8-sig')
top=mask.loc[mask.groupby(['maxfreq','snr','model','fold','axis']).r2_drop.idxmax()]
top.to_csv(OUT/'mask_top_per_fold.csv',index=False,encoding='utf-8-sig')
ig=read(SNAP/'ig_diagnostics.csv')
ig.groupby(['maxfreq','model']).completeness_relative_error.agg(['count','median','min','max']).to_csv(OUT/'ig_quality.csv',encoding='utf-8-sig')
(OUT/'verification.json').write_text(json.dumps({'scope':'September 14 only; six conditions; no training',
    'checks':['same 1080 chunk identities, labels and folds in all six conditions',
              '18 source WAVs; all five WAV median predictions reproduced',
              'equal-weight predictions match chunk arithmetic means',
              'exact ensemble-vs-CNN squared error decomposition verified'],
    'ig_relative_error_over_005':int((ig.completeness_relative_error>0.05).sum()),
    'ig_count':len(ig),'sources_sha256':sources},ensure_ascii=False,indent=2),encoding='utf-8')
print(pd.DataFrame(metrics).query("unit=='wav_median'").pivot(index=['frequency','snr'],columns='model',values='r2').round(5).to_string())
print('Decomposition',pd.DataFrame(decompositions).query("unit=='wav_median'").round(5).to_string(index=False),sep='\n')
print('Top masks',means.loc[means.groupby(['maxfreq','snr','model','axis']).mean_r2_drop.idxmax()].round(4).to_string(index=False),sep='\n')

# Diagnostics retain every WAV. Excluding an endpoint below is sensitivity
# analysis, not a replacement score or a recommendation to discard data.
wav_df = pd.DataFrame(wav_rows)
recovery, sensitivity, ranges = [], [], []
for freq, g in wav_df.groupby('frequency'):
    a=g[g.snr.eq('0')].set_index('source_wav_id')
    b=g[g.snr.eq('-20')].set_index('source_wav_id')
    for model in MODELS+ENSEMBLES:
        gain=a[model+'_sq_error']-b[model+'_sq_error']
        for idx, value in gain.items():
            recovery.append({'frequency':freq,'model':model,'source_wav_id':idx,
                             'y_true':a.loc[idx,'y_true'],'sse_reduction_0_to_minus20':value})
        for selected in [a.index, a.index[a.y_true>1]]:
            sensitivity.append({'frequency':freq,'model':model,'n_wavs':len(selected),
                                'mse_change_minus20_minus0':-gain.loc[selected].mean()})
    base=g[g.snr.eq('no_noise')]
    for fold in [1,2,3]:
        tr=base[base.fold.ne(fold)]
        for row in base[base.fold.eq(fold)].itertuples():
            ranges.append({'frequency':freq,'fold':fold,'source_wav_id':row.source_wav_id,
                           'y_true':row.y_true,'training_min':tr.y_true.min(),'training_max':tr.y_true.max(),
                           'outside_training_label_range':not tr.y_true.min()<=row.y_true<=tr.y_true.max()})
save('recovery_by_wav.csv',recovery)
save('endpoint_sensitivity.csv',sensitivity)
save('outer_training_ranges.csv',ranges)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
colors=['#777777','#0072B2','#D55E00','#009E73','#CC79A7']
labels=['RF','CNN + Transformer','AlexNet','Equal ensemble','Inner holdout']
fig,axes=plt.subplots(2,2,figsize=(12,8.5))
md=pd.DataFrame(metrics).query("unit=='wav_median'")
for ax,freq in zip(axes[0],['maxfreq=3kHz','maxfreq=22kHz']):
    for model,color,label in zip(MODELS+ENSEMBLES,colors,labels):
        z=md[(md.frequency==freq)&(md.model==model)].set_index('snr').loc[['no_noise','0','-20']]
        ax.plot(range(3),z.r2,'o-',color=color,label=label)
    ax.set(xticks=range(3),xticklabels=['Clean','0','-20'],ylim=(.88,.975),
           title=freq.replace('maxfreq=','')+' | September 14 only',ylabel='Pooled WAV median R2',xlabel='Reference SNR (categorical)')
    ax.grid(alpha=.2)
axes[0,0].legend(fontsize=8,loc='lower left')
g=wav_df.query("frequency=='maxfreq=22kHz' and snr=='0'").sort_values('y_true')
for model,color,label in zip(MODELS+['ensemble__inner_holdout'],colors[:3]+[colors[4]],labels[:3]+[labels[4]]):
    axes[1,0].plot(g.y_true/1000,(g[model]-g.y_true)/1000,'o-',color=color,label=label)
axes[1,0].axhline(0,color='black',lw=.8)
axes[1,0].set(title='22 kHz / 0: shared residual structure',xlabel='True heat flux (kW/m2)',ylabel='Prediction minus truth (kW/m2)')
axes[1,0].grid(alpha=.2)
rc=pd.DataFrame(recovery).query("frequency=='maxfreq=22kHz' and model=='ensemble__inner_holdout'").sort_values('y_true')
axes[1,1].bar(range(len(rc)),rc.sse_reduction_0_to_minus20/1e9,color=np.where(rc.sse_reduction_0_to_minus20>0,'#009E73','#D55E00'))
axes[1,1].axhline(0,color='black',lw=.8)
axes[1,1].set(xticks=range(len(rc)),xticklabels=[f'{x/1000:.0f}' for x in rc.y_true],
              title='22 kHz inner holdout: 0 to -20 error reduction',xlabel='True heat flux (kW/m2)',ylabel='Squared-error reduction (1e9 W2/m4)')
axes[1,1].tick_params(axis='x',labelrotation=60,labelsize=8)
fig.suptitle('Same 18 recordings / 3 folds / 6 conditions; separately trained at each noise level')
fig.tight_layout()
fig.savefig(OUT/'cause_overview.png',dpi=180)
fig.savefig(OUT/'cause_overview.pdf')
plt.close(fig)
