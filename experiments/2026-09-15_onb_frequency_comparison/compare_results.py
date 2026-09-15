"""Read frozen 3/22 kHz results; verify, compare, and plot without training.

The original September 15 snapshots are preserved. Use a new --output directory.
"""
import argparse
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, roc_auc_score
from analyze_3khz import ROOT, MODELS, LABELS, NOISES, longpath, relative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit('Use a new empty output directory.')
    snapshots = {
        '3': Path(__file__).parent/'snapshot_3khz',
        '22': ROOT/'experiments/2026-09-15_onb_22khz_noise_sweep/snapshot_20260915',
    }
    hashes, counts, frames = {}, {}, {}

    def read(p):
        data = p.read_bytes()
        hashes[relative(p)] = hashlib.sha256(data).hexdigest()
        return data

    def csv(p):
        return pd.read_csv(io.BytesIO(read(p)), keep_default_na=False, dtype={'snr': str})

    def save(df, name):
        df.to_csv(out/f'{name}.csv', index=False, encoding='utf-8-sig')

    def figsave(fig, name):
        for ext in ('png', 'pdf'):
            fig.savefig(out/f'{name}.{ext}', dpi=170, bbox_inches='tight')
        plt.close(fig)

    split_reference = None
    for freq, snap in snapshots.items():
        frames[freq] = {n:csv(snap/f'{n}.csv') for n in [
            'wav_median_metrics', 'wav_predictions', 'model_comparison', 'mask_aggregates',
            'top_mask_bands', 'ig_diagnostics', 'treeshap_pca_summary', 'input_stability',
            'top_layer_randomization_sanity', 'run_inventory', 'explainability_summary',
            'sample_correspondence', 'group_mask_performance', 'residual_correlations']}
        f = frames[freq]
        counts[freq] = {'performance':0, 'mask_baselines':0, 'ig_array_sums':0}
        for _, inv in f['run_inventory'].iterrows():
            d = longpath(ROOT/inv.run_manifest).parent
            split = json.loads(read(d/'split_manifest.json'))['folds']
            identity = [(s['fold'],sorted(s['training_wav_groups']), sorted(s['evaluation_wav_groups'])) for s in split]
            if split_reference is None:
                split_reference = identity
            assert identity == split_reference
            for fold in range(1,4):
                p = csv(d/f'fold_pred/pred_f{fold}_{inv.snr}.csv')
                g = p.groupby('source_wav_id')[['y_true',*MODELS]].median()
                for key in MODELS[:3]:
                    mask = f['group_mask_performance']
                    base = mask[(mask.snr==inv.snr)&(mask.fold==fold)&(mask.model==key)].base_r2
                    assert np.allclose(base,r2_score(g.y_true,g[key]),rtol=0,atol=1e-7)
                    counts[freq]['mask_baselines'] += 1
            g = f['wav_predictions'].query('snr == @inv.snr')
            y = g.y_true.to_numpy()
            for key in MODELS:
                row = f['wav_median_metrics'].query('snr == @inv.snr and model_key == @key').iloc[0]
                p = g[key+'_pred_median'].to_numpy()
                assert np.isclose(row.r2,r2_score(y,p))
                assert np.isclose(row.auc_binary,roc_auc_score(y>=368978.105,p>=368978.105))
                counts[freq]['performance'] += 1
        ex = f['explainability_summary'].query('method == "integrated_gradients"')
        ex = ex.merge(f['sample_correspondence'][['snr','fold','model','sample_id','sample_dir']],
                      on=['snr','fold','model','sample_id'],validate='one_to_one')
        for _, row in ex.iterrows():
            a = np.load(io.BytesIO(read(longpath(ROOT/row.sample_dir)/'integrated_gradients_signed.npy')))
            assert np.isclose(float(a.sum()),float(row.attribution_sum),rtol=2e-6,atol=.1)
            counts[freq]['ig_array_sums'] += 1

    metrics = pd.concat([f['wav_median_metrics'].assign(maxfreq_khz=freq) for freq,f in frames.items()])
    save(metrics,'wav_metrics_both_frequencies')
    degradation, xai, errors, contributions = [], [], [], []
    for freq,f in frames.items():
        for key in MODELS:
            g = f['wav_median_metrics'].query('model_key == @key').set_index('snr').loc[NOISES]
            clean, end = g.iloc[0],g.iloc[-1]
            degradation.append(dict(maxfreq_khz=freq, model=key, clean_r2=clean.r2, noise20_r2=end.r2,
                r2_drop_clean_to_noise20=clean.r2-end.r2,
                r2_relative_drop_percent=100*(clean.r2-end.r2)/clean.r2,
                rmse_clean=clean.rmse_all,rmse_noise20=end.rmse_all,
                rmse_increase=end.rmse_all-clean.rmse_all,
                r2_rebounds_across_six_steps=int((np.diff(g.r2)>0).sum())))
        for key in MODELS[:3]:
            top = f['top_mask_bands'].query('model == @key').group.value_counts().to_dict()
            if key=='rf':
                v=f['treeshap_pca_summary'].completeness_error.abs()
                xai.append(dict(maxfreq_khz=freq,model=key,top_bands=json.dumps(top),n=len(v),
                    shap_median_absolute_error=v.median(),shap_max_absolute_error=v.max()))
            else:
                ig=f['ig_diagnostics'].query('model == @key')
                st=f['input_stability'].query('model == @key')
                rand=f['top_layer_randomization_sanity'].query('model == @key')
                xai.append(dict(maxfreq_khz=freq,model=key,top_bands=json.dumps(top),n=ig.n.sum(),
                    ig_failed=ig.n_relative_error_gt_005.sum(),ig_min_condition_median=ig.median_relative_error.min(),
                    ig_max_condition_median=ig.median_relative_error.max(),
                    stability_median=st.pearson_abs_map.median(),randomization_median=rand.pearson_abs_map.median()))
        for snr in NOISES:
            g=f['wav_predictions'].query('snr == @snr').sort_values('source_wav_id')
            e=np.column_stack([g[key+'_pred_median']-g.y_true for key in MODELS[:3]])
            avg=e.mean(axis=1)
            # Exact squared-error identity for the mean of WAV-median predictions.
            # The production ensemble takes the median after combining chunks, so report both.
            mse_base=np.mean(e**2)
            diversity=np.mean((e-avg[:,None])**2)
            mse_mean=np.mean(avg**2)
            assert np.isclose(mse_mean,mse_base-diversity)
            actual=g['ensemble__simple_equal_pred_median']-g.y_true
            errors.append(dict(maxfreq_khz=freq,snr=snr,mean_single_mse=mse_base,
                prediction_diversity=diversity,mse_of_mean_wav_medians=mse_mean,
                production_equal_mse=np.mean(actual**2),
                max_median_order_difference=np.max(np.abs(actual-avg))))
            for key in MODELS:
                e1=g[key+'_pred_median']-g.y_true
                for (_,r),err in zip(g.iterrows(),e1):
                    contributions.append(dict(maxfreq_khz=freq,snr=snr,model=key,
                        source_wav_id=r.source_wav_id,y_true=r.y_true,error=err,
                        fraction_squared_error=err**2/np.sum(e1**2)))
    save(pd.DataFrame(degradation),'degradation_summary')
    save(pd.DataFrame(xai),'xai_summary')
    save(pd.DataFrame(errors),'ensemble_error_decomposition')
    save(pd.DataFrame(contributions),'wav_error_contributions')

    # Thesis Table 4.2.1 and 4.2.2: parse means and SE directly from the extracted PDF.
    text=read(Path(__file__).parent/'thesis_extracted_text.txt').decode('utf-8').replace('\r\n','\n')
    page=text.split('## PDF page 25\n')[1].split('## PDF page 26')[0]
    parts=page.split('Table 4.2.1 Comparison')[1].split('Table 4.2.2 Comparison')
    thesis=[]
    for kind,part in zip(['r2','auc_binary'],parts):
        vals=re.findall(r'(0\.\d+)\s*±\s*(0\.\d+)',part)
        assert len(vals)==28
        for i,(mean,se) in enumerate(vals):
            thesis.append(dict(metric=kind,snr=NOISES[i//4],model=['AlexNet','ResNet50','VGG16','Ensemble'][i%4],
                               mean=float(mean),se=float(se),source='Table 4.2.1/4.2.2; PDF page 25, printed page 21'))
    th=pd.DataFrame(thesis)
    save(th,'thesis_table_values')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    xx=np.arange(7); ticks=['Clean','0','-4','-8','-12','-16','-20']
    fig,axs=plt.subplots(3,2,figsize=(13,12),sharex=True)
    for row,(metric,title) in enumerate([('r2','R2'),('auc_binary','AUC after ONB binarization'),('roc_auc_cont','Continuous-score ROC-AUC')]):
        for col,freq in enumerate(['3','22']):
            ax=axs[row,col]
            for key,label in zip(MODELS,LABELS):
                g=metrics.query('maxfreq_khz == @freq and model_key == @key').set_index('snr').loc[NOISES]
                ax.plot(xx,g[metric],marker='o',label=label,alpha=.85)
            ax.set_title(f'{freq} kHz / {title}');ax.set_xticks(xx,ticks);ax.grid(alpha=.25)
            if metric=='r2': ax.set_ylim(.89,.97)
            else: ax.set_ylim(.86,1.01)
            if row==2:ax.set_xlabel('Reference SNR (dB); noise increases to the right')
    axs[0,0].legend(fontsize=8)
    fig.suptitle('2025-06-11 | matched training | 18 source-WAV OOF medians | seed 42\nOverlapping AUC curves are retained; these are two bandwidths of the same recordings')
    fig.tight_layout(rect=(0,0,1,.95));figsave(fig,'frequency_metric_comparison')
    fig,axs=plt.subplots(2,3,figsize=(16,8),sharex=True)
    for row,(metric,title) in enumerate([('r2','R2'),('auc_binary','Binary-score AUC')]):
        for model,g in th.query('metric == @metric').groupby('model',sort=False):
            g=g.set_index('snr').loc[NOISES]
            axs[row,0].errorbar(xx,g['mean'],yerr=g.se,marker='o',capsize=3,label=model)
        axs[row,0].set_title('Thesis (5-fold chunk mean +/- SE)')
        for col,freq in enumerate(['3','22'],start=1):
            for key,label in zip(MODELS,LABELS):
                g=metrics.query('maxfreq_khz == @freq and model_key == @key').set_index('snr').loc[NOISES]
                axs[row,col].plot(xx,g[metric],marker='o',label=label)
            axs[row,col].set_title(f'Current {freq} kHz (pooled WAV medians)')
        for ax in axs[row]:
            ax.set_ylim(.86,1.01);ax.set_xticks(xx,ticks);ax.grid(alpha=.25)
            ax.set_ylabel(title)
    axs[0,0].legend(fontsize=8);axs[0,1].legend(fontsize=8)
    fig.suptitle('Historical context: different recordings, models, preprocessing, splits and SNR definitions\nAUC type aligned; absolute performance is NOT a controlled before/after comparison')
    fig.tight_layout(rect=(0,0,1,.92));figsave(fig,'thesis_current_comparison')
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    for ax,freq in zip(axs,['3','22']):
        for key,label in zip(MODELS,LABELS):
            g=metrics.query('maxfreq_khz == @freq and model_key == @key').set_index('snr').loc[NOISES]
            ax.plot(xx,g.iloc[0].r2-g.r2,marker='o',label=label)
        ax.axhline(0,color='black',lw=.7);ax.set_title(f'{freq} kHz');ax.set_xticks(xx,ticks);ax.grid(alpha=.2)
        ax.set_ylabel('R2(clean) - R2(noise)');ax.set_xlabel('Reference SNR (dB)')
        ax.set_ylim(-.02,.045)
    axs[0].legend(fontsize=8)
    fig.suptitle('Degradation from each model\'s own clean baseline; lower is less drop\nRead together with absolute R2: a weaker clean baseline can make the drop smaller')
    fig.tight_layout();figsave(fig,'degradation_comparison')
    (out/'verification.json').write_text(json.dumps(dict(collected_at=datetime.now().astimezone().isoformat(),
        counts=counts,same_outer_splits_between_frequencies=True,source_sha256=hashes),ensure_ascii=False,indent=2),encoding='utf-8')
    print(pd.DataFrame(degradation).to_string(index=False))
    print(pd.DataFrame(xai).to_string(index=False))
    print(metrics.groupby('maxfreq_khz')[['false_negatives','false_positives','roc_auc_cont','auc_binary']].agg(['min','max','sum']).to_string())


if __name__=='__main__':
    main()
