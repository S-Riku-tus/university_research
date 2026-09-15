"""Complement the WAV analysis with the historical chunk-fold mean/SE view."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,roc_auc_score
from analyze_3khz import ROOT,MODELS,NOISES,longpath


def main():
    base=Path(__file__).parent
    out=base/'chunk_comparison'
    if out.exists() and any(out.iterdir()):raise SystemExit('Preserve existing output.')
    out.mkdir(exist_ok=True)
    snapshots={'3':base/'snapshot_3khz','22':ROOT/'experiments/2026-09-15_onb_22khz_noise_sweep/snapshot_20260915'}
    th=pd.read_csv(base/'comparison/thesis_table_values.csv',dtype={'snr':str})
    labels=['RandomForest','CNN+Tf v2 GAP','AlexNet','Ensemble simple equal','Ensemble inner holdout']
    data=[];checks=[];curves=[]
    for freq,snap in snapshots.items():
        metrics=pd.read_csv(snap/'chunk_metrics.csv',dtype={'snr':str})
        print(freq,metrics.model.unique())
        data.append(metrics.assign(maxfreq_khz=freq))
        for _,inv in pd.read_csv(snap/'run_inventory.csv',dtype={'snr':str}).iterrows():
            folder=longpath(ROOT/inv.run_manifest).parent
            folded={key:[] for key in MODELS}
            for fold in range(1,4):
                pred=pd.read_csv(folder/f'fold_pred/pred_f{fold}_{inv.snr}.csv')
                for key in MODELS:
                    folded[key].append((r2_score(pred.y_true,pred[key]),roc_auc_score(pred.y_true>=368978.105,pred[key]>=368978.105)))
            # CSV order matches the runtime model catalog; assert explicit labels below.
            subset=metrics[metrics.snr==inv.snr]
            assert list(subset.model)==labels,list(subset.model)
            for key,(_,row) in zip(MODELS,subset.iterrows()):
                arr=np.array(folded[key])
                assert abs(row.r2_mean-arr[:,0].mean())<1e-6
                assert abs(row.auc_binary_mean-arr[:,1].mean())<1e-6
                checks.append(dict(maxfreq_khz=freq,snr=inv.snr,model=key,verified=True))
        ex=pd.read_csv(snap/'explainability_summary.csv')
        for (model,method),g in ex.groupby(['model','method']):
            for metric in ['deletion_area_between_curve','insertion_area_between_curve']:
                v=pd.to_numeric(g[metric],errors='coerce').dropna()
                if len(v):curves.append(dict(maxfreq_khz=freq,model=model,method=method,metric=metric,n=len(v),median=v.median(),min=v.min(),max=v.max()))
    allmetrics=pd.concat(data)
    allmetrics.to_csv(out/'chunk_metrics_both.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(curves).to_csv(out/'deletion_insertion_summary.csv',index=False,encoding='utf-8-sig')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axs=plt.subplots(2,3,figsize=(16,8),sharex=True)
    xx=np.arange(7);ticks=['Clean','0','-4','-8','-12','-16','-20']
    for row,metric in enumerate(['r2','auc_binary']):
        for model,g in th[th.metric==metric].groupby('model',sort=False):
            g=g.set_index('snr').loc[NOISES]
            axs[row,0].errorbar(xx,g['mean'],yerr=g.se,marker='o',capsize=2,label=model)
        for col,freq in enumerate(['3','22'],start=1):
            for model,g in allmetrics[allmetrics.maxfreq_khz==freq].groupby('model',sort=False):
                g=g.set_index('snr').loc[NOISES]
                axs[row,col].errorbar(xx,g[metric+'_mean'],yerr=g[metric+'_se'],marker='o',capsize=2,label=model)
        for col,title in enumerate(['Thesis: 5-fold','Current 3 kHz: 3-fold','Current 22 kHz: 3-fold']):
            ax=axs[row,col];ax.set_title(title);ax.set_xticks(xx,ticks);ax.grid(alpha=.2)
            ax.set_ylabel('R2' if metric=='r2' else 'AUC after ONB binarization')
            ax.set_ylim(.84,1.02)
    axs[0,0].legend(fontsize=8);axs[0,1].legend(fontsize=8)
    fig.suptitle('Chunk fold means +/- SE in all panels; AUC definition aligned\nRecordings, preprocessing, models and split groups still differ; fold SE is not independent-run uncertainty')
    fig.tight_layout(rect=(0,0,1,.92))
    for ext in ['png','pdf']:fig.savefig(out/f'thesis_chunk_comparison.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    (out/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(allmetrics[allmetrics.snr.isin(['no_noise','-20'])][['maxfreq_khz','snr','model','r2_mean','auc_binary_mean']].to_string(index=False))


if __name__=='__main__':main()
