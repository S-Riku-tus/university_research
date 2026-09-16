"""Numerical verification only: no training; no edits to historical runs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
from utils.models.regression.base_regression import RegressionModelMaker, LogPowerCompression
from utils.explainability.spectrogram_explainers import integrated_gradients


def old_ig(model,x):
    # Reproduce the old 64-step rule, batched to avoid a large activation tensor.
    alpha=np.linspace(0,1,65)
    total=np.zeros_like(x,dtype=np.float64)
    for i in range(0,65,4):
        inputs=tf.convert_to_tensor(alpha[i:i+4,None,None,None]*x[None],dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            y=model(inputs,training=False)
        gradients=tape.gradient(y,inputs).numpy()
        for j,g in enumerate(gradients):
            total+=g*(.5 if i+j in [0,64] else 1.)/64
    return (x*total).sum(axis=-1)


def main():
    records=[]
    # Analytic check spanning the same power scales as the experiment.
    x=np.array([1e-12,1e-9,1e-6,1e-4],np.float32).reshape(1,4,1)
    inp=tf.keras.Input(x.shape)
    model=tf.keras.Model(inp,tf.reduce_sum(LogPowerCompression()(inp),axis=(1,2,3))[:,None])
    old=old_ig(model,x)
    new,diag=integrated_gradients(model,x,return_diagnostics=True)
    exact=np.log1p(x.astype(float)/1e-12).sum()
    records.append({'case':'analytic_log_sum','exact_delta':exact,'old_sum':old.sum(),
                    'old_relative_error':abs(old.sum()-exact)/exact,
                    'new_sum':new.sum(),'new_relative_error':abs(new.sum()-exact)/exact,
                    'diagnostics':diag})
    inv=pd.read_csv(ROOT/'experiments/2026-09-15_research_status_snapshot/run_inventory.csv')
    r=inv[(inv.run=='20260914_selected_log_architecture') & (inv.maxfreq=='maxfreq=22kHz') & (inv.snr=='no_noise')].iloc[0]
    path=Path('\\\\?\\'+str(ROOT/r.run_manifest)).parent
    manifest=json.loads((path/'run_manifest.json').read_text(encoding='utf-8'))
    pred=pd.read_csv(path/'fold_pred/pred_f1_no_noise.csv')
    row=pred.iloc[180]
    data_path=Path('\\\\?\\'+manifest['dataset']['data_path'])/row.sample_filename
    x=np.load(data_path)
    if x.ndim==2:
        x=x[...,None]
    x=x.astype(np.float32)
    print('Real input',x.shape,'power range',x.min(),x.max(),flush=True)
    for name,method in [('cnntf_v2_gap','cnn_transformer_v2'),('alexnet','alexnet')]:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)
        model=getattr(RegressionModelMaker(x.shape),method)()
        print('Checking',name,'with random weights (not historical trained model)',flush=True)
        old=old_ig(model,x)
        new,diag=integrated_gradients(model,x,return_diagnostics=True)
        delta=diag['output_delta_model_units']
        record={'case':name,'weights':'random initialization; no training',
                'input_source':str(data_path),'input_sha256':hashlib.sha256(data_path.read_bytes()).hexdigest(),
                'old_sum':float(old.sum()),'old_relative_error':float(abs(old.sum()-delta)/(abs(delta)+1e-12)),
                'new_relative_error':float(abs(new.sum()-delta)/(abs(delta)+1e-12)),
                'diagnostics':diag}
        records.append(record)
        print(name,record['old_relative_error'],'->',record['new_relative_error'],diag['converged'],diag['nodes'],flush=True)
    (OUT/'numerical_verification.json').write_text(json.dumps(records,indent=2,ensure_ascii=False),encoding='utf-8')


if __name__=='__main__':
    if '--refine' not in sys.argv:
        main()
    else:
        path=OUT/'numerical_verification.json'
        records=json.loads(path.read_text(encoding='utf-8'))
        for record in records:
            if record['diagnostics']['converged']:
                continue
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(42)
            x=np.load(record['input_source']).astype(np.float32)
            if x.ndim==2:
                x=x[...,None]
            method='alexnet' if record['case']=='alexnet' else 'cnn_transformer_v2'
            model=getattr(RegressionModelMaker(x.shape),method)()
            print('Refining',record['case'],flush=True)
            new,diag=integrated_gradients(model,x,steps=1024,max_steps=4096,return_diagnostics=True)
            record['diagnostics_at_1024']=record['diagnostics']
            record['diagnostics']=diag
            record['new_relative_error']=float(abs(new.sum()-diag['output_delta_model_units'])/(abs(diag['output_delta_model_units'])+1e-12))
            print(record['case'],record['new_relative_error'],diag['converged'],diag['nodes'],flush=True)
            path.write_text(json.dumps(records,indent=2,ensure_ascii=False),encoding='utf-8')
