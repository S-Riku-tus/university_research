"""Record the legacy PCA version2 BLAS sensitivity; do not change old models."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES","-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")
from pathlib import Path
import sys
import csv

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"code"));sys.path.insert(0,str(ROOT/"experiments/2026-10-06_chunk_complementary_model_pilots"))
import pilot as p
from utils.training.model_training import ModelTrainer


def main():
    rows,folds,data,outer,base,y,days=p.setup()
    root=p.RUN/"maxfreq=3kHz/heatflux_no_noise/fitted_state/fold1"
    model=joblib.load(p.lp(root/"randomforest.joblib"));pca=joblib.load(p.lp(root/"pca.joblib"));scaler=joblib.load(p.lp(root/"target_scaler.joblib"));trainer=ModelTrainer(42)
    output=[]
    for noise in ["clean","-4","-20"]:
        lookup={p.identity(r):r for r in p.read_csv(data/p.NOISE_DIRS[noise]/"chunk_manifest.csv")}
        raw=np.asarray([np.load(p.lp(data/p.NOISE_DIRS[noise]/lookup[p.identity(row)]["sample_filename"])) for row in outer],dtype=np.float32)[...,None]
        suffix="no_noise" if noise=="clean" else noise
        saved={p.identity(r):r for r in p.read_csv(p.RUN/"maxfreq=3kHz"/p.NOISE_DIRS[noise]/"fold_pred"/f"pred_f1_{suffix}.csv")}
        expected=np.asarray([float(saved[p.identity(row)]["randomforest"])/1000 for row in outer])
        with threadpool_limits(limits=20,user_api="blas"):reference=trainer.transform_pca(pca,raw)
        for count in [1,2,8,16,20]:
            with threadpool_limits(limits=count,user_api="blas"):
                transformed=trainer.transform_pca(pca,raw)
                prediction=scaler.inverse_transform(model.predict(transformed).reshape(-1,1)).ravel()/1000
            difference=np.abs(prediction-expected)
            output.append({"noise":noise,"blas_threads":count,"input_rows":540,"different_pca_elements_vs20":int(np.count_nonzero(transformed!=reference)),
                "predictions_changed_over_0.01_kW_m2":int((difference>.01).sum()),"max_prediction_difference_kW_m2":float(difference.max()),
                "pca_transform_version":2,"original_model_retrained":False})
    with (OUT/"rf_thread_context_diagnostic.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
    print("RF projection thread diagnostic saved; original models preserved")


if __name__=="__main__":main()
