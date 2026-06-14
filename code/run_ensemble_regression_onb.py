"""
run_ensemble_regression_onb.py

中心スクリプト 3.run_ensemble_ROC_100%_analysis.py の作り直し版。
2026-06-12 の研究計画 (研究進捗報告/2026/612/今後の研究計画_2026-06-12.md) の
Phase 0「評価軸とコードの整理」に対応する。

旧スクリプトに対して、次の3点を直したうえで作り直している。

  ① モデル名の取り違えをなくす
     旧コードは RandomForest の予測を alexnet_pred という変数に入れ、以降の
     R2/AUC/グラフ/txt すべてに "AlexNet" というラベルで記録していた。
     本スクリプトでは MODEL_SPECS というレジストリで各モデルを定義し、
     ラベルが必ず実体のモデルに追従するようにした。モデルの差し替え・
     有効/無効化は MODEL_SPECS を編集するだけで済む。

  ② AUC を「連続スコア版」と「二値化後の分類指標」に分ける
     旧コードは予測を閾値で 0/1 化してから ROC を計算していたため、ROC が
     2点しか持たず AUC が実質バランス精度になっていた。本スクリプトでは
       - 連続スコア版 AUC: 予測熱流束をそのままスコアにした ROC-AUC / PR-AUC
       - 二値化後の分類指標: Accuracy / Precision / Recall / F1
     を分けて算出する。旧来の二値化 AUC も後方比較用に残してある。

  ③ アンサンブル重みの決め方を選択式にする (リーク対策)
     旧コードは評価対象である検証 fold (y_val) の誤差から重みを決めており、
     データリークになっていた。本スクリプトでは WEIGHT_STRATEGY で
       - simple          : 単純平均 (重みなし)
       - fixed           : 固定重み (FIXED_WEIGHTS で指定)
       - inner_holdout   : 学習 fold 内 holdout の誤差から重み (リークなし)
       - val_fold_legacy : 旧来どおり検証 fold 誤差から重み (リークあり/再現用)
     を切り替えられる。

  ④ データパスは別マシン運用のためそのまま (旧コードと同じハードコード)

旧スクリプト (code/3.run_ensemble_ROC_100%_analysis.py) は再現性のため残す。

再利用可能な処理 (指標計算・学習/予測・重み付け・作図) は、既存の utils 方針に
合わせて用途別のクラスに分離してある。本ファイルにはこの実験固有の設定と
main() のオーケストレーションだけを置く。
    - 指標計算    : utils/calculation/regression_detection_metrics.py
    - 学習/予測   : utils/training/model_training.py
    - 重み付け    : utils/ensemble/ensemble_weighting.py
    - 作図        : utils/plotting/regression_plots.py
"""

import os
import gc
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

from tensorflow.keras import backend as K

from utils.models.regression.base_regression import RegressionModelMaker
from utils.dataloading.dataloading_and_conversion import DataLoadingConversion
from utils.calculation.regression_detection_metrics import RegressionDetectionMetrics
from utils.training.model_training import ModelTrainer
from utils.ensemble.ensemble_weighting import EnsembleWeighting
from utils.plotting.regression_plots import RegressionPlotter


#######################################################################
#                              変数の指定
#######################################################################

EPOCH_NUM = 500
DIVISIONS = 5            # 交差検証の fold 数
COLOR_CHANNEL = 1
RANDOM_SEED = 42

# 閾値 (沸騰開始点 ONB の熱流束)。旧コードと同じ値。
threshold_list = [275174.6641]
THRESHOLD = sum(threshold_list) / len(threshold_list)

# ONB 近傍 band の幅 (閾値に対する相対割合)。
# |y - THRESHOLD| <= THRESHOLD * ONB_BAND_FRAC を ONB 近傍サンプルとして
# 別途 RMSE/MAE を見る (計画 RQ で要求されている ONB 近傍誤差に対応)。
ONB_BAND_FRAC = 0.10

# パラメータをループさせて検証するかどうか (旧コード踏襲)
FLG_ROOP = True
BATCH_SIZES_ALL = [48]
LEARNING_RATE_ALL = [0.005]

# ホワイトノイズ (=0) か水流動音 (=1) か (旧コード踏襲)
NOISE = 1
PREVIOUS_MODEL = False
SAVE_DATE = "20260208"
DATA_DATE = "20251219"

# 周波数解析のパラメータ
CHUNK = 1
max_freq_hz = "maxfreq=22kHz"


# ===================================================================
# ① モデルレジストリ
#    各モデルを 1 つの辞書で定義する。label が必ず実体に追従するので、
#    旧コードのような「RF の予測を alexnet と呼ぶ」取り違えが起きない。
#    モデルの追加・差し替え・無効化は、このリストを編集するだけでよい。
#
#    kind:
#       "keras"   ... 画像 (224,224,C) をそのまま入力する深層モデル
#       "sklearn" ... 平坦化 + PCA した特徴量を入力する非深層モデル
#                     (RandomForest / XGBRF など)
#    builder: RegressionModelMaker のインスタンスを受け取りモデルを返す関数
# ===================================================================
MODEL_SPECS = [
    {
        "key": "rf",
        "label": "RandomForest",
        "kind": "sklearn",
        "builder": lambda mm: mm.random_forest(),
        "enabled": True,
    },
    {
        "key": "cnntf_v1",
        "label": "CNN+Tf (AttnPool)",
        "kind": "keras",
        "builder": lambda mm: mm.cnn_transformer_v1(),
        "lr": 0.005,
        "batch_size": 48,
        "enabled": True,
    },
    {
        "key": "cnntf_v2",
        "label": "CNN+Tf (GAP)",
        "kind": "keras",
        "builder": lambda mm: mm.cnn_transformer_v2(),
        "lr": 0.005,
        "batch_size": 48,
        "enabled": True,
    },
]


# ===================================================================
# ③ アンサンブル重み戦略
#    "simple"          : 単純平均 (重みなし)。最も安全で挙動が明快。
#    "fixed"           : 固定重み。FIXED_WEIGHTS (key -> 重み) で指定。
#    "inner_holdout"   : 学習 fold をさらに内部 holdout に分け、その内部検証
#                        誤差から重みを決める。検証 fold を一切見ないので
#                        リークなし。各モデルは内部学習データのみで学習する点に注意。
#    "val_fold_legacy" : 旧コードと同じく検証 fold 誤差から重みを決める。
#                        リークありなので新たな主張には使わない。旧結果の再現用。
# ===================================================================
WEIGHT_STRATEGY = "simple"
FIXED_WEIGHTS = {"rf": 1.0, "cnntf_v1": 1.0, "cnntf_v2": 1.0}
INNER_HOLDOUT_FRAC = 0.2   # inner_holdout のときの内部検証の割合

# アンサンブルの統合方法 ("mean" ... 重み付き平均 | "min" ... 各サンプル最小値)
ENSEMBLE_COMBINE = "mean"

# PCA 次元 (sklearn 系モデル用)
PCA_COMPONENTS = 100


#### データフォルダの設定 (④: 別マシン運用のためそのまま) ####
noise = "whitenoise" if NOISE == 0 else "waterflow"
highpass = f"_{DATA_DATE}_{CHUNK}s"
noise = noise + highpass

BASE_PATH = r"C:\Users\Casper4\Python\ueki\shibasaki\研究\Pool_boiling\Subcooling_20_degrees\0.3\2025.07.09_0.3_1"
base_path = Path(BASE_PATH)

BASE_DATA_PATH = base_path / "data" / "npy" / noise / str(max_freq_hz)
DATA_PATH = [
    BASE_DATA_PATH / "heatflux_no_noise",
    # BASE_DATA_PATH / "heatflux_SNR=0",
    # BASE_DATA_PATH / "heatflux_SNR=-4",
    # BASE_DATA_PATH / "heatflux_SNR=-8",
    # BASE_DATA_PATH / "heatflux_SNR=-12",
    # BASE_DATA_PATH / "heatflux_SNR=-16",
    # BASE_DATA_PATH / "heatflux_SNR=-20"
]

BASE_SAVE_PATH = base_path / "regression_result" / "npy" / "ensemble"

# matplotlib の設定
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


#######################################################################
#                                実行部
#######################################################################

def main():
    # 再利用ヘルパー (用途別の utils クラス) を用意する
    metrics = RegressionDetectionMetrics()
    trainer = ModelTrainer(random_seed=RANDOM_SEED)
    weighting = EnsembleWeighting()
    plotter = RegressionPlotter()

    enabled_specs = [s for s in MODEL_SPECS if s.get("enabled", True)]
    use_sklearn = any(s["kind"] == "sklearn" for s in enabled_specs)
    all_keys = [s["key"] for s in enabled_specs] + ["ensemble"]
    label_of = {s["key"]: s["label"] for s in enabled_specs}
    label_of["ensemble"] = "Ensemble"

    print(f"有効なモデル: {[s['label'] for s in enabled_specs]}")
    print(f"重み戦略: {WEIGHT_STRATEGY} | 統合: {ENSEMBLE_COMBINE}")
    if WEIGHT_STRATEGY == "val_fold_legacy":
        print("【警告】val_fold_legacy は検証 fold の正解から重みを決めるリークあり方式です。"
              "旧結果の再現用にのみ使用してください。")

    for data_path in DATA_PATH:
        K.clear_session()
        gc.collect()

        if "no_noise" in str(data_path):
            snr_value = "no_noise"
        else:
            snr_value = str(data_path).split("SNR=")[-1]

        print(f"\n{'='*40}\nデータセット読み込み開始: {snr_value}")

        start_time = time.time()
        data_loading = DataLoadingConversion()
        if COLOR_CHANNEL == 1:
            x, y = data_loading.load_npy_data(data_path)
        else:
            x, y = data_loading.load_image_data(data_path)
        print(f"x shape: {x.shape} | y shape: {y.shape} | "
              f"読み込み {time.time() - start_time:.2f} 秒")

        for all_bs in BATCH_SIZES_ALL:
            for all_lr in LEARNING_RATE_ALL:
                noise_dir_name = os.path.basename(data_path)
                SAVE_PATH = os.path.join(
                    BASE_SAVE_PATH, noise_dir_name, max_freq_hz,
                    f"{SAVE_DATE}_ep{EPOCH_NUM}_bs{all_bs}_lr{all_lr}_{WEIGHT_STRATEGY}")
                os.makedirs(SAVE_PATH, exist_ok=True)

                kf = KFold(n_splits=DIVISIONS, shuffle=True, random_state=RANDOM_SEED)

                # 指標の保存先 (key -> metric -> [fold ごとの値])
                store = {k: defaultdict(list) for k in all_keys}
                # 重みの記録 (fold ごと)
                weight_log = []

                output_file = os.path.join(SAVE_PATH, f'validation_results_{snr_value}.txt')
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write("K-fold Cross-Validation Results\n")
                    f.write(f"models={[s['label'] for s in enabled_specs]}\n")
                    f.write(f"weight_strategy={WEIGHT_STRATEGY}, combine={ENSEMBLE_COMBINE}\n")
                    f.write("=" * 30 + "\n")

                    fold = 1
                    for train_index, val_index in kf.split(x):
                        x_train, x_val = x[train_index], x[val_index]
                        y_train, y_val = y[train_index], y[val_index]

                        # --- inner_holdout のときだけ学習 fold を内部分割 ---
                        if WEIGHT_STRATEGY == "inner_holdout":
                            x_fit, x_inner, y_fit, y_inner = train_test_split(
                                x_train, y_train, test_size=INNER_HOLDOUT_FRAC,
                                random_state=RANDOM_SEED)
                        else:
                            x_fit, y_fit = x_train, y_train
                            x_inner = y_inner = None

                        # スケーラは学習に使うラベル (y_fit) のみで fit する
                        scaler = MinMaxScaler()
                        y_fit_scaled = scaler.fit_transform(y_fit.reshape(-1, 1))

                        # sklearn 系モデル用の PCA (学習データのみで fit)
                        if use_sklearn:
                            x_fit_pca, (x_val_pca, x_inner_pca) = trainer.make_pca(
                                x_fit, [x_val, x_inner], PCA_COMPONENTS)
                        else:
                            x_fit_pca = x_val_pca = x_inner_pca = None

                        mm = RegressionModelMaker((224, 224, COLOR_CHANNEL))

                        # --- 各モデルの学習と予測 ---
                        val_preds = {}        # key -> 検証 fold への予測 (元スケール)
                        errors_for_weight = {}  # key -> 重み用の誤差 (1 - R2)
                        for spec in enabled_specs:
                            print(f"[{spec['label']}] Fold {fold}/{DIVISIONS} 学習開始")
                            model, history = trainer.train_one_model(
                                spec, mm, x_fit, y_fit_scaled, x_fit_pca,
                                all_bs, EPOCH_NUM)
                            plotter.plot_loss_history(history, EPOCH_NUM, spec["label"],
                                                      fold, SAVE_PATH, snr_value)

                            # 検証 fold への予測
                            val_preds[spec["key"]] = trainer.predict_one_model(
                                spec, model, x_val, x_val_pca, scaler)

                            # --- 重み用の誤差 (③ 戦略ごとにリークしない/する を切替) ---
                            if WEIGHT_STRATEGY == "inner_holdout":
                                inner_pred = trainer.predict_one_model(
                                    spec, model, x_inner, x_inner_pca, scaler)
                                errors_for_weight[spec["key"]] = 1.0 - r2_score(y_inner, inner_pred)
                            elif WEIGHT_STRATEGY == "val_fold_legacy":
                                # 旧来どおり検証 fold 誤差 (リークあり)
                                errors_for_weight[spec["key"]] = 1.0 - r2_score(y_val, val_preds[spec["key"]])

                            del model
                            K.clear_session()
                            gc.collect()

                        # --- アンサンブル ---
                        weights = weighting.compute_weights(
                            WEIGHT_STRATEGY, enabled_specs, errors_for_weight, FIXED_WEIGHTS)
                        weight_log.append((fold, dict(weights)))
                        ensemble_pred = weighting.combine_predictions(
                            val_preds, weights, ENSEMBLE_COMBINE)

                        # --- 指標の算出 (② 3 種類に分離) ---
                        preds_all = dict(val_preds)
                        preds_all["ensemble"] = ensemble_pred
                        for key in all_keys:
                            pred = preds_all[key]
                            reg = metrics.regression_metrics(y_val, pred, THRESHOLD, ONB_BAND_FRAC)
                            det_c = metrics.detection_metrics_continuous(y_val, pred, THRESHOLD)
                            det_b = metrics.detection_metrics_binary(y_val, pred, THRESHOLD)
                            for d in (reg, det_c, det_b):
                                for mk, mv in d.items():
                                    store[key][mk].append(mv)

                        # --- 作図 (アンサンブルの散布図) ---
                        ens_fold_metrics = {mk: store["ensemble"][mk][-1]
                                            for mk in store["ensemble"]}
                        plotter.plot_regression_scatter(
                            y_val, ensemble_pred, y, ens_fold_metrics,
                            THRESHOLD, SAVE_PATH, snr_value, fold)

                        # --- fold 結果を txt に追記 ---
                        f.write(f"Recorded at: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
                        f.write(f"Fold {fold} Results | weights={ {k: round(v,3) for k,v in weights.items()} }\n")
                        for key in all_keys:
                            m = {mk: store[key][mk][-1] for mk in store[key]}
                            f.write(
                                f"  [{label_of[key]}] "
                                f"R2={m.get('r2', float('nan')):.4f} "
                                f"RMSE={m.get('rmse_all', float('nan')):.1f} "
                                f"MAE={m.get('mae_all', float('nan')):.1f} | "
                                f"R2_high={m.get('r2_high', float('nan')):.4f} "
                                f"RMSE_onb={m.get('rmse_onb', float('nan')):.1f} "
                                f"(n_onb={m.get('n_onb', 0)}) | "
                                f"ROC_cont={m.get('roc_auc_cont', float('nan')):.4f} "
                                f"PR_cont={m.get('pr_auc_cont', float('nan')):.4f} | "
                                f"Acc={m.get('accuracy', float('nan')):.4f} "
                                f"Prec={m.get('precision', float('nan')):.4f} "
                                f"Rec={m.get('recall', float('nan')):.4f} "
                                f"F1={m.get('f1', float('nan')):.4f} "
                                f"AUC_bin={m.get('auc_binary', float('nan')):.4f}\n")
                        f.write("-" * 30 + "\n")

                        fold += 1
                        del x_train, x_val, y_train, y_val, y_fit_scaled
                        K.clear_session()
                        gc.collect()

                    # --- 平均結果 (mean ± SE) ---
                    f.write(f"\nRecorded at: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
                    f.write("Average Results (mean ± SE):\n")
                    summary_metrics = [
                        "r2", "rmse_all", "mae_all", "r2_high", "rmse_high", "mae_high",
                        "rmse_onb", "mae_onb",
                        "roc_auc_cont", "pr_auc_cont",
                        "accuracy", "precision", "recall", "f1", "auc_binary",
                    ]
                    for key in all_keys:
                        f.write(f"  [{label_of[key]}]\n")
                        for mk in summary_metrics:
                            mean, se = metrics.mean_se(store[key][mk])
                            f.write(f"    {mk:14s}: {mean:.4f} ± {se:.4f}\n")
                    f.write("=" * 30 + "\n\n")

                # --- 棒グラフ (モデル別: R2 と 連続スコア ROC-AUC) ---
                labels = [label_of[k] for k in all_keys]
                r2_means = [metrics.mean_se(store[k]["r2"])[0] for k in all_keys]
                r2_ses = [metrics.mean_se(store[k]["r2"])[1] for k in all_keys]
                roc_means = [metrics.mean_se(store[k]["roc_auc_cont"])[0] for k in all_keys]
                roc_ses = [metrics.mean_se(store[k]["roc_auc_cont"])[1] for k in all_keys]
                plotter.plot_bar("R2 Score", labels, r2_means, r2_ses, EPOCH_NUM, SAVE_PATH, snr_value)
                plotter.plot_bar("ROC-AUC (continuous)", labels, roc_means, roc_ses,
                                 EPOCH_NUM, SAVE_PATH, snr_value)

                # --- 指標 CSV (fold 平均をモデル別に保存。後で比較しやすくする) ---
                csv_path = os.path.join(SAVE_PATH, f'metrics_summary_{snr_value}.csv')
                with open(csv_path, 'w', encoding='utf-8') as cf:
                    header = ["model"] + [f"{mk}_mean" for mk in summary_metrics] \
                                       + [f"{mk}_se" for mk in summary_metrics]
                    cf.write(",".join(header) + "\n")
                    for key in all_keys:
                        means = [f"{metrics.mean_se(store[key][mk])[0]:.6f}" for mk in summary_metrics]
                        ses = [f"{metrics.mean_se(store[key][mk])[1]:.6f}" for mk in summary_metrics]
                        cf.write(",".join([label_of[key]] + means + ses) + "\n")
                print(f"指標 CSV を保存: {csv_path}")

                if not FLG_ROOP:
                    break
            if not FLG_ROOP:
                break

        del x, y
        gc.collect()


if __name__ == '__main__':
    main()
