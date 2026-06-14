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
"""

import os
import gc
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from sklearn.model_selection import KFold, train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)

from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD

from utils.models.regression.base_regression import RegressionModelMaker
from utils.dataloading.dataloading_and_conversion import DataLoadingConversion


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
#                              指標の計算
#   ② 回帰 / 連続スコア検知 / 二値化後分類 の 3 種類に分けて算出する。
#######################################################################

def _safe_roc_auc(y_true_bin, y_score):
    """連続スコア ROC-AUC。fold 内に片方のクラスしかない場合は nan を返す。"""
    if len(np.unique(y_true_bin)) < 2:
        return np.nan
    return roc_auc_score(y_true_bin, y_score)


def _safe_pr_auc(y_true_bin, y_score):
    """連続スコア PR-AUC。正例が無い場合は nan を返す。"""
    if y_true_bin.sum() == 0:
        return np.nan
    return average_precision_score(y_true_bin, y_score)


def regression_metrics(y_true, y_pred, threshold, band_frac):
    """
    回帰指標を返す。
      - 全体        : r2 / rmse / mae
      - 高熱流束域   : 閾値以上 (沸騰域) の rmse / mae と r2_high
      - ONB 近傍     : |y - threshold| <= threshold * band_frac の rmse / mae
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    m_high = y_true >= threshold
    band = np.abs(y_true - threshold) <= threshold * band_frac

    out = {
        "r2": r2_score(y_true, y_pred),
        "rmse_all": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae_all": mean_absolute_error(y_true, y_pred),
    }
    # 高熱流束域 (沸騰域)
    if m_high.sum() >= 2:
        out["r2_high"] = r2_score(y_true[m_high], y_pred[m_high])
        out["rmse_high"] = np.sqrt(mean_squared_error(y_true[m_high], y_pred[m_high]))
        out["mae_high"] = mean_absolute_error(y_true[m_high], y_pred[m_high])
    else:
        out["r2_high"] = np.nan
        out["rmse_high"] = np.nan
        out["mae_high"] = np.nan
    # ONB 近傍 band
    if band.sum() >= 1:
        out["rmse_onb"] = np.sqrt(mean_squared_error(y_true[band], y_pred[band]))
        out["mae_onb"] = mean_absolute_error(y_true[band], y_pred[band])
        out["n_onb"] = int(band.sum())
    else:
        out["rmse_onb"] = np.nan
        out["mae_onb"] = np.nan
        out["n_onb"] = 0
    return out


def detection_metrics_continuous(y_true, y_score, threshold):
    """
    連続スコア検知指標。予測熱流束 (連続値) をそのままスコアとして使う。
    正解は y_true を閾値で二値化したもの。
      - roc_auc_cont : ROC-AUC (連続スコア)
      - pr_auc_cont  : PR-AUC (連続スコア)
    """
    y_true_bin = (np.asarray(y_true).ravel() >= threshold).astype(int)
    y_score = np.asarray(y_score, dtype=float).ravel()
    return {
        "roc_auc_cont": _safe_roc_auc(y_true_bin, y_score),
        "pr_auc_cont": _safe_pr_auc(y_true_bin, y_score),
    }


def detection_metrics_binary(y_true, y_pred, threshold):
    """
    二値化後の分類指標。予測も正解も閾値で 0/1 化してから算出する。
      - accuracy / precision / recall / f1
      - auc_binary : 旧コードと同じ二値化後 AUC (後方比較用。意味は限定的)
    """
    y_true_bin = (np.asarray(y_true).ravel() >= threshold).astype(int)
    y_pred_bin = (np.asarray(y_pred).ravel() >= threshold).astype(int)
    out = {
        "accuracy": accuracy_score(y_true_bin, y_pred_bin),
        "precision": precision_score(y_true_bin, y_pred_bin, zero_division=0),
        "recall": recall_score(y_true_bin, y_pred_bin, zero_division=0),
        "f1": f1_score(y_true_bin, y_pred_bin, zero_division=0),
    }
    # 旧コード互換の二値化後 AUC (ROC が 2 点しか持たないので参考値)
    out["auc_binary"] = _safe_roc_auc(y_true_bin, y_pred_bin)
    return out


def _mean_se(arr):
    """平均と標準誤差 (nan を除外)。"""
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    if len(arr) == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1) / np.sqrt(len(arr)))


#######################################################################
#                          モデルの学習と予測
#######################################################################

def make_pca(x_fit, x_other_list, n_components):
    """sklearn 系モデル用に平坦化 + PCA。学習データのみで fit する。"""
    x_fit_flat = x_fit.reshape(x_fit.shape[0], -1)
    pca = PCA(n_components=min(n_components, x_fit_flat.shape[0], x_fit_flat.shape[1]),
              random_state=RANDOM_SEED)
    x_fit_pca = pca.fit_transform(x_fit_flat)
    others = []
    for x_other in x_other_list:
        if x_other is None:
            others.append(None)
        else:
            others.append(pca.transform(x_other.reshape(x_other.shape[0], -1)))
    return x_fit_pca, others


def train_one_model(spec, mm, x_fit, y_fit_scaled,
                    x_fit_pca, batch_size, epochs):
    """1 モデルを学習して返す。kind に応じて入力形態を変える。"""
    model = spec["builder"](mm)
    if spec["kind"] == "keras":
        lr = spec.get("lr", 0.005)
        model.compile(optimizer=SGD(learning_rate=lr, momentum=0.9, clipnorm=1.0),
                      loss='mean_squared_error')
        history = model.fit(x_fit, y_fit_scaled,
                            batch_size=spec.get("batch_size", batch_size),
                            epochs=epochs, verbose=1)
        return model, history
    else:  # sklearn / xgboost
        model.fit(x_fit_pca, y_fit_scaled.ravel())
        return model, None


def predict_one_model(spec, model, x, x_pca, scaler):
    """学習済みモデルで予測し、元スケールの熱流束に戻して返す。"""
    if spec["kind"] == "keras":
        pred_scaled = model.predict(x)
    else:
        pred_scaled = model.predict(x_pca).reshape(-1, 1)
    return scaler.inverse_transform(pred_scaled).ravel()


#######################################################################
#                        アンサンブル重みの決定 (③)
#######################################################################

def compute_weights(strategy, enabled_specs, errors_for_weight):
    """
    戦略に応じてモデル重みを返す (合計 1 に正規化)。
      errors_for_weight: key -> 誤差 (1 - R2)。simple/fixed では使わない。
    """
    keys = [s["key"] for s in enabled_specs]
    n = len(keys)

    if strategy == "simple":
        return {k: 1.0 / n for k in keys}

    if strategy == "fixed":
        raw = np.array([max(FIXED_WEIGHTS.get(k, 0.0), 0.0) for k in keys], dtype=float)
        if raw.sum() == 0:
            return {k: 1.0 / n for k in keys}
        raw = raw / raw.sum()
        return {k: float(w) for k, w in zip(keys, raw)}

    # inner_holdout / val_fold_legacy : 誤差の逆数で重み付け
    weights = []
    for k in keys:
        err = errors_for_weight.get(k, np.nan)
        if err is None or np.isinf(err) or np.isnan(err) or err <= 0:
            weights.append(1e-6)
        else:
            weights.append(1.0 / err)
    weights = np.array(weights, dtype=float)
    if weights.sum() == 0:
        weights = np.ones(n) / n
    else:
        weights = weights / weights.sum()
    return {k: float(w) for k, w in zip(keys, weights)}


def combine_predictions(preds_by_key, weights, combine):
    """preds_by_key: key -> 1D 予測配列。weights: key -> 重み。"""
    keys = list(preds_by_key.keys())
    stacked = np.stack([preds_by_key[k] for k in keys], axis=0)  # (n_models, n_samples)
    if combine == "min":
        return np.min(stacked, axis=0)
    w = np.array([weights[k] for k in keys], dtype=float).reshape(-1, 1)
    return np.sum(stacked * w, axis=0)


#######################################################################
#                               作図
#######################################################################

def plot_loss_history(history, epochs, label, fold, save_path, snr_value):
    if history is None:
        return
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title(f'{label} Loss History (Epochs: {epochs}, Fold: {fold}, SNR: {snr_value})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Mean Squared Error)')
    plt.legend()
    plt.grid(True)
    out_dir = os.path.join(save_path, "loss_histories")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f'{label}_ep{epochs}_fold{fold}_SNR={snr_value}.png'))
    plt.close()


def plot_bar(metric_name, labels, values, errors, epochs, save_path, snr_value):
    """モデル別の指標を棒グラフで保存 (R2 / 連続スコア ROC-AUC など)。"""
    plt.figure(figsize=(8, 6))
    colors = ['c', 'cadetblue', 'skyblue', 'dodgerblue', 'steelblue', 'lightblue']
    plt.bar(labels, values, color=colors[:len(labels)],
            yerr=errors, capsize=5, width=0.5)
    plt.ylim(0.0, 1.05)
    plt.ylabel(metric_name, fontsize=20)
    plt.xticks(fontsize=13, rotation=20)
    plt.yticks(fontsize=18)
    for i, v in enumerate(values):
        if not np.isnan(v):
            plt.text(i, 0.03, f'{v:.3f}', ha='center', va='bottom',
                     fontsize=18, color='black', rotation=90)
    out_dir = os.path.join(save_path, "bar_results")
    os.makedirs(out_dir, exist_ok=True)
    safe = metric_name.replace(" ", "_").replace("/", "_")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{safe}_ep{epochs}_SNR={snr_value}.png'))
    plt.close()


def plot_regression_scatter(y_val, ensemble_pred, y_all, metrics_ens,
                            threshold, save_path, snr_value, fold):
    """アンサンブル予測の回帰散布図 + 閾値 + 100% 分類閾値線。"""
    y_val = np.asarray(y_val).ravel()
    ensemble_pred = np.asarray(ensemble_pred).ravel()

    plt.figure(figsize=(12, 9))
    plt.scatter(y_val, ensemble_pred, label='Data', alpha=0.6)
    plt.plot([min(y_all), max(y_all)], [min(y_all), max(y_all)], 'r--')
    plt.xlabel('True Heat Flux MW/m²', fontsize=40)
    plt.ylabel('Predicted Heat Flux MW/m²', fontsize=40)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="plain", axis="both")
    ax.xaxis.offsetText.set_visible(False)
    ax.yaxis.offsetText.set_visible(False)

    plt.axvline(x=threshold, color='k', linestyle='dashed',
                label=f'Threshold (Boiling Point):\n{threshold / 1e6:.4f} MW/m²')
    plt.axhline(y=threshold, color='k', linestyle='dashed', label=None)

    # --- 100% 分類性能の評価 (連続して同じ y_val を 1 ブロックとみなす) ---
    blocks, current_block = [], [0]
    for i in range(1, len(y_val)):
        if y_val[i] == y_val[i - 1]:
            current_block.append(i)
        else:
            blocks.append(current_block)
            current_block = [i]
    blocks.append(current_block)

    block_labels = [y_val[b[0]] for b in blocks]
    sorted_blocks = [blocks[i] for i in np.argsort(block_labels)]

    block_flags, block_val_values = [], []
    for b in sorted_blocks:
        bp = ensemble_pred[b].flatten()
        block_flags.append(np.all(bp > threshold))
        block_val_values.append(y_val[b[0]])

    for i, val_value in enumerate(block_val_values):
        if not block_flags[i]:
            continue
        if all(block_flags[i:]):
            plt.axvline(x=val_value, linestyle='dashdot', color='green',
                        label=f"100% Classification Threshold:\n{val_value / 1e6:.4f} MW/m²")
            break

    r2 = metrics_ens.get("r2", np.nan)
    r2_high = metrics_ens.get("r2_high", np.nan)
    plt.text(0.72, 0.10, f'R² All :  {r2:.4f}\nR² High: {r2_high:.4f}',
             ha='center', va='center', transform=ax.transAxes, fontsize=40)
    legend = plt.legend(loc=(0.007, 0.72), fontsize=20)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(0.7)

    xticks = np.arange(0, 1.3e6, step=2e5)
    plt.xticks(xticks, fontsize=24, labels=[f'{x/1e6:.1f}' for x in xticks])
    plt.yticks(xticks, fontsize=24, labels=[f'{x/1e6:.1f}' for x in xticks])
    plt.tick_params(axis='both', labelsize=30)

    out_dir = os.path.join(save_path, "regression_results")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f'regression_split_{snr_value}_{fold}.png'))
    plt.close()


#######################################################################
#                                実行部
#######################################################################

def main():
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
                            x_fit_pca, (x_val_pca, x_inner_pca) = make_pca(
                                x_fit, [x_val, x_inner], PCA_COMPONENTS)
                        else:
                            x_fit_pca = x_val_pca = x_inner_pca = None

                        mm = RegressionModelMaker((224, 224, COLOR_CHANNEL))

                        # --- 各モデルの学習と予測 ---
                        val_preds = {}        # key -> 検証 fold への予測 (元スケール)
                        errors_for_weight = {}  # key -> 重み用の誤差 (1 - R2)
                        for spec in enabled_specs:
                            print(f"[{spec['label']}] Fold {fold}/{DIVISIONS} 学習開始")
                            model, history = train_one_model(
                                spec, mm, x_fit, y_fit_scaled, x_fit_pca,
                                all_bs, EPOCH_NUM)
                            plot_loss_history(history, EPOCH_NUM, spec["label"],
                                              fold, SAVE_PATH, snr_value)

                            # 検証 fold への予測
                            val_preds[spec["key"]] = predict_one_model(
                                spec, model, x_val, x_val_pca, scaler)

                            # --- 重み用の誤差 (③ 戦略ごとにリークしない/する を切替) ---
                            if WEIGHT_STRATEGY == "inner_holdout":
                                inner_pred = predict_one_model(
                                    spec, model, x_inner, x_inner_pca, scaler)
                                errors_for_weight[spec["key"]] = 1.0 - r2_score(y_inner, inner_pred)
                            elif WEIGHT_STRATEGY == "val_fold_legacy":
                                # 旧来どおり検証 fold 誤差 (リークあり)
                                errors_for_weight[spec["key"]] = 1.0 - r2_score(y_val, val_preds[spec["key"]])

                            del model
                            K.clear_session()
                            gc.collect()

                        # --- アンサンブル ---
                        weights = compute_weights(WEIGHT_STRATEGY, enabled_specs, errors_for_weight)
                        weight_log.append((fold, dict(weights)))
                        ensemble_pred = combine_predictions(val_preds, weights, ENSEMBLE_COMBINE)

                        # --- 指標の算出 (② 3 種類に分離) ---
                        preds_all = dict(val_preds)
                        preds_all["ensemble"] = ensemble_pred
                        for key in all_keys:
                            pred = preds_all[key]
                            reg = regression_metrics(y_val, pred, THRESHOLD, ONB_BAND_FRAC)
                            det_c = detection_metrics_continuous(y_val, pred, THRESHOLD)
                            det_b = detection_metrics_binary(y_val, pred, THRESHOLD)
                            for d in (reg, det_c, det_b):
                                for mk, mv in d.items():
                                    store[key][mk].append(mv)

                        # --- 作図 (アンサンブルの散布図) ---
                        ens_fold_metrics = {mk: store["ensemble"][mk][-1]
                                            for mk in store["ensemble"]}
                        plot_regression_scatter(
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
                            mean, se = _mean_se(store[key][mk])
                            f.write(f"    {mk:14s}: {mean:.4f} ± {se:.4f}\n")
                    f.write("=" * 30 + "\n\n")

                # --- 棒グラフ (モデル別: R2 と 連続スコア ROC-AUC) ---
                labels = [label_of[k] for k in all_keys]
                r2_means = [_mean_se(store[k]["r2"])[0] for k in all_keys]
                r2_ses = [_mean_se(store[k]["r2"])[1] for k in all_keys]
                roc_means = [_mean_se(store[k]["roc_auc_cont"])[0] for k in all_keys]
                roc_ses = [_mean_se(store[k]["roc_auc_cont"])[1] for k in all_keys]
                plot_bar("R2 Score", labels, r2_means, r2_ses, EPOCH_NUM, SAVE_PATH, snr_value)
                plot_bar("ROC-AUC (continuous)", labels, roc_means, roc_ses,
                         EPOCH_NUM, SAVE_PATH, snr_value)

                # --- 指標 CSV (fold 平均をモデル別に保存。後で比較しやすくする) ---
                csv_path = os.path.join(SAVE_PATH, f'metrics_summary_{snr_value}.csv')
                with open(csv_path, 'w', encoding='utf-8') as cf:
                    header = ["model"] + [f"{mk}_mean" for mk in summary_metrics] \
                                       + [f"{mk}_se" for mk in summary_metrics]
                    cf.write(",".join(header) + "\n")
                    for key in all_keys:
                        means = [f"{_mean_se(store[key][mk])[0]:.6f}" for mk in summary_metrics]
                        ses = [f"{_mean_se(store[key][mk])[1]:.6f}" for mk in summary_metrics]
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
