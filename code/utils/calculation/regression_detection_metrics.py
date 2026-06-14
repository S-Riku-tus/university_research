import numpy as np
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)


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


class RegressionDetectionMetrics:
    """
    熱流束回帰と ONB 検知の評価指標をまとめたクラス。
    2026-06-12 計画 Phase 0 の「連続スコア版 AUC」と「二値化後の分類指標」の
    分離に対応する。各メソッドは状態を持たず引数だけで計算する
    (calc_r2_auc.py の AUCorR2Calculation と同じ作法)。
    """

    def regression_metrics(self, y_true, y_pred, threshold, band_frac):
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

    def detection_metrics_continuous(self, y_true, y_score, threshold):
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

    def detection_metrics_binary(self, y_true, y_pred, threshold):
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

    def mean_se(self, arr):
        """平均と標準誤差 (nan を除外)。"""
        arr = np.asarray(arr, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return float("nan"), float("nan")
        if len(arr) == 1:
            return float(arr[0]), 0.0
        return float(np.mean(arr)), float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
