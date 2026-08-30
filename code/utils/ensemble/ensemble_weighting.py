import numpy as np


class EnsembleWeighting:
    """
    アンサンブルのモデル重み決定と予測統合をまとめたクラス。
    2026-06-12 計画 Phase 0 の「重みの決め方を選択式にする (リーク対策)」に対応する。
    どの戦略を使うかは呼び出し側 (run スクリプト) が持ち、ここでは戦略名を
    受け取って計算するだけにする。
    """

    def compute_weights(self, strategy, enabled_specs, errors_for_weight):
        """
        戦略に応じてモデル重みを返す (合計 1 に正規化)。
          strategy          : "simple" / "inner_holdout" / "val_fold_legacy"
          enabled_specs     : 有効なモデル spec のリスト (key を持つ)
          errors_for_weight : key -> 誤差 (1 - R2)。simple では使わない。
        """
        keys = [s["key"] for s in enabled_specs]
        n = len(keys)

        if strategy == "simple":
            return {k: 1.0 / n for k in keys}

        # inner_holdout / val_fold_legacy : 誤差の逆数で重み付け
        weights = []
        for k in keys:
            err = errors_for_weight.get(k, np.nan)
            if err is None or np.isinf(err) or np.isnan(err):
                weights.append(1e-6)
            elif err <= 0:
                # A perfect or numerically-above-one R2 should receive the
                # largest weight, not the smallest one.
                weights.append(1.0 / 1e-6)
            else:
                weights.append(1.0 / err)
        weights = np.array(weights, dtype=float)
        if weights.sum() == 0:
            weights = np.ones(n) / n
        else:
            weights = weights / weights.sum()
        return {k: float(w) for k, w in zip(keys, weights)}

    def combine_predictions(self, preds_by_key, weights, combine):
        """preds_by_key: key -> 1D 予測配列。weights: key -> 重み。"""
        keys = list(preds_by_key.keys())
        stacked = np.stack([preds_by_key[k] for k in keys], axis=0)  # (n_models, n_samples)
        if combine == "min":
            return np.min(stacked, axis=0)
        if combine == "max":
            return np.max(stacked, axis=0)
        w = np.array([weights[k] for k in keys], dtype=float).reshape(-1, 1)
        return np.sum(stacked * w, axis=0)
