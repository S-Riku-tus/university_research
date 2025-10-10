import os
import numpy as np
from sklearn.metrics import roc_curve, auc, r2_score


class AUCorR2Calculation:
    def __init__(self):
        pass

    def calc_r2_score(self, y_val, y_pred, r2_scores, fold, model_name):
        """
        回帰分析の結果をプロットし、R^2スコアを表示
        """
        r2 = r2_score(y_val, y_pred)
        r2_scores.append(r2)
        print(f'Model : {model_name} | Fold : {fold} | r2 scores : {r2}')
        return 1 - r2  # 決定係数を誤差率(値が小さいほど精度が良い)として、各モデルの重み計算に用いる

    def calc_roc_curve(self, y_true_binary, y_pred_binary, auc_scores, fold, model_name):
        """
        各foldのROC曲線をプロットし、AUCを計算して保存
        """
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_binary)
        roc_variable = auc(fpr, tpr)
        auc_scores.append(roc_variable)
        print(f'Model : {model_name} | Fold : {fold} | AUC scores : {auc_scores}')