import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def _safe_stem(text, max_len=32):
    """Return a short Windows-friendly file stem."""
    aliases = {
        "RandomForest": "rf",
        "AlexNet": "alexnet",
        "CNN+Tf (AttnPool)": "cnntf_v1",
        "CNN+Tf (GAP)": "cnntf_v2",
        "ROC-AUC (continuous)": "roc_auc_cont",
        "R2 Score": "r2",
    }
    text = aliases.get(text, str(text))
    safe = []
    for ch in text:
        if ch.isalnum() or ch in "-_.":
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_")[:max_len] or "plot"


class RegressionPlotter:
    """
    熱流束回帰・アンサンブル評価まわりの作図をまとめたクラス。
      - plot_loss_history     : 学習損失曲線
      - plot_bar              : モデル別指標の棒グラフ (R2 / 連続スコア AUC など)
      - plot_regression_scatter : 予測散布図 + 閾値線 + 100% 分類閾値線
    matplotlib の rcParams (フォント等) は呼び出し側で設定する想定。
    """

    def plot_loss_history(self, history, epochs, label, fold, save_path, snr_value):
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
        label_stem = _safe_stem(label)
        snr_stem = _safe_stem(snr_value, max_len=16)
        plt.savefig(os.path.join(out_dir, f'loss_{label_stem}_f{fold}_{snr_stem}.png'))
        plt.close()

    def plot_bar(self, metric_name, labels, values, errors, epochs, save_path, snr_value):
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
        safe = _safe_stem(metric_name)
        plt.tight_layout()
        snr_stem = _safe_stem(snr_value, max_len=16)
        plt.savefig(os.path.join(out_dir, f'{safe}_ep{epochs}_{snr_stem}.png'))
        plt.close()

    def plot_regression_scatter(self, y_val, ensemble_pred, y_all, metrics_ens,
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
        snr_stem = _safe_stem(snr_value, max_len=16)
        plt.savefig(os.path.join(out_dir, f'scatter_{snr_stem}_f{fold}.png'))
        plt.close()
