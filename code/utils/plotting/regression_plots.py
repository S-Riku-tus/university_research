import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def _safe_stem(text, max_len=32):
    """Return a short Windows-friendly file stem."""
    aliases = {
        "RandomForest": "rf",
        "AlexNet": "alexnet",
        "Conformer": "conformer",
        "ROC-AUC (continuous)": "roc_auc_cont",
        "AUC (binary legacy)": "auc_bin",
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


def _windows_long_path(path):
    path = os.path.abspath(path)
    if os.name == "nt" and not path.startswith("\\\\?\\"):
        return "\\\\?\\" + path
    return path


def _save_current_figure(path):
    """Save the current Matplotlib figure, allowing long absolute paths on Windows."""
    save_path = _windows_long_path(path)
    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(save_path)


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
        out_dir = os.path.join(save_path, "loss")
        label_stem = _safe_stem(label)
        snr_stem = _safe_stem(snr_value, max_len=16)
        _save_current_figure(os.path.join(out_dir, f'loss_{label_stem}_f{fold}_{snr_stem}.png'))
        plt.close()

    def plot_bar(self, metric_name, labels, values, errors, epochs, save_path, snr_value):
        """モデル別の指標を棒グラフで保存 (R2 / 連続スコア ROC-AUC など)。"""
        plt.figure(figsize=(8, 6))
        colors = ['c', 'cadetblue', 'skyblue', 'dodgerblue', 'steelblue', 'lightblue']
        display_label_aliases = {
            "RandomForest": "RandomForest",
            "Conformer": "Conformer",
        }
        display_labels = [display_label_aliases.get(label, label) for label in labels]
        plt.bar(display_labels, values, color=colors[:len(labels)],
                yerr=errors, capsize=5, width=0.5)
        plt.ylim(0.0, 1.05)
        display_metric_name = metric_name
        if metric_name == "R2 Score":
            display_metric_name = "R\u00b2 Score"
            ylabel_size = 23
            xtick_size = 20
        elif metric_name.startswith("AUC"):
            display_metric_name = "AUC"
            ylabel_size = 20
            xtick_size = 19
        else:
            ylabel_size = 20
            xtick_size = 19
        plt.ylabel(display_metric_name, fontsize=ylabel_size)
        plt.xticks(fontsize=xtick_size)
        plt.yticks(fontsize=18)
        for i, v in enumerate(values):
            if not np.isnan(v):
                err = errors[i] if errors is not None and i < len(errors) else np.nan
                if np.isnan(err):
                    text = f'{v:.4f}'
                else:
                    text = f'{v:.4f} \u00b1 {err:.4f}'
                plt.text(i, 0.03, text, ha='center', va='bottom',
                         fontsize=25, color='black', rotation=90)
        out_dir = os.path.join(save_path, "bar")
        safe = _safe_stem(metric_name)
        plt.tight_layout()
        snr_stem = _safe_stem(snr_value, max_len=16)
        _save_current_figure(os.path.join(out_dir, f'{safe}_ep{epochs}_{snr_stem}.png'))
        plt.close()

    def plot_ensemble_strategy_improvements(self, comparison_rows, save_path,
                                            snr_value):
        """Plot signed deltas against the best individual model.

        Positive values always mean that the ensemble is better, including for
        error metrics where a smaller raw value is preferable.
        """
        metrics = [
            ("r2", "R²", "Δ R²"),
            ("rmse_onb", "ONB RMSE", "Improvement in RMSE"),
            ("recall", "Recall", "Δ Recall"),
            ("f1", "F1", "Δ F1"),
        ]
        strategy_aliases = {
            "simple_equal": "Equal",
            "fixed_rf90_even": "RF90 / 5 / 5",
            "fixed_rf95_even": "RF95 / 2.5 / 2.5",
            "fixed_rf98_even": "RF98 / 1 / 1",
            "fixed_rf95_cnntf": "RF95 / CNN-Tf5",
            "fixed_rf95_alex": "RF95 / Alex5",
            "prediction_max": "Prediction max",
            "inner_holdout": "Inner holdout",
        }
        by_metric = {}
        strategy_order = []
        for row in comparison_rows:
            strategy_name = str(row[0])
            metric_name = str(row[3])
            if strategy_name not in strategy_order:
                strategy_order.append(strategy_name)
            by_metric.setdefault(metric_name, {})[strategy_name] = float(row[10])

        if not comparison_rows:
            return
        labels = [strategy_aliases.get(name, name) for name in strategy_order]
        y_pos = np.arange(len(strategy_order))
        fig, axes = plt.subplots(2, 2, figsize=(17, 13))
        for ax, (metric_key, title, xlabel) in zip(axes.flat, metrics):
            values = np.asarray([
                by_metric.get(metric_key, {}).get(name, np.nan)
                for name in strategy_order
            ], dtype=float)
            colors = [
                "#2b6cb0" if np.isfinite(value) and value > 0 else "#c05640"
                for value in values
            ]
            ax.barh(y_pos, np.nan_to_num(values, nan=0.0), color=colors, alpha=0.9)
            ax.axvline(0.0, color="black", linewidth=1.2)
            ax.set_yticks(y_pos, labels=labels, fontsize=15)
            ax.invert_yaxis()
            ax.set_title(title, fontsize=22, pad=10)
            ax.set_xlabel(xlabel + " vs best single model", fontsize=17)
            ax.tick_params(axis="x", labelsize=14)
            ax.grid(axis="x", alpha=0.25)
            finite = np.abs(values[np.isfinite(values)])
            axis_scale = float(np.max(finite)) if finite.size else 0.0
            text_offset = max(axis_scale * 0.025, 1e-6)
            for index, value in enumerate(values):
                if not np.isfinite(value):
                    continue
                fits_inside = abs(value) >= axis_scale * 0.15
                if fits_inside:
                    ha = "right" if value >= 0 else "left"
                    x = value - text_offset if value >= 0 else value + text_offset
                    text_color = "white"
                else:
                    ha = "left" if value >= 0 else "right"
                    x = value + text_offset if value >= 0 else value - text_offset
                    text_color = "black"
                ax.text(x, index, f"{value:+.4g}", va="center", ha=ha,
                        fontsize=13, color=text_color)

        fig.suptitle(
            "Ensemble improvement over the best individual model\n"
            "Positive values indicate improvement",
            fontsize=24,
            y=0.99,
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
        out_dir = os.path.join(save_path, "bar")
        snr_stem = _safe_stem(snr_value, max_len=16)
        _save_current_figure(
            os.path.join(out_dir, f"ensemble_improvement_{snr_stem}.png")
        )
        plt.close(fig)

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

        out_dir = os.path.join(save_path, "scatter")
        snr_stem = _safe_stem(snr_value, max_len=16)
        _save_current_figure(os.path.join(out_dir, f'scatter_{snr_stem}_f{fold}.png'))
        plt.close()
