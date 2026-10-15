import os
import textwrap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter


# Fixed comparisons, independent of the metric values or model ranking.
BAR_COMPARISON_GROUPS = {
    "legacy4": ("RandomForest", "Conformer", "AlexNet", "Original three performance"),
    "addition4": ("Original three MSE", "Original three + HGB",
                  "Original three + ExtraTrees", "Original three + HGB + ExtraTrees"),
    "integration5": ("Original three MSE", "HGB frequency34", "ExtraTrees frequency34",
                     "Ensemble simple equal", "Original three + HGB + ExtraTrees"),
}


def _safe_stem(text, max_len=32):
    """Return a short Windows-friendly file stem."""
    aliases = {
        "RandomForest": "randomforest",
        "AlexNet": "alexnet",
        "Conformer": "conformer",
        "ROC-AUC (continuous)": "roc_auc_cont",
        "PR-AUC (continuous)": "pr_auc_cont",
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

    def plot_noise_trends(self, run_paths, output_dir, **kwargs):
        """保存済みの各ノイズ条件を読み、性能の折れ線グラフを更新する。"""
        from utils.plotting.noise_trend_plots import plot_noise_trends_from_runs

        return plot_noise_trends_from_runs(run_paths, output_dir, **kwargs)

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
        """Save all models, fixed comparison subsets, and explicitly marked zooms.

        Subsets reuse the exact input means, errors and colors. A comparison is
        omitted when any of its required models is absent; no ranking is used.
        """
        labels = list(labels)
        values = np.asarray(values, dtype=float)
        errors = (np.full(len(labels), np.nan) if errors is None
                  else np.asarray(errors, dtype=float))
        if values.shape != (len(labels),) or errors.shape != values.shape:
            raise ValueError("Bar labels, values and errors must have matching lengths")
        if np.any(np.isfinite(errors) & (errors < 0)):
            raise ValueError("Bar error sizes must be nonnegative")
        if not labels:
            return []
        palette = ['c', 'cadetblue', 'skyblue', 'dodgerblue', 'steelblue', 'lightblue']
        colors = [palette[i % len(palette)] for i in range(len(labels))]
        views = [("", list(range(len(labels))))]
        for name, selected in BAR_COMPARISON_GROUPS.items():
            if all(labels.count(label) == 1 for label in selected):
                views.append((name, [labels.index(label) for label in selected]))
        files = []
        for name, indices in views:
            selected_values, selected_errors = values[indices], errors[indices]
            limits = self._bar_limits(selected_values, selected_errors)
            for zoom, ylim in ((False, limits[0]), (True, limits[1])):
                if ylim is None:
                    continue
                files.append(self._plot_bar_view(
                    metric_name, [labels[i] for i in indices], selected_values, selected_errors,
                    [colors[i] for i in indices], epochs, save_path, snr_value, name, zoom, ylim))
        return files

    @staticmethod
    def _bar_limits(values, errors):
        finite = np.isfinite(values)
        if not finite.any():
            return (0., 1.05), None
        error_sizes = np.where(np.isfinite(errors), errors, 0.)[finite]
        low = float(np.min(values[finite] - error_sizes))
        high = float(np.max(values[finite] + error_sizes))
        span = max(high - low, max(abs(low), abs(high)) * .002, .002)
        full = (min(0., low - .05 * span) if low < 0 else 0., max(1.05, high + .05 * span))
        zoom_low = low - .15 * span
        if low >= 0:
            zoom_low = max(0., zoom_low)
        zoom = (zoom_low, high + .65 * span)
        # A broad or undefined metric range does not benefit from a second view.
        if zoom[1] - zoom[0] >= .85 * (full[1] - full[0]):
            zoom = None
        return full, zoom

    def _plot_bar_view(self, metric_name, labels, values, errors, colors, epochs,
                       save_path, snr_value, view_name, zoom, ylim):
        axis_fontsize = 23
        value_fontsize = 30
        display_label_aliases = {
            "RandomForest": "Random\nForest",
            "Conformer": "Conformer",
            "HGB frequency34": "HGB34",
            "ExtraTrees frequency34": "ExtraTrees\n34",
            "Original three MSE": "Original3\nMSE",
            "Original three performance": "Original3\nperf",
            "Original three + HGB": "Original3\n+ HGB",
            "Original three + ExtraTrees": "Original3\n+ ET",
            "Original three + HGB + ExtraTrees": "Original3\n+ HGB + ET",
            "Ensemble simple equal": "Equal\nmean",
        }
        display_labels = [display_label_aliases.get(label, textwrap.fill(
            str(label), width=12, break_long_words=True, break_on_hyphens=False)) for label in labels]
        label_lines = max((label.count("\n")+1 for label in display_labels), default=1)
        fig, ax = plt.subplots(figsize=(max(9, 1.4 * len(labels)), 6.6 + .35*(label_lines-1)))
        positions = np.arange(len(labels))
        error_sizes = np.where(np.isfinite(errors), errors, 0.)
        ax.bar(positions, values, color=colors,
               yerr=error_sizes if np.any(np.isfinite(errors)) else None, capsize=5, width=0.5)
        ax.set_ylim(*ylim)
        display_metric_name = metric_name
        if metric_name == "R2 Score":
            display_metric_name = "R\u00b2 Score"
        ax.set_ylabel(display_metric_name, fontsize=axis_fontsize)
        ax.set_xticks(positions, display_labels, fontsize=axis_fontsize,
                      rotation=0, ha="center", multialignment="center")
        ax.tick_params(axis="x", pad=9)
        ax.tick_params(axis="y", labelsize=18)
        if zoom:
            ax.text(0., 1.04, "Zoomed y-axis", transform=ax.transAxes, fontsize=17)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            formatter = ScalarFormatter(useOffset=False)
            formatter.set_scientific(False)
            ax.yaxis.set_major_formatter(formatter)
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
            if ylim[0] > 0:
                for x in (0., 1.):
                    ax.plot([x-.009, x+.009], [-.012, .012], transform=ax.transAxes,
                            color="black", linewidth=1.5, clip_on=False)
        annotations = []
        for i, v in enumerate(values):
            if np.isfinite(v):
                err = errors[i]
                if not np.isfinite(err):
                    text = f'{v:.4f}'
                else:
                    separator = '\n' if zoom else ' '
                    text = f'{v:.4f}{separator}\u00b1 {err:.4f}'
                if zoom:
                    annotation = ax.annotate(text, (i, v + error_sizes[i]), xytext=(0, 10),
                                             textcoords="offset points", ha="center", va="bottom",
                                             fontsize=value_fontsize, linespacing=1.)
                else:
                    annotation = ax.text(i, .035, text, transform=ax.get_xaxis_transform(),
                                         ha="center", va="bottom", fontsize=value_fontsize,
                                         color="black", rotation=90)
                annotations.append((i, annotation))
        out_dir = os.path.join(save_path, "bar", view_name)
        safe = _safe_stem(metric_name)
        plt.tight_layout()
        # Keep large labels; give them more horizontal space if the active font
        # is wider than expected. Text bounds are measured after wrapping.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bounds = [label.get_window_extent(renderer) for label in ax.get_xticklabels()]
        widths = [bound.width for bound in bounds]
        for i, annotation in annotations:
            widths[i] = max(widths[i], annotation.get_window_extent(renderer).width)
        required_scale = max((
            (widths[i]/2 + widths[i+1]/2 + 18) / ((right.x0+right.x1-left.x0-left.x1)/2)
            for i, (left, right) in enumerate(zip(bounds, bounds[1:]))
        ), default=1.)
        if required_scale > 1:
            fig.set_size_inches(fig.get_figwidth()*required_scale*1.04, fig.get_figheight())
            plt.tight_layout()
        snr_stem = _safe_stem(snr_value, max_len=16)
        suffix = "_zoom" if zoom else ""
        path = os.path.join(out_dir, f'{safe}_ep{epochs}_{snr_stem}{suffix}.png')
        try:
            _save_current_figure(path)
        finally:
            plt.close(fig)
        return path

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
                                threshold, save_path, snr_value, fold,
                                model_key="model", model_label=None):
        """1秒chunk予測の回帰散布図 + 閾値 + 100%分類閾値線。"""
        y_val = np.asarray(y_val).ravel()
        ensemble_pred = np.asarray(ensemble_pred).ravel()

        plt.figure(figsize=(12, 9))
        plt.scatter(y_val, ensemble_pred, label='Data', alpha=0.6)
        plt.plot([min(y_all), max(y_all)], [min(y_all), max(y_all)], 'r--')
        plt.xlabel('True Heat Flux MW/m²', fontsize=40)
        plt.ylabel('Predicted Heat Flux MW/m²', fontsize=40)
        if model_label:
            plt.title(str(model_label), fontsize=28)

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
        model_stem = _safe_stem(model_key, max_len=32)
        _save_current_figure(os.path.join(
            out_dir, f'scatter_{model_stem}_{snr_stem}_f{fold}.png'))
        plt.close()
