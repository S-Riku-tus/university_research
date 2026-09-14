"""保存済み指標から、モデルごとのノイズ強度と性能の関係を描く。学習は行わない。"""

import csv
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRIC_LABELS = {
    "r2": "R² score",
    "roc_auc_cont": "ROC-AUC (continuous score)",
    "auc_binary": "AUC (binary prediction)",
}
MODEL_LABELS = {
    "rf": "RandomForest",
    "cnntf_v2_gap": "CNN+Transformer",
    "alexnet": "AlexNet",
}
STRATEGY_LABELS = {
    "simple_equal": "equal mean",
    "prediction_max": "prediction max",
    "inner_holdout": "inner holdout",
    "val_fold_legacy": "validation-fold legacy",
}


def _path(path):
    """Windowsの長いパスにも対応する。"""
    value = os.path.abspath(os.fspath(path))
    if os.name == "nt" and not value.startswith("\\\\?\\"):
        value = "\\\\?\\" + value
    return Path(value)


def _json(path):
    with _path(path).open(encoding="utf-8-sig") as source:
        return json.load(source)


def _csv(path):
    with _path(path).open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _noise_value(value):
    """フォルダ名と数値表記を、比較に使う共通のSNR表記へ揃える。"""
    value = str(value)
    if value in ("heatflux_no_noise", "no_noise", "No_noise"):
        return "no_noise"
    if "SNR=" in value:
        value = value.split("SNR=", 1)[1]
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"ノイズ条件は有限のSNRかno_noiseにしてください: {value}")
    return f"{number:g}"


def ordered_noise_values(values):
    """無雑音、0、負のSNRの順に並べ、右ほどノイズを強くする。"""
    unique = set(_noise_value(value) for value in values)
    return sorted(unique, key=lambda value: (0, 0) if value == "no_noise" else (1, -float(value)))


def collect_noise_trend_rows(run_paths, *, noise_order, model_keys,
                             ensemble_strategy_names="primary", metrics=("r2", "roc_auc_cont"),
                             evaluation_units=("chunk", "wav"), wav_aggregation="median",
                             expected_run_hash=None):
    """同一実験・周波数・学習設定の保存結果を、評価単位を分けて集める。

    chunkは従来のfold平均と標準誤差、WAVは全OOFを集めた集約値を使う。
    WAVの一点推定へchunkの標準誤差を付けることはしない。
    未実行条件はNaNとして残し、欠測箇所を線で結ばない。
    """
    unknown_metrics = set(metrics) - set(METRIC_LABELS)
    unknown_units = set(evaluation_units) - {"chunk", "wav"}
    if unknown_metrics or unknown_units:
        raise ValueError(f"未対応の作図設定です: metrics={unknown_metrics}, units={unknown_units}")
    if wav_aggregation not in ("mean", "median", "p90"):
        raise ValueError(f"未対応のWAV集約方法です: {wav_aggregation}")
    noise_values = ordered_noise_values(noise_order)
    loaded = {}
    reference = None
    for directory in run_paths:
        directory = _path(directory)
        manifest_path = directory / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = _json(manifest_path)
        completion = {}
        if manifest.get("execution_schema_version", 1) >= 2:
            completed_path = directory / "completed.json"
            if not completed_path.is_file():
                continue
            completion = _json(completed_path)
            if completion.get("run_hash") != manifest.get("run_hash"):
                continue
        if expected_run_hash is not None and manifest.get("run_hash") != expected_run_hash:
            continue
        dataset = manifest["dataset"]
        noise = _noise_value(dataset["snr_value"])
        if noise not in noise_values:
            continue
        context = manifest.get("learning_context", {})
        policy = manifest["validation_config"].get("learning_policy", {
            "split_mode": "within_day", "training_noise": "matched"})
        signature = (dataset["experiment_name"], dataset["max_freq_hz"],
                     manifest.get("run_hash"), manifest["run_dir"],
                     json.dumps(policy, sort_keys=True), tuple(context.get("training_experiments", [])),
                     tuple(completion.get("fit_ids", [])) if policy["training_noise"] == "clean_only" else ())
        if reference is None:
            reference = (signature, manifest)
        elif signature != reference[0]:
            raise ValueError("異なる実験日・周波数・学習設定を一つのノイズ曲線へ混在させられません。")
        if noise in loaded:
            raise ValueError(f"同じノイズ条件が重複しています: {noise}")
        loaded[noise] = (directory, manifest)
    if not loaded:
        return []

    manifest = reference[1]
    policy = manifest["validation_config"].get("learning_policy", {
        "split_mode": "within_day", "training_noise": "matched"})
    held_out_day = policy["split_mode"] == "leave_one_day_out"
    ensemble = manifest["validation_config"].get("ensemble", {})
    plans = ensemble.get("resolved_strategy_plan", [])
    available = {plan["name"]: plan for plan in plans}
    if ensemble_strategy_names == "primary":
        selected = [ensemble.get("primary_strategy")]
    elif ensemble_strategy_names == "all":
        selected = list(available)
    elif isinstance(ensemble_strategy_names, (list, tuple)):
        selected = list(ensemble_strategy_names)
    else:
        raise ValueError("ensemble_strategy_namesはprimary、all、または方式名のリストです。")
    selected = [name for name in selected if name is not None]
    if set(selected) - set(available):
        raise ValueError(f"実行されていないアンサンブル方式です: {set(selected) - set(available)}")
    # 単体モデルだけの実行にも対応する。
    if not selected:
        selected = [None]
    labels = {spec["key"]: spec["label"] for spec in manifest["run_specs"]}
    if set(model_keys) - set(labels):
        raise ValueError("要求された単体モデルが保存manifestにありません。")
    labels.update({plan["result_key"]: plan["label"] for plan in plans})
    cached = {}
    for noise, (directory, run_manifest) in loaded.items():
        snr = str(run_manifest["dataset"]["snr_value"])
        chunk_path = directory / f"metrics_summary_{snr}.csv"
        wav_path = directory / "wav_eval" / f"wav_metrics_{snr}.csv"
        eval_path = directory / "wav_eval" / f"evaluation_manifest_{snr}.json"
        cached[noise] = {
            "chunk": {row["model"]: row for row in _csv(chunk_path)} if chunk_path.is_file() else {},
            "wav": {row["model_key"]: row for row in _csv(wav_path)
                    if row["aggregation"] == wav_aggregation} if wav_path.is_file() else {},
            "chunk_path": chunk_path,
            "wav_path": wav_path,
            "chunk_threshold": run_manifest["dataset"].get("threshold"),
            "wav_threshold": _json(eval_path).get("threshold") if eval_path.is_file() else None,
        }

    rows = []
    for strategy in selected:
        keys = list(model_keys)
        if strategy is not None:
            keys.append(available[strategy]["result_key"])
        for unit in evaluation_units:
            # WAV評価を無効にした実行では、空のWAVグラフを作らない。
            if not any(data[unit] for data in cached.values()):
                continue
            thresholds = {data[f"{unit}_threshold"] for data in cached.values() if data[unit]}
            if len(thresholds) > 1:
                raise ValueError(f"{unit}評価のONB閾値がノイズ条件間で一致しません。")
            for metric in metrics:
                for key in keys:
                    for noise in noise_values:
                        data = cached.get(noise, {})
                        record = data.get(unit, {}).get(labels[key] if unit == "chunk" else key, {})
                        value = _number(record.get(f"{metric}_mean" if unit == "chunk" else metric))
                        se = _number(record.get(f"{metric}_se")) if unit == "chunk" and not held_out_day else float("nan")
                        rows.append({
                            "experiment": manifest["dataset"]["experiment_name"],
                            "maxfreq": manifest["dataset"]["max_freq_hz"],
                            "run_hash": manifest.get("run_hash", ""),
                            "split_mode": policy["split_mode"],
                            "training_noise": policy["training_noise"],
                            "strategy": strategy or "single_models",
                            "evaluation_unit": unit,
                            "aggregation": ("held_out_day" if held_out_day else "fold_mean") if unit == "chunk" else wav_aggregation,
                            "metric": metric, "model_key": key,
                            "model_label": MODEL_LABELS.get(key, "Ensemble" if key.startswith("ensemble__") else labels[key]),
                            "noise": noise, "value": value, "standard_error": se,
                            "error_bar_definition": "fold_standard_error" if unit == "chunk" and not held_out_day else "not_estimated",
                            "threshold": data.get(f"{unit}_threshold", ""),
                            "n_wavs": record.get("n_wavs", "") if unit == "wav" else "",
                            "claim_safe": record.get("claim_safe", "") if unit == "wav" else "",
                            "source_csv": str(data.get(f"{unit}_path", "")),
                        })
    return rows


def plot_noise_trends_from_runs(run_paths, output_dir, *, formats=("png", "pdf"), **kwargs):
    """各方式で単体3モデル＋アンサンブルの曲線と、その数値CSVを保存する。"""
    if not formats or set(formats) - {"png", "pdf", "svg"}:
        raise ValueError("画像形式はpng、pdf、svgから選んでください。")
    rows = collect_noise_trend_rows(run_paths, **kwargs)
    if not rows:
        return []
    groups = {}
    for row in rows:
        groups.setdefault((row["strategy"], row["evaluation_unit"], row["metric"]), []).append(row)
    artifacts = []
    styles = [("#12bfc4", "o", "-"), ("#469ca8", "s", "--"),
              ("#65bdeb", "^", ":"), ("#2588fa", "D", "-.")]
    # 既存の大きな散布図用設定に影響されないよう、この図だけの書式を指定する。
    with plt.rc_context({"font.family": "DejaVu Serif", "font.size": 11,
                         "axes.labelsize": 13, "xtick.labelsize": 10,
                         "ytick.labelsize": 11, "legend.fontsize": 10}):
        for (strategy, unit, metric), group in groups.items():
            if not any(math.isfinite(row["value"]) for row in group):
                continue
            directory = _path(output_dir) / strategy
            directory.mkdir(parents=True, exist_ok=True)
            stem = f"{unit}_{group[0]['aggregation']}_{metric}"
            csv_path = directory / f"{stem}.csv"
            with csv_path.open("w", encoding="utf-8-sig", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=list(group[0]))
                writer.writeheader()
                writer.writerows(group)
            noises = ordered_noise_values(row["noise"] for row in group)
            model_order = list(dict.fromkeys(row["model_key"] for row in group))
            fig, ax = plt.subplots(figsize=(7.0, 4.8))
            try:
                for index, model in enumerate(model_order):
                    by_noise = {row["noise"]: row for row in group if row["model_key"] == model}
                    values = np.asarray([by_noise[noise]["value"] for noise in noises])
                    errors = np.asarray([by_noise[noise]["standard_error"] for noise in noises])
                    color, marker, line = styles[index % len(styles)]
                    ax.errorbar(np.arange(len(noises)), values,
                                yerr=errors if unit == "chunk" and np.any(np.isfinite(errors)) else None,
                                color=color, marker=marker, linestyle=line, markersize=4,
                                linewidth=1.5, capsize=3, label=by_noise[noises[0]]["model_label"])
                ax.set_xticks(np.arange(len(noises)))
                ax.set_xticklabels(["No_noise" if noise == "no_noise" else noise for noise in noises])
                ax.set_xlabel("Noise level (reference SNR [dB])")
                ax.set_ylabel(METRIC_LABELS[metric])
                unit_label = "Chunk: fold mean ± SE" if unit == "chunk" else f"WAV: pooled OOF {group[0]['aggregation']}"
                if group[0]["split_mode"] == "leave_one_day_out":
                    unit_label = "Chunk: held-out day" if unit == "chunk" else f"WAV: held-out day {group[0]['aggregation']}"
                ensemble_label = STRATEGY_LABELS.get(strategy, strategy)
                policy_label = "clean train" if group[0]["training_noise"] == "clean_only" else "matched-noise train"
                ax.set_title(f"{group[0]['experiment']} | {group[0]['maxfreq']} | {policy_label}\n{unit_label} | {ensemble_label}", fontsize=10)
                # 負のR²や誤差棒も含め、自動スケールで全点を表示する。
                ax.margins(x=0.08, y=0.12)
                ax.tick_params(direction="in")
                ax.legend(loc="best", framealpha=1, edgecolor="black")
                fig.tight_layout()
                paths = []
                for extension in formats:
                    path = directory / f"{stem}.{extension}"
                    fig.savefig(path, dpi=220, bbox_inches="tight")
                    paths.append(str(path))
                artifacts.append({"strategy": strategy, "unit": unit, "metric": metric,
                                  "csv": str(csv_path), "figures": paths})
            finally:
                plt.close(fig)
    return artifacts
