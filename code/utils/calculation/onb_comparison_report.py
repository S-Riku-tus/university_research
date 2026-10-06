"""One source-day comparison table for saved studies and the normal ONB run.

This module consumes already evaluated rows; it never fits or selects models.
The historical combined-threshold summaries and raw-score AUCs are retained.
"""
import csv
import json
from pathlib import Path

import numpy as np

from utils.experiment.run_helpers import makedirs, open_text

ERRORS = ["rmse_all", "mae_all", "rmse_pre_onb", "rmse_onb", "mae_onb", "rmse_high",
          "mae_high", "bias_all", "bias_pre_onb", "bias_onb", "bias_high", "q100", "g100", "onb_threshold"]
RATES = ["recall", "precision", "accuracy", "f1", "fpr"]
SCORES = ["r2", "r2_high", "roc_auc_cont", "pr_auc_cont"]
COUNTS = ["n", "n_pre_onb", "n_post_onb", "n_onb", "n_source_wavs", "tp", "fp", "tn", "fn"]
POOLED = "pooled_source_day_thresholds"


def noise_key(value):
    value = str(value)
    if value in {"clean", "no_noise", "heatflux_no_noise"}:
        return "clean"
    return value.split("SNR=", 1)[-1] if "SNR=" in value else value


def _number(value):
    return float(value) if value not in (None, "") else np.nan


def comparison_tables(rows, protocol):
    normalized = []
    for row in rows:
        if row.get("threshold_basis") != "source-day":
            raise ValueError("Main ONB tables require source-day thresholds")
        if row.get("heat_flux_unit") != "W/m2":
            raise ValueError("Main ONB tables expect W/m2 inputs")
        item = {"seed": int(row["seed"]), "fold": int(row["fold"]), "noise": noise_key(row["noise"]),
                "model_key": row["model_key"], "model_label": row.get("model_label", row["model_key"]),
                "source_day": row["source_day"], "threshold_basis": "source-day", "heat_flux_unit": "kW/m2",
                "source_result": row.get("source_result", "")}
        item.update({key: _number(row.get(key)) for key in COUNTS+RATES+SCORES})
        item.update({key+"_kW_m2": _number(row.get(key))/1000 for key in ERRORS})
        normalized.append(item)
    lookup = {}
    for row in normalized:
        key = tuple(row[k] for k in ["seed", "fold", "noise", "source_day", "model_key"])
        if key in lookup:
            raise ValueError(f"Duplicate ONB comparison row: {key}")
        lookup[key] = row
    required = set(protocol["model_keys"]) | {"ensemble__"+k for k in protocol["ensemble_names"]}
    units = {key[:4] for key in lookup}
    for unit in units:
        absent = required - {key[-1] for key in lookup if key[:4] == unit}
        if absent:
            raise ValueError(f"Missing comparison methods for {unit}: {sorted(absent)}")
    changes, degradation, overview = [], [], []
    for row in normalized:
        unit = tuple(row[k] for k in ["seed", "fold", "noise", "source_day"])
        for baseline in protocol["comparison_baselines"]:
            if row["model_key"] == baseline:
                continue
            before = lookup[(*unit, baseline)]
            changes.append({**{k: row[k] for k in ["seed", "fold", "noise", "source_day", "model_key"]},
                "baseline_key": baseline,
                **{"delta_"+k+"_kW_m2": row[k+"_kW_m2"]-before[k+"_kW_m2"] for k in ERRORS if k != "onb_threshold"},
                **{"delta_"+k+"_pp": 100*(row[k]-before[k]) for k in RATES},
                **{"delta_"+k: row[k]-before[k] for k in SCORES+["fp", "fn"]}})
        clean = lookup.get((unit[0], unit[1], "clean", unit[3], row["model_key"]))
        if row["noise"] != "clean" and clean is not None:
            degradation.append({**{k: row[k] for k in ["seed", "fold", "noise", "source_day", "model_key"]},
                "clean_rmse_kW_m2": clean["rmse_all_kW_m2"], "noisy_rmse_kW_m2": row["rmse_all_kW_m2"],
                "increase_rmse_from_clean_kW_m2": row["rmse_all_kW_m2"]-clean["rmse_all_kW_m2"],
                "increase_near_onb_rmse_from_clean_kW_m2": row["rmse_onb_kW_m2"]-clean["rmse_onb_kW_m2"],
                "delta_fn_from_clean": row["fn"]-clean["fn"], "delta_fp_from_clean": row["fp"]-clean["fp"]})
        if row["noise"] == "clean":
            noisy = [lookup.get((unit[0], unit[1], n, unit[3], row["model_key"])) for n in protocol["noise_conditions"] if n != "clean"]
            noisy = [r for r in noisy if r is not None]
            overview.append({**{k: row[k] for k in ["seed", "fold", "source_day", "model_key"]},
                "clean_rmse_kW_m2": row["rmse_all_kW_m2"], "noise_conditions_observed": len(noisy),
                "mean_noise_rmse_kW_m2": float(np.mean([r["rmse_all_kW_m2"] for r in noisy])) if noisy else np.nan,
                "max_observed_fp": max([r["fp"] for r in noisy]+[row["fp"]]),
                "complete_six_noise_mean": len(noisy) == len(protocol["noise_conditions"])-1})
    return {"main_metrics.csv": normalized, "main_deltas.csv": changes,
            "noise_degradation.csv": degradation, "noise_overview.csv": overview,
            "q100_by_source_day.csv": [r for r in normalized if r["source_day"] != POOLED]}


def _csv(path, rows):
    if not rows:
        return
    with open_text(path, "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: "" if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v
                         for k, v in r.items()} for r in rows)


def _json(path, value):
    with open_text(path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def write_comparison_report(directory, rows, protocol, conditions, weights, verification):
    tables = comparison_tables(rows, protocol)
    directory = Path(directory)
    makedirs(directory)
    for name, values in tables.items():
        _csv(directory/name, values)
    _csv(directory/"ensemble_weights.csv", weights)
    _json(directory/"evaluation_protocol.json", protocol)
    _json(directory/"run_conditions.json", conditions)
    _json(directory/"verification.json", verification)
    lines = ["# ONBの主比較表", "", f"評価定義：`{protocol['protocol_id']}`。誤差・熱流束はkW/m²、率は%で表示。",
        "", "対象："+verification.get("scope", "saved evaluation results"),
        "", "日別ONB閾値による主表。元3MSEを基準とし、ET4・ET単体も主要対照に含める。",
        "q100/g100は日別に報告し、複数日のq100を平均して全体の検知点とはしない。",
        "連続ROC/PR-AUCは従来通り予測熱流束の生スコア。日別閾値が異なる全体集計では補助的な順位指標として読む。",
        "", f"出力確認：**{verification['status']}**。性能の改善方向は合否条件に含めない。",
        "研究条件一致・実装確認の区別、学習/評価/再読込の実件数は[verification.json](verification.json)と[run_conditions.json](run_conditions.json)。",
        "", "## 全体の回帰・ONB判定（各seed内で同じ評価chunk）", "",
        "| seed | 条件 | モデル | RMSE | MAE | 近傍RMSE | FN | FP | Recall | F1 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    metric_rows = tables["main_metrics.csv"]
    for r in metric_rows:
        if r["source_day"] == POOLED and r["noise"] in {"clean", "-20"}:
            lines.append(f"| {r['seed']} | {r['noise']} | {r['model_label']} | {r['rmse_all_kW_m2']:.2f} | {r['mae_all_kW_m2']:.2f} | {r['rmse_onb_kW_m2']:.2f} | {r['fn']:.0f} | {r['fp']:.0f} | {100*r['recall']:.2f}% | {r['f1']:.5f} |")
    lines += ["", "## 日別q100・g100", "", "| seed | 条件 | 出典日 | モデル | ONB閾値 | q100 | g100 | FN | FP |",
              "|---|---|---|---|---:|---:|---:|---:|---:|"]
    main_keys = set(protocol["comparison_baselines"]) | {protocol["primary_model_key"], "ensemble__original3_hgb"}
    fmt = lambda value: f"{value:.2f}" if np.isfinite(value) else "未到達"
    for r in metric_rows:
        if r["source_day"] != POOLED and r["noise"] in {"clean", "-20"} and r["model_key"] in main_keys:
            lines.append(f"| {r['seed']} | {r['noise']} | {r['source_day']} | {r['model_label']} | {r['onb_threshold_kW_m2']:.2f} | {fmt(r['q100_kW_m2'])} | {fmt(r['g100_kW_m2'])} | {r['fn']:.0f} | {r['fp']:.0f} |")
    lines += ["", "## 強雑音の絶対誤差とcleanからの劣化量", "",
              "| seed | モデル | clean RMSE | −20 RMSE | RMSE増加量 |", "|---|---|---:|---:|---:|"]
    for r in tables["noise_degradation.csv"]:
        if r["source_day"] == POOLED and r["noise"] == "-20" and r["model_key"] in main_keys:
            lines.append(f"| {r['seed']} | {r['model_key']} | {r['clean_rmse_kW_m2']:.2f} | {r['noisy_rmse_kW_m2']:.2f} | {r['increase_rmse_from_clean_kW_m2']:.2f} |")
    lines += ["", "## 全条件の出力", "",
              "- [全7条件・全モデル・日別/全体の主指標](main_metrics.csv)",
              "- [元3MSE・ET4・ET単体との指標差](main_deltas.csv)",
              "- [自身のcleanからの劣化量](noise_degradation.csv)",
              "- [6雑音の平均と最大観測FP](noise_overview.csv)",
              "- [日別q100/g100](q100_by_source_day.csv)",
              "- [clean OOF由来の統合重み](ensemble_weights.csv)", "",
              "既知WAV内の未使用chunk、反復使用した研究データの記述結果。",
              "seed・雑音加工を独立実験として合算しない。q100は有限標本の到達段階で、時間遅延や将来の100%保証ではない。",
              "従来の平均ONB閾値によるmetrics_summary CSV・散布図は補助出力として別に保存される。", ""]
    with open_text(directory/"main_comparison.md", "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    return directory/"main_comparison.md"
