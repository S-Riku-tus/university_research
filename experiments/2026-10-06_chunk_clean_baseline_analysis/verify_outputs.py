"""Independently verify saved comparisons using raw prediction records."""

import csv
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent


def long_path(path):
    value = str(path.resolve())
    return Path("\\\\?\\" + value) if os.name == "nt" else path


def read_csv(path):
    with long_path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(long_path(path).read_text(encoding="utf-8-sig"))


def day(row):
    return "2025.06.11_0.3_2" if row["source_wav_id"].startswith("20250611::") else "2025.06.18_0.3_3"


def main():
    scope = read_json(OUTPUT / "run_scope.json")
    metric_rows = read_csv(OUTPUT / "full_metrics.csv")
    metric_index = {(row["run"], row["unit"], row["noise"], row["scope"], row["model"]): row for row in metric_rows}
    ep_rows = read_csv(OUTPUT / "endpoints.csv")
    endpoint_index = {(row["run"], row["unit"], row["noise"], row["scope"], row["model"]): row for row in ep_rows}
    checked_metrics = checked_endpoints = 0
    for tag, relative in scope["runs"].items():
        run = ROOT / relative
        for noise in ("clean", "0", "-4", "-8", "-12", "-16", "-20"):
            dirname = "heatflux_no_noise" if noise == "clean" else "heatflux_reference_SNR=" + noise
            suffix = "no_noise" if noise == "clean" else noise
            rows = read_csv(run / "maxfreq=3kHz" / dirname / "fold_pred" / f"pred_f1_{suffix}.csv")
            for source_day in ("two_day", *scope["thresholds_W_m2"]):
                selected = rows if source_day == "two_day" else [row for row in rows if day(row) == source_day]
                y = np.array([float(row["y_true"]) for row in selected])
                thresholds = np.array([scope["thresholds_W_m2"][day(row)] for row in selected])
                for model in ("randomforest", "conformer", "alexnet", "ensemble__performance_kfold", "ensemble__simple_equal"):
                    p = np.array([float(row[model]) for row in selected])
                    saved = metric_index[tag, "outer", noise, source_day, model]
                    assert np.isclose(float(saved["rmse"]), np.linalg.norm(p - y) / np.sqrt(len(y)))
                    assert int(saved["fp"]) == sum((y < thresholds) & (p >= thresholds))
                    assert int(saved["fn"]) == sum((y >= thresholds) & (p < thresholds))
                    checked_metrics += 1
                    if source_day != "two_day":
                        negative_levels = y[p < thresholds]
                        eligible = np.unique(y) if not len(negative_levels) else np.unique(y)[np.unique(y) > negative_levels.max()]
                        q = float(eligible[0]) / 1000 if len(eligible) else np.nan
                        saved_ep = endpoint_index[tag, "outer", noise, source_day, model]
                        assert np.isclose(float(saved_ep["q100_kW_m2"]), q, equal_nan=True)
                        assert np.isclose(float(saved_ep["g100_kW_m2"]), q - scope["thresholds_W_m2"][source_day] / 1000, equal_nan=True)
                        checked_endpoints += 1
    perf_eps = [row for row in ep_rows if row["unit"] == "outer" and row["model"] == "ensemble__performance_kfold"]
    assert len(perf_eps) == 28
    for day_value in scope["thresholds_W_m2"]:
        assert len({row["q100_kW_m2"] for row in perf_eps if row["scope"] == day_value}) == 1
    oof_checks = []
    complementarity = read_csv(OUTPUT / "complementarity.csv")
    for tag, relative in scope["runs"].items():
        directory = ROOT / relative / "maxfreq=3kHz/heatflux_no_noise"
        samples = read_json(directory / "internal_validation_fold1.json")["samples"]
        wrow = next(row for row in read_csv(directory / "ensemble_weights_no_noise.csv") if row["strategy_name"] == "performance_kfold")
        keys = ("randomforest", "conformer", "alexnet")
        matrix = np.array([[row[key] for key in keys] for row in samples])
        y = np.array([row["heat_flux"] for row in samples])
        thresholds = np.array([scope["thresholds_W_m2"][day(row)] for row in samples])
        p = matrix @ np.array([float(wrow[key]) for key in keys])
        saved = metric_index[tag, "training_oof", "clean", "two_day", "ensemble__performance_kfold"]
        assert np.isclose(float(saved["rmse"]), np.linalg.norm(p - y) / np.sqrt(len(y)))
        assert int(saved["fn"]) == sum((y >= thresholds) & (p < thresholds))
        assert int(saved["fp"]) == sum((y < thresholds) & (p >= thresholds))
        common = int(sum((y >= thresholds) & (matrix.max(axis=1) < thresholds)))
        saved_common = next(row for row in complementarity if row["run"] == tag and row["unit"] == "training_oof" and row["scope"] == "two_day")
        assert common == int(saved_common["all_three_fn"])
        oof_checks.append({"run": tag, "n": len(y), "ensemble_fn": int(saved["fn"]), "all_three_fn": common})
    result = {"passed": True, "raw_outer_metric_checks": checked_metrics, "independent_q100_checks": checked_endpoints,
        "clean_oof_checks": oof_checks,
        "performance_q100_unchanged_in_all_14_day_noise_pairs": True,
        "research_model_training_executed": False}
    (OUTPUT / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
