"""Recompute clean-to-noise changes from existing 2025 metric tables only."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
from utils.experiment.run_helpers import open_text

POOLED = "pooled_source_day_thresholds"
CORE = "ensemble__original3_mse"
FIVE = "ensemble__original3_hgb_extra_trees"
NOISE = ["0", "-4", "-8", "-12", "-16", "-20"]
KEYS = ["seed", "fold", "source_day", "model_key"]


def read(path):
    with open_text(path, "r", encoding="utf-8-sig") as handle:
        return pd.read_csv(handle, dtype={"noise": str})


def digest(path):
    with open_text(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def main():
    comparison_path = ROOT / "experiments/2026-10-07_onb_200epoch_result_analysis/comparison_across_runs.csv"
    comparison = read(comparison_path)
    latest = comparison.loc[comparison.seed.eq(42)].iloc[0]
    latest_path = Path(latest.source_result).parent / "onb_comparison/main_metrics.csv"
    previous_path = ROOT / "experiments/2026-10-06_onb_main_comparison_ready/saved_main_tables/main_metrics.csv"
    paths = [latest_path, previous_path]
    before_hashes = [digest(path) for path in paths]
    data = pd.concat([read(path) for path in paths], ignore_index=True)
    assert len(data) == 693 and not data.duplicated(KEYS + ["noise"]).any()
    assert set(data.seed) == {42, 43, 44}
    assert set(data.noise) == {"clean", *NOISE}
    assert len(data.model_key.unique()) == 11 and len(data.source_day.unique()) == 3
    assert data.threshold_basis.eq("source-day").all() and data.heat_flux_unit.eq("kW/m2").all()
    pooled = data.loc[data.source_day.eq(POOLED)]
    assert pooled.n.eq(540).all() and pooled.n_pre_onb.eq(225).all() and pooled.n_post_onb.eq(315).all()
    metrics = [
        "rmse_all_kW_m2", "mae_all_kW_m2", "rmse_onb_kW_m2", "mae_onb_kW_m2",
        "rmse_pre_onb_kW_m2", "rmse_high_kW_m2", "r2", "fn", "fp", "fpr",
        "recall", "q100_kW_m2", "g100_kW_m2",
    ]
    clean = data.loc[data.noise.eq("clean"), KEYS + metrics]
    paired = data.loc[data.noise.ne("clean")].merge(
        clean, on=KEYS, suffixes=("", "__clean"), validate="many_to_one")
    assert len(paired) == 594
    for metric in metrics:
        paired["change_" + metric] = paired[metric] - paired[metric + "__clean"]
    paired["cnn_epochs"] = paired.seed.map({42: 200, 43: 150, 44: 150})
    cols = KEYS + ["noise", "cnn_epochs", "n", "n_pre_onb", "n_post_onb"]
    for metric in metrics:
        cols.extend([metric + "__clean", metric, "change_" + metric])
    paired[cols].to_csv(OUT / "clean_to_noise_changes.csv", index=False, encoding="utf-8-sig")

    # Check all corresponding saved degradation rows; do not rerun fitting or inference.
    saved = read(ROOT / "experiments/2026-10-06_onb_main_comparison_ready/saved_main_tables/noise_degradation.csv")
    checks = saved.merge(paired, on=KEYS + ["noise"], validate="one_to_one")
    assert len(checks) == 396
    check_pairs = [
        ("increase_rmse_from_clean_kW_m2", "change_rmse_all_kW_m2"),
        ("increase_near_onb_rmse_from_clean_kW_m2", "change_rmse_onb_kW_m2"),
        ("delta_fn_from_clean", "change_fn"), ("delta_fp_from_clean", "change_fp"),
    ]
    largest_difference = 0.0
    for saved_col, computed_col in check_pairs:
        differences = abs(checks[saved_col] - checks[computed_col])
        largest_difference = max(largest_difference, float(differences.max()))
        assert np.allclose(checks[saved_col], checks[computed_col], atol=1e-10, rtol=0)

    rows = []
    current = paired.loc[paired.seed.eq(42) & paired.source_day.eq(POOLED) & paired.noise.eq("-20")]
    for _, row in current.iterrows():
        rows.append({"model_key": row.model_key, "clean_rmse": row.rmse_all_kW_m2__clean,
                     "minus20_rmse": row.rmse_all_kW_m2, "increase_rmse": row.change_rmse_all_kW_m2,
                     "clean_near_rmse": row.rmse_onb_kW_m2__clean,
                     "minus20_near_rmse": row.rmse_onb_kW_m2,
                     "increase_near_rmse": row.change_rmse_onb_kW_m2,
                     "clean_fn": row.fn__clean, "minus20_fn": row.fn,
                     "clean_fp": row.fp__clean, "minus20_fp": row.fp})
    main_table = pd.DataFrame(rows)
    main_table.to_csv(OUT / "latest_minus20_summary.csv", index=False, encoding="utf-8-sig")

    comparisons = []
    for (seed, noise), group in paired.loc[paired.source_day.eq(POOLED)].groupby(["seed", "noise"]):
        indexed = group.set_index("model_key")
        five = indexed.loc[FIVE]
        for model, baseline in indexed.drop(index=FIVE).iterrows():
            comparisons.append({"seed": int(seed), "cnn_epochs": int(five.cnn_epochs), "noise": noise,
                                "baseline": model,
                                "baseline_clean_rmse": baseline.rmse_all_kW_m2__clean,
                                "five_clean_rmse": five.rmse_all_kW_m2__clean,
                                "baseline_noisy_rmse": baseline.rmse_all_kW_m2,
                                "five_noisy_rmse": five.rmse_all_kW_m2,
                                "baseline_increase_rmse": baseline.change_rmse_all_kW_m2,
                                "five_increase_rmse": five.change_rmse_all_kW_m2,
                                "five_minus_baseline_noisy_rmse": five.rmse_all_kW_m2-baseline.rmse_all_kW_m2,
                                "five_minus_baseline_increase_rmse": five.change_rmse_all_kW_m2-baseline.change_rmse_all_kW_m2})
    gains = pd.DataFrame(comparisons)
    gains.to_csv(OUT / "five_vs_each_baseline.csv", index=False, encoding="utf-8-sig")
    verification = {
        "scope": "2025 June 11+18 saved within_wav_chunk clean_only results; no new fitting or inference",
        "source_rows": len(data), "clean_to_noise_rows": len(paired), "models": 11,
        "source_days": ["2025.06.11_0.3_2", "2025.06.18_0.3_3"],
        "pooled_n": 540, "pooled_pre_onb": 225, "pooled_post_onb": 315,
        "cnn_epochs_by_training_seed": {"42": 200, "43": 150, "44": 150},
        "saved_degradation_rows_checked": len(checks), "metrics_checked": len(check_pairs),
        "max_saved_degradation_difference": largest_difference,
        "sources": [{"path": str(path.relative_to(ROOT)), "sha256": sha}
                    for path, sha in zip(paths, before_hashes)],
        "sources_unchanged": before_hashes == [digest(path) for path in paths],
        "interpretation": "Smaller noisy RMSE and smaller clean-to-noise RMSE increase are distinct claims. Conditions and final model adoption remain provisional.",
    }
    assert verification["sources_unchanged"]
    (OUT / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    print(main_table.round(3).to_string(index=False))
    print(gains.loc[gains.baseline.eq(CORE) & gains.noise.eq("-20"),
                    ["seed", "cnn_epochs", "baseline_increase_rmse", "five_increase_rmse", "five_minus_baseline_increase_rmse"]].round(3).to_string(index=False))
    print(gains.loc[gains.baseline.eq(CORE) & gains.seed.eq(42),
                    ["noise", "baseline_noisy_rmse", "five_noisy_rmse", "baseline_increase_rmse", "five_increase_rmse"]].round(3).to_string(index=False))
    print(json.dumps({"checked_rows": len(checks), "max_error": largest_difference, "sources_unchanged": True}))


if __name__ == "__main__":
    main()
