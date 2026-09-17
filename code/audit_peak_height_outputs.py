"""Verify peak export, variant alignment, and the new real-data smoke run."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.experiment.acoustic_selection import AcousticTrainingSelector
from utils.experiment.learning_policy import checked_metadata, sample_key
from utils.experiment.onb_thresholds import onb_threshold_by_experiment
from utils.experiment.spectral_peaks import PEAK_WINDOWS_HZ, PEAK_FEATURE, peak_features

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/2026-09-16_peak_height_selection"
OLD = ROOT / "experiments/2026-09-16_day_split_spectral_selection"
TRAIN = {"2025.06.11_0.3_2", "2025.07.09_0.3_1"}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as inp:
        return list(csv.DictReader(inp))


def main():
    base = ROOT / "Pool_boiling/Subcooling_20_degrees/0.3"
    cfg = {"enabled": True, "mode": "peak_height", "features_csv": str(OUT / "peak_features.csv"),
           "feature": PEAK_FEATURE, "peak_height_threshold": 1e-9}
    selector = AcousticTrainingSelector(cfg, onb_threshold_by_experiment())
    former = AcousticTrainingSelector({"enabled": True, "features_csv": str(OLD / "spectral_features.csv"),
        "feature": "band_2000_3000_db", "background_quantile": .99}, onb_threshold_by_experiment())
    table = pd.read_csv(OUT / "peak_features.csv")
    manifest = json.loads((OUT / "peak_manifest.json").read_text(encoding="utf-8"))
    counts, references, disagreements, datasets, background = {}, {}, {}, [], []
    for source in manifest["datasets"]:
        day = source["experiment_name"]
        path = ROOT / source["source_psd"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source["source_psd_sha256"]
        assert hashlib.sha256((OLD / f"chunks_{day}.csv").read_bytes()).hexdigest() == source["source_metadata_sha256"]
        archive = np.load(path)
        chunk_rows = read_csv(OLD / f"chunks_{day}.csv")
        for row, spectrum in zip(chunk_rows, archive["psd"]):
            actual = selector.rows[sample_key(row)]
            expected = peak_features(archive["frequency_hz"], spectrum)
            for key, value in expected.items():
                if isinstance(value, str):
                    assert actual[key] == value
                else:
                    np.testing.assert_allclose(float(actual[key]), value, rtol=1e-12, atol=0)
        day_table = table[table.experiment_name == day]
        assert len(day_table) == len(chunk_rows) == source["n_chunks"]
        for lo, hi in PEAK_WINDOWS_HZ:
            feature = f"peak_{lo}_{hi}_psd"
            for name, mask in (("early_background_q_below_half_ONB", day_table.q_over_onb < .5),
                               ("late_pre_ONB", (day_table.q_over_onb >= .5) & (day_table.q_over_onb < 1)),
                               ("ONB_and_above", day_table.q_over_onb >= 1)):
                values = day_table.loc[mask, feature].to_numpy() * 1e9
                background.append({"day": day, "role": "train" if day in TRAIN else "test_description_only",
                    "region": name, "feature": feature, "n": len(values),
                    "minimum": float(values.min()), "median": float(np.median(values)),
                    "maximum": float(values.max()), "unit": "1e-9 digital amplitude squared / Hz"})
        for path in sorted((base / day / "data/npy/waterflow_20260817_1s").glob("*/*/chunk_manifest.csv")):
            metadata = checked_metadata(read_csv(path), day)
            retained, audit = selector.select(metadata)
            keys = {sample_key(metadata[i]) for i in retained}
            if day not in references:
                references[day] = keys
                old_indices, _ = former.select(metadata)
                old_keys = {sample_key(metadata[i]) for i in old_indices}
                disagreements[day] = {"old_kept": len(old_keys), "new_kept": len(keys),
                    "newly_kept": len(keys - old_keys), "newly_excluded": len(old_keys - keys),
                    "applies_to_training": day in TRAIN}
            assert keys == references[day]
            assert audit["by_experiment"][day]["threshold_psd"] == 1e-9
            datasets.append({"day": day, "path": str(path.relative_to(ROOT)), "n": len(metadata),
                             "n_above_rule_or_pre_onb": len(keys), "applied_in_training": day in TRAIN})
        images = list((base / day / "data/spectrum_peak_png/waterflow_20260817_1s").glob("*/*.png"))
        assert len(images) == len(chunk_rows)
        counts[day] = {"chunks": len(chunk_rows), "images": len(images), "n_training_retained": len(references[day]) if day in TRAIN else None,
                      "n_test_evaluated": len(chunk_rows) if day not in TRAIN else None}
    assert len(datasets) == 105
    assert len(list((OUT / "timelines").glob("*.png"))) == 49
    assert sum(counts[d]["n_training_retained"] for d in TRAIN) == 1649
    pd.DataFrame(background).to_csv(OUT / "peak_height_regions.csv", index=False)
    sensitivity = pd.read_csv(OUT / "threshold_sensitivity.csv")
    for _, row in sensitivity.iterrows():
        g = table[(table.experiment_name == row.experiment_name) & (table.source_wav_id == row.source_wav_id)]
        assert int((g[PEAK_FEATURE] >= row.threshold_in_1e9_units * 1e-9).sum()) == row.n_peak_pass
    sensitivity[sensitivity.role == "train"].groupby("threshold_in_1e9_units")["n_training_retained_if_applied"].sum().to_csv(OUT / "training_threshold_counts.csv")
    make_example(table)
    smoke_root = base / "2025.06.18_0.3_3/regression_result/npy/day_split_smoke/20260916_peak_height_smoke_selected"
    if os.name == "nt":
        smoke_root = Path("\\\\?\\" + str(smoke_root))
    paths = list(smoke_root.rglob("completed.json"))
    assert len(paths) == 1, "Expected one completed new peak-selection smoke run"
    directory = paths[0].parent
    run = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    split = json.loads((directory / "split_manifest.json").read_text(encoding="utf-8"))["folds"][0]
    assert split["n_training_chunks"] == 1649 and split["n_evaluation_chunks"] == 1080
    assert len(split["evaluation_wav_groups"]) == 18
    assert not set(split["training_wav_groups"]) & set(split["evaluation_wav_groups"])
    assert {json.loads(g)[0] for g in split["training_wav_groups"]} == TRAIN
    internal = json.loads((directory / "internal_validation_fold1.json").read_text(encoding="utf-8"))
    assert len(internal["samples"]) == 1860
    assert sorted(i for f in internal["folds"] for i in f["validation_indices"]) == list(range(1860))
    for fold in internal["folds"]:
        assert not set(fold["fit_indices"]) & set(fold["validation_indices"])
        assert fold["selection"]["mode"] == "peak_height"
        assert fold["selection"]["threshold_source"] == "fixed_config"
        assert all(d["threshold_psd"] == 1e-9 for d in fold["selection"]["by_experiment"].values())
        assert set(fold["selection"]["by_experiment"]) == TRAIN
    predictions = read_csv(directory / "fold_pred/pred_f1_no_noise.csv")
    model_keys = ["randomforest", "conformer", "alexnet"]
    assert len(predictions) == 1080
    assert all(np.isfinite(float(row[key])) for row in predictions for key in model_keys + ["ensemble__performance_kfold"])
    weights = read_csv(directory / "ensemble_weights_no_noise.csv")[0]
    expected_weights = np.asarray([1 / max(internal["individual_errors"][key], 1e-6) for key in model_keys])
    expected_weights /= expected_weights.sum()
    np.testing.assert_allclose([float(weights[key]) for key in model_keys], expected_weights)
    selection = json.loads((directory / "training_selection_fold1.json").read_text(encoding="utf-8"))
    assert selection["mode"] == "peak_height" and selection["threshold_source"] == "fixed_config"
    assert selection["config"]["features_sha256"] == hashlib.sha256((OUT / "peak_features.csv").read_bytes()).hexdigest()
    assert set(selection["by_experiment"]) == TRAIN
    assert not selection["test_filtering"] and not selection["labels_changed"]
    snapshot = OUT / "smoke_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in ["completed.json", "run_manifest.json", "split_manifest.json", "training_selection_fold1.json",
                 "internal_validation_fold1.json", "ensemble_weights_no_noise.csv", "metrics_summary_no_noise.csv"]:
        shutil.copyfile(directory / name, snapshot / name)
    shutil.copyfile(directory / "fold_pred/pred_f1_no_noise.csv", snapshot / "test_predictions.csv")
    result = {"counts": counts, "image_count": 2940, "timeline_count": 49, "dataset_manifest_count": len(datasets),
        "all_peak_values_recomputed": True, "same_selection_across_all_variants": True,
        "comparison_to_previous_rule": disagreements, "smoke_run": {"run_hash": run["run_hash"], "path": str(directory),
        "n_train": 1649, "n_unfiltered_internal_validation": 1860, "n_test": 1080,
        "finite_predictions_all_models": True, "weights_recomputed_from_training_oof": True,
        "scope": "2 epochs integration verification, not a converged performance comparison"}, "datasets": datasets}
    (OUT / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "datasets"}, ensure_ascii=True, indent=2))


def make_example(table):
    day, wav = "2025.07.09_0.3_1", "index=14.643516"
    g = table[(table.experiment_name == day) & (table.source_wav_id == wav)].sort_values("chunk_index")
    spectra = np.load(OLD / f"spectra_{day}.npz")
    meta = pd.read_csv(OLD / f"chunks_{day}.csv")
    ids = meta.index[meta.source_wav_id == wav]
    freq, psd = spectra["frequency_hz"], spectra["psd"][ids] * 1e9
    heights = g[PEAK_FEATURE].to_numpy() * 1e9
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, i in zip(axes.flat, [2, 4, 9, 11]):
        ax.plot(freq, psd[i], color="#266ea6", lw=1)
        ax.axvspan(2100, 2500, alpha=.12, color="orange")
        ax.hlines(1, 2100, 2500, color="red", ls="--", label="Inclusion line = 1")
        ax.set(xlim=(0, 3000), ylim=(0, 12), xlabel="Frequency (Hz)", ylabel="PSD (x 1e-9)",
               title=f"{i}-{i+1} s | height={heights[i]:.3f} | {'include' if heights[i] >= 1 else 'exclude'}")
        ax.grid(alpha=.2)
    fig.suptitle("2025/07/09, q=643.52 kW/m2 | Same linear scale (tall peaks clipped)")
    fig.tight_layout()
    fig.savefig(OUT / "example_spectra.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
