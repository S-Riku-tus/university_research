"""2実験日・70条件のmanifestで、4方針の分割とノイズ間対応を確認する。学習は行わない。"""

import json
from pathlib import Path
import runpy
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))

from utils.calculation.wav_event_metrics import load_sample_metadata_without_arrays
from utils.experiment.learning_policy import (
    aligned_indices, build_learning_families, checked_metadata, outer_splits, wav_groups,
)


def main():
    settings = runpy.run_path(str(REPO_ROOT / "code/run_ensemble_regression_onb.py"), run_name="data_check_only")
    days = ["2025.06.11_0.3_2", "2025.06.18_0.3_3"]
    frequencies = [f"maxfreq={value}kHz" for value in (3, 5, 10, 15, 22)]
    noises = ["heatflux_no_noise"] + [f"heatflux_reference_SNR={value}" for value in (0, -4, -8, -12, -16, -20)]
    jobs = settings["make_dataset_jobs"](
        experiment_root=settings["EXPERIMENT_ROOT"], experiment_names=days,
        max_freq_hz_list=frequencies, noise_dir_names=noises,
        data_source_dir_by_experiment=settings["DATA_SOURCE_DIR_BY_EXPERIMENT"],
        noise_source_prefix=settings["NOISE_SOURCE_PREFIX"], chunk_seconds=settings["CHUNK"],
        threshold_by_experiment=settings["THRESHOLD_BY_EXPERIMENT"],
        result_model_group="ensemble", result_date_dir="metadata_check_only",
        color_channel=1, require_experiment_threshold=True, skip_missing_datasets=False,
    )
    cache = {str(job["data_path"]): checked_metadata(load_sample_metadata_without_arrays(job["data_path"]),
              job["experiment_name"]) for job in jobs}
    by_condition = {(job["experiment_name"], job["max_freq_hz"], job["noise_dir_name"]): cache[str(job["data_path"])]
                    for job in jobs}
    for day in days:
        for frequency in frequencies:
            reference = by_condition[(day, frequency, "heatflux_no_noise")]
            for noise in noises:
                aligned_indices(reference, by_condition[(day, frequency, noise)])
    reports = []
    for split in ("within_day", "leave_one_day_out"):
        for training_noise in ("matched", "clean_only"):
            policy = {"split_mode": split, "training_noise": training_noise}
            families = build_learning_families(jobs, policy, days)
            evaluation_splits = 0
            for family in families:
                training_metadata = [row for job in family["training_jobs"] for row in cache[str(job["data_path"])]]
                for job in family["evaluation_jobs"]:
                    evaluation_metadata = cache[str(job["data_path"])]
                    splits = outer_splits(training_metadata, evaluation_metadata, split, 3)
                    evaluation_splits += len(splits)
                    evaluated_indices = sorted(int(index) for _, test in splits for index in test)
                    assert evaluated_indices == list(range(len(evaluation_metadata)))
            reports.append({**policy, "training_families": len(families),
                            "outer_fits_per_model": len(families) * (3 if split == "within_day" else 1),
                            "evaluation_conditions": len(jobs), "evaluation_splits": evaluation_splits})
    result = {"training_started": False, "n_conditions": len(jobs), "policies": reports,
              "source_wavs_per_day": {day: len(set(wav_groups(by_condition[(day, frequencies[0], noises[0])]))) for day in days},
              "noise_alignment": "all source WAVs, chunk indices, time intervals and heat-flux labels match"}
    output = Path(__file__).with_name("real_data_preflight.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
