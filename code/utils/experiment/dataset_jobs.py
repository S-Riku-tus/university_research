from pathlib import Path

from utils.experiment.run_helpers import snr_value_from_noise_dir


def chunk_tag(chunk_seconds):
    if isinstance(chunk_seconds, (int, float)):
        return f"{chunk_seconds:g}s"
    return f"{chunk_seconds}s"


def has_input_files(data_path, color_channel):
    extension = "*.npy" if color_channel == 1 else "*.png"
    return data_path.is_dir() and any(data_path.glob(extension))


def find_data_source_dir(
    experiment_root,
    experiment_name,
    data_source_dir_by_experiment,
    noise_source_prefix,
    chunk_seconds,
):
    npy_root = experiment_root / "data" / "npy"
    configured = data_source_dir_by_experiment.get(experiment_name)
    if configured:
        configured_path = npy_root / configured
        if configured_path.is_dir():
            return configured_path

    candidates = sorted(npy_root.glob(f"{noise_source_prefix}_*_{chunk_tag(chunk_seconds)}"))
    if not candidates:
        return None
    return candidates[-1]


def build_dataset_jobs(
    experiment_root,
    experiment_names,
    max_freq_hz_list,
    noise_dir_names,
    data_source_dir_by_experiment,
    noise_source_prefix,
    chunk_seconds,
    threshold_by_experiment,
    result_model_group,
    result_date_dir,
    color_channel,
    require_experiment_threshold,
    skip_missing_datasets,
    learning_policy=None,
):
    jobs = []
    missing = []
    root = Path(experiment_root)
    policy = learning_policy or {}
    explicit_days = policy.get("split_mode") == "explicit_days"
    test_days = set(policy["test_experiments"]) if explicit_days else set(experiment_names)
    intended = 0
    for experiment_name in experiment_names:
        exp_root = root / experiment_name
        source_dir = find_data_source_dir(
            exp_root,
            experiment_name,
            data_source_dir_by_experiment,
            noise_source_prefix,
            chunk_seconds,
        )
        threshold = threshold_by_experiment.get(experiment_name)
        # 学習専用日のclean_onlyでは評価ノイズを全件要求しない。
        # clean条件を評価一覧から外しても、学習専用日では検査する。
        if experiment_name in test_days:
            required_noises = list(noise_dir_names)
        elif policy.get("training_noise") == "clean_only":
            required_noises = ["heatflux_no_noise"]
        else:
            required_noises = list(noise_dir_names)
        for max_freq_name in max_freq_hz_list:
            for noise_dir_name in required_noises:
                intended += 1
                data_path = None if source_dir is None else source_dir / max_freq_name / noise_dir_name
                job = {
                    "experiment_name": experiment_name,
                    "experiment_root": exp_root,
                    "source_dir": source_dir,
                    "threshold": threshold,
                    "max_freq_hz": max_freq_name,
                    "noise_dir_name": noise_dir_name,
                    "snr_value": snr_value_from_noise_dir(noise_dir_name),
                    "data_path": data_path,
                    "save_base_path": (
                        exp_root / "regression_result" / "npy" / result_model_group / result_date_dir
                    ),
                }
                if require_experiment_threshold and threshold is None:
                    missing.append({**job, "missing_reason": "threshold"})
                elif data_path is not None and has_input_files(data_path, color_channel):
                    jobs.append(job)
                else:
                    missing.append({**job, "missing_reason": "data"})

    print(f"dataset plan: existing={len(jobs)} / intended={intended}")
    if missing:
        print("missing datasets:")
        for job in missing:
            missing_path = job["data_path"] if job["data_path"] is not None else job["experiment_root"] / "data" / "npy"
            reason = job.get("missing_reason", "data")
            print(f"  - {reason} | {job['experiment_name']} | {job['max_freq_hz']} | {job['noise_dir_name']} | {missing_path}")
        if not skip_missing_datasets:
            raise FileNotFoundError("Some intended datasets are missing. Set SKIP_MISSING_DATASETS=True to continue.")
    return jobs
