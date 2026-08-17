"""Controlled full-SNR test with label-independent water-flow amplitude.

Unlike the production 20260722 data, the injected noise power here is shared
by every source WAV at a given reference-SNR level.  Each selected water-flow
chunk is normalized to that exact target power, and its offset is held fixed
across SNR levels.  Results include both per-condition retraining and a
clean-trained cross-SNR transfer test.
"""

import csv
import json
from math import gcd
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from skimage.transform import resize

from utils.dataloading.waterflow_preprocessing import (
    build_experiment_context,
    calc_stft,
    get_sorted_wav_files,
    heat_flux_label_from_wav,
    highpass_filter,
    select_noise_chunk,
    stable_noise_seed,
)
from utils.diagnostics.noise_shortcut import (
    _make_rf,
    _safe_pearson,
    _split_plan,
    evaluate_noise_shortcut_dataset,
    old_style_magnitude_minmax,
    summarize_fold_metrics,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = (
    REPO_ROOT / "Pool_boiling" / "Subcooling_20_degrees" / "0.3"
)
EXPERIMENTS = [
    "2025.06.18_0.3_3",
    "2025.07.09_0.3_1",
    "2025.06.11_0.3_2",
]
RECORDING_DIR = "録音データ_熱流束"
WATERFLOW_PATH = REPO_ROOT / "water_flow" / "water_flow_125.wav"
RESULT_DIR = (
    REPO_ROOT / "experiments" / "2026-07-24_noise_shortcut_diagnostic"
)

SAMPLERATE = 44100
AUDIO_SAMPLES_USED = 2646000
CHUNK_SECONDS = 1
CHUNKS_PER_SOURCE = 5
SAMPLE_NUMBER = 672
MAX_FREQ_HZ = 22050
SNR_LEVELS = [None, 0, -4, -8, -12, -16, -20]
RANDOM_SEED = 42
FILTER_CONFIG = {
    "fp": 500,
    "fs": 400,
    "gpass": 0.00001,
    "gstop": 0.0001,
}
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bynode": 0.6,
}
FEATURE_SETS = [
    "total_power_only",
    "full_spectrogram_pca",
    "old_style_magnitude_minmax_pca",
]


def _pcm_to_float(data):
    data = np.asarray(data)
    if np.issubdtype(data.dtype, np.floating):
        return data.astype(np.float64, copy=False)
    if data.dtype == np.uint8:
        return (data.astype(np.float64) - 128.0) / 128.0
    info = np.iinfo(data.dtype)
    scale = float(max(abs(info.min), info.max))
    return data.astype(np.float64) / scale


def _load_resampled(path, first_channel=False):
    source_sr, data = wavfile.read(path, mmap=True)
    data = _pcm_to_float(data)
    if data.ndim > 1:
        data = data[:, 0] if first_channel else data.mean(axis=1)
    if int(source_sr) != SAMPLERATE:
        divisor = gcd(int(source_sr), SAMPLERATE)
        data = signal.resample_poly(
            data,
            SAMPLERATE // divisor,
            int(source_sr) // divisor,
        )
    return np.asarray(data[:AUDIO_SAMPLES_USED], dtype=float)


def _load_filtered(path, first_channel=False):
    data = _load_resampled(path, first_channel=first_channel)
    return highpass_filter(
        data,
        SAMPLERATE,
        FILTER_CONFIG["fp"],
        FILTER_CONFIG["fs"],
        FILTER_CONFIG["gpass"],
        FILTER_CONFIG["gstop"],
    )


def _experiment_context(experiment):
    return build_experiment_context(
        experiment_name=experiment,
        base_experiment_dir=str(EXPERIMENT_ROOT),
        recording_dir_name=RECORDING_DIR,
        waterflow_path=str(WATERFLOW_PATH),
        save_date=20260817,
        dataset_version="controlled_global_fixed_chunk_rms",
        chunk_seconds=CHUNK_SECONDS,
        script_path=__file__,
    )


def _source_paths(experiment):
    context = _experiment_context(experiment)
    names = get_sorted_wav_files(context["folder_path"])
    return context, [Path(context["folder_path"]) / name for name in names]


def _global_reference_rms():
    values = []
    for experiment in EXPERIMENTS:
        _, paths = _source_paths(experiment)
        for path in paths:
            filtered = _load_filtered(path)
            values.append(float(np.sqrt(np.mean(filtered**2))))
    return float(np.median(values)), values


def _selected_chunk_starts(sample_count):
    data_length = SAMPLERATE * CHUNK_SECONDS
    starts = list(range(0, sample_count, data_length))
    indices = np.linspace(
        0, len(starts) - 1, num=min(CHUNKS_PER_SOURCE, len(starts)), dtype=int
    )
    return [starts[index] for index in sorted(set(indices.tolist()))]


def _spectrogram(chunk):
    power = calc_stft(chunk, SAMPLE_NUMBER, SAMPLERATE)
    max_k = int(MAX_FREQ_HZ * SAMPLE_NUMBER * 2 / SAMPLERATE)
    return resize(power[:, : max_k + 1], (224, 224)).astype(np.float32)


def _noise_label(snr):
    return "no_noise" if snr is None else f"reference_SNR={snr}"


def _generate_experiment(experiment, reference_rms, waterflow):
    context, paths = _source_paths(experiment)
    data_by_level = {
        _noise_label(snr): {"x": [], "y": [], "metadata": []}
        for snr in SNR_LEVELS
    }
    data_length = SAMPLERATE * CHUNK_SECONDS

    for source_index, path in enumerate(paths, start=1):
        source = _load_filtered(path)
        label = float(heat_flux_label_from_wav(str(path), context))
        source_id = f"{experiment}:{path.stem}"
        starts = _selected_chunk_starts(len(source))
        print(
            f"  {experiment} source {source_index}/{len(paths)}: {path.name}"
        )
        for chunk_slot, start in enumerate(starts):
            chunk_index = start // data_length
            signal_chunk = source[start : start + data_length]
            if len(signal_chunk) < data_length:
                signal_chunk = np.pad(
                    signal_chunk, (0, data_length - len(signal_chunk))
                )
            seed = stable_noise_seed(
                RANDOM_SEED,
                "shared_across_all_sources",
                "shared_across_snr",
                chunk_slot,
            )
            noise_chunk, noise_offset = select_noise_chunk(
                waterflow,
                data_length,
                randomize_offset=True,
                random_seed=seed,
            )
            raw_noise_power = float(np.mean(noise_chunk**2))

            for snr in SNR_LEVELS:
                level = _noise_label(snr)
                if snr is None:
                    model_input = signal_chunk
                    scaled_noise_power = 0.0
                    realized_snr = np.nan
                else:
                    target_noise_power = reference_rms**2 / 10 ** (
                        float(snr) / 10
                    )
                    scale = np.sqrt(target_noise_power / raw_noise_power)
                    scaled_noise = noise_chunk * scale
                    model_input = signal_chunk + scaled_noise
                    scaled_noise_power = float(np.mean(scaled_noise**2))
                    signal_power = float(np.mean(signal_chunk**2))
                    realized_snr = float(
                        10 * np.log10(signal_power / scaled_noise_power)
                    )

                sample_name = f"{path.stem}_chunk-{chunk_index:04d}.npy"
                data_by_level[level]["x"].append(_spectrogram(model_input))
                data_by_level[level]["y"].append(label)
                data_by_level[level]["metadata"].append(
                    {
                        "sample_filename": sample_name,
                        "source_wav_id": source_id,
                        "model_input_power": float(np.mean(model_input**2)),
                        "signal_chunk_power": float(np.mean(signal_chunk**2)),
                        "scaled_noise_chunk_power": scaled_noise_power,
                        "realized_snr_db": realized_snr,
                        "noise_offset_samples": noise_offset,
                    }
                )

    for payload in data_by_level.values():
        payload["x"] = np.asarray(payload["x"], dtype=np.float32)
        payload["y"] = np.asarray(payload["y"], dtype=float)
    return data_by_level


def _feature_train_and_transform(clean, other, train_index, val_index, feature):
    if feature == "total_power_only":
        clean_power = np.asarray(
            [float(row["model_input_power"]) for row in clean["metadata"]]
        ).reshape(-1, 1)
        other_power = np.asarray(
            [float(row["model_input_power"]) for row in other["metadata"]]
        ).reshape(-1, 1)
        return clean_power[train_index], other_power[val_index]

    clean_x = clean["x"]
    other_x = other["x"]
    if feature == "old_style_magnitude_minmax_pca":
        clean_x = old_style_magnitude_minmax(clean_x)
        other_x = old_style_magnitude_minmax(other_x)
    elif feature != "full_spectrogram_pca":
        raise ValueError(feature)

    train_flat = clean_x[train_index].reshape(len(train_index), -1)
    other_flat = other_x[val_index].reshape(len(val_index), -1)
    components = min(100, train_flat.shape[0], train_flat.shape[1])
    pca = PCA(n_components=components, svd_solver="randomized", random_state=42)
    return pca.fit_transform(train_flat), pca.transform(other_flat)


def _clean_trained_transfer(data_by_level):
    clean = data_by_level["no_noise"]
    fold_rows = []
    for split_strategy in ("kfold", "group_kfold"):
        splits = _split_plan(
            len(clean["y"]), clean["metadata"], split_strategy, 3, 42
        )
        for fold, (train_index, val_index) in enumerate(splits, start=1):
            scaler = MinMaxScaler()
            y_train_scaled = scaler.fit_transform(
                clean["y"][train_index].reshape(-1, 1)
            ).ravel()
            for feature in FEATURE_SETS:
                pca = None
                if feature == "total_power_only":
                    clean_features = np.asarray(
                        [
                            float(row["model_input_power"])
                            for row in clean["metadata"]
                        ]
                    ).reshape(-1, 1)
                    x_train = clean_features[train_index]
                else:
                    clean_x = clean["x"]
                    if feature == "old_style_magnitude_minmax_pca":
                        clean_x = old_style_magnitude_minmax(clean_x)
                    elif feature != "full_spectrogram_pca":
                        raise ValueError(feature)
                    train_flat = clean_x[train_index].reshape(
                        len(train_index), -1
                    )
                    components = min(
                        100, train_flat.shape[0], train_flat.shape[1]
                    )
                    pca = PCA(
                        n_components=components,
                        svd_solver="randomized",
                        random_state=42,
                    )
                    x_train = pca.fit_transform(train_flat)

                model = _make_rf(42 + fold, RF_PARAMS)
                model.fit(x_train, y_train_scaled)
                for level, payload in data_by_level.items():
                    if feature == "total_power_only":
                        other_features = np.asarray(
                            [
                                float(row["model_input_power"])
                                for row in payload["metadata"]
                            ]
                        ).reshape(-1, 1)
                        x_val = other_features[val_index]
                    else:
                        other_x = payload["x"]
                        if feature == "old_style_magnitude_minmax_pca":
                            other_x = old_style_magnitude_minmax(other_x)
                        other_flat = other_x[val_index].reshape(
                            len(val_index), -1
                        )
                        x_val = pca.transform(other_flat)
                    prediction = scaler.inverse_transform(
                        model.predict(x_val).reshape(-1, 1)
                    ).ravel()
                    y_val = clean["y"][val_index]
                    fold_rows.append(
                        {
                            "test_noise_level": level,
                            "split_strategy": split_strategy,
                            "feature_set": feature,
                            "fold": fold,
                            "r2": float(r2_score(y_val, prediction)),
                            "rmse": float(
                                np.sqrt(mean_squared_error(y_val, prediction))
                            ),
                            "mae": float(mean_absolute_error(y_val, prediction)),
                            "prediction_pearson": _safe_pearson(
                                y_val, prediction
                            ),
                        }
                    )
    return fold_rows


def _summarize_transfer(rows):
    grouped = {}
    for row in rows:
        key = (
            row["test_noise_level"],
            row["split_strategy"],
            row["feature_set"],
        )
        grouped.setdefault(key, []).append(row)
    output = []
    for key, group in sorted(grouped.items()):
        item = {
            "test_noise_level": key[0],
            "split_strategy": key[1],
            "feature_set": key[2],
            "folds": len(group),
        }
        for metric in ("r2", "rmse", "mae", "prediction_pearson"):
            values = np.asarray([row[metric] for row in group], dtype=float)
            finite = values[np.isfinite(values)]
            item[f"{metric}_mean"] = float(np.mean(finite))
            item[f"{metric}_std"] = (
                float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
            )
        output.append(item)
    return output


def _write_csv(path, rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    reference_rms, source_rms_values = _global_reference_rms()
    waterflow = _load_filtered(WATERFLOW_PATH, first_channel=True)
    print(f"global_reference_rms={reference_rms:.10g}")

    retrained_fold_rows = []
    retrained_summary_rows = []
    transfer_fold_rows = []
    transfer_summary_rows = []
    snr_rows = []

    for experiment in EXPERIMENTS:
        print(f"experiment={experiment}")
        data_by_level = _generate_experiment(
            experiment, reference_rms, waterflow
        )
        for level, payload in data_by_level.items():
            evaluation = evaluate_noise_shortcut_dataset(
                payload["x"],
                payload["y"],
                payload["metadata"],
                split_strategies=("kfold", "group_kfold"),
                feature_sets=FEATURE_SETS,
                folds=3,
                random_seed=42,
                pca_components=100,
                rf_params=RF_PARAMS,
            )
            context = {
                "experiment_name": experiment,
                "noise_level": level,
            }
            retrained_fold_rows.extend(
                {**context, **row} for row in evaluation["fold_metrics"]
            )
            retrained_summary_rows.extend(
                {**context, **row}
                for row in summarize_fold_metrics(evaluation["fold_metrics"])
            )
            realized = np.asarray(
                [
                    float(row["realized_snr_db"])
                    for row in payload["metadata"]
                    if np.isfinite(float(row["realized_snr_db"]))
                ],
                dtype=float,
            )
            snr_rows.append(
                {
                    **context,
                    "count": int(realized.size),
                    "mean_db": (
                        float(np.mean(realized)) if realized.size else np.nan
                    ),
                    "std_db": (
                        float(np.std(realized, ddof=1))
                        if realized.size > 1
                        else np.nan
                    ),
                }
            )

        transfer = _clean_trained_transfer(data_by_level)
        transfer_fold_rows.extend(
            {"experiment_name": experiment, **row} for row in transfer
        )
        transfer_summary_rows.extend(
            {"experiment_name": experiment, **row}
            for row in _summarize_transfer(transfer)
        )

    _write_csv(
        RESULT_DIR / "controlled_retrained_fold_metrics.csv",
        retrained_fold_rows,
    )
    _write_csv(
        RESULT_DIR / "controlled_retrained_metrics_summary.csv",
        retrained_summary_rows,
    )
    _write_csv(
        RESULT_DIR / "controlled_clean_transfer_fold_metrics.csv",
        transfer_fold_rows,
    )
    _write_csv(
        RESULT_DIR / "controlled_clean_transfer_summary.csv",
        transfer_summary_rows,
    )
    _write_csv(RESULT_DIR / "controlled_realized_snr.csv", snr_rows)

    fn = SAMPLERATE / 2
    order, wn = signal.buttord(
        FILTER_CONFIG["fp"] / fn,
        FILTER_CONFIG["fs"] / fn,
        FILTER_CONFIG["gpass"],
        FILTER_CONFIG["gstop"],
    )
    manifest = {
        "purpose": "label-independent controlled noise curve",
        "experiments": EXPERIMENTS,
        "source_wav_count": len(source_rms_values),
        "global_reference_rms": reference_rms,
        "global_reference_definition": (
            "median RMS of all 49 high-pass-filtered source WAVs"
        ),
        "noise_power_definition": (
            "each selected noise chunk is normalized to "
            "global_reference_rms**2 / 10**(reference_snr/10)"
        ),
        "noise_offset_pairing": (
            "the same five water-flow chunks are used for every source WAV, "
            "experiment, and SNR; only their prescribed level changes"
        ),
        "snr_levels": SNR_LEVELS,
        "chunks_per_source": CHUNKS_PER_SOURCE,
        "feature_sets": FEATURE_SETS,
        "split_strategies": ["kfold", "group_kfold"],
        "filter": {
            **FILTER_CONFIG,
            "order": int(order),
            "effective_cutoff_hz": float(wn * fn),
        },
        "audio_loading": "scipy.io.wavfile + scipy.signal.resample_poly",
        "outputs": [
            "controlled_retrained_fold_metrics.csv",
            "controlled_retrained_metrics_summary.csv",
            "controlled_clean_transfer_fold_metrics.csv",
            "controlled_clean_transfer_summary.csv",
            "controlled_realized_snr.csv",
        ],
    }
    with (RESULT_DIR / "controlled_noise_manifest.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"results={RESULT_DIR}")


if __name__ == "__main__":
    main()
