import csv
import hashlib
import json
import math
import os
import re
from datetime import datetime

import librosa as lr
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from skimage.transform import resize


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["xtick.top"] = True
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["ytick.right"] = True
plt.rcParams["font.size"] = 30


def chunk_label(chunk_seconds):
    return f"{chunk_seconds:g}" if isinstance(chunk_seconds, float) else str(chunk_seconds)


def sanitize_label_for_filename(label):
    return re.sub(r'[<>:"/\\|?*]', "_", str(label).strip())


def wav_index_from_name(name):
    match = re.search(r"index=(\d+)", os.path.splitext(os.path.basename(name))[0])
    if match:
        return int(match.group(1))
    return float("inf")


def load_heat_flux_labels(csv_path):
    if not os.path.exists(csv_path):
        print(f"Warning: heat flux csv not found. Fallback to wav filename labels: {csv_path}")
        return {}

    labels = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for index, row in enumerate(reader, start=1):
            if "q" not in row:
                raise ValueError(f"'q' column not found in heat flux csv: {csv_path}")
            labels[index] = sanitize_label_for_filename(row["q"])
    return labels


def build_experiment_context(
    experiment_name,
    base_experiment_dir,
    recording_dir_name,
    waterflow_path,
    save_date,
    dataset_version,
    chunk_seconds,
    script_path,
):
    experiment_root = os.path.join(base_experiment_dir, experiment_name)
    recording_folder_path = os.path.join(experiment_root, recording_dir_name)
    heat_flux_csv_path = os.path.join(
        experiment_root,
        f"実験結果{experiment_name}",
        f"heat_flux_{experiment_name}.csv",
    )
    output_name = f"waterflow_{save_date}_{chunk_label(chunk_seconds)}s_{dataset_version}"
    base_npy_save_folder_path = os.path.join(
        experiment_root,
        "data",
        "npy",
        output_name,
    )
    base_png_save_folder_path = os.path.join(
        experiment_root,
        "data",
        "spectrogram_png",
        output_name,
    )

    return {
        "experiment_name": experiment_name,
        "experiment_root": experiment_root,
        "folder_path": recording_folder_path,
        "waterflow_path": waterflow_path,
        "heat_flux_csv_path": heat_flux_csv_path,
        "heat_flux_label_by_index": load_heat_flux_labels(heat_flux_csv_path),
        "base_npy_save_folder_path": base_npy_save_folder_path,
        "base_png_save_folder_path": base_png_save_folder_path,
        "save_date": save_date,
        "dataset_version": dataset_version,
        "script_path": script_path,
    }


def heat_flux_label_from_wav(file_path, context):
    text = os.path.splitext(os.path.basename(file_path))[0]
    match = re.search(r"index=(\d+)", text)
    heat_flux_label_by_index = context["heat_flux_label_by_index"]
    active_heat_flux_csv_path = context["heat_flux_csv_path"]

    if match and heat_flux_label_by_index:
        wav_index = int(match.group(1))
        if wav_index not in heat_flux_label_by_index:
            raise ValueError(
                f"wav index {wav_index} is not present in heat flux csv: {active_heat_flux_csv_path}"
            )
        return heat_flux_label_by_index[wav_index]

    dot_index = text.find(".")
    if dot_index != -1:
        return sanitize_label_for_filename(text[dot_index + 1 :])
    return sanitize_label_for_filename(text)


def fourier_transform(data, samplerate):
    del samplerate
    n = len(data)
    window = np.hanning(n)
    input_data = data * window
    spectrum = np.fft.fft(input_data)
    amplitude = np.sqrt((spectrum.real**2) + (spectrum.imag**2)) / (n / 2)
    amplitude = 1 / (sum(window) / n) * amplitude
    amplitude = amplitude[: int(len(amplitude) / 2)]
    return amplitude**2


def calc_stft(data, sample_number, samplerate):
    stft = []
    for i in range(sample_number):
        b1 = i * ((len(data) - sample_number * 2) // sample_number)
        b2 = b1 + sample_number * 2
        power = fourier_transform(data[b1:b2], samplerate)
        stft.append(power)
    return np.array(stft)


def highpass_filter(x, samplerate, fp, fs, gpass, gstop):
    fn = samplerate / 2
    wp = fp / fn
    ws = fs / fn
    n, wn = signal.buttord(wp, ws, gpass, gstop)
    sos = signal.butter(n, wn, "high", output="sos")
    y = signal.sosfiltfilt(sos, x, axis=0)
    return y


def add_waterflow_noise(y, waterflow_noise, snr):
    y_power = np.mean(y**2)
    waterflow_noise_power = np.mean(waterflow_noise**2)
    if waterflow_noise_power == 0:
        raise ValueError("waterflow_noise_power is zero. Cannot scale noise by SNR.")
    scaled_noise = waterflow_noise * np.sqrt(
        (y_power / 10 ** (snr / 10)) / waterflow_noise_power
    )
    return y + scaled_noise


def stable_noise_seed(random_seed, source_wav_id, snr, chunk_index):
    """Return a reproducible seed for one source/SNR/chunk combination."""
    payload = f"{int(random_seed)}|{source_wav_id}|{snr}|{int(chunk_index)}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def select_noise_chunk(
    waterflow_noise,
    chunk_samples,
    start_sample=0,
    randomize_offset=False,
    random_seed=42,
):
    """Select a fixed-length noise segment, wrapping when necessary."""
    waterflow_noise = np.asarray(waterflow_noise, dtype=float).reshape(-1)
    if waterflow_noise.size == 0:
        raise ValueError("waterflow_noise is empty.")
    if int(chunk_samples) <= 0:
        raise ValueError("chunk_samples must be positive.")

    if randomize_offset:
        rng = np.random.default_rng(int(random_seed))
        offset = int(rng.integers(0, waterflow_noise.size))
    else:
        offset = int(start_sample) % waterflow_noise.size

    indices = (offset + np.arange(int(chunk_samples))) % waterflow_noise.size
    return waterflow_noise[indices], offset


def scale_noise_chunk(
    signal_reference,
    noise_reference,
    noise_chunk,
    snr,
    scaling_mode="relative_source_rms",
    fixed_reference_signal_rms=None,
):
    """Scale a selected noise chunk.

    ``relative_source_rms`` reproduces the current relative-SNR design: each
    source WAV determines its own noise amplitude. ``fixed_reference_rms`` uses
    one signal RMS for every WAV but retains full-recording noise power.
    ``fixed_global_rms`` also normalizes each selected noise chunk, making the
    target noise power identical for every source WAV.
    """
    noise_reference_power = float(
        np.mean(np.asarray(noise_reference, dtype=float) ** 2)
    )
    noise_chunk_power = float(
        np.mean(np.asarray(noise_chunk, dtype=float) ** 2)
    )
    if noise_reference_power <= 0:
        raise ValueError("noise_reference power is zero. Cannot scale noise.")
    if noise_chunk_power <= 0:
        raise ValueError("noise_chunk power is zero. Cannot scale noise.")

    if scaling_mode == "relative_source_rms":
        reference_signal_power = float(
            np.mean(np.asarray(signal_reference, dtype=float) ** 2)
        )
        normalization_noise_power = noise_reference_power
    elif scaling_mode == "fixed_reference_rms":
        if fixed_reference_signal_rms is None:
            raise ValueError(
                "fixed_reference_signal_rms is required for fixed_reference_rms."
            )
        reference_signal_power = float(fixed_reference_signal_rms) ** 2
        normalization_noise_power = noise_reference_power
    elif scaling_mode == "fixed_global_rms":
        if fixed_reference_signal_rms is None:
            raise ValueError(
                "fixed_reference_signal_rms is required for fixed_global_rms."
            )
        reference_signal_power = float(fixed_reference_signal_rms) ** 2
        normalization_noise_power = noise_chunk_power
    else:
        raise ValueError(
            "noise_scaling_mode must be 'relative_source_rms', "
            "'fixed_reference_rms', or 'fixed_global_rms'; "
            f"got {scaling_mode!r}."
        )

    scale = np.sqrt(
        (reference_signal_power / 10 ** (float(snr) / 10))
        / normalization_noise_power
    )
    return np.asarray(noise_chunk, dtype=float) * scale, float(scale)


def realized_snr_db(signal_chunk, scaled_noise_chunk):
    signal_power = float(np.mean(np.asarray(signal_chunk, dtype=float) ** 2))
    noise_power = float(np.mean(np.asarray(scaled_noise_chunk, dtype=float) ** 2))
    if signal_power <= 0 or noise_power <= 0:
        return None
    return float(10 * np.log10(signal_power / noise_power))


def estimate_filtered_reference_signal_rms(file_paths, config, samplerate=44100):
    """Estimate one fixed RMS from the median source-WAV RMS.

    This is a diagnostic absolute-amplitude reference, not a calibrated SPL.
    The returned value must be recorded in the preprocessing manifest.
    """
    rms_values = []
    for file_path in file_paths:
        y, sr = lr.load(file_path, sr=samplerate)
        filtered = highpass_filter(
            y[: config["audio_samples_used"]],
            sr,
            config["fp"],
            config["fs"],
            config["gpass"],
            config["gstop"],
        )
        rms_values.append(float(np.sqrt(np.mean(filtered**2))))
    if not rms_values:
        raise ValueError("No source WAVs were supplied for RMS estimation.")
    return float(np.median(rms_values))


def write_chunk_manifest(out_dir, rows):
    """Upsert per-chunk provenance and realized-SNR diagnostics."""
    if not rows:
        return
    manifest_path = os.path.join(out_dir, "chunk_manifest.csv")
    fieldnames = list(rows[0])
    merged = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                key = row.get("sample_filename")
                if key:
                    merged[key] = row
    for row in rows:
        merged[row["sample_filename"]] = row

    def sort_key(row):
        return (
            str(row.get("source_wav_id", "")),
            int(float(row.get("chunk_index", 0))),
        )

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(merged.values(), key=sort_key))


def write_preprocess_manifest(
    out_dir,
    max_freq_hz,
    samplerate,
    data_length,
    fft_window_size,
    max_k,
    snr_db,
    chunk_seconds,
    context,
    config,
):
    fn = samplerate / 2
    n, wn = signal.buttord(
        config["fp"] / fn,
        config["fs"] / fn,
        config["gpass"],
        config["gstop"],
    )
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": os.path.relpath(context["script_path"], os.getcwd()),
        "source_audio_folder": context["folder_path"],
        "waterflow_path": context["waterflow_path"],
        "heat_flux_csv_path": context["heat_flux_csv_path"],
        "label_source": "heat_flux_csv.q keyed by wav index",
        "save_date": context["save_date"],
        "dataset_version": context["dataset_version"],
        "output_formats": {
            "npy": config["save_npy"],
            "spectrogram_png": config["save_spectrogram_png"],
            "png_with_axes": config["png_with_axes"],
        },
        "chunk_seconds": chunk_seconds,
        "samplerate_hz": samplerate,
        "audio_samples_used": config["audio_samples_used"],
        "audio_loading": config.get("audio_loading", "librosa.load default"),
        "segment_samples": data_length,
        "chunk_selection": {
            "max_chunks_per_source": config.get("max_chunks_per_source"),
            "strategy": (
                "evenly spaced over available source chunks"
                if config.get("max_chunks_per_source")
                else "all sequential chunks"
            ),
        },
        "max_freq_hz": max_freq_hz,
        "max_freq_khz_display": max_freq_hz / 1000.0,
        "max_frequency_bin_index": max_k,
        "stft": {
            "frames": config["sample_number"],
            "n_fft": fft_window_size,
            "window": "hann",
            "hop_length_formula": "(len(chunk) - n_fft) // frames",
            "hop_length_samples": (data_length - fft_window_size) // config["sample_number"],
            "spectrum": "linear power, amplitude**2",
            "resize_shape": [224, 224],
            "resize_library": "skimage.transform.resize",
            "saved_array_dtype": config.get("saved_array_dtype", "float64"),
            "saved_array_axes_before_resize": ["time_frame", "frequency_bin"],
        },
        "highpass_filter": {
            "type": "butterworth",
            "fp_hz": config["fp"],
            "fs_hz": config["fs"],
            "gpass_db": config["gpass"],
            "gstop_db": config["gstop"],
            "buttord_order": int(n),
            "buttord_Wn_normalized": float(wn),
            "buttord_cutoff_hz": float(wn * fn),
            "application": "scipy.signal.sosfiltfilt",
        },
        "noise": {
            "snr_list_db": snr_db,
            "scaling_mode": config.get(
                "noise_scaling_mode", "relative_source_rms"
            ),
            "output_signal_mode": config.get("output_signal_mode", "mixture"),
            "fixed_reference_signal_rms": config.get(
                "fixed_reference_signal_rms"
            ),
            "fixed_reference_provenance": config.get(
                "fixed_reference_provenance"
            ),
            "randomize_noise_offset_per_chunk": config.get(
                "randomize_noise_offset_per_chunk", False
            ),
            "random_seed": config.get("random_seed", 42),
            "pair_noise_across_snr": config.get(
                "pair_noise_across_snr", False
            ),
            "noise_seed_scope": config.get(
                "noise_seed_scope", "per_source"
            ),
            "level_folder_prefix": config.get(
                "noise_level_folder_prefix", "SNR"
            ),
            "level_definition": config.get(
                "noise_level_definition",
                "SNR relative to each source WAV",
            ),
            "relative_formula": (
                "scaled_noise = waterflow_noise_chunk * "
                "sqrt((mean(source_wav**2) / 10**(snr/10)) / "
                "mean(waterflow_noise_recording**2))"
            ),
            "fixed_formula": (
                "scaled_noise = waterflow_noise_chunk * "
                "sqrt((fixed_reference_signal_rms**2 / 10**(snr/10)) / "
                "mean(waterflow_noise_recording**2))"
            ),
            "fixed_global_formula": (
                "scaled_noise = waterflow_noise_chunk * "
                "sqrt((fixed_reference_signal_rms**2 / "
                "10**(level_db/10)) / mean(waterflow_noise_chunk**2))"
            ),
            "realized_snr": (
                "10*log10(mean(signal_chunk**2)/mean(scaled_noise_chunk**2)); "
                "saved for every chunk in chunk_manifest.csv"
            ),
        },
    }
    manifest_path = os.path.join(out_dir, "preprocess_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def save_spectrogram_chunks_with_snr(
    file_path,
    max_freq_hz,
    chunk_seconds,
    context,
    config,
    snr_db,
    prefiltered_signal=None,
    prefiltered_waterflow_noise=None,
):
    sr = 44100
    if prefiltered_signal is None:
        y, sr = lr.load(file_path, sr=sr)
        y = highpass_filter(
            y[: config["audio_samples_used"]],
            sr,
            config["fp"],
            config["fs"],
            config["gpass"],
            config["gstop"],
        )
    else:
        y = np.asarray(prefiltered_signal, dtype=float)

    if prefiltered_waterflow_noise is None:
        waterflow_noise, _ = lr.load(
            context["waterflow_path"], sr=sr, mono=False
        )
        if waterflow_noise.ndim > 1:
            waterflow_noise = waterflow_noise[0]
        waterflow_noise = highpass_filter(
            waterflow_noise[: config["audio_samples_used"]],
            sr,
            config["fp"],
            config["fs"],
            config["gpass"],
            config["gstop"],
        )
    else:
        waterflow_noise = np.asarray(
            prefiltered_waterflow_noise, dtype=float
        )

    data_length = int(44100 * chunk_seconds)
    fft_window_size = config["sample_number"] * 2
    max_k = int(max_freq_hz * fft_window_size / sr)
    max_freq_khz_display = max_freq_hz / 1000.0

    npy_max_freq_parent_folder = os.path.join(
        context["base_npy_save_folder_path"], f"maxfreq={max_freq_khz_display:.0f}kHz"
    )
    png_max_freq_parent_folder = os.path.join(
        context["base_png_save_folder_path"], f"maxfreq={max_freq_khz_display:.0f}kHz"
    )

    if config["save_npy"]:
        os.makedirs(npy_max_freq_parent_folder, exist_ok=True)
        write_preprocess_manifest(
            npy_max_freq_parent_folder,
            max_freq_hz,
            sr,
            data_length,
            fft_window_size,
            max_k,
            snr_db,
            chunk_seconds,
            context,
            config,
        )
    if config["save_spectrogram_png"]:
        os.makedirs(png_max_freq_parent_folder, exist_ok=True)
        write_preprocess_manifest(
            png_max_freq_parent_folder,
            max_freq_hz,
            sr,
            data_length,
            fft_window_size,
            max_k,
            snr_db,
            chunk_seconds,
            context,
            config,
        )

    extracted_label = heat_flux_label_from_wav(file_path, context)
    source_wav_name = os.path.basename(file_path)
    source_wav_id = os.path.splitext(source_wav_name)[0]
    noise_scaling_mode = config.get(
        "noise_scaling_mode", "relative_source_rms"
    )
    output_signal_mode = config.get("output_signal_mode", "mixture")
    if output_signal_mode not in {"mixture", "noise_only"}:
        raise ValueError(
            "output_signal_mode must be 'mixture' or 'noise_only', got "
            f"{output_signal_mode!r}."
        )
    randomize_noise_offset = bool(
        config.get("randomize_noise_offset_per_chunk", False)
    )
    base_random_seed = int(config.get("random_seed", 42))
    fixed_reference_signal_rms = config.get("fixed_reference_signal_rms")
    include_source_in_filename = bool(
        config.get("include_source_in_filename", False)
    )
    pair_noise_across_snr = bool(
        config.get("pair_noise_across_snr", False)
    )
    noise_seed_scope = config.get("noise_seed_scope", "per_source")
    if noise_seed_scope not in {"per_source", "shared_across_sources"}:
        raise ValueError(
            "noise_seed_scope must be 'per_source' or "
            f"'shared_across_sources', got {noise_seed_scope!r}."
        )
    noise_level_folder_prefix = config.get(
        "noise_level_folder_prefix", "SNR"
    )
    saved_array_dtype = np.dtype(
        config.get("saved_array_dtype", "float64")
    )

    for snr in snr_db:
        if snr is None and output_signal_mode == "noise_only":
            # A zero-valued "no noise only" dataset is not a useful negative
            # control and would create a misleading no-noise condition.
            continue
        snr_label = (
            "no_noise"
            if snr is None
            else f"{noise_level_folder_prefix}={snr}"
        )

        if config["save_npy"]:
            npy_snr_save_folder_path = os.path.join(
                npy_max_freq_parent_folder, f"heatflux_{snr_label}"
            )
            os.makedirs(npy_snr_save_folder_path, exist_ok=True)
        else:
            npy_snr_save_folder_path = None

        if config["save_spectrogram_png"]:
            png_snr_save_folder_path = os.path.join(
                png_max_freq_parent_folder, f"heatflux_{snr_label}"
            )
            os.makedirs(png_snr_save_folder_path, exist_ok=True)
        else:
            png_snr_save_folder_path = None

        all_chunk_starts = list(range(0, len(y), data_length))
        max_chunks_per_source = config.get("max_chunks_per_source")
        if (
            max_chunks_per_source is not None
            and int(max_chunks_per_source) < len(all_chunk_starts)
        ):
            selected_indices = np.linspace(
                0,
                len(all_chunk_starts) - 1,
                num=int(max_chunks_per_source),
                dtype=int,
            )
            chunk_starts = [
                all_chunk_starts[index]
                for index in sorted(set(selected_indices.tolist()))
            ]
        else:
            chunk_starts = all_chunk_starts

        chunk_manifest_rows = []
        for i in chunk_starts:
            chunk_index = i // data_length
            signal_chunk = y[i : i + data_length]
            unpadded_signal_samples = len(signal_chunk)
            if len(signal_chunk) < data_length:
                signal_chunk = np.pad(
                    signal_chunk,
                    (0, data_length - len(signal_chunk)),
                    "constant",
                )

            noise_offset_samples = None
            noise_seed = None
            noise_scale = None
            raw_noise_chunk_power = None
            scaled_noise_chunk_power = None
            chunk_realized_snr_db = None
            if snr is None:
                scaled_noise_chunk = np.zeros_like(signal_chunk)
                model_input_chunk = signal_chunk
            else:
                seed_source_id = (
                    "shared_across_all_sources"
                    if noise_seed_scope == "shared_across_sources"
                    else f"{context['experiment_name']}:{source_wav_id}"
                )
                seed_snr = (
                    "paired_across_snr" if pair_noise_across_snr else snr
                )
                noise_seed = stable_noise_seed(
                    base_random_seed, seed_source_id, seed_snr, chunk_index
                )
                raw_noise_chunk, noise_offset_samples = select_noise_chunk(
                    waterflow_noise,
                    data_length,
                    start_sample=i,
                    randomize_offset=randomize_noise_offset,
                    random_seed=noise_seed,
                )
                scaled_noise_chunk, noise_scale = scale_noise_chunk(
                    signal_reference=y,
                    noise_reference=waterflow_noise,
                    noise_chunk=raw_noise_chunk,
                    snr=snr,
                    scaling_mode=noise_scaling_mode,
                    fixed_reference_signal_rms=fixed_reference_signal_rms,
                )
                model_input_chunk = (
                    scaled_noise_chunk
                    if output_signal_mode == "noise_only"
                    else signal_chunk + scaled_noise_chunk
                )
                raw_noise_chunk_power = float(np.mean(raw_noise_chunk**2))
                scaled_noise_chunk_power = float(
                    np.mean(scaled_noise_chunk**2)
                )
                chunk_realized_snr_db = realized_snr_db(
                    signal_chunk, scaled_noise_chunk
                )

            amplitude = calc_stft(
                model_input_chunk, config["sample_number"], sr
            )
            amplitude_cropped = amplitude[:, : max_k + 1]
            resized_amplitude = resize(
                amplitude_cropped, (224, 224)
            ).astype(saved_array_dtype, copy=False)

            if include_source_in_filename:
                safe_source = sanitize_label_for_filename(source_wav_id).replace(
                    " ", "-"
                )
                file_stem = (
                    f"{extracted_label}_src-{safe_source}_"
                    f"chunk-{chunk_index:04d}"
                )
            else:
                file_stem = f"{extracted_label}_{chunk_index}"
            if config["save_npy"]:
                npy_save_path = os.path.join(
                    npy_snr_save_folder_path, f"{file_stem}.npy"
                )
                np.save(npy_save_path, resized_amplitude)
            if config["save_spectrogram_png"]:
                time_axis = np.linspace(
                    0, chunk_seconds, resized_amplitude.shape[0]
                )
                plt.figure(figsize=(6, 6))
                plt.imshow(
                    # Internal arrays are (time, frequency). Transpose only for
                    # conventional display: x=time, y=frequency.
                    resized_amplitude.T,
                    cmap="jet",
                    aspect="auto",
                    origin="lower",
                    extent=[
                        time_axis[0],
                        time_axis[-1],
                        0,
                        max_freq_khz_display,
                    ],
                )
                if not config["png_with_axes"]:
                    plt.axis("off")

                tick_positions_sec = [0, chunk_seconds / 2, chunk_seconds]
                tick_labels_str = [f"{p:g}" for p in tick_positions_sec]
                plt.xticks(
                    tick_positions_sec, labels=tick_labels_str, fontsize=20
                )

                y_ticks_positions = np.arange(
                    0, max_freq_khz_display + 1, step=1
                )
                if len(y_ticks_positions) > 10:
                    step = math.ceil(len(y_ticks_positions) / 5)
                    y_ticks_positions = np.arange(
                        0, max_freq_khz_display + 1, step=step
                    )
                y_ticks_labels = [f"{int(p)}" for p in y_ticks_positions]
                plt.yticks(
                    y_ticks_positions, labels=y_ticks_labels, fontsize=20
                )
                plt.xlabel("Time s", fontsize=25, labelpad=2)
                plt.ylabel("Frequency kHz", fontsize=25, labelpad=2)
                plt.subplots_adjust(
                    left=0.25, right=0.85, top=0.85, bottom=0.25
                )

                png_save_path = os.path.join(
                    png_snr_save_folder_path, f"{file_stem}.png"
                )
                pad_inches = 0.02 if config["png_with_axes"] else 0
                plt.savefig(
                    png_save_path,
                    dpi=150,
                    bbox_inches="tight",
                    pad_inches=pad_inches,
                )
                plt.close()

            row = {
                "sample_filename": f"{file_stem}.npy",
                "heat_flux": extracted_label,
                "experiment_name": context["experiment_name"],
                "source_wav_id": source_wav_id,
                "source_wav_name": source_wav_name,
                "chunk_index": chunk_index,
                "chunk_start_sample": i,
                "chunk_start_seconds": i / sr,
                "chunk_duration_seconds": chunk_seconds,
                "unpadded_signal_samples": unpadded_signal_samples,
                "requested_snr_db": "" if snr is None else snr,
                "realized_snr_db": (
                    ""
                    if chunk_realized_snr_db is None
                    else chunk_realized_snr_db
                ),
                "signal_chunk_power": float(np.mean(signal_chunk**2)),
                "raw_noise_chunk_power": (
                    "" if raw_noise_chunk_power is None
                    else raw_noise_chunk_power
                ),
                "scaled_noise_chunk_power": (
                    "" if scaled_noise_chunk_power is None
                    else scaled_noise_chunk_power
                ),
                "model_input_power": float(
                    np.mean(model_input_chunk**2)
                ),
                "noise_scale": "" if noise_scale is None else noise_scale,
                "noise_offset_samples": (
                    "" if noise_offset_samples is None
                    else noise_offset_samples
                ),
                "noise_seed": "" if noise_seed is None else noise_seed,
                "noise_seed_scope": noise_seed_scope,
                "pair_noise_across_snr": pair_noise_across_snr,
                "noise_scaling_mode": noise_scaling_mode,
                "output_signal_mode": output_signal_mode,
                "fixed_reference_signal_rms": (
                    "" if fixed_reference_signal_rms is None
                    else fixed_reference_signal_rms
                ),
            }
            chunk_manifest_rows.append(row)

        if config["save_npy"]:
            write_chunk_manifest(
                npy_snr_save_folder_path, chunk_manifest_rows
            )
        if config["save_spectrogram_png"]:
            png_rows = [
                {
                    **row,
                    "sample_filename": row["sample_filename"].replace(
                        ".npy", ".png"
                    ),
                }
                for row in chunk_manifest_rows
            ]
            write_chunk_manifest(
                png_snr_save_folder_path, png_rows
            )


def get_sorted_wav_files(folder_path):
    wav_files = sorted(
        [name for name in os.listdir(folder_path) if name.endswith(".wav")],
        key=wav_index_from_name,
    )
    if not wav_files:
        raise FileNotFoundError(f"wav files not found in: {folder_path}")
    return wav_files
