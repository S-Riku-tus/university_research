import csv
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
    b, a = signal.butter(n, wn, "high")
    y = signal.filtfilt(b, a, x, axis=0)
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
        "segment_samples": data_length,
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
            "application": "scipy.signal.filtfilt",
        },
        "noise": {
            "snr_list_db": snr_db,
            "formula": "y + waterflow_noise * sqrt((mean(y**2) / 10**(snr/10)) / mean(waterflow_noise**2))",
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
):
    y, sr = lr.load(file_path, sr=44100)
    waterflow_noise, _ = lr.load(context["waterflow_path"], sr=sr, mono=False)

    y = highpass_filter(
        y[: config["audio_samples_used"]],
        sr,
        config["fp"],
        config["fs"],
        config["gpass"],
        config["gstop"],
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

    for snr in snr_db:
        if snr is None:
            y_noisy = y
            snr_label = "no_noise"
        else:
            repeats = int(np.ceil(len(y) / len(waterflow_noise)))
            current_waterflow_noise = np.tile(waterflow_noise, repeats)[: len(y)]
            y_noisy = add_waterflow_noise(y, current_waterflow_noise, snr)
            snr_label = f"SNR={snr}"

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

        print(f"Debug: Type of y_noisy: {type(y_noisy)}")
        if isinstance(y_noisy, np.ndarray):
            print(f"Debug: Shape of y_noisy: {y_noisy.shape}")
        else:
            print("Debug: y_noisy is not a numpy array")

        for i in range(0, len(y_noisy), data_length):
            chunk = y_noisy[i : i + data_length]
            if len(chunk) < data_length:
                chunk = np.pad(chunk, (0, data_length - len(chunk)), "constant")

            amplitude = calc_stft(chunk, config["sample_number"], sr)
            amplitude_cropped = amplitude[:, : max_k + 1]
            resized_amplitude = resize(amplitude_cropped, (224, 224))

            time_axis = np.linspace(0, chunk_seconds, resized_amplitude.shape[0])
            plt.figure(figsize=(6, 6))
            plt.imshow(
                # Internal arrays are (time, frequency). Transpose only for
                # conventional display: x=time, y=frequency.
                resized_amplitude.T,
                cmap="jet",
                aspect="auto",
                origin="lower",
                extent=[time_axis[0], time_axis[-1], 0, max_freq_khz_display],
            )

            if not config["png_with_axes"]:
                plt.axis("off")

            tick_positions_sec = [0, chunk_seconds / 2, chunk_seconds]
            tick_labels_str = [f"{p:g}" for p in tick_positions_sec]
            plt.xticks(tick_positions_sec, labels=tick_labels_str, fontsize=20)

            y_ticks_positions = np.arange(0, max_freq_khz_display + 1, step=1)
            if len(y_ticks_positions) > 10:
                step = math.ceil(len(y_ticks_positions) / 5)
                y_ticks_positions = np.arange(0, max_freq_khz_display + 1, step=step)

            y_ticks_labels = [f"{int(p)}" for p in y_ticks_positions]
            plt.yticks(y_ticks_positions, labels=y_ticks_labels, fontsize=20)

            plt.xlabel("Time s", fontsize=25, labelpad=2)
            plt.ylabel("Frequency kHz", fontsize=25, labelpad=2)
            plt.subplots_adjust(left=0.25, right=0.85, top=0.85, bottom=0.25)

            file_stem = f"{extracted_label}_{i // data_length}"
            if config["save_npy"]:
                npy_save_path = os.path.join(
                    npy_snr_save_folder_path, f"{file_stem}.npy"
                )
                np.save(npy_save_path, resized_amplitude)
            if config["save_spectrogram_png"]:
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


def get_sorted_wav_files(folder_path):
    wav_files = sorted(
        [name for name in os.listdir(folder_path) if name.endswith(".wav")],
        key=wav_index_from_name,
    )
    if not wav_files:
        raise FileNotFoundError(f"wav files not found in: {folder_path}")
    return wav_files
