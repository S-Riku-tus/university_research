"""Read-only checks of the new experiment's generated acoustic inputs."""
import csv
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.transform import resize

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
from utils.experiment.run_helpers import open_text, path_exists, windows_long_path

EXPERIMENT = "2026.10.07_0.3_1"
BASE = ROOT / "Pool_boiling/Subcooling_20_degrees/0.3" / EXPERIMENT
LABELS = BASE / f"実験結果{EXPERIMENT}/heat_flux_{EXPERIMENT}.csv"


def read_csv(path):
    with open_text(path, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def digest(path):
    value = hashlib.sha256()
    with open(windows_long_path(path), "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def main():
    labels = read_csv(LABELS)
    label_map = {i: row for i, row in enumerate(labels, 1)}
    onset_rows = [row for row in labels if np.isclose(float(row["volt"].rstrip("Vv")), 1.2)]
    if len(onset_rows) != 1:
        raise ValueError("ONB observation voltage must select exactly one label row")
    threshold = float(onset_rows[0]["q"])
    datasets, errors, main_rows, spectra, examples = [], [], [], {}, {}
    npy_root = BASE / "data/npy"
    for dataset in sorted(npy_root.glob("waterflow_20261007_*s")):
        for frequency in sorted(dataset.glob("maxfreq=*")):
            preprocess = json.loads((frequency / "preprocess_manifest.json").read_text(encoding="utf-8"))
            clean = frequency / "heatflux_no_noise"
            rows = read_csv(clean / "chunk_manifest.csv")
            png_clean = BASE / "data/spectrogram_png" / dataset.name / frequency.name / "heatflux_no_noise"
            files = {path.name: path for path in clean.glob("*.npy")}
            png_files = {path.stem for path in png_clean.glob("*.png")}
            expected = {row["sample_filename"] for row in rows}
            is_main = dataset.name.endswith("_1s") and frequency.name == "maxfreq=3kHz"
            if expected != set(files):
                errors.append(f"{dataset.name}/{frequency.name}: NPY/manifest mismatch")
            if png_files != {Path(name).stem for name in expected}:
                errors.append(f"{dataset.name}/{frequency.name}: PNG/manifest mismatch")
            shape_counts, dtype_counts, source_chunks = Counter(), Counter(), defaultdict(list)
            source_sum, source_count = {}, Counter()
            array_min, array_max = float("inf"), float("-inf")
            for row in rows:
                name, source = row["sample_filename"], row["source_wav_id"]
                array = np.load(windows_long_path(files[name]), allow_pickle=False)
                shape_counts[str(array.shape)] += 1
                dtype_counts[str(array.dtype)] += 1
                array_min, array_max = min(array_min, float(array.min())), max(array_max, float(array.max()))
                if array.shape != (224, 224) or array.dtype != np.float32:
                    errors.append(f"{dataset.name}/{frequency.name}/{name}: shape or dtype")
                if not np.isfinite(array).all() or (array < 0).any() or not (array > 0).any():
                    errors.append(f"{dataset.name}/{frequency.name}/{name}: invalid power")
                index = int(re.search(r"index=(\d+)", source).group(1))
                if float(row["heat_flux"]) != float(label_map[index]["q"]):
                    errors.append(f"{name}: label differs from q CSV")
                if float(name.split("_src-")[0]) != float(row["heat_flux"]):
                    errors.append(f"{name}: filename label differs")
                seconds, chunk = float(row["chunk_duration_seconds"]), int(row["chunk_index"])
                if seconds != float(preprocess["chunk_seconds"]) or float(row["chunk_start_seconds"]) != seconds * chunk:
                    errors.append(f"{name}: chunk time mapping")
                if int(row["unpadded_signal_samples"]) != int(seconds * 44100):
                    errors.append(f"{name}: padded/short chunk")
                if row["requested_snr_db"] or row["scaled_noise_chunk_power"]:
                    errors.append(f"{name}: unexpected noise in clean dataset")
                if float(row["signal_chunk_power"]) != float(row["model_input_power"]):
                    errors.append(f"{name}: clean input power differs from signal")
                with Image.open(windows_long_path(png_clean / (Path(name).stem + ".png"))) as picture:
                    picture.verify()
                source_chunks[source].append(chunk)
                if is_main:
                    record = {**row, "voltage": label_map[index]["volt"], "positive_by_note": int(float(row["heat_flux"]) >= threshold)}
                    main_rows.append(record)
                    source_sum.setdefault(source, np.zeros(224, dtype=np.float64))
                    source_sum[source] += array.mean(axis=0, dtype=np.float64)
                    source_count[source] += 1
                    if chunk == 0 and label_map[index]["volt"] in ["1.0V", "1.1V", "1.2V", "1.3V"]:
                        examples[label_map[index]["volt"]] = (array.copy(), record)
            expected_chunks = round(60 / float(preprocess["chunk_seconds"]))
            if len(source_chunks) != 11 or any(sorted(indices) != list(range(expected_chunks)) for indices in source_chunks.values()):
                errors.append(f"{dataset.name}/{frequency.name}: incomplete/duplicate source chunk coverage")
            datasets.append({"dataset": dataset.name, "frequency": frequency.name,
                "npy": len(files), "png": len(png_files), "manifest_rows": len(rows), "source_wavs": len(source_chunks),
                "chunks_per_wav": sorted({len(indices) for indices in source_chunks.values()}),
                "shapes": dict(shape_counts), "dtypes": dict(dtype_counts), "array_min": array_min, "array_max": array_max,
                "noise_folders": sorted(path.name for path in frequency.glob("heatflux_*") if path.is_dir()),
                "manifest_created_at": preprocess["created_at"], "preprocess_manifest_sha256": digest(frequency / "preprocess_manifest.json"),
                "chunk_manifest_sha256": digest(clean / "chunk_manifest.csv"),
                "fixed_reference_rms": preprocess["noise"]["fixed_reference_signal_rms"]})
            if is_main:
                spectra = {source: source_sum[source] / source_count[source] for source in source_sum}

    by_source = defaultdict(list)
    for row in main_rows:
        by_source[row["source_wav_id"]].append(row)
    source_summary = []
    for source in sorted(by_source, key=lambda value: int(re.search(r"index=(\d+)", value).group(1))):
        rows = by_source[source]
        source_summary.append({"source_wav_id": source, "voltage": rows[0]["voltage"],
            "heat_flux_kw_m2": float(rows[0]["heat_flux"]) / 1000, "chunks": len(rows),
            "positive_by_note": rows[0]["positive_by_note"],
            "rms_median_60_chunks": float(np.median([np.sqrt(float(row["signal_chunk_power"])) for row in rows])),
            "labelled_wav_sha256": digest(BASE / "録音データ_熱流束" / rows[0]["source_wav_name"])})

    # Reproduce only three predetermined clean chunks, without rewriting any input.
    entry = ROOT / "code/2.run_npy_waterflow_2つhighpass.py"
    spec = importlib.util.spec_from_file_location("audited_generation_entry", entry)
    generation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generation)
    reload_checks = []
    for voltage in ["1.0V", "1.2V", "1.3V"]:
        saved, row = examples[voltage]
        filtered = generation._load_and_filter(str(BASE / "録音データ_熱流束" / row["source_wav_name"]))
        regenerated = generation.save_spectrogram_chunks_with_snr.__globals__["calc_stft"](filtered[:44100], 672, 44100)
        max_k = round(3000 / (44100 / 1344))
        regenerated = resize(regenerated[:, :max_k + 1], (224, 224)).astype(np.float32)
        maximum = float(np.max(np.abs(regenerated.astype(np.float64) - saved)))
        reload_checks.append({"voltage": voltage, "chunk_index": 0, "maximum_absolute_difference": maximum,
            "bitwise_equal": bool(np.array_equal(saved, regenerated))})
        if not np.array_equal(saved, regenerated):
            errors.append(f"{voltage}: raw-WAV/STFT reproduction differs")

    with (OUT / "source_summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(source_summary[0]))
        writer.writeheader()
        writer.writerows(source_summary)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    vmax = max(float(np.percentile(np.log10(array + 1e-20), 99.5)) for array, _ in examples.values())
    vmin = vmax - 6
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3), constrained_layout=True)
    for axis, voltage in zip(axes, ["1.0V", "1.1V", "1.2V", "1.3V"]):
        array, row = examples[voltage]
        picture = axis.imshow(np.log10(array.T + 1e-20), origin="lower", aspect="auto", extent=[0, 1, 0, 3], vmin=vmin, vmax=vmax, cmap="magma")
        axis.set(title=f"{voltage} | q={float(row['heat_flux'])/1000:.2f}\nchunk 0", xlabel="Time (s)")
    axes[0].set_ylabel("Frequency (kHz; resized grid)")
    fig.colorbar(picture, ax=axes, label="log10 STFT power (uncalibrated)", shrink=0.85)
    for extension in ["png", "svg"]:
        fig.savefig(OUT / f"onb_neighbour_spectrograms.{extension}", dpi=150)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for voltage in ["1.0V", "1.1V", "1.2V", "1.3V"]:
        _, row = examples[voltage]
        axis.plot(np.linspace(0, 3, 224), spectra[row["source_wav_id"]], label=voltage)
    axis.set(xlabel="Frequency (kHz; approximate resized grid)", ylabel="Mean STFT power (uncalibrated)", yscale="log", title="Mean over all 60 one-second chunks per WAV")
    axis.axvspan(2.1, 2.5, alpha=0.12, color="grey")
    axis.legend()
    for extension in ["png", "svg"]:
        fig.savefig(OUT / f"onb_neighbour_mean_spectra.{extension}", dpi=150)
    plt.close(fig)
    summary = {"experiment": EXPERIMENT, "audit_date": "2026-10-08", "label_csv_rows": len(labels),
        "label_csv_sha256": digest(LABELS), "generation_entry_sha256": digest(entry),
        "onb_voltage_observation": "1.2V", "onb_q_w_m2": threshold, "datasets": datasets,
        "total_npy": sum(row["npy"] for row in datasets), "total_png": sum(row["png"] for row in datasets),
        "main_1s_3khz_chunks": len(main_rows), "main_positive_chunks": sum(int(row["positive_by_note"]) for row in main_rows),
        "main_negative_chunks": sum(not int(row["positive_by_note"]) for row in main_rows),
        "near_onb_plus_minus_10_percent_chunks": sum(abs(float(row["heat_flux"]) - threshold) <= threshold * .1 for row in main_rows),
        "regression_output_files": sum(path.is_file() for path in (BASE / "regression_result").rglob("*")),
        "raw_wav_reproduction": reload_checks, "errors": errors,
        "scope": "Input audit and descriptive plots only; no training, noise generation, or physical source identification."}
    write_json("verification.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key not in ["datasets", "raw_wav_reproduction"]}, ensure_ascii=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
