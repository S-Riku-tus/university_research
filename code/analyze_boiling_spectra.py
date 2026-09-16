"""Export manifest-aligned, one-second clean spectra without per-image scaling.

Uses exactly the source loading and 500 Hz high-pass of the production generator.
No model inputs are overwritten. Welch PSD is in digital amplitude squared / Hz,
not calibrated sound pressure. All 49 source WAVs / 2,940 seconds are exported.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from utils.experiment.onb_thresholds import onb_threshold_by_experiment

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / "Pool_boiling/Subcooling_20_degrees/0.3"
BANDS = [(500, 1000), (1000, 2000), (2000, 3000), (3000, 5000),
         (5000, 10000), (10000, 15000), (15000, 22050), (1000, 5000), (2000, 5000)]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/2026-09-16_day_split_spectral_selection")
    parser.add_argument("--skip-images", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("production_audio", ROOT / "code/2.run_npy_waterflow_2つhighpass.py")
    production = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(production)
    # The legacy preprocessing module changes global plotting fonts on import.
    plt.rcdefaults()
    thresholds = onb_threshold_by_experiment()
    all_rows, wav_rows, audit = [], [], []
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    lines = []
    for ax, limit in zip(axes, [5000, 22050]):
        lines.append(ax.plot([], [], lw=0.8)[0])
        ax.set(xlim=(500, limit), ylim=(-160, -50), xlabel="Frequency (Hz)", ylabel="PSD (dB re digital amplitude^2/Hz)")
        ax.grid(alpha=0.25)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    for day, onb in thresholds.items():
        folder = EXPERIMENT_ROOT / day
        source = folder / "data/npy/waterflow_20260817_1s/maxfreq=22kHz/heatflux_no_noise/chunk_manifest.csv"
        with source.open(encoding="utf-8-sig", newline="") as inp:
            rows = list(csv.DictReader(inp))
        by_wav = defaultdict(list)
        for row in rows:
            by_wav[row["source_wav_id"]].append(row)
        image_root = folder / "data/spectrum_png/waterflow_20260817_1s"
        day_psds, day_rows = [], []
        for wav_id, wav_chunks in by_wav.items():
            wav_path = folder / "録音データ_熱流束" / wav_chunks[0]["source_wav_name"]
            signal = production._load_and_filter(wav_path)
            psds = []
            wav_image_dir = image_root / wav_id
            if not args.skip_images:
                wav_image_dir.mkdir(parents=True, exist_ok=True)
            for row in sorted(wav_chunks, key=lambda r: int(r["chunk_index"])):
                start = int(row["chunk_start_sample"])
                count = int(round(float(row["chunk_duration_seconds"]) * 44100))
                chunk = signal[start:start + count]
                if len(chunk) != count:
                    raise ValueError(f"Short source chunk: {wav_path}, {start}")
                power = float(np.mean(chunk ** 2))
                if not np.isclose(power, float(row["signal_chunk_power"]), rtol=1e-7, atol=1e-18):
                    raise ValueError(f"Source/manifest power mismatch: {wav_path}, {start}")
                f, psd = welch(chunk, fs=44100, window="hann", nperseg=2048,
                               noverlap=1024, detrend="constant", scaling="density")
                record = {key: row[key] for key in ("experiment_name", "source_wav_id", "chunk_index",
                          "chunk_start_seconds", "chunk_duration_seconds", "heat_flux", "sample_filename")}
                record.update(onb_threshold=onb, q_over_onb=float(row["heat_flux"]) / onb,
                              signal_power=power)
                for low, high in BANDS:
                    bandpower = float(psd[(f >= low) & (f < high)].sum() * (f[1] - f[0]))
                    record[f"band_{low}_{high}_db"] = float(10 * np.log10(max(bandpower, 1e-30)))
                all_rows.append(record)
                day_rows.append(record)
                psds.append(psd)
                if not args.skip_images:
                    for line in lines:
                        line.set_data(f, 10 * np.log10(np.maximum(psd, 1e-30)))
                    fig.suptitle(f"{day} | {wav_id} | {start / 44100:.0f}-{(start + count) / 44100:.0f} s\n"
                                 f"q={float(row['heat_flux']) / 1000:.2f} kW/m2; q/q_ONB={record['q_over_onb']:.3f}", fontsize=10)
                    fig.savefig(wav_image_dir / f"chunk-{int(row['chunk_index']):04d}.png", dpi=100)
            stack = np.asarray(psds)
            day_psds.extend(psds)
            wav_rows.append({"experiment_name": day, "source_wav_id": wav_id,
                             "heat_flux": wav_chunks[0]["heat_flux"], "n_chunks": len(psds),
                             **{f"band_{lo}_{hi}_db_median": float(np.median([r[f"band_{lo}_{hi}_db"] for r in day_rows[-len(psds):]])) for lo, hi in BANDS}})
            print(f"{day} {wav_id}: {len(psds)} spectra", flush=True)
        np.savez_compressed(args.output / f"spectra_{day}.npz", frequency_hz=f, psd=np.asarray(day_psds))
        write_csv(args.output / f"chunks_{day}.csv", day_rows)
        audit.append({"experiment": day, "n_wavs": len(by_wav), "n_chunks": len(day_rows),
                      "manifest": str(source.relative_to(ROOT)),
                      "manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                      "image_directory": str(image_root.relative_to(ROOT))})
        # One overview per day includes every second, arranged in heat-flux order.
        order = sorted(range(len(day_rows)), key=lambda i: (float(day_rows[i]["heat_flux"]), int(day_rows[i]["chunk_index"])))
        summary, ax = plt.subplots(figsize=(11, 5))
        im = ax.imshow(10 * np.log10(np.maximum(np.asarray(day_psds)[order].T, 1e-30)),
                       aspect="auto", origin="lower", extent=(0, len(order), 0, f[-1] / 1000), vmin=-145, vmax=-80)
        ax.set(xlabel="All 1 s chunks, ordered by heat flux then time", ylabel="Frequency (kHz)", title=day)
        summary.colorbar(im, ax=ax, label="PSD (dB re digital amplitude^2/Hz)")
        summary.tight_layout()
        summary.savefig(args.output / f"overview_{day}.png", dpi=150)
        plt.close(summary)
    plt.close(fig)
    write_csv(args.output / "spectral_features.csv", all_rows)
    write_csv(args.output / "wav_band_summary.csv", wav_rows)
    (args.output / "spectrum_manifest.json").write_text(json.dumps({
        "method": "Welch Hann 2048 / overlap 1024 / 44100 Hz; production 500 Hz high-pass",
        "unit": "uncalibrated digital amplitude squared per Hz",
        "normalization": "none; fixed image scales", "noise": "clean source only",
        "source_power_verified_for_every_chunk": True, "datasets": audit,
        "images_written": not args.skip_images,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
