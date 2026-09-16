"""Export linear 0-3 kHz spectra, 60 s peak traces, and threshold sensitivity.

Reuses the already source-verified PSD, without changing raw audio or model NPY.
The selected height is an exploratory acoustic inclusion rule, not event truth.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.experiment.spectral_peaks import PEAK_WINDOWS_HZ, PEAK_FEATURE, peak_features, classify_peak_height

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/2026-09-16_day_split_spectral_selection"
OUT = ROOT / "experiments/2026-09-16_peak_height_selection"
THRESHOLD = 1e-9
STRONG = 10e-9
CANDIDATES = (0.1, 0.3, 1., 3., 10., 30.)  # plot units: 10^-9 digital amplitude^2/Hz
TRAIN_DAYS = {"2025.06.11_0.3_2", "2025.07.09_0.3_1"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-images", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "timelines").mkdir(exist_ok=True)
    rows, sensitivity, manifests, browser_wavs = [], [], [], []
    for archive in sorted(SOURCE.glob("spectra_*.npz")):
        day = archive.stem[len("spectra_"):]
        data = np.load(archive)
        f, psds = data["frequency_hz"], data["psd"]
        meta = pd.read_csv(SOURCE / f"chunks_{day}.csv")
        if len(meta) != len(psds):
            raise ValueError("PSD / metadata mismatch")
        for i, row in meta.iterrows():
            record = {k: row[k] for k in ("experiment_name", "source_wav_id", "chunk_index", "chunk_start_seconds",
                      "chunk_duration_seconds", "heat_flux", "sample_filename", "onb_threshold", "q_over_onb")}
            record.update(peak_features(f, psds[i]))
            rows.append(record)
        peaks = pd.DataFrame(rows[-len(meta):])
        for wav, g in peaks.groupby("source_wav_id", sort=False):
            g = g.sort_values("chunk_index")
            if len(g) != 60 or not np.array_equal(g.chunk_index, np.arange(60)):
                raise ValueError("Expected 60 contiguous one-second chunks")
            ids = g.index.to_numpy()
            wave_psds = psds[ids]
            heights = g[PEAK_FEATURE].to_numpy()
            peak_hz = g[PEAK_FEATURE.replace("psd", "hz")].to_numpy()
            q = float(g.heat_flux.iloc[0])
            eligible = q >= float(g.onb_threshold.iloc[0])
            statuses = classify_peak_height(heights, THRESHOLD, STRONG)
            for threshold in CANDIDATES:
                passes = heights >= threshold * 1e-9
                sensitivity.append({"experiment_name": day, "role": "train" if day in TRAIN_DAYS else "test_description_only",
                    "source_wav_id": wav, "heat_flux": q, "q_over_onb": float(g.q_over_onb.iloc[0]),
                    "threshold_in_1e9_units": threshold, "n_peak_pass": int(passes.sum()),
                    "n_training_retained_if_applied": int(passes.sum()) if eligible else 60,
                    "n_strong": int((statuses == "strong").sum()),
                    "passing_chunk_indices": "|".join(map(str, g.chunk_index.to_numpy()[passes]))})
            stem = f"{day}_{wav}"
            t = g.chunk_start_seconds.to_numpy()
            fig, ax = plt.subplots(3, 1, figsize=(12, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 2, 1]})
            for lo, hi in PEAK_WINDOWS_HZ:
                ax[0].plot(t + .5, g[f"peak_{lo}_{hi}_psd"] * 1e9, marker=".", label=f"{lo}-{hi} Hz")
            for panel in ax[:2]:
                panel.axhline(THRESHOLD * 1e9, c="tab:red", ls="--", label="Include >= 1")
                panel.grid(alpha=.2)
            ax[0].axhline(STRONG * 1e9, c="gray", ls=":", label="Strong >= 10 (description)")
            ax[0].set_ylabel("Peak PSD (x 1e-9)")
            ax[0].set_title(f"{day} | {wav} | q={q/1000:.2f} kW/m2 | q/qONB={g.q_over_onb.iloc[0]:.3f}")
            ax[0].legend(fontsize=8, ncol=3)
            ax[1].plot(t + .5, heights * 1e9, color="tab:blue", marker="o", markersize=3)
            ax[1].set(ylim=(0, 12), ylabel="2100-2500 Hz peak\n(threshold detail)")
            colors = {"strong": "#174a7e", "weak_included": "#38a88c", "below_threshold": "#cccccc"}
            ax[2].bar(t, np.ones(len(t)), width=1, align="edge", color=[colors[s] for s in statuses], edgecolor="white")
            ax[2].set(xlim=(0, 60), ylim=(0, 1), yticks=[], xlabel="Time within this 60 s recording (s)")
            ax[2].set_title(f"Blue: >=10 | Green: 1 to <10 | Gray: <1 | Acoustic pass {(heights >= THRESHOLD).sum()}/60"
                            + (" | Used for training selection" if eligible and day in TRAIN_DAYS else " | Description only / pre-ONB kept"), fontsize=9)
            fig.tight_layout()
            fig.savefig(OUT / "timelines" / f"{stem}.png", dpi=130)
            plt.close(fig)
            visible = (f >= 0) & (f <= 3000)
            spectrum = wave_psds[:, visible] * 1e9
            browser_wavs.append({"day": day, "wav": wav, "q": q, "q_onb": float(g.onb_threshold.iloc[0]),
                "role": "training" if day in TRAIN_DAYS else "test: description only",
                "peaks": np.round(heights * 1e9, 6).tolist(), "peak_hz": np.round(peak_hz, 2).tolist(),
                "frequency": np.round(f[visible], 3).tolist(), "spectra": np.round(spectrum, 6).tolist()})
            if not args.skip_images:
                img_dir = ROOT / "Pool_boiling/Subcooling_20_degrees/0.3" / day / "data/spectrum_peak_png/waterflow_20260817_1s" / wav
                img_dir.mkdir(parents=True, exist_ok=True)
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                lines = []
                for panel, ymax in zip(axes, [max(12, float(spectrum.max()) * 1.05), 12]):
                    lines.append(panel.plot([], [], lw=.9)[0])
                    panel.set(xlim=(0, 3000), ylim=(0, ymax), xlabel="Frequency (Hz)", ylabel="PSD (x 1e-9 digital amplitude^2/Hz)")
                    panel.axvspan(2100, 2500, color="tab:orange", alpha=.1)
                    panel.hlines(1., 2100, 2500, color="tab:red", linestyle="--")
                    panel.grid(alpha=.25)
                axes[0].set_title("Full height (fixed for this WAV)", fontsize=10)
                axes[1].set_title("Threshold detail (same scale for all WAVs)", fontsize=10)
                fig.tight_layout(rect=(0, 0, 1, .87))
                for i in range(60):
                    for line in lines:
                        line.set_data(f[visible], spectrum[i])
                    fig.suptitle(f"{day} | {wav} | {i}-{i+1} s\nPeak={heights[i]*1e9:.3g} at {peak_hz[i]:.0f} Hz | {statuses[i]}", fontsize=10)
                    fig.savefig(img_dir / f"chunk-{i:04d}.png", dpi=100)
                plt.close(fig)
            print(f"{day} {wav}: peak pass {(heights >= THRESHOLD).sum()}/60", flush=True)
        manifests.append({"experiment_name": day, "n_chunks": len(meta), "n_wavs": peaks.source_wav_id.nunique(),
                          "source_psd": str(archive.relative_to(ROOT)), "source_psd_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                          "source_metadata_sha256": hashlib.sha256((SOURCE / f"chunks_{day}.csv").read_bytes()).hexdigest()})
    pd.DataFrame(rows).to_csv(OUT / "peak_features.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(OUT / "threshold_sensitivity.csv", index=False)
    browser_wavs.sort(key=lambda w: (w["day"], w["q"]))
    (OUT / "review_data.json").write_text(json.dumps(browser_wavs, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    template = (ROOT / "templates/peak_threshold_review.html").read_text(encoding="utf-8")
    (OUT / "peak_threshold_review.html").write_text(template.replace("__REVIEW_DATA__", json.dumps(browser_wavs, ensure_ascii=False, separators=(",", ":"))), encoding="utf-8")
    image_count = sum(len(list((ROOT / "Pool_boiling/Subcooling_20_degrees/0.3" / m["experiment_name"] /
        "data/spectrum_peak_png/waterflow_20260817_1s").glob("*/*.png"))) for m in manifests)
    (OUT / "peak_manifest.json").write_text(json.dumps({"datasets": manifests, "peak_windows_hz": PEAK_WINDOWS_HZ,
        "rule": "maximum linear PSD ordinate in 2100-2500 Hz >= fixed 1e-9",
        "image_count": image_count, "images_written_this_run": 0 if args.skip_images else len(rows), "temporal_padding": False,
        "strong_threshold_for_description_only": STRONG, "retention_threshold": THRESHOLD,
        "source_verified_against_production_manifest": str((SOURCE / "spectrum_manifest.json").relative_to(ROOT)),
        "scientific_status": "exploratory threshold sensitivity, no bubble-event ground truth"}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
