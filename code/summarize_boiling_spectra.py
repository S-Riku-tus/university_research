"""Create review figures and training-only selection audits from exported PSDs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.experiment.acoustic_selection import AcousticTrainingSelector
from utils.experiment.onb_thresholds import onb_threshold_by_experiment

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/2026-09-16_day_split_spectral_selection"
TRAIN = ["2025.06.11_0.3_2", "2025.07.09_0.3_1"]
FEATURE = "band_2000_3000_db"


def main():
    data = pd.read_csv(OUT / "spectral_features.csv")
    candidates = []
    for day, group in data.groupby("experiment_name"):
        for feat in [c for c in data if c.startswith("band_")]:
            threshold = group.loc[group.q_over_onb < 1, feat].quantile(.99)
            for q, wav in group.groupby("heat_flux"):
                candidates.append(dict(day=day, role="training" if day in TRAIN else "test_descriptive_only",
                    feature=feat, heat_flux=q, q_ratio=wav.q_over_onb.iloc[0], threshold_db=threshold,
                    median_db=wav[feat].median(), n=len(wav), kept=int((wav[feat] > threshold).sum())))
    pd.DataFrame(candidates).to_csv(OUT / "candidate_band_retention.csv", index=False)
    cfg = {"enabled": True, "features_csv": str((OUT / "spectral_features.csv").relative_to(ROOT)),
           "feature": FEATURE, "background_quantile": .99, "margin_db": 0.,
           "apply_max_heat_flux_by_experiment": {}}
    selector = AcousticTrainingSelector(cfg, onb_threshold_by_experiment())
    meta = data[data.experiment_name.isin(TRAIN)].to_dict("records")
    _, audit = selector.select(meta)
    (OUT / "training_selection_preview.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    decisions = pd.DataFrame(audit["decisions"])
    summary = decisions.groupby(["experiment_name", "source_wav_id", "heat_flux"]).keep.agg(["size", "sum"]).reset_index()
    summary.rename(columns={"size": "n_before", "sum": "n_retained"}).to_csv(OUT / "training_retention_by_wav.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    representatives = []
    for ax, (day, g) in zip(axes, data.groupby("experiment_name")):
        threshold = g.loc[g.q_over_onb < 1, FEATURE].quantile(.99)
        ax.scatter(g.heat_flux / 1000, g[FEATURE], s=5, alpha=.35)
        ax.axhline(threshold, color="tab:red", ls="--", label="Day pre-ONB 99th percentile")
        ax.axvline(g.onb_threshold.iloc[0] / 1000, color="black", ls=":", label="Recorded ONB")
        ax.set(title=day + (" (train)" if day in TRAIN else " (test: description only)"),
               xlabel="Measured heat flux (kW/m2)")
        ax.grid(alpha=.2)
        onset = g[np.isclose(g.q_over_onb, 1)]
        representatives.extend(onset.loc[[onset[FEATURE].idxmin(), onset[FEATURE].idxmax()]].to_dict("records"))
    axes[0].set_ylabel("2-3 kHz band power (dB re digital amplitude^2)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "band_power_by_heat_flux.png", dpi=160)
    plt.close(fig)
    pd.DataFrame(representatives).to_csv(OUT / "onb_quiet_loud_representatives.csv", index=False)
    # Six exact ONB seconds: PSD and existing model-input spectrogram side by side.
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for row_idx, (day, g) in enumerate(data.groupby("experiment_name")):
        archive = np.load(OUT / f"spectra_{day}.npz")
        onset = g[np.isclose(g.q_over_onb, 1)]
        pair = onset.loc[[onset[FEATURE].idxmin(), onset[FEATURE].idxmax()]]
        for col, (index, row) in enumerate(pair.iterrows(), 1):
            local_index = g.index.get_loc(index)
            axes[row_idx, 0].plot(archive["frequency_hz"] / 1000,
                                  10 * np.log10(np.maximum(archive["psd"][local_index], 1e-30)),
                                  label=f"chunk {int(row.chunk_index)} ({'quiet' if col == 1 else 'loud'})")
            png = ROOT / "Pool_boiling/Subcooling_20_degrees/0.3" / day / "data/spectrogram_png/waterflow_20260817_1s/maxfreq=22kHz/heatflux_no_noise" / Path(row.sample_filename).with_suffix(".png")
            axes[row_idx, col].imshow(plt.imread(png))
            axes[row_idx, col].set_title(f"Existing STFT: chunk {int(row.chunk_index)}")
            axes[row_idx, col].axis("off")
        axes[row_idx, 0].set(xlim=(.5, 5), ylim=(-150, -60), title=f"{day}: recorded ONB", xlabel="Frequency (kHz)", ylabel="PSD (dB / Hz)")
        axes[row_idx, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "onb_spectra_and_existing_spectrograms.png", dpi=140)
    plt.close(fig)
    print(json.dumps({"n_before": audit["n_before"], "n_after": audit["n_after"], "days": audit["by_experiment"]}, indent=2))


if __name__ == "__main__":
    main()
