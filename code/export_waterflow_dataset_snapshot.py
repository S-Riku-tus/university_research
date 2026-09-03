"""Audit an ignored water-flow dataset and export Git-friendly evidence.

The generated NPY arrays and per-condition chunk manifests are intentionally
kept outside Git.  This script records their counts, hashes, realized-SNR
distribution, and paired-noise invariants without copying the large dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLUMNS = ["source_wav_id", "chunk_index"]
PAIR_COLUMNS = ["noise_seed", "noise_offset_samples", "raw_noise_chunk_power"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        action="append",
        required=True,
        type=Path,
        help="Experiment-specific waterflow dataset root (repeatable).",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def condition_sort_key(path: Path) -> tuple[int, float]:
    if path.name == "heatflux_no_noise":
        return (0, 0.0)
    try:
        return (1, -float(path.name.rsplit("=", 1)[1]))
    except (IndexError, ValueError):
        return (2, 0.0)


def max_relative_noise_power_error(frame: pd.DataFrame) -> float:
    noisy = frame.dropna(subset=["requested_snr_db", "scaled_noise_chunk_power"])
    if noisy.empty:
        return float("nan")
    expected = (
        noisy["fixed_reference_signal_rms"].astype(float) ** 2
        / np.power(10.0, noisy["requested_snr_db"].astype(float) / 10.0)
    )
    actual = noisy["scaled_noise_chunk_power"].astype(float)
    return float(np.max(np.abs(actual - expected) / expected))


def all_values_match(frame: pd.DataFrame, value_column: str) -> bool:
    counts = frame.groupby(KEY_COLUMNS, dropna=False)[value_column].nunique(dropna=False)
    return bool((counts == 1).all())


def global_manifest_projection(manifest: dict) -> dict:
    return {
        "reference_rms": manifest["reference_rms"],
        "reference_scope": manifest["reference_scope"],
        "source_count": manifest["source_count"],
        "label_or_individual_source_used_for_noise_level": manifest[
            "label_or_individual_source_used_for_noise_level"
        ],
        "calibrated_spl": manifest["calibrated_spl"],
        "filter": manifest["filter"],
        "source_rms": manifest["source_rms"],
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    condition_rows: list[dict] = []
    pairing_rows: list[dict] = []
    source_manifests: list[dict] = []
    global_projections: list[dict] = []

    for dataset_root in args.dataset_root:
        global_path = dataset_root / "global_reference_manifest.json"
        if not global_path.is_file():
            raise FileNotFoundError(f"Missing global manifest: {global_path}")
        global_manifest = json.loads(global_path.read_text(encoding="utf-8"))
        global_projections.append(global_manifest_projection(global_manifest))
        source_manifests.append(
            {
                "dataset_root": dataset_root.as_posix(),
                "global_reference_manifest_sha256": sha256(global_path),
            }
        )

        experiment_name = dataset_root.parents[2].name
        for frequency_root in sorted(
            (path for path in dataset_root.iterdir() if path.is_dir()),
            key=lambda path: float(path.name.split("=")[1].replace("kHz", "")),
        ):
            condition_frames = []
            condition_roots = sorted(
                (path for path in frequency_root.iterdir() if path.is_dir()),
                key=condition_sort_key,
            )
            for condition_root in condition_roots:
                manifest_path = condition_root / "chunk_manifest.csv"
                if not manifest_path.is_file():
                    raise FileNotFoundError(f"Missing chunk manifest: {manifest_path}")
                frame = pd.read_csv(manifest_path)
                frame["condition_name"] = condition_root.name
                condition_frames.append(frame)

                requested = frame["requested_snr_db"].dropna().unique()
                if len(requested) > 1:
                    raise ValueError(f"Multiple requested SNR values in {manifest_path}")
                realized = pd.to_numeric(frame["realized_snr_db"], errors="coerce")
                npy_count = sum(1 for path in condition_root.iterdir() if path.suffix == ".npy")
                condition_rows.append(
                    {
                        "experiment_name": experiment_name,
                        "max_frequency": frequency_root.name,
                        "condition_name": condition_root.name,
                        "requested_snr_db": float(requested[0]) if len(requested) else np.nan,
                        "manifest_rows": int(len(frame)),
                        "npy_file_count": int(npy_count),
                        "npy_count_matches_manifest": bool(npy_count == len(frame)),
                        "source_wav_groups": int(frame["source_wav_id"].nunique()),
                        "heat_flux_levels": int(frame["heat_flux"].nunique()),
                        "realized_snr_mean_db": float(realized.mean()),
                        "realized_snr_median_db": float(realized.median()),
                        "realized_snr_std_db": float(realized.std()),
                        "realized_snr_min_db": float(realized.min()),
                        "realized_snr_max_db": float(realized.max()),
                        "mean_signal_chunk_power": float(frame["signal_chunk_power"].mean()),
                        "mean_scaled_noise_chunk_power": float(
                            frame["scaled_noise_chunk_power"].mean()
                        ),
                        "fixed_reference_signal_rms": float(
                            frame["fixed_reference_signal_rms"].iloc[0]
                        ),
                        "noise_scaling_mode": str(frame["noise_scaling_mode"].iloc[0]),
                        "pair_noise_across_snr": bool(
                            frame["pair_noise_across_snr"].iloc[0]
                        ),
                        "max_relative_noise_power_error": max_relative_noise_power_error(
                            frame
                        ),
                        "chunk_manifest_sha256": sha256(manifest_path),
                    }
                )

            combined = pd.concat(condition_frames, ignore_index=True)
            noisy = combined.dropna(subset=["requested_snr_db"])
            noisy_counts = noisy.groupby(KEY_COLUMNS, dropna=False).size()
            all_counts = combined.groupby(KEY_COLUMNS, dropna=False).size()
            pairing_rows.append(
                {
                    "experiment_name": experiment_name,
                    "max_frequency": frequency_root.name,
                    "sample_keys": int(len(all_counts)),
                    "condition_count": int(combined["condition_name"].nunique()),
                    "all_sample_keys_present_in_7_conditions": bool(
                        (all_counts == 7).all()
                    ),
                    "all_sample_keys_present_in_6_noisy_conditions": bool(
                        (noisy_counts == 6).all()
                    ),
                    "signal_power_matches_across_conditions": all_values_match(
                        combined, "signal_chunk_power"
                    ),
                    "heat_flux_matches_across_conditions": all_values_match(
                        combined, "heat_flux"
                    ),
                    **{
                        f"{column}_matches_across_snr": all_values_match(noisy, column)
                        for column in PAIR_COLUMNS
                    },
                }
            )

    if any(item != global_projections[0] for item in global_projections[1:]):
        raise ValueError("Global reference manifests do not share the same data definition")

    conditions = pd.DataFrame(condition_rows).sort_values(
        ["experiment_name", "max_frequency", "condition_name"]
    )
    pairing = pd.DataFrame(pairing_rows).sort_values(
        ["experiment_name", "max_frequency"]
    )
    conditions.to_csv(
        output_dir / "condition_audit.csv", index=False, float_format="%.10g"
    )
    pairing.to_csv(output_dir / "paired_noise_audit.csv", index=False)

    source_rows = []
    for item in global_projections[0]["source_rms"]:
        source_path = Path(item["path"])
        source_rows.append(
            {
                "experiment_name": source_path.parts[3],
                "source_wav": source_path.as_posix(),
                "filtered_rms": item["filtered_rms"],
            }
        )
    pd.DataFrame(source_rows).to_csv(
        output_dir / "source_wav_filtered_rms.csv", index=False, float_format="%.10g"
    )

    audit = {
        "schema_version": 1,
        "large_dataset_is_git_ignored": True,
        "source_manifests": source_manifests,
        "semantic_global_manifests_match": True,
        "fixed_reference_signal_rms": global_projections[0]["reference_rms"],
        "source_wav_count": int(len(source_rows)),
        "condition_count": int(len(conditions)),
        "manifest_row_count": int(conditions["manifest_rows"].sum()),
        "npy_file_count": int(conditions["npy_file_count"].sum()),
        "all_npy_counts_match_manifests": bool(
            conditions["npy_count_matches_manifest"].all()
        ),
        "all_paired_noise_checks_pass": bool(
            pairing.drop(columns=["experiment_name", "max_frequency", "sample_keys", "condition_count"])
            .all()
            .all()
        ),
        "generated_files": [
            "condition_audit.csv",
            "paired_noise_audit.csv",
            "source_wav_filtered_rms.csv",
        ],
    }
    (output_dir / "dataset_snapshot_manifest.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
