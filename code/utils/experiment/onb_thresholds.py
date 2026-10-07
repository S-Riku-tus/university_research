"""Experiment-specific ONB thresholds with auditable local provenance."""

from copy import deepcopy
import csv
import hashlib
import math
from pathlib import Path

from utils.calculation.heatflux_preprocessing import resolve_heat_flux_csv_path


# These are the experiment-specific ONB values confirmed for the current runs.
# The automatic linearity-loss estimates emitted by the old ``0.*`` notebooks
# are diagnostics, not authoritative ONB labels.
ONB_THRESHOLD_RECORDS = {
    "2025.06.11_0.3_2": {
        "threshold": 221505.1102,
        "source": "experiments/2026-09-24_selection_onb_ig_review/README.md#2-onb値の確定",
        "definition": "experiment-specific ONB confirmed for the current run",
    },
    "2025.06.18_0.3_3": {
        "threshold": 271677.6816,
        "source": "experiments/2026-09-24_selection_onb_ig_review/README.md#2-onb値の確定",
        "definition": "experiment-specific ONB confirmed for the current run",
    },
    "2025.06.11_0.3_2_6.18_0.3_3": {
        "threshold": 246591.3959,
        "source": "experiments/2026-09-29_within_wav_chunk_combined/README.md",
        "definition": (
            "user-directed arithmetic mean of the confirmed 2025-06-11 and "
            "2025-06-18 ONB thresholds for the combined-dataset comparison"
        ),
    },
    "2025.07.09_0.3_1": {
        "threshold": 571694.252491167,
        "source": "experiments/2026-09-24_selection_onb_ig_review/README.md#2-onb値の確定",
        "definition": "experiment-specific ONB confirmed for the current run",
    },
}


REPO_ROOT = Path(__file__).resolve().parents[3]

# Opt-in records for new experiments: existing audit scripts retain their original scope.
# The observed voltage is an experimental observation, not an automatic linearity estimate.
CSV_ONB_THRESHOLD_RECORDS = {
    "2026.10.07_0.3_1": {
        "threshold": None,
        "applied_voltage_v": 1.2,
        "heat_flux_csv": (
            "Pool_boiling/Subcooling_20_degrees/0.3/2026.10.07_0.3_1/"
            "実験結果2026.10.07_0.3_1/heat_flux_2026.10.07_0.3_1.csv"
        ),
        "source": (
            "Pool_boiling/Subcooling_20_degrees/0.3/2026.10.07_0.3_1/"
            "raw/experiment_notes/2026-10-07_06-57-53-export.elabftw.csv"
        ),
        "definition": "Heat flux at the 1.2 V boiling-start observation in the experiment notes",
        "heat_flux_unit": "W/m2",
    },
}


def resolve_csv_onb_threshold_record(experiment_name, repo_root=None):
    """Resolve one observed ONB voltage against a completed, reviewed label CSV."""
    record = deepcopy(CSV_ONB_THRESHOLD_RECORDS[experiment_name])
    root = Path(repo_root or REPO_ROOT)
    nominal_path = root / record["heat_flux_csv"]
    path = resolve_heat_flux_csv_path(nominal_path.parents[1], repo_root=root)
    record["heat_flux_csv"] = path.relative_to(root).as_posix()
    if not path.is_file():
        record["status"] = "waiting_for_heat_flux_csv"
        return record
    candidates = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"volt", "q"}.issubset(reader.fieldnames or []):
            raise ValueError(f"ONB label CSV needs volt and q columns: {path}")
        for index, row in enumerate(reader, start=1):
            text = str(row["volt"]).strip().casefold()
            if text.endswith("v"):
                text = text[:-1].strip()
            try:
                voltage = float(text)
            except ValueError as error:
                raise ValueError(f"Invalid applied voltage in ONB label CSV: row {index}, {path}") from error
            if math.isclose(voltage, record["applied_voltage_v"], rel_tol=0, abs_tol=1e-9):
                candidates.append((index, row))
    if len(candidates) != 1:
        raise ValueError(
            f"ONB label CSV must have exactly one {record['applied_voltage_v']:g} V row; "
            f"found {len(candidates)}: {path}"
        )
    index, row = candidates[0]
    try:
        threshold = float(row["q"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid ONB heat flux q in W/m²: {path}, row {index}") from error
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError(f"ONB heat flux q must be finite positive W/m²: {path}, row {index}")
    record.update(threshold=threshold, status="resolved_from_heat_flux_csv",
                  csv_row_index=index, heat_flux_csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return record


def onb_threshold_by_experiment(csv_experiments=()):
    thresholds = {
        experiment_name: float(record["threshold"])
        for experiment_name, record in ONB_THRESHOLD_RECORDS.items()
    }
    for name in csv_experiments:
        thresholds[name] = resolve_csv_onb_threshold_record(name)["threshold"]
    return thresholds


def onb_threshold_provenance_by_experiment(csv_experiments=()):
    records = deepcopy(ONB_THRESHOLD_RECORDS)
    for name in csv_experiments:
        records[name] = resolve_csv_onb_threshold_record(name)
    return records
