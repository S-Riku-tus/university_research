"""Experiment-specific ONB thresholds with auditable local provenance."""

from copy import deepcopy


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
    "2025.07.09_0.3_1": {
        "threshold": 571694.252491167,
        "source": "experiments/2026-09-24_selection_onb_ig_review/README.md#2-onb値の確定",
        "definition": "experiment-specific ONB confirmed for the current run",
    },
}


def onb_threshold_by_experiment():
    return {
        experiment_name: float(record["threshold"])
        for experiment_name, record in ONB_THRESHOLD_RECORDS.items()
    }


def onb_threshold_provenance_by_experiment():
    return deepcopy(ONB_THRESHOLD_RECORDS)
