"""Experiment-specific ONB thresholds with auditable local provenance."""

from copy import deepcopy


# ``threshold`` is the exact heat-flux label of the first recorded source WAV
# identified as nucleate-boiling onset. ``reported_value`` preserves the rounded
# number written in the experiment result text file.
ONB_THRESHOLD_RECORDS = {
    "2025.06.11_0.3_2": {
        "threshold": 221505.1102,
        "reported_value": 3.69e5,
        "source": (
            "Pool_boiling/Subcooling_20_degrees/0.3/2025.06.11_0.3_2/"
            "実験結果2025.06.11_0.3_2/2025.06.11_0.3_2.txt:3"
        ),
        "definition": "first measured heat-flux level identified as ONB",
    },
    "2025.06.18_0.3_3": {
        "threshold": 271677.6816,
        "reported_value": 3.76e5,
        "source": (
            "Pool_boiling/Subcooling_20_degrees/0.3/2025.06.18_0.3_3/"
            "実験結果2025.06.18_0.3_3/2025.06.18_0.3_3.txt:3"
        ),
        "definition": "first measured heat-flux level identified as ONB",
    },
    "2025.07.09_0.3_1": {
        "threshold": 571694.252491167,
        "reported_value": 4.42e5,
        "source": (
            "Pool_boiling/Subcooling_20_degrees/0.3/2025.07.09_0.3_1/"
            "実験結果2025.07.09_0.3_1/2025.07.09_0.3_1.txt:3"
        ),
        "definition": "first measured heat-flux level identified as ONB",
    },
}


def onb_threshold_by_experiment():
    return {
        experiment_name: float(record["threshold"])
        for experiment_name, record in ONB_THRESHOLD_RECORDS.items()
    }


def onb_threshold_provenance_by_experiment():
    return deepcopy(ONB_THRESHOLD_RECORDS)
