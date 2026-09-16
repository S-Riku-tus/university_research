"""Two-epoch integration run on all real samples, baseline and selected training.

Not a converged accuracy experiment. Production config and older results stay
unchanged. Writes a dedicated smoke-test directory with predictions/audits.
"""
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import run_ensemble_regression_onb as onb
from utils.ensemble.ensemble_runtime import EnsembleManager
from utils.experiment.learning_runner import run_learning_experiments
from utils.plotting.regression_plots import RegressionPlotter
from utils.training.model_training import ModelTrainer


def main():
    all_jobs = onb.build_dataset_jobs()
    specs = [s for s in onb.MODEL_SPECS if s["key"] in onb.ACTIVE_MODEL_KEYS]
    for enabled in (False, True):
        config = deepcopy(onb.validation_config_snapshot())
        config["run"].update(epochs=2, folds=3, smoke_test=True)
        config["data"].update(max_freq_hz_list=["maxfreq=3kHz"], noise_dir_names=["heatflux_no_noise"])
        config["explainability"] = {"enabled": False}
        config["acoustic_selection"]["enabled"] = enabled
        config["output"]["noise_trend_plots"] = {"enabled": False}
        config["output"]["result_date_dir"] = "20260916_day_split_smoke_" + ("selected" if enabled else "baseline")
        config["output"]["run_instance_id"] = "day_split_integration"
        jobs = []
        for job in all_jobs:
            if job["max_freq_hz"] == "maxfreq=3kHz" and job["noise_dir_name"] == "heatflux_no_noise":
                jobs.append({**job, "save_base_path": Path(job["experiment_root"]) / "regression_result/npy/day_split_smoke" / config["output"]["result_date_dir"]})
        run_learning_experiments(jobs, config["learning_policy"], config, specs, onb.PARAMETER_SETS,
                                 onb.ENSEMBLE_MANAGER, ModelTrainer(42), RegressionPlotter(), lambda *a: None)


if __name__ == "__main__":
    main()
