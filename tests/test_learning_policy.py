"""4種類の方針で、実データ形式の読込から両評価の保存・再開までを確認する。"""

import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from utils.ensemble.ensemble_runtime import EnsembleManager
from utils.experiment.learning_policy import (
    aligned_indices, build_learning_families, checked_metadata,
    normalize_learning_policy, outer_splits, policy_result_date_dir,
)
from utils.experiment.learning_runner import run_learning_experiments
from utils.experiment.result_paths import (
    existing_result_run_path,
    normalize_result_date_dir,
    result_run_path,
    scoped_result_job,
)
from utils.experiment.result_paths import MAX_STUDY_DIR_LENGTH, result_scope_dir_name
from utils.experiment.run_helpers import is_completed_run, run_config_digest, run_dir_name
from utils.plotting.noise_trend_plots import collect_noise_trend_rows
from utils.training.model_training import ModelTrainer
from reorganize_onb_results import migrate_results


class ObservedTrainer(ModelTrainer):
    def __init__(self):
        super().__init__(42)
        self.fits, self.predictions, self.pca_fits = [], [], []

    def make_pca(self, x_fit, others, n_components, return_pca=False):
        self.pca_fits.append(x_fit.copy())
        return super().make_pca(x_fit, others, n_components, return_pca)

    def train_one_model(self, spec, mm, x_fit, y_scaled, x_pca, epochs):
        model = Ridge(alpha=0.1).fit(x_pca, y_scaled.ravel())
        model.fit_number = len(self.fits)
        self.fits.append(x_fit.copy())
        return model, None

    def predict_one_model(self, spec, model, x, x_pca, scaler):
        self.predictions.append((model.fit_number, x.copy(), scaler.data_min_.copy(), scaler.data_max_.copy()))
        return super().predict_one_model(spec, model, x, x_pca, scaler)


class SilentPlotter:
    # 数値処理と保存は実処理を通す。描画自体は既存の作図テストで確認する。
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def fixture(root, policy, evaluated_noises=("heatflux_no_noise", "heatflux_reference_SNR=-20"),
            days=("day-a", "day-b")):
    jobs = []
    for day_number, day in enumerate(days, 1):
        source = root / day / "source"
        for noise_number, noise in enumerate(("heatflux_no_noise", "heatflux_reference_SNR=-20")):
            directory = source / "maxfreq=3kHz" / noise
            directory.mkdir(parents=True)
            rows = []
            for wav in range(6):
                for chunk in range(2):
                    filename = f"{10 + wav * 10 + day_number}_{wav}_{chunk}.npy"
                    value = day_number * 100 + wav * 10 + noise_number * 1000 + chunk * 0.01
                    np.save(directory / filename, np.asarray([[value, value * 0.3], [wav, chunk]], dtype=np.float32))
                    rows.append({"sample_filename": filename, "experiment_name": day,
                                 "source_wav_id": f"wav-{wav}", "source_wav_name": f"wav-{wav}.wav",
                                 "chunk_index": chunk, "chunk_start_seconds": chunk, "chunk_duration_seconds": 1})
            with (directory / "chunk_manifest.csv").open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            if noise in evaluated_noises:
                jobs.append({"experiment_name": day, "source_dir": source, "data_path": directory,
                             "experiment_root": root / day, "max_freq_hz": "maxfreq=3kHz",
                             "noise_dir_name": noise, "snr_value": "no_noise" if noise_number == 0 else "-20",
                             "threshold": 35 + day_number,
                             "save_base_path": root / day / policy_result_date_dir("results", policy)})
    specs = [{"key": key, "label": key, "kind": "sklearn"} for key in ("randomforest", "second")]
    manager = EnsembleManager(
        {"enabled_strategy_names": ["simple_equal", "inner_holdout"]},
        [s["key"] for s in specs],
    )
    config = {
        "learning_policy": policy,
        "run": {"smoke_test": False, "epochs": 1, "folds": 3, "random_seed": 42, "loop_parameter_sets": True},
        "data": {"experiment_names": days}, "models": {}, "ensemble": manager.snapshot(),
        "features": {"pca_components": 2},
        "thresholds": {"onb_band_frac": 0.1, "provenance_by_experiment": {}},
        "output": {"save_fold_predictions": True, "save_tuning_summary": True,
                   "resume_completed_runs": True, "run_instance_id": "test-instance"},
        "explainability": {"enabled": False},
    }
    return jobs, specs, manager, config


class LearningPolicyTest(unittest.TestCase):
    def test_weight_scope_is_per_noise_for_matched_and_shared_for_clean_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            days = ("day-a", "day-b")

            matched = {"split_mode": "explicit_days", "training_noise": "matched",
                       "train_experiments": ["day-a"], "test_experiments": ["day-b"]}
            jobs, _, _, _ = fixture(root / "matched", matched, days=days)
            families = build_learning_families(jobs, matched, list(days))
            self.assertEqual(len(families), 2)
            for family in families:
                context = family["context"]
                evaluation_noises = {job["noise_dir_name"] for job in family["evaluation_jobs"]}
                self.assertEqual(context["ensemble_weight_scope"], "per_training_noise")
                self.assertEqual(evaluation_noises, {context["training_noise_dir"]})
                self.assertEqual(
                    {job["noise_dir_name"] for job in family["training_jobs"]},
                    evaluation_noises,
                )

            clean_only = {"split_mode": "explicit_days", "training_noise": "clean_only",
                          "train_experiments": ["day-a"], "test_experiments": ["day-b"]}
            jobs, _, _, _ = fixture(root / "clean_only", clean_only, days=days)
            families = build_learning_families(jobs, clean_only, list(days))
            self.assertEqual(len(families), 1)
            family = families[0]
            self.assertEqual(
                family["context"]["ensemble_weight_scope"],
                "shared_clean_across_evaluation_noises",
            )
            self.assertEqual(
                {job["noise_dir_name"] for job in family["training_jobs"]},
                {"heatflux_no_noise"},
            )
            self.assertEqual(
                {job["noise_dir_name"] for job in family["evaluation_jobs"]},
                {"heatflux_no_noise", "heatflux_reference_SNR=-20"},
            )

    def test_scoped_result_directory_separates_executions_and_parameters(self):
        job = {"save_base_path": Path("ensemble/202609/17/onb__days_matched"),
               "experiment_name": "2025.06.18_0.3_3",
               "max_freq_hz": "maxfreq=3kHz", "noise_dir_name": "heatflux_no_noise"}
        config = {"learning_policy": {"split_mode": "explicit_days", "training_noise": "matched",
                                      "internal_validation": "wav_kfold",
                                      "train_experiments": ["2025.06.11_0.3_2"],
                                      "test_experiments": ["2025.06.18_0.3_3"]},
                  "data": {"experiment_names": ["2025.06.11_0.3_2", "2025.06.18_0.3_3"]},
                  "run": {"epochs": 150, "folds": 3},
                  "acoustic_selection": {"enabled": True, "peak_height_threshold": 1e-9}}
        first = scoped_result_job(job, "010203", "config-a", config)
        self.assertEqual(first, scoped_result_job(job, "010203", "config-a", config))
        self.assertNotEqual(first["save_base_path"], scoped_result_job(job, "010204", "config-a", config)["save_base_path"])
        first_parameter = scoped_result_job(
            job, "010203", "config-a", config,
            parameter_index=1, parameter_count=2,
        )
        second_parameter = scoped_result_job(
            job, "010203", "config-b", config,
            parameter_index=2, parameter_count=2,
        )
        self.assertIn("_p01_010203", first_parameter["save_base_path"].name)
        self.assertNotEqual(first_parameter["save_base_path"], second_parameter["save_base_path"])
        self.assertIn("_p02_010203", second_parameter["save_base_path"].name)
        self.assertEqual(first["save_base_path"].parent, job["save_base_path"].parent)
        self.assertEqual(
            first["save_base_path"].name,
            "onb_xd-t0611-v0618_iw3-nm_s1e-9_e150_010203",
        )
        self.assertLessEqual(len(first["save_base_path"].name), MAX_STUDY_DIR_LENGTH)
        self.assertEqual(result_run_path(first, ""), first["save_base_path"] / "maxfreq=3kHz" / "heatflux_no_noise")

        two_day_config = {**config, "learning_policy": {
            **config["learning_policy"],
            "train_experiments": ["2025.06.11_0.3_2", "2025.07.09_0.3_1"],
        }}
        name = result_scope_dir_name("onb", job, two_day_config, "010203", "config-a")
        self.assertIn("t0611+0709", name)
        self.assertTrue(name.endswith("_010203"))
        self.assertLessEqual(len(name), MAX_STUDY_DIR_LENGTH)
        with self.assertRaisesRegex(ValueError, "HHMMSS"):
            result_scope_dir_name("onb", job, config, "execution-a", "config-a")

    def test_scoped_result_layout_writes_directly_below_noise(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            policy = {"split_mode": "within_day", "training_noise": "matched"}
            jobs, specs, manager, config = fixture(root, policy,
                                                    evaluated_noises=("heatflux_no_noise",), days=("day-a",))
            config["output"].update(run_scoped_result_dir=True, execution_id="execution-a")
            trainer = ObservedTrainer()
            for execution_id in ("010203", "010204"):
                config["output"]["execution_id"] = execution_id
                with contextlib.redirect_stdout(io.StringIO()), patch("gc.collect"), patch("tensorflow.keras.backend.clear_session"):
                    run_learning_experiments(jobs, policy, config, specs, [{"name": "test"}],
                                             manager, trainer, SilentPlotter(), lambda *args: None)
            base = jobs[0]["save_base_path"]
            scoped = sorted(base.parent.glob(base.name + "_*"))
            self.assertEqual(len(scoped), 2)
            for directory in scoped:
                result = directory / "maxfreq=3kHz" / "heatflux_no_noise"
                self.assertTrue((result / "completed.json").is_file())
                self.assertTrue((result / "run_manifest.json").is_file())
                manifest = json.loads((result / "run_manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["run_dir"], "")
                self.assertEqual(manifest["study_directory_naming"]["directory"], directory.name)
                self.assertEqual(manifest["study_directory_naming"]["max_component_length"], 52)
                self.assertTrue((directory / "tuning_summary.csv").is_file())
                self.assertFalse(any(path.name.startswith("e1_") for path in result.iterdir()))

    def test_nested_analysis_date_path_and_config_scoped_run_name(self):
        policy = {"split_mode": "explicit_days", "training_noise": "matched"}
        self.assertEqual(
            normalize_result_date_dir("20260916/selected_log_architecture"),
            "202609/16/selected_log_architecture",
        )
        self.assertEqual(
            normalize_result_date_dir("20260626_cf3m"),
            "202606/26/cf3m",
        )
        self.assertEqual(
            normalize_result_date_dir("202609/16/selected_log_architecture"),
            "202609/16/selected_log_architecture",
        )
        self.assertEqual(
            policy_result_date_dir("202609/16/selected_log_architecture", policy),
            "202609/16/selected_log_architecture__days_matched",
        )
        first_hash = run_config_digest(
            {"models": {}, "output": {}, "learning_policy": {**policy, "train_experiments": ["day-a"]}},
            {}, [], "randomforest", True,
        )
        second_hash = run_config_digest(
            {"models": {}, "output": {}, "learning_policy": {**policy, "train_experiments": ["day-a", "day-b"]}},
            {}, [], "randomforest", True,
        )
        self.assertNotEqual(first_hash, second_hash)
        first_dir = run_dir_name(
            300, "fixed", "randomforest", False, "simple", first_hash, "run-a"
        )
        second_dir = run_dir_name(
            300, "fixed", "randomforest", False, "simple", second_hash, "run-a"
        )
        self.assertNotEqual(first_dir, second_dir)
        self.assertTrue(first_dir.endswith(f"{first_hash}_run-a"))

        repeated_condition = run_dir_name(
            300, "fixed", "randomforest", False, "simple", first_hash, "run-b"
        )
        self.assertNotEqual(first_dir, repeated_condition)

    def run_fixture(self, root, policy, evaluated_noises=None):
        args = {} if evaluated_noises is None else {"evaluated_noises": evaluated_noises}
        jobs, specs, manager, config = fixture(root, policy, **args)
        trainer = ObservedTrainer()
        call = lambda: run_learning_experiments(jobs, policy, config, specs, [{"name": "test"}],
                                                manager, trainer, SilentPlotter(), lambda *args: None)
        with contextlib.redirect_stdout(io.StringIO()), patch("gc.collect"), patch("tensorflow.keras.backend.clear_session"):
            call()
            fit_count = len(trainer.fits)
            call()
        self.assertEqual(len(trainer.fits), fit_count, "完了再開で再学習してはいけない")
        return jobs, trainer, config, call

    def test_all_four_policies_fit_counts_leakage_and_both_evaluations(self):
        for split in ("within_day", "leave_one_day_out"):
            for noise in ("matched", "clean_only"):
                with self.subTest(split=split, noise=noise), tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    policy = {"split_mode": split, "training_noise": noise}
                    jobs, trainer, config, _ = self.run_fixture(root, policy)
                    folds = 3 if split == "within_day" else 1
                    # 2日 × 学習ノイズ数 × fold × 2モデル × 内部/最終学習。
                    expected = 2 * (2 if noise == "matched" else 1) * folds * 2 * 2
                    self.assertEqual(len(trainer.fits), expected)
                    self.assertEqual(len(trainer.pca_fits), expected // 2)
                    if noise == "clean_only":
                        self.assertTrue(all(np.max(x[:, 0, 0, 0]) < 1000 for x in trainer.fits + trainer.pca_fits))
                        # 同じモデルとscalerのまま、無雑音と強雑音の両方へ予測する。
                        used = {}
                        for model_id, x, minimum, maximum in trainer.predictions:
                            used.setdefault(model_id, []).append((np.max(x[:, 0, 0, 0]) >= 1000, minimum, maximum))
                        shared = [items for items in used.values() if len(items) == 2]
                        self.assertEqual(len(shared), 2 * folds * 2)
                        for first, second in shared:
                            self.assertNotEqual(first[0], second[0])
                            np.testing.assert_array_equal(first[1], second[1])
                            np.testing.assert_array_equal(first[2], second[2])
                    by_day = {}
                    for job in jobs:
                        directories = list((job["save_base_path"] / job["max_freq_hz"] / job["noise_dir_name"]).iterdir())
                        self.assertEqual(len(directories), 1)
                        directory = directories[0]
                        manifest = json.loads(
                            (directory / "run_manifest.json").read_text(encoding="utf-8")
                        )
                        expected_scope = (
                            "per_training_noise"
                            if noise == "matched"
                            else "shared_clean_across_evaluation_noises"
                        )
                        self.assertEqual(
                            manifest["learning_context"]["ensemble_weight_scope"],
                            expected_scope,
                        )
                        split_data = json.loads((directory / "split_manifest.json").read_text(encoding="utf-8"))
                        self.assertEqual(len(split_data["folds"]), folds)
                        seen = []
                        for fold in split_data["folds"]:
                            training, evaluation = set(fold["training_wav_groups"]), set(fold["evaluation_wav_groups"])
                            self.assertFalse(training & evaluation)
                            seen.extend(fold["evaluation_sample_indices"])
                            if split == "leave_one_day_out":
                                self.assertNotIn(job["experiment_name"], {json.loads(group)[0] for group in training})
                        self.assertEqual(sorted(seen), list(range(12)))
                        self.assertTrue((directory / f"metrics_summary_{job['snr_value']}.csv").is_file())
                        self.assertFalse((directory / "wav_eval").exists())
                        by_day.setdefault(job["experiment_name"], []).append(directory)
                    for directories in by_day.values():
                        rows = collect_noise_trend_rows(directories, noise_order=["no_noise", "-20"],
                                                        model_keys=["randomforest", "second"])
                        self.assertTrue(rows)
                        if split == "leave_one_day_out":
                            self.assertTrue(all(np.isnan(row["standard_error"]) for row in rows))

    def test_clean_training_does_not_require_clean_in_evaluation_list(self):
        with tempfile.TemporaryDirectory() as temp:
            _, trainer, _, _ = self.run_fixture(Path(temp), {"split_mode": "within_day", "training_noise": "clean_only"},
                                                 ["heatflux_reference_SNR=-20"])
            self.assertTrue(all(np.max(x[:, 0, 0, 0]) < 1000 for x in trainer.fits))

    def test_actual_keras_trainer_reuses_one_fit_per_held_out_day(self):
        from tensorflow import keras
        with tempfile.TemporaryDirectory() as temp:
            policy = {"split_mode": "leave_one_day_out", "training_noise": "clean_only"}
            jobs, _, _, config = fixture(Path(temp), policy)
            specs = [{"key": "tiny", "label": "Tiny Keras", "kind": "keras",
                      "builder": lambda mm: keras.Sequential([
                          keras.layers.Input(shape=(2, 2, 1)), keras.layers.Flatten(), keras.layers.Dense(1)])}]
            parameter_sets = [{"name": "tiny", "default_keras": {"lr": 0.000001, "batch_size": 4, "fit_verbose": 0}}]
            manager = EnsembleManager(
                {"enabled_strategy_names": ["simple_equal"]}, ["tiny"])
            config["ensemble"] = manager.snapshot()
            trainer = ModelTrainer()
            with contextlib.redirect_stdout(io.StringIO()), patch.object(trainer, "train_one_model", wraps=trainer.train_one_model) as fit:
                run_learning_experiments(jobs, policy, config, specs, parameter_sets, manager,
                                         trainer, SilentPlotter(), lambda *args: None)
                self.assertEqual(fit.call_count, 2)
            for job in jobs:
                path = next((job["save_base_path"] / job["max_freq_hz"] / job["noise_dir_name"]).iterdir())
                with (path / "fold_pred" / f"pred_f1_{job['snr_value']}.csv").open(encoding="utf-8") as source:
                    rows = list(csv.DictReader(source))
                self.assertEqual(len(rows), 12)
                self.assertTrue(all(np.isfinite(float(row["tiny"])) for row in rows))

    def test_single_day_leave_out_and_typo_are_rejected(self):
        with self.assertRaises(ValueError):
            normalize_learning_policy({"split_mode": "leave_one_day_out"}, ["day-a"])
        with self.assertRaises(ValueError):
            normalize_learning_policy({"training_noise": "clean"}, ["day-a"])

    def test_three_day_leave_out_pools_two_training_days_without_id_collision(self):
        with tempfile.TemporaryDirectory() as temp:
            policy = {"split_mode": "leave_one_day_out", "training_noise": "clean_only"}
            jobs, specs, manager, config = fixture(Path(temp), policy, days=["day-a", "day-b", "day-c"])
            trainer = ObservedTrainer()
            with contextlib.redirect_stdout(io.StringIO()), patch("gc.collect"), patch("tensorflow.keras.backend.clear_session"):
                run_learning_experiments(jobs, policy, config, specs, [{"name": "test"}],
                                         manager, trainer, SilentPlotter(), lambda *args: None)
            self.assertEqual(len(trainer.fits), 3 * 2 * 2)
            for job in jobs:
                path = next((job["save_base_path"] / job["max_freq_hz"] / job["noise_dir_name"]).iterdir())
                record = json.loads((path / "split_manifest.json").read_text(encoding="utf-8"))["folds"][0]
                training_days = {json.loads(group)[0] for group in record["training_wav_groups"]}
                self.assertEqual(len(training_days), 2)
                self.assertNotIn(job["experiment_name"], training_days)
                self.assertEqual(len(record["training_wav_groups"]), 12)

    def test_alignment_uses_recording_and_chunk_and_checks_labels(self):
        rows = checked_metadata([{"sample_filename": f"{i}_x.npy", "source_wav_id": f"w{i}", "chunk_index": 0}
                                 for i in range(4)], "a")
        other = list(reversed(rows))
        np.testing.assert_array_equal(aligned_indices(rows, other), [3, 2, 1, 0])
        for train, test in outer_splits(rows, other, "within_day", 2):
            self.assertFalse({rows[i]["source_wav_id"] for i in train} & {other[i]["source_wav_id"] for i in test})
        with self.assertRaises(ValueError):
            aligned_indices(rows, other[:-1])
        with self.assertRaises(ValueError):
            aligned_indices(rows, [{**other[0], "sample_filename": "999_x.npy"}] + other[1:])
        with self.assertRaises(ValueError):
            outer_splits(rows, rows, "leave_one_day_out", 2)

    def test_default_hash_is_compatible_and_other_policies_change_hash(self):
        config = {"models": {}, "output": {}}
        old = run_config_digest(config, {}, [], "randomforest", True)
        default = {"split_mode": "within_day", "training_noise": "matched"}
        self.assertEqual(old, run_config_digest({**config, "learning_policy": default}, {}, [], "randomforest", True))
        self.assertNotEqual(old, run_config_digest({**config, "learning_policy": {**default, "training_noise": "clean_only"}}, {}, [], "randomforest", True))

    def test_new_and_legacy_paths_and_resume_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            job = {"save_base_path": Path(temp), "max_freq_hz": "frequency", "noise_dir_name": "noise"}
            legacy = Path(temp) / "noise" / "frequency" / "run"
            legacy.mkdir(parents=True)
            self.assertEqual(existing_result_run_path(job, "run"), legacy)
            current = result_run_path(job, "run")
            current.mkdir(parents=True)
            self.assertEqual(existing_result_run_path(job, "run"), current)
            (current / "metrics_summary_no_noise.csv").write_text("model,r2_mean\nrandomforest,0.9\n")
            manifest = {"run_dir": "run", "run_hash": "a", "dataset": {"snr_value": "no_noise"}, "execution_schema_version": 2}
            (current / "run_manifest.json").write_text(json.dumps(manifest))
            check = lambda h: is_completed_run(Path(temp) / "summary.csv", "run", current, "no_noise", True, False, h)
            self.assertFalse(check("a"))
            (current / "completed.json").write_text(json.dumps({"run_hash": "a"}))
            self.assertTrue(check("a"))
            self.assertFalse(check("b"))

    def test_partial_clean_resume_recomputes_one_family_with_shared_models(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            jobs, trainer, config, call = self.run_fixture(root, {"split_mode": "within_day", "training_noise": "clean_only"})
            target = next((jobs[0]["save_base_path"] / jobs[0]["max_freq_hz"] / jobs[0]["noise_dir_name"]).iterdir())
            (target / "completed.json").unlink()
            # 同じ実行IDを指定した場合だけ、中断した保存先を再開する。
            before = len(trainer.fits)
            with contextlib.redirect_stdout(io.StringIO()), patch("gc.collect"), patch("tensorflow.keras.backend.clear_session"):
                call()
            self.assertEqual(len(trainer.fits) - before, 3 * 2 * 2)
            markers = [json.loads(path.read_text(encoding="utf-8")) for path in
                       jobs[0]["save_base_path"].rglob("completed.json")]
            self.assertEqual(len(markers), 2)
            self.assertEqual(markers[0]["fit_ids"], markers[1]["fit_ids"])
            with (jobs[0]["save_base_path"] / "tuning_summary.csv").open(encoding="utf-8") as source:
                self.assertEqual(len(list(csv.DictReader(source))), 8)

    def test_migration_preserves_metric_values_updates_paths_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old = root / "heatflux_no_noise" / "maxfreq=3kHz"
            run = old / "run"
            run.mkdir(parents=True)
            content = b"model,r2_mean\nrandomforest,0.9\n"
            (run / "metrics_summary_no_noise.csv").write_bytes(content)
            (root / "tuning_summary.csv").write_text(str(run), encoding="utf-8")
            self.assertFalse(migrate_results(root)["applied"])
            self.assertTrue(run.is_dir())
            report = migrate_results(root, apply=True)
            new = root / "maxfreq=3kHz" / "heatflux_no_noise" / "run"
            self.assertEqual((new / "metrics_summary_no_noise.csv").read_bytes(), content)
            self.assertEqual((root / "tuning_summary.csv").read_text(encoding="utf-8"), str(new))
            self.assertEqual(report["metric_files_verified"], 1)
            old.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                migrate_results(root, apply=True)


if __name__ == "__main__":
    unittest.main()
