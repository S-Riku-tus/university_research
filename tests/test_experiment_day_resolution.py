"""ONB実行の実験日指定と、学習専用日のデータ計画を確認する。"""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from utils.experiment.dataset_jobs import build_dataset_jobs
from utils.experiment.learning_policy import (
    build_learning_families,
    normalize_learning_policy,
    resolve_experiment_names,
)


class ExperimentDayResolutionTest(unittest.TestCase):
    def setUp(self):
        self.policy = {
            "split_mode": "explicit_days",
            "train_experiments": ["2025.06.11_0.3_2", "2025.07.09_0.3_1"],
            "test_experiments": ["2025.06.18_0.3_3"],
            "training_noise": "matched",
        }
        self.days = ["2025.06.11_0.3_2", "2025.06.18_0.3_3", "2025.07.09_0.3_1"]

    def test_explicit_days_derive_the_previous_three_day_order(self):
        names = resolve_experiment_names({}, self.policy)
        self.assertEqual(names, self.days)
        normalize_learning_policy(self.policy, names)
        self.assertEqual(resolve_experiment_names({"experiment_names": self.days}, self.policy), names)

    def test_unused_or_missing_explicit_day_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "和集合"):
            resolve_experiment_names({"experiment_names": self.days + ["unused"]}, self.policy)
        with self.assertRaisesRegex(ValueError, "未使用"):
            normalize_learning_policy(self.policy, self.days + ["unused"])
        with self.assertRaisesRegex(ValueError, "test_experiments"):
            resolve_experiment_names({}, {**self.policy, "test_experiments": []})
        with self.assertRaisesRegex(ValueError, "分離"):
            normalize_learning_policy({**self.policy, "test_experiments": [self.days[0]]}, self.days)

    def test_older_split_modes_still_require_experiment_names(self):
        for mode in ("within_day", "leave_one_day_out"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "experiment_names"):
                resolve_experiment_names({}, {"split_mode": mode})
            self.assertEqual(resolve_experiment_names({"experiment_names": self.days}, {"split_mode": mode}), self.days)

    def test_clean_only_requires_only_clean_training_days(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = "waterflow_20260817_1s"
            noises = ["heatflux_no_noise", "heatflux_reference_SNR=-20"]
            for day in self.days:
                day_noises = noises if day == self.days[1] else noises[:1]
                for noise in day_noises:
                    directory = root / day / "data" / "npy" / source / "maxfreq=3kHz" / noise
                    directory.mkdir(parents=True)
                    (directory / "sample.npy").touch()
            args = dict(
                experiment_root=root, experiment_names=self.days,
                max_freq_hz_list=["maxfreq=3kHz"], noise_dir_names=noises,
                data_source_dir_by_experiment={day: source for day in self.days},
                noise_source_prefix="waterflow", chunk_seconds=1,
                threshold_by_experiment={day: 100 for day in self.days},
                result_model_group="ensemble", result_date_dir="test",
                color_channel=1, require_experiment_threshold=True,
                skip_missing_datasets=False,
            )
            clean_policy = normalize_learning_policy({**self.policy, "training_noise": "clean_only"}, self.days)
            jobs = build_dataset_jobs(**args, learning_policy=clean_policy)
            self.assertEqual(len(jobs), 4)
            families = build_learning_families(jobs, clean_policy, self.days)
            self.assertEqual(len(families), 1)
            self.assertEqual(len(families[0]["training_jobs"]), 2)
            self.assertEqual(len(families[0]["evaluation_jobs"]), 2)
            with self.assertRaises(FileNotFoundError):
                build_dataset_jobs(**args, learning_policy=self.policy)


if __name__ == "__main__":
    unittest.main()
