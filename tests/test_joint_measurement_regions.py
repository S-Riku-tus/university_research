"""Physical-time selection preserves row pairing and applied-voltage identity."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from utils.calculation.heatflux_preprocessing import (
    joint_stable_intervals, summarize_stages, add_heat_flux,
    read_logger_csv, resolve_heat_flux_csv_path, backup_results_before_rerun,
)


def frame_at(times, shunt=None, electrode=None):
    times = np.asarray(times, dtype=float)
    return pd.DataFrame({
        'elapsed_s': times, 'timestamp': pd.Timestamp('2026-01-01') + pd.to_timedelta(times, unit='s'),
        'water': np.full(len(times), 80.),
        'shunt': np.full(len(times), .01) if shunt is None else shunt,
        'Pt': np.full(len(times), 1.) if electrode is None else electrode,
    }, index=pd.Index(range(1, len(times)+1), name='scan'))


class JointMeasurementRegionsTest(unittest.TestCase):
    def test_same_time_span_at_different_logger_cadences(self):
        for dt in [1, 5]:
            frame = frame_at(np.arange(0, 36, dt))
            self.assertEqual(joint_stable_intervals(frame, 35, 1e-4, 1e-3), [[0, len(frame)-1]])
        self.assertEqual(joint_stable_intervals(frame_at(range(8)), 35, 1e-4, 1e-3), [])

    def test_gaps_or_ineligible_rows_split_equal_voltage_regions(self):
        frame = frame_at([0, 1, 2, 10, 11, 12])
        self.assertEqual(joint_stable_intervals(frame, 2, 1e-4, 1e-3), [[0, 2], [3, 5]])
        frame = frame_at(range(7))
        self.assertEqual(joint_stable_intervals(frame, 2, 1e-4, 1e-3,
                          [True, True, True, False, True, True, True]), [[0, 2], [4, 6]])

    def test_both_channels_must_be_stable(self):
        frame = frame_at(range(5), electrode=[1, 2, 1, 2, 1])
        self.assertEqual(joint_stable_intervals(frame, 2, 1e-4, 1e-3), [])

    def ramp(self):
        frame = frame_at(range(240), shunt=[0.]*60 + [.01]*60 + [.02]*60 + [0.]*60,
                         electrode=[0.]*60 + [.085]*60 + [.255]*60 + [0.]*60)
        frame.loc[62, 'Pt'] = .15  # One-sample overshoot is not an extra stage.
        return frame

    def test_sequence_skips_voltages_and_keeps_common_rows(self):
        summary, _ = summarize_stages(self.ramp(), [0, .1, .3])
        self.assertEqual(summary.volt.tolist(), [0, .1, .3])
        self.assertEqual(summary.n_samples.tolist(), [60, 55, 55])
        frame = self.ramp()
        for row in summary.itertuples():
            selected = frame.loc[row.start_scan:row.end_scan]
            self.assertAlmostEqual(row.shunt, selected.shunt.mean())
            self.assertAlmostEqual(row.Pt, selected.Pt.mean())
        q = add_heat_flux(summary, .002, .0003, .04)
        self.assertAlmostEqual(q.q.iloc[1], .01/.002*.085/(np.pi*.0003*.04))

    def test_mismatch_and_unstable_stage_stop_without_relabelling(self):
        with self.assertRaisesRegex(ValueError, 'Detected 3 stages'):
            summarize_stages(self.ramp(), [0, .1, .2, .3])
        frame = self.ramp()
        frame.loc[121:180, 'shunt'] = np.tile([.01, .02], 30)
        with self.assertRaisesRegex(ValueError, 'No joint stable interval at 0.3 V'):
            summarize_stages(frame, [0, .1, .3])

    def test_temperature_outliers_are_not_compressed_into_adjacent_samples(self):
        frame = self.ramp()
        frame.loc[140, 'water'] = 85
        summary, _ = summarize_stages(frame, [0, .1, .3], window_seconds=15)
        stage = summary.iloc[-1]
        self.assertFalse(stage.start_scan <= 140 <= stage.end_scan)

    def test_gl860_units_scan_numbers_and_real_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'logger.csv'
            with path.open('w', encoding='cp932', newline='') as stream:
                csv.writer(stream).writerows([
                    ['number','datetime','ms','CH1','CH2','CH3'],
                    ['NO.','Time','ms','C','mV','V'],
                    [10,'2026-01-01 12:00:00',100,80,16.4,.855],
                    [11,'2026-01-01 12:00:01',200,80,16.5,.856],
                ])
            frame = read_logger_csv(path)
            self.assertEqual(frame.index.tolist(), [10, 11])
            self.assertAlmostEqual(frame.shunt.iloc[0], .0164)
            self.assertAlmostEqual(frame.elapsed_s.iloc[1], 1.1)

    def test_legacy_timestamp_keeps_milliseconds_and_voltage_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'legacy.csv'
            with path.open('w', encoding='cp932', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerows([['metadata']]*12)
                writer.writerow(['scan','time','water','alarm1','shunt','alarm2','Pt'])
                writer.writerow([1,'2025/06/18 14:00:00:204',80,0,.0164,0,.855])
                writer.writerow([2,'2025/06/18 14:00:05:195',80,0,.0165,0,.856])
            frame = read_logger_csv(path)
            self.assertAlmostEqual(frame.elapsed_s.iloc[1], 4.991)
            self.assertAlmostEqual(frame.shunt.iloc[0], .0164)

    def test_processing_config_keeps_the_established_csv_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp = root / 'experiment'
            legacy = resolve_heat_flux_csv_path(exp, root)
            legacy.parent.mkdir(parents=True)
            legacy.write_text('volt,q\n1.2V,123\n', encoding='utf-8')
            cfg = root / 'configs/datasets/experiment_heatflux_processing.json'
            cfg.parent.mkdir(parents=True)
            cfg.write_text(json.dumps({'experiment_name':'experiment', 'stability_window_seconds':35}))
            self.assertEqual(resolve_heat_flux_csv_path(exp, root), legacy)
            self.assertTrue(resolve_heat_flux_csv_path(exp, root).exists())

    def test_stft_context_requires_and_uses_the_established_label_csv(self):
        from utils.dataloading.waterflow_preprocessing import build_experiment_context
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / 'configs/datasets/experiment_heatflux_processing.json'
            cfg.parent.mkdir(parents=True)
            cfg.write_text(json.dumps({'experiment_name':'experiment', 'stability_window_seconds':35}))
            with patch('utils.calculation.heatflux_preprocessing.REPO_ROOT', root):
                arguments = ('experiment', tmp, 'recordings', '', 20261007, 1, 'script.py')
                with self.assertRaisesRegex(FileNotFoundError, 'Configured heat flux CSV'):
                    build_experiment_context(*arguments)
                path = resolve_heat_flux_csv_path(root / 'experiment')
                path.parent.mkdir(parents=True)
                path.write_text('volt,q\n1.2V,200987.654321\n', encoding='utf-8')
                context = build_experiment_context(*arguments)
                self.assertEqual(context['heat_flux_csv_path'], str(path))
                self.assertEqual(context['heat_flux_label_by_index'][1], '200987.654321')

    def test_rerun_backup_preserves_all_results_and_leaves_the_standard_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = root / 'experiment'
            label = resolve_heat_flux_csv_path(experiment)
            self.assertIsNone(backup_results_before_rerun(experiment, root))
            label.parent.mkdir(parents=True)
            label.write_bytes(b'volt,q\n1.2V,123.456\n')
            debug = label.parent / 'debug.csv'
            debug.write_bytes(b'original debug')
            plots = label.parent / 'plots'
            plots.mkdir()
            (plots / 'plot.png').write_bytes(b'original plot')
            archive = backup_results_before_rerun(experiment, root)
            self.assertTrue(archive.is_relative_to(root / 'experiments'))
            self.assertEqual((archive / label.name).read_bytes(), label.read_bytes())
            self.assertEqual((archive / 'debug.csv').read_bytes(), debug.read_bytes())
            self.assertEqual((archive / 'plots/plot.png').read_bytes(), b'original plot')
            self.assertEqual(resolve_heat_flux_csv_path(experiment), label)


if __name__ == '__main__':
    unittest.main()
