"""Joint, time-based logger selection; never changes original measurements."""

import csv
import json
import re
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]


def load_processing_config(experiment_root, repo_root=None):
    path = Path(repo_root or REPO_ROOT) / 'configs' / 'datasets' / (
        Path(experiment_root).name + '_heatflux_processing.json'
    )
    if not path.is_file():
        return {}
    config = json.loads(path.read_text(encoding='utf-8'))
    if config['experiment_name'] != Path(experiment_root).name:
        raise ValueError(f'Experiment name does not match processing config: {path}')
    tag = config.get('results_run_tag', '')
    if tag and not re.fullmatch(r'[A-Za-z0-9_-]+', tag):
        raise ValueError('results_run_tag must be a single safe folder name')
    return config


def resolve_heat_flux_csv_path(experiment_root, repo_root=None):
    root = Path(experiment_root)
    config = load_processing_config(root, repo_root)
    folder = root / ('実験結果' + root.name)
    if config.get('results_run_tag'):
        folder /= config['results_run_tag']
    return folder / ('heat_flux_' + root.name + '.csv')


def read_logger_csv(csv_path):
    """Read GL860 mV or legacy V channels, retaining scan IDs and real time."""
    with open(csv_path, newline='', encoding='cp932') as stream:
        rows = list(islice(csv.reader(stream), 64))
    header_index = next((i for i, row in enumerate(rows)
                         if all(name in row for name in ('CH1', 'CH2', 'CH3'))), None)
    if header_index is None:
        raw = pd.read_csv(csv_path, skiprows=12, encoding='cp932')
        values = raw.iloc[:, [2, 4, 6]].copy()
        values.columns = ['water', 'shunt', 'Pt']
        timestamps = raw.iloc[:, 1].astype(str).str.replace(
            r':(\d{3})$', r'.\1', regex=True)
        timestamps = pd.to_datetime(timestamps, format='%Y/%m/%d %H:%M:%S.%f')
    else:
        header, units = rows[header_index], rows[header_index + 1]
        shunt_unit, electrode_unit = units[header.index('CH2')], units[header.index('CH3')]
        if shunt_unit not in ('mV', 'V') or electrode_unit != 'V':
            raise ValueError('Expected shunt mV/V and electrode V')
        raw = pd.read_csv(csv_path, skiprows=list(range(header_index)) + [header_index + 1],
                          encoding='cp932')
        values = raw[['CH1', 'CH2', 'CH3']].copy()
        values.columns = ['water', 'shunt', 'Pt']
        values['shunt'] = pd.to_numeric(values['shunt']) / (1000 if shunt_unit == 'mV' else 1)
        timestamps = pd.to_datetime(raw.iloc[:, 1], format='%Y-%m-%d %H:%M:%S')
        timestamps += pd.to_timedelta(pd.to_numeric(raw.iloc[:, 2]), unit='ms')
    values = values.apply(pd.to_numeric, errors='raise')
    values.index = pd.Index(pd.to_numeric(raw.iloc[:, 0]), name='scan')
    values['timestamp'] = timestamps.to_numpy()
    values['elapsed_s'] = (timestamps - timestamps.iloc[0]).dt.total_seconds().to_numpy()
    if not values.index.is_unique or np.any(np.diff(values['elapsed_s']) <= 0):
        raise ValueError('Logger scan IDs must be unique and timestamps strictly increasing')
    return values


def detect_voltage_stages(frame, min_step_v=0.04, confirm_seconds=5.0,
                          baseline_seconds=60.0):
    """Detect sustained upward electrode steps, ending at sustained shutdown.

    These are chronological boundaries, not estimates of the applied voltage.
    Applied voltages must be supplied independently and match the stage count.
    """
    t = frame['elapsed_s'].to_numpy()
    u = frame['Pt'].to_numpy()
    if not np.isfinite(u).all():
        raise ValueError('Nonfinite electrode measurements; inspect original logger data')
    dt = float(np.median(np.diff(t)))
    half_window = confirm_seconds / 2
    smooth = np.array([np.median(u[np.searchsorted(t, x - half_window):
                                   np.searchsorted(t, x + half_window, side='right')])
                       for x in t])
    changes = []
    level = float(smooth[0])
    stop = len(frame)
    i = 1
    while i < len(frame):
        delta = smooth[i] - level
        if abs(delta) >= min_step_v:
            end = np.searchsorted(t, t[i] + confirm_seconds, side='left')
            if end >= len(frame):
                break
            future = smooth[i:end + 1]
            sustained = np.all(future - level >= min_step_v) if delta > 0 else np.all(
                future - level <= -min_step_v)
            if sustained:
                if delta < 0 and changes:
                    stop = i
                    break
                if delta > 0:
                    changes.append(i)
                    level = float(np.median(future))
                    i = end + 1
                    continue
        # Track slow drift within a stage without allowing step-sized changes.
        if abs(delta) < min_step_v:
            left = np.searchsorted(t, t[i] - confirm_seconds)
            level = float(np.median(smooth[left:i + 1]))
        i += 1
    if not changes:
        raise ValueError('No sustained powered stages found')
    baseline_start = int(np.searchsorted(t, t[changes[0]] - baseline_seconds))
    starts = [baseline_start] + changes
    ends = changes + [stop]
    return pd.DataFrame({'start_pos': starts, 'end_pos_exclusive': ends})


def joint_stable_intervals(frame, window_seconds, shunt_limit_v, electrode_limit_v,
                           eligible=None, cadence_seconds=None):
    """Union jointly stable windows on unfiltered time; gaps are never joined."""
    if window_seconds <= 0 or min(shunt_limit_v, electrode_limit_v) <= 0:
        raise ValueError('Stability window and limits must be positive')
    t = frame['elapsed_s'].to_numpy()
    sh = frame['shunt'].to_numpy()
    u = frame['Pt'].to_numpy()
    if len(t) < 2:
        return []
    dt = cadence_seconds or float(np.median(np.diff(t)))
    eligible = np.ones(len(t), dtype=bool) if eligible is None else np.asarray(eligible, bool)
    intervals = []
    for left in range(len(t)):
        right = int(np.searchsorted(t, t[left] + window_seconds - dt * 0.01, side='left'))
        if right >= len(t):
            break
        selection = slice(left, right + 1)
        if (right <= left or not eligible[selection].all()
                or np.any(np.diff(t[selection]) > dt * 1.5)
                or not np.isfinite(sh[selection]).all() or not np.isfinite(u[selection]).all()):
            continue
        if np.std(sh[selection]) < shunt_limit_v and np.std(u[selection]) < electrode_limit_v:
            if intervals and left <= intervals[-1][1]:
                intervals[-1][1] = max(intervals[-1][1], right)
            else:
                intervals.append([left, right])
    return intervals


def summarize_stages(frame, voltage_sequence, *, window_seconds=35.0,
                     baseline_seconds=60.0, min_step_v=0.04, confirm_seconds=5.0,
                     settling_seconds=5.0, water_target_c=80.0, water_tolerance_c=1.0,
                     shunt_limit_v=1e-4, electrode_limit_v=1e-3):
    """Select one common representative interval per explicitly named stage.

    An unmatched stage count or a stage without a valid interval raises, rather
    than dropping stages, interpolating labels, or relaxing tolerances.
    """
    voltage = np.asarray(voltage_sequence, dtype=float)
    if (len(voltage) < 2 or voltage[0] != 0 or not np.isfinite(voltage).all()
            or np.any(np.diff(voltage) <= 0)):
        raise ValueError('Supply the actual ascending voltage sequence, beginning with 0 V')
    boundaries = detect_voltage_stages(frame, min_step_v, confirm_seconds, baseline_seconds)
    if len(boundaries) != len(voltage):
        raise ValueError(f'Detected {len(boundaries)} stages, but voltage_sequence has '
                         f'{len(voltage)} values. Inspect logger transitions; never number by count.')
    dt = float(np.median(np.diff(frame['elapsed_s'])))
    summaries, diagnostics = [], []
    for number, (start, end) in enumerate(boundaries.itertuples(index=False, name=None)):
        stage = frame.iloc[start:end]
        t = stage['elapsed_s'].to_numpy()
        eligible = (np.abs(stage['water'].to_numpy() - water_target_c) <= water_tolerance_c)
        if number:
            eligible &= (stage['shunt'].to_numpy() > 0.001) & (t >= t[0] + settling_seconds)
        intervals = joint_stable_intervals(stage, window_seconds, shunt_limit_v,
                                           electrode_limit_v, eligible, dt)
        for left, right in intervals:
            diagnostics.append({'volt': voltage[number], 'start_scan': int(stage.index[left]),
                                'end_scan': int(stage.index[right]),
                                'duration_s': t[right] - t[left], 'n_samples': right - left + 1})
        if not intervals:
            raise ValueError(f'No joint stable interval at {voltage[number]:g} V for '
                             f'{window_seconds:g} s; inspect stage {stage.index[0]}..{stage.index[-1]}')
        left, right = max(intervals, key=lambda pair: t[pair[1]] - t[pair[0]])
        selected = stage.iloc[left:right + 1]
        full = stage.loc[eligible]
        summaries.append({
            'volt': voltage[number], 'stage_start_scan': int(stage.index[0]),
            'stage_end_scan': int(stage.index[-1]),
            'start_scan': int(selected.index[0]), 'end_scan': int(selected.index[-1]),
            'start_time': selected['timestamp'].iloc[0], 'end_time': selected['timestamp'].iloc[-1],
            'duration_s': float(t[right] - t[left]), 'n_samples': len(selected),
            'eligible_samples': len(full), 'coverage_fraction': len(selected) / len(full),
            'stable_components': len(intervals),
            'shunt': selected['shunt'].mean(), 'Pt': selected['Pt'].mean(),
            'shunt_std': selected['shunt'].std(ddof=0), 'Pt_std': selected['Pt'].std(ddof=0),
            'water': selected['water'].mean(),
            'full_stage_shunt': full['shunt'].mean(), 'full_stage_Pt': full['Pt'].mean(),
            'full_stage_shunt_std': full['shunt'].std(ddof=0),
            'full_stage_Pt_std': full['Pt'].std(ddof=0),
            'mean_instant_product_v2': (selected['shunt'] * selected['Pt']).mean(),
        })
    return pd.DataFrame(summaries), pd.DataFrame(diagnostics)


def add_heat_flux(summary, resistance_ohm, diameter_m, length_m):
    """Keep the legacy product-of-means definition and W/m² units."""
    if not all(np.isfinite(x) and x > 0 for x in (resistance_ohm, diameter_m, length_m)):
        raise ValueError('Shunt resistance, diameter and length must be finite positive')
    result = summary.copy()
    scale = resistance_ohm * np.pi * diameter_m * length_m
    result['q'] = result['shunt'] * result['Pt'] / scale
    result['q_full_stage'] = result['full_stage_shunt'] * result['full_stage_Pt'] / scale
    result['q_mean_instant_product'] = result['mean_instant_product_v2'] / scale
    result['selection_delta_pct'] = (result['q'] / result['q_full_stage'] - 1) * 100
    return result
