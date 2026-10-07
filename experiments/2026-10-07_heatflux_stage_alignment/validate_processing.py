"""Reproduce the bounded comparison without touching production labels or audio."""
import contextlib
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))
from utils.calculation.heatflux_preprocessing import (
    read_logger_csv, summarize_stages, add_heat_flux, load_processing_config,
)


def run():
    rows = []
    new_config = load_processing_config('2026.10.07_0.3_1')
    for day in ['2026.10.07_0.3_1', '2025.06.18_0.3_3']:
        root = next((ROOT / 'Pool_boiling').glob('*/0.3/' + day))
        current = day.startswith('2026')
        raw = next(root.glob('raw/logger/*')) if current else root / (day + '.csv')
        old_csv = root / ('実験結果' + day) / ('heat_flux_' + day + '.csv')
        reference = None if current else pd.read_csv(old_csv)
        voltages = new_config['voltage_sequence'] if current else reference.volt.str.rstrip('V').astype(float).tolist()
        frame = read_logger_csv(raw)
        for seconds in new_config['sensitivity_windows_seconds']:
            folder = OUT / 'comparison' / day / f'{seconds}s'
            folder.mkdir(parents=True, exist_ok=True)
            try:
                summary, candidates = summarize_stages(frame, voltages, window_seconds=seconds)
            except ValueError as error:
                rows.append({'experiment': day, 'window_seconds': seconds,
                             'status': 'no_valid_result', 'reason': str(error)})
                continue
            result = add_heat_flux(summary, 0.001667, 0.0003, 0.03827 if current else 0.03954)
            if reference is not None:
                result['old_q'] = reference.q.to_numpy()
                result['delta_from_old_pct'] = (result.q / result.old_q - 1) * 100
            result.to_csv(folder / 'stage_quality.csv', index=False)
            candidates.to_csv(folder / 'stable_interval_candidates.csv', index=False)
            labels = result[['volt', 'q']].copy()
            labels['volt'] = labels.volt.map(lambda value: f'{value:.1f}V')
            labels['q(e)'] = labels.q.map(lambda value: f'{value:.2e}')
            labels.to_csv(folder / ('heat_flux_' + day + '.csv'), index=False)
            positive = result.loc[result.volt > 0]
            row = {'experiment': day, 'window_seconds': seconds, 'status': 'complete',
                   'stage_count': len(result),
                   'min_coverage_fraction': positive.coverage_fraction.min(),
                   'max_selected_vs_full_pct': positive.selection_delta_pct.abs().max(),
                   'max_product_definition_difference_pct': ((positive.q_mean_instant_product / positive.q - 1)*100).abs().max(),
                   'logger_sha256': hashlib.sha256(raw.read_bytes()).hexdigest()}
            if reference is not None:
                row['max_delta_from_old_pct'] = positive.delta_from_old_pct.abs().max()
            else:
                row['q_at_observed_onb_1p2v_w_m2'] = float(result.loc[result.volt == 1.2, 'q'].iloc[0])
                row['max_delta_vs_35s_pct'] = None
            rows.append(row)
    new_primary = pd.read_csv(OUT / 'comparison/2026.10.07_0.3_1/35.0s/stage_quality.csv')
    new_short = pd.read_csv(OUT / 'comparison/2026.10.07_0.3_1/15.0s/stage_quality.csv')
    difference = ((new_short.q.iloc[1:].to_numpy() / new_primary.q.iloc[1:].to_numpy() - 1) * 100)
    for row in rows:
        if row['experiment'].startswith('2026') and row['window_seconds'] == 15:
            row['max_delta_vs_35s_pct'] = float(np.abs(difference).max())
    (OUT / 'comparison_summary.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')

    # Execute the existing notebook's thermal cells only, redirecting all outputs.
    notebook = json.loads((ROOT / 'code/0.run_auto_heatflux_analysis_v2.ipynb').read_text(encoding='utf-8'))
    audit = OUT / 'notebook_validation'
    if (audit / 'heat_flux_2026.10.07_0.3_1.csv').exists():
        raise FileExistsError('Validation output exists; use a different OUT for another comparison')
    namespace = {}
    log = io.StringIO()
    plt.show = lambda: plt.close('all')
    with contextlib.redirect_stdout(log):
        for index in range(8):
            source = ''.join(notebook['cells'][index]['source'])
            if index == 0:
                source = source.replace(
                    'save_path = str(resolve_heat_flux_csv_path(dir_path, repo_root).parent)',
                    'save_path = ' + repr(str(audit)))
            exec(compile(source, f'notebook_cell_{index+1}', 'exec'), namespace)
    (OUT / 'notebook_validation_log.txt').write_text(log.getvalue(), encoding='utf-8')
    labels = pd.read_csv(audit / 'heat_flux_2026.10.07_0.3_1.csv')
    assert len(labels) == 21
    assert np.allclose(labels.q, new_primary.q, rtol=1e-12, atol=1e-12)
    waves = next((ROOT / 'Pool_boiling').glob('*/0.3/2026.10.07_0.3_1')) / '録音データ'
    voltage_names = {f.stem.casefold() for f in waves.glob('*.wav')}
    assert voltage_names.issubset({name.casefold() for name in labels.volt})
    mapping = labels.loc[labels.volt.str.casefold().isin(voltage_names)].copy()
    mapping['csv_row_index'] = mapping.index + 1
    mapping.to_csv(OUT / 'wav_label_mapping.csv', index=False)
    assert len(mapping) == 11
    print('Comparison saved; notebook cells 1-8 complete; all 11 WAV names have explicit labels.')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    run()
