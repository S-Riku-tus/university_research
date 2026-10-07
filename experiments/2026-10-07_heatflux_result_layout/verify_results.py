"""Check the user's completed outputs and their downstream references, without generating audio."""
import ast
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))
from utils.calculation.heatflux_preprocessing import (
    read_logger_csv, summarize_stages, add_heat_flux, resolve_heat_flux_csv_path,
)
from utils.dataloading.waterflow_preprocessing import build_experiment_context, heat_flux_label_from_wav
from utils.experiment.onb_thresholds import resolve_csv_onb_threshold_record


def main():
    experiment = next((ROOT / 'Pool_boiling').glob('*/0.3/2026.10.07_0.3_1'))
    csv_path = resolve_heat_flux_csv_path(experiment)
    folder = csv_path.parent
    manifest = json.loads((OUT / 'before_changes_manifest.json').read_text(encoding='utf-8'))
    integrity = []
    for old in manifest['files']:
        if 'relative_result_path' in old:
            new = folder / old['relative_result_path']
            matched = hashlib.sha256(new.read_bytes()).hexdigest() == old['sha256']
            assert matched, new
            integrity.append({'path':new.relative_to(ROOT).as_posix(), 'sha256':old['sha256'], 'unchanged':matched})
    provenance = json.loads((folder / 'processing_provenance.json').read_text(encoding='utf-8'))
    logger = Path(provenance['logger_csv'])
    assert hashlib.sha256(logger.read_bytes()).hexdigest() == provenance['logger_sha256']
    summary, _ = summarize_stages(
        read_logger_csv(logger), provenance['voltage_sequence'],
        window_seconds=provenance['stability_window_seconds'],
        settling_seconds=provenance['settling_seconds'],
        water_target_c=provenance['water_target_c'],
        shunt_limit_v=provenance['shunt_std_limit_v'], electrode_limit_v=provenance['electrode_std_limit_v'],
    )
    independent = add_heat_flux(summary, provenance['R_shunt_ohm'], provenance['diameter_m'], provenance['length_m'])
    labels = pd.read_csv(csv_path)
    quality = pd.read_csv(folder / ('joint_stable_regions_' + experiment.name + '.csv'))
    temperature = pd.read_csv(folder / ('Volt_T_' + experiment.name + '.csv'))
    resonance = pd.read_csv(folder / ('T_Reso_' + experiment.name + '.csv'))
    volts = labels.volt.str.rstrip('Vv').astype(float).to_numpy()
    assert len(labels) == len(temperature) == len(resonance) == len(quality) == 21
    assert np.allclose(volts, np.arange(21)/10, rtol=0, atol=1e-12)
    assert np.isfinite(labels.q).all() and np.all(np.diff(labels.q) > 0)
    assert np.allclose(labels.q, independent.q, rtol=1e-12, atol=1e-10)
    assert np.array_equal(quality.start_scan, independent.start_scan)
    assert np.array_equal(quality.end_scan, independent.end_scan)
    assert np.allclose(quality.q, labels.q, rtol=1e-12)
    assert np.array_equal(temperature.volt, labels.volt)
    assert np.array_equal(resonance.volt, labels.volt)
    assert np.isfinite(temperature['T']).all()
    frequencies = resonance.drop(columns='volt').to_numpy()
    assert frequencies.shape == (21, 12) and np.isfinite(frequencies).all() and (frequencies > 0).all()

    notebook = json.loads((OUT / 'before_changes/0.run_auto_heatflux_analysis_v2.ipynb').read_text(encoding='utf-8'))
    executed = [i+1 for i,c in enumerate(notebook['cells']) if c.get('execution_count') is not None]
    errors = [o for c in notebook['cells'] for o in c.get('outputs',[]) if o.get('output_type') == 'error']
    assert executed == list(range(1,12)) and not errors
    thermal_reference = ROOT / 'experiments/2026-10-07_heatflux_stage_alignment/notebook_validation'
    for prefix in ['Volt_T_', 'T_Reso_']:
        actual = pd.read_csv(folder / (prefix + experiment.name + '.csv'))
        expected = pd.read_csv(thermal_reference / (prefix + experiment.name + '.csv'))
        pd.testing.assert_frame_equal(actual, expected)

    # Exercise the actual rename entrypoint configuration without copying audio.
    rename = json.loads((ROOT / 'code/1.run_rename_files.ipynb').read_text(encoding='utf-8'))
    tree = ast.parse(''.join(rename['cells'][0]['source']))
    tree.body = [node for node in tree.body if not (isinstance(node, ast.Expr)
                 and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                 and node.value.func.id in {'copy_folder','rename_files'})]
    namespace = {}
    exec(compile(tree, 'rename_configuration_only', 'exec'), namespace)
    context = build_experiment_context(experiment.name, str(experiment.parent), '録音データ_熱流束', '', 20261007, 1, 'configuration_check')
    onb = resolve_csv_onb_threshold_record(experiment.name)
    assert Path(namespace['csv_file_path']) == Path(context['heat_flux_csv_path']) == csv_path
    assert ROOT / onb['heat_flux_csv'] == csv_path
    assert onb['threshold'] == float(labels.loc[labels.volt == '1.2V', 'q'].iloc[0])
    import run_ensemble_regression_onb as entrypoint
    entrypoint.validate_validation_config(entrypoint.MODEL_SPECS)
    assert entrypoint.THRESHOLD_BY_EXPERIMENT[experiment.name] == onb['threshold']
    recordings = experiment / '録音データ'
    names = {p.stem.casefold() for p in recordings.glob('*.wav')}
    mapping = []
    for index, row in labels.iterrows():
        if row.volt.casefold() in names:
            copied_name = f'index={index+1}.{int(row.q)}.wav'
            stft_label = heat_flux_label_from_wav(copied_name, context)
            assert np.isclose(float(stft_label), row.q, rtol=1e-12, atol=1e-10)
            mapping.append({'recording_voltage':row.volt, 'csv_row':index+1, 'copied_filename':copied_name,'q_w_m2':row.q})
    assert len(mapping) == 11

    original_manifest = json.loads((ROOT / 'experiments/2026-10-07_pool_boiling_data_setup/original_files_manifest.json').read_text(encoding='utf-8-sig'))
    for entry in original_manifest['files']:
        assert hashlib.sha256((ROOT / entry['current_path']).read_bytes()).hexdigest() == entry['sha256']
    previous = json.loads((ROOT / 'experiments/2026-10-07_heatflux_stage_alignment/before_changes_manifest.json').read_text(encoding='utf-8-sig'))
    for entry in previous['files']:
        if entry['path'].startswith('Pool_boiling/'):
            assert hashlib.sha256((ROOT / entry['path']).read_bytes()).hexdigest() == entry['sha256']
    drops = [{'from_volt':labels.volt.iloc[i-1], 'to_volt':labels.volt.iloc[i],
              'from_c':temperature['T'].iloc[i-1], 'to_c':temperature['T'].iloc[i]}
             for i in range(1,len(labels)) if temperature['T'].iloc[i] < temperature['T'].iloc[i-1]]
    resistance_check = []
    for voltage in [1.1, 1.2]:
        row = quality.loc[quality.volt == voltage].iloc[0]
        resistance_check.append({
            'volt':voltage,
            'selected_wire_resistance_ohm':float(row.Pt / (row.shunt / provenance['R_shunt_ohm'])),
            'full_stage_wire_resistance_ohm':float(row.full_stage_Pt / (row.full_stage_shunt / provenance['R_shunt_ohm'])),
        })
    report = {
        'date':'2026-10-07','output_layout':'established hierarchy',
        'heat_flux_csv':csv_path.relative_to(ROOT).as_posix(),
        'user_executed_cells':executed,'notebook_saved_errors':0,
        'csv_stage_count':21,'matched_recording_count':11,
        'max_recomputed_q_abs_difference_w_m2':float(np.max(np.abs(labels.q-independent.q))),
        'thermal_outputs_match_separate_notebook_validation':True,
        'onb_observed_voltage_v':1.2,'onb_threshold_w_m2':onb['threshold'],
        'rename_stft_onb_csv_paths_match':True,
        'onb_entrypoint_config_validation_passed':True,
        'moved_results_unchanged':integrity,'original_and_previous_results_sha256_match':True,
        'temperature_decreases_observed':drops,
        'selected_vs_full_stage_resistance_check':resistance_check,
        'physical_validation_note':'Successful execution and numeric consistency do not establish actual shunt resistance or WAV/logger synchronization.',
        'wav_mapping':mapping,
    }
    (OUT / 'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('All 11 relocated outputs unchanged; all 21 stages match recalculation; 11 WAVs match exact CSV labels.')
    print('Rename/STFT/ONB use the same established CSV. ONB:',onb['threshold'],'W/m2.')
    print('Temperature decreases (recorded, not automatically corrected):',drops)


if __name__ == '__main__':
    main()
