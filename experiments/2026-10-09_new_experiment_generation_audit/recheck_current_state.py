"""Recheck saved audit metadata and live outputs without rerunning content tests."""
import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'Pool_boiling/Subcooling_20_degrees/0.3/2026.10.07_0.3_1'


def long_path(path):
    value = str(Path(path).resolve())
    return '\\\\?\\' + value if os.name == 'nt' else value


def digest(path):
    with open(long_path(path), 'rb') as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def csv_rows(path):
    with open(long_path(path), encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def files_in(path, extension):
    with os.scandir(long_path(path)) as entries:
        return {entry.name for entry in entries if entry.is_file() and entry.name.endswith(extension)}


def main():
    previous = json.loads((OUT / 'verification.json').read_text(encoding='utf-8'))
    conditions = csv_rows(OUT / 'condition_counts.csv')
    errors, records = [], []
    for row in conditions:
        npy_parent = BASE / 'data/npy' / row['dataset'] / row['band']
        png_parent = BASE / 'data/spectrogram_png' / row['dataset'] / row['band']
        npy_folder, png_folder = npy_parent / row['noise'], png_parent / row['noise']
        names_npy, names_png = files_in(npy_folder, '.npy'), files_in(png_folder, '.png')
        manifest = csv_rows(npy_folder / 'chunk_manifest.csv')
        expected = {item['sample_filename'] for item in manifest}
        metadata_unchanged = (
            digest(npy_parent / 'preprocess_manifest.json') == row['preprocess_sha256']
            and digest(npy_folder / 'chunk_manifest.csv') == row['chunk_manifest_sha256'])
        matching_files = (names_npy == expected and names_png == {Path(name).stem + '.png' for name in expected}
            and len(names_npy) == int(row['npy_count']) and len(names_png) == int(row['png_count']))
        if not metadata_unchanged or not matching_files:
            errors.append(f"{row['dataset']}/{row['band']}/{row['noise']}: changed metadata or outputs")
        records.append({'dataset': row['dataset'], 'band': row['band'], 'noise': row['noise'],
            'npy_count': len(names_npy), 'png_count': len(names_png),
            'metadata_unchanged': metadata_unchanged, 'files_match_manifest': matching_files})

    labels_path = BASE / '実験結果2026.10.07_0.3_1/heat_flux_2026.10.07_0.3_1.csv'
    labels_unchanged = digest(labels_path) == previous['label_csv_sha256']
    labelled_recordings = list((BASE / '録音データ_熱流束').glob('*.wav'))
    model_files = [path for path in (BASE / 'regression_result').rglob('*') if path.is_file()]
    main_path = ROOT / 'code/run_ensemble_regression_onb.py'
    tree = ast.parse(main_path.read_text(encoding='utf-8-sig'))
    config_expression = next(node.value.args[0] for node in tree.body
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'VALIDATION_CONFIG' for target in node.targets))
    config_nodes = {key.value: value for key, value in zip(config_expression.keys, config_expression.values)}
    data, policy = ast.literal_eval(config_nodes['data']), ast.literal_eval(config_nodes['learning_policy'])
    old_ranges = []
    for experiment in policy['evaluation_settings']['cross_day']['train_experiments']:
        manifest_path = ROOT / 'Pool_boiling/Subcooling_20_degrees/0.3' / experiment / 'data/npy/waterflow_20260817_1s/maxfreq=3kHz/preprocess_manifest.json'
        metadata = json.loads(manifest_path.read_text(encoding='utf-8'))
        heatflux = [float(row['q']) for row in csv_rows(metadata['heat_flux_csv_path'])]
        old_ranges.append({'experiment': experiment, 'heat_flux_min_kw_m2': min(heatflux)/1000,
            'heat_flux_max_kw_m2': max(heatflux)/1000, 'reference_rms': metadata['noise']['fixed_reference_signal_rms']})
    heatflux_new = [float(row['q']) for row in csv_rows(labels_path)[10:]]
    rms_new = previous['global_reference_records'][0]['reference_rms']
    rms_old = old_ranges[0]['reference_rms']
    if not labels_unchanged:
        errors.append('Heat-flux CSV changed after the full audit')
    if len(conditions) != 70:
        errors.append('Expected all 70 conditions')
    report = {'record_date_jst': '2026-10-09', 'status': 'passed' if not errors else 'needs_review',
        'previous_full_audit_checked_at_jst': previous['checked_at_jst'],
        'condition_count': len(records), 'total_npy': sum(row['npy_count'] for row in records),
        'total_png': sum(row['png_count'] for row in records), 'labelled_wavs': len(labelled_recordings),
        'label_csv_unchanged': labels_unchanged, 'regression_result_files': len(model_files),
        'regression_result_file_paths': [path.relative_to(ROOT).as_posix() for path in model_files],
        'current_onb_evaluation_mode': policy['evaluation_mode'],
        'current_onb_training_noise': policy['training_noise'],
        'current_onb_data': {'chunk_seconds': data['chunk_seconds'], 'max_freq_hz_list': data['max_freq_hz_list'],
            'noise_dir_names': data['noise_dir_names']},
        'current_onb_evaluation_settings': policy['evaluation_settings'],
        'current_onb_sha256': digest(main_path), 'old_day_ranges': old_ranges,
        'new_recorded_heat_flux_min_kw_m2': min(heatflux_new)/1000,
        'new_recorded_heat_flux_max_kw_m2': max(heatflux_new)/1000,
        'new_to_old_reference_rms_ratio': rms_new/rms_old,
        'reference_amplitude_difference_db': 20*math.log10(rms_new/rms_old),
        'prior_content_checks': {'sampled_pairs': previous['sampled_npy_contents_and_png_files'],
            'reproduced_chunks': len(previous['reproduction_checks']), 'prior_error_count': previous['error_count']},
        'scope': 'Live counts and filename/manifest correspondence, saved metadata hashes, labels, model-output presence, and configuration. Prior content samples were not rerun; no generation or model training/prediction.',
        'errors': errors, 'conditions': records}
    (OUT / 'current_state_recheck.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key not in ['conditions', 'current_onb_evaluation_settings', 'regression_result_file_paths']}, ensure_ascii=True))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
