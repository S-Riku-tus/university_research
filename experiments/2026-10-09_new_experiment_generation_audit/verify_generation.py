"""Read-only audit of the requested new-day inputs; write evidence here only."""
import ast
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EXPERIMENT = '2026.10.07_0.3_1'
BASE = ROOT / 'Pool_boiling/Subcooling_20_degrees/0.3' / EXPERIMENT
LABELS = BASE / ('\u5b9f\u9a13\u7d50\u679c' + EXPERIMENT) / f'heat_flux_{EXPERIMENT}.csv'
WAVS = BASE / '\u9332\u97f3\u30c7\u30fc\u30bf_\u71b1\u6d41\u675f'
LEVELS = [None, 0, -4, -8, -12, -16, -20]
BANDS = [3000, 5000, 10000, 15000, 22050]
ENTRY = ROOT / 'code/2.run_npy_waterflow_2\u3064highpass.py'


def long_path(path):
    value = str(Path(path).resolve())
    return '\\\\?\\' + value if os.name == 'nt' and not value.startswith('\\\\?\\') else value


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_rows(path):
    with open(long_path(path), encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def files_in(folder, extension):
    if not folder.is_dir():
        return {}
    with os.scandir(long_path(folder)) as entries:
        return {entry.name: entry.stat().st_size for entry in entries
                if entry.is_file() and entry.name.endswith(extension)}


def main():
    errors = []
    conditions, reproduction, sampled = [], [], []
    def check(ok, message):
        if not ok:
            errors.append(message)

    labels = csv_rows(LABELS)
    label_map = {index: row for index, row in enumerate(labels, 1)}
    source_names = {path.name for path in WAVS.glob('*.wav')}
    source_ids = {Path(name).stem for name in source_names}
    old = json.loads((ROOT / 'experiments/2026-10-08_new_experiment_input_audit/verification.json').read_text(encoding='utf-8'))
    check(digest(LABELS) == old['label_csv_sha256'], 'Heat-flux CSV changed since the previous audit')
    check(len(source_ids) == 11, 'Expected 11 labelled source WAVs')
    row_lookup, selected_for_reproduction = {}, []
    paired_noise, ref_records = {}, []
    max_noise_relative_error = 0.0

    for seconds in (0.5, 1):
        dataset = f'waterflow_20261007_{seconds:g}s'
        dataset_dir = BASE / 'data/npy' / dataset
        reference_path = dataset_dir / 'global_reference_manifest.json'
        reference = json.loads(reference_path.read_text(encoding='utf-8'))
        rms = float(reference['reference_rms'])
        check(reference['source_count'] == 11 and len(reference['source_rms']) == 11,
              f'{dataset}: reference source count')
        check(math.isclose(rms, float(np.median([r['filtered_rms'] for r in reference['source_rms']])), rel_tol=1e-14),
              f'{dataset}: reference is not the recorded source RMS median')
        ref_records.append({'dataset': dataset, 'reference_rms': rms,
                            'created_at': reference['created_at'], 'sha256': digest(reference_path)})
        chunks = round(60 / seconds)
        expected_count = len(source_ids) * chunks
        for hz in BANDS:
            band = f'maxfreq={round(hz/1000)}kHz'
            parent = dataset_dir / band
            png_parent = BASE / 'data/spectrogram_png' / dataset / band
            prep_path = parent / 'preprocess_manifest.json'
            prep = json.loads(prep_path.read_text(encoding='utf-8'))
            check(prep['chunk_seconds'] == seconds and prep['max_freq_hz'] == hz,
                  f'{dataset}/{band}: preprocessing dimensions')
            check(prep['noise']['scaling_mode'] == 'fixed_global_rms'
                  and prep['noise']['fixed_reference_signal_rms'] == rms,
                  f'{dataset}/{band}: noise reference')
            check(json.loads((png_parent / 'preprocess_manifest.json').read_text(encoding='utf-8')) == prep,
                  f'{dataset}/{band}: NPY/PNG preprocessing metadata differ')
            clean_arrays = {}
            baseline_rows = {}
            for level in LEVELS:
                noise = 'heatflux_no_noise' if level is None else f'heatflux_reference_SNR={level}'
                folder, png_folder = parent / noise, png_parent / noise
                prefix = f'{dataset}/{band}/{noise}'
                rows = csv_rows(folder / 'chunk_manifest.csv')
                png_rows = csv_rows(png_folder / 'chunk_manifest.csv')
                # Each format records its own extension; compare the sample stem
                # and all provenance/noise fields after normalizing that suffix.
                normalized_png_rows = [{**row, 'sample_filename':Path(row['sample_filename']).stem + '.npy'}
                                       for row in png_rows]
                check(normalized_png_rows == rows, prefix + ': NPY/PNG chunk records differ')
                npys, pngs = files_in(folder, '.npy'), files_in(png_folder, '.png')
                expected_names = {r['sample_filename'] for r in rows}
                check(len(rows) == expected_count == len(expected_names) == len(npys) == len(pngs),
                      prefix + ': counts or duplicate filenames')
                check(set(npys) == expected_names, prefix + ': NPY filenames/manifest')
                check(set(pngs) == {Path(name).stem + '.png' for name in expected_names}, prefix + ': PNG filenames/manifest')
                check(all(size > 0 for size in [*npys.values(), *pngs.values()]), prefix + ': empty files')
                check(all(size == 200832 for size in npys.values()), prefix + ': unexpected NPY file size')
                coverage = defaultdict(list)
                row_by_key = {}
                for row in rows:
                    source, chunk = row['source_wav_id'], int(row['chunk_index'])
                    key = (source, chunk)
                    coverage[source].append(chunk)
                    row_by_key[key] = row
                    index = int(re.search(r'index=(\d+)', source).group(1))
                    q = float(row['heat_flux'])
                    check(q == float(label_map[index]['q']) == float(row['sample_filename'].split('_src-')[0]),
                          prefix + ': label or filename mismatch')
                    check(row['experiment_name'] == EXPERIMENT and row['source_wav_name'] in source_names,
                          prefix + ': source provenance')
                    check(float(row['chunk_duration_seconds']) == seconds
                          and float(row['chunk_start_seconds']) == chunk * seconds
                          and int(row['chunk_start_sample']) == round(chunk * seconds * 44100)
                          and int(row['unpadded_signal_samples']) == round(seconds * 44100),
                          prefix + ': chunk time/length')
                    if level is None:
                        check(not row['requested_snr_db'] and not row['scaled_noise_chunk_power']
                              and float(row['model_input_power']) == float(row['signal_chunk_power']),
                              prefix + ': clean metadata')
                        baseline_rows[key] = row
                    else:
                        check(float(row['requested_snr_db']) == level
                              and row['noise_scaling_mode'] == 'fixed_global_rms'
                              and float(row['fixed_reference_signal_rms']) == rms,
                              prefix + ': noise level/scaling')
                        power = float(row['scaled_noise_chunk_power'])
                        target = rms**2 / 10**(level/10)
                        error = abs(power - target) / target
                        max_noise_relative_error = max(max_noise_relative_error, error)
                        check(error < 1e-12, prefix + ': scaled noise power differs from requested target')
                        check(math.isclose(float(row['raw_noise_chunk_power']) * float(row['noise_scale'])**2,
                                           power, rel_tol=1e-12), prefix + ': scale/power mismatch')
                        check(math.isclose(10*math.log10(float(row['signal_chunk_power']) / power),
                                           float(row['realized_snr_db']), abs_tol=1e-10), prefix + ': realized SNR')
                        check(key in baseline_rows and row['signal_chunk_power'] == baseline_rows[key]['signal_chunk_power'],
                              prefix + ': noise/clean source power differs')
                        payload = f'42|shared_across_all_sources|paired_across_snr|{chunk}'
                        expected_seed = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], 'little')
                        check(int(row['noise_seed']) == expected_seed, prefix + ': seed mismatch')
                        pair = (row['noise_seed'], row['noise_offset_samples'], row['raw_noise_chunk_power'])
                        pair_key = (seconds, chunk)
                        check(paired_noise.setdefault(pair_key, pair) == pair, prefix + ': unpaired noise across sources/levels/bands')
                check(set(coverage) == source_ids and all(sorted(indices) == list(range(chunks)) for indices in coverage.values()),
                      prefix + ': missing/duplicate source chunks')
                # Fixed sampling covers low heat flux, onset, and highest heat flux;
                # first, middle, and final chunks in every one of the 70 conditions.
                selected_sources = [source for source in sorted(source_ids)
                                    if int(re.search(r'index=(\d+)', source).group(1)) in (11, 13, 21)]
                for source in selected_sources:
                    for chunk in (0, chunks//2, chunks-1):
                        key = (source, chunk)
                        row = row_by_key[key]
                        array = np.load(long_path(folder / row['sample_filename']), allow_pickle=False)
                        check(array.shape == (224, 224) and array.dtype == np.float32
                              and np.isfinite(array).all() and (array >= 0).all() and (array > 0).any(),
                              prefix + ': invalid sampled NPY')
                        if level is None:
                            clean_arrays[key] = array
                        else:
                            check(not np.array_equal(array, clean_arrays[key]), prefix + ': noisy sample equals clean')
                        with Image.open(long_path(png_folder / (Path(row['sample_filename']).stem + '.png'))) as picture:
                            picture.verify()
                        sampled.append({'condition': prefix, 'source_wav_id':source, 'chunk_index':chunk})
                        if (seconds == 1 and hz == 3000 and chunk == 0) or (seconds == 0.5 and hz == 22050 and chunk == chunks-1 and level == -20):
                            selected_for_reproduction.append((seconds, hz, level, row, folder / row['sample_filename']))
                row_lookup[(seconds, hz, level)] = row_by_key
                conditions.append({'dataset':dataset, 'band':band, 'max_frequency_hz':hz, 'chunk_seconds':seconds,
                    'noise':noise, 'reference_snr_db':level, 'npy_count':len(npys), 'png_count':len(pngs),
                    'manifest_rows':len(rows), 'source_wavs':len(coverage), 'chunks_per_source':chunks,
                    'sampled_npy_and_png':9, 'preprocess_created_at':prep['created_at'],
                    'preprocess_configured_snrs':prep['noise']['snr_list_db'],
                    'preprocess_sha256':digest(prep_path), 'chunk_manifest_sha256':digest(folder/'chunk_manifest.csv')})
            print(f'Checked {dataset}/{band}: seven complete conditions; contents sampled.', flush=True)

    # Reconstruct 24 predetermined inputs in memory without saving to the data tree.
    sys.path.insert(0, str(ROOT / 'code'))
    spec = importlib.util.spec_from_file_location('generation_for_read_only_audit', ENTRY)
    generation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generation)
    from utils.dataloading import waterflow_preprocessing as preprocessing
    from skimage.transform import resize
    shared_noise = generation._load_and_filter(generation.WATERFLOW_PATH, first_channel=True)
    sources = {}
    for seconds, hz, level, row, path in selected_for_reproduction:
        source_name = row['source_wav_name']
        if source_name not in sources:
            sources[source_name] = generation._load_and_filter(str(WAVS / source_name))
        signal = sources[source_name]
        start, length = int(row['chunk_start_sample']), round(seconds * 44100)
        signal_chunk = signal[start:start+length]
        model_input = signal_chunk
        if level is not None:
            noise_chunk, offset = preprocessing.select_noise_chunk(shared_noise, length, start_sample=start,
                randomize_offset=True, random_seed=int(row['noise_seed']))
            noise_scaled, scale = preprocessing.scale_noise_chunk(signal, shared_noise, noise_chunk, level,
                scaling_mode='fixed_global_rms', fixed_reference_signal_rms=float(row['fixed_reference_signal_rms']))
            check(offset == int(row['noise_offset_samples'])
                  and math.isclose(scale, float(row['noise_scale']), rel_tol=1e-12), 'Reproduction noise parameters differ')
            model_input = signal_chunk + noise_scaled
        stft = preprocessing.calc_stft(model_input, 672, 44100)
        max_k = round(hz / (44100/1344))
        rebuilt = resize(stft[:, :max_k+1], (224, 224)).astype(np.float32)
        saved = np.load(long_path(path), allow_pickle=False)
        equal = bool(np.array_equal(saved, rebuilt))
        check(equal, f'Reproduction differs: {path.relative_to(BASE)}')
        reproduction.append({'chunk_seconds':seconds, 'max_frequency_hz':hz, 'reference_snr_db':level,
            'source_wav_id':row['source_wav_id'], 'chunk_index':int(row['chunk_index']),
            'bitwise_equal':equal, 'maximum_absolute_difference':float(np.max(np.abs(saved.astype(float)-rebuilt)))})

    total_npy = sum(c['npy_count'] for c in conditions)
    total_png = sum(c['png_count'] for c in conditions)
    check(len(conditions) == 70 and total_npy == total_png == 69300, 'Total condition/file counts')
    main_rows = row_lookup[(1, 3000, None)].values()
    threshold = float(next(row['q'] for row in labels if row['volt'] == '1.2V'))
    raw_root = BASE / '\u9332\u97f3\u30c7\u30fc\u30bf'
    summary = {'checked_at_jst':datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds'),
        'experiment':EXPERIMENT, 'status':'passed' if not errors else 'failed',
        'generation_save_tag':20261007, 'levels_db':LEVELS, 'bands_hz':BANDS, 'chunk_seconds':[0.5,1],
        'raw_wavs':sum(1 for p in raw_root.glob('*.wav')), 'labelled_wavs':len(source_names),
        'label_csv_rows':len(labels), 'label_csv_sha256':digest(LABELS),
        'label_csv_unchanged_since_previous_audit':digest(LABELS) == old['label_csv_sha256'],
        'condition_count':len(conditions), 'total_npy':total_npy, 'total_png':total_png,
        'clean_per_format':sum(c['npy_count'] for c in conditions if c['reference_snr_db'] is None),
        'noise_per_format':sum(c['npy_count'] for c in conditions if c['reference_snr_db'] is not None),
        'full_metadata_rows_checked':sum(c['manifest_rows'] for c in conditions),
        'sampled_npy_contents_and_png_files':len(sampled), 'content_sampling_policy':'indices 11,13,21; first,middle,last chunks; every condition',
        'noise_target_maximum_relative_error':max_noise_relative_error,
        'global_reference_records':ref_records, 'onb_q_w_m2':threshold,
        'main_1s_3khz_chunks_per_noise':660, 'main_positive_chunks':sum(float(row['heat_flux']) >= threshold for row in main_rows),
        'regression_result_files':sum(p.is_file() for p in (BASE/'regression_result').rglob('*')),
        'generation_entry_sha256':digest(ENTRY), 'shared_preprocessing_sha256':digest(ROOT/'code/utils/dataloading/waterflow_preprocessing.py'),
        'reproduction_checks':reproduction, 'conditions':conditions, 'error_count':len(errors), 'errors':errors[:50],
        'scope':'All file counts, names, sizes, source/chunk coverage, labels and noise metadata; stratified content samples, 24 in-memory reconstructions. No data generation or model training. Not an exhaustive content scan of every NPY/PNG.'}
    (OUT/'verification.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    with (OUT/'condition_counts.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        fields = [key for key in conditions[0] if key not in ('preprocess_configured_snrs',)]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(conditions)
    print(json.dumps({key:summary[key] for key in ('status','condition_count','total_npy','total_png',
        'clean_per_format','noise_per_format','sampled_npy_contents_and_png_files','regression_result_files','error_count')}, ensure_ascii=True))
    if errors:
        print(json.dumps(errors[:10], ensure_ascii=True))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
