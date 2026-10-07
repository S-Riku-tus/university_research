"""One-time notebook edit record. Run from repository root; not an analysis entrypoint."""
import json
from pathlib import Path


path = Path('code/0.run_auto_heatflux_analysis_v2.ipynb')
snapshot_dir = Path(__file__).resolve().parent / 'before_changes'
notebook = json.loads((snapshot_dir / path.name).read_text(encoding='utf-8-sig'))
source0 = ''.join(notebook['cells'][0]['source'])
parameters = source0[:source0.index('dir_path=os.path.dirname(input_folder)')]
parameters = '\n'.join(line if not line.startswith('VOLTAGE_SEQUENCE = ') else
    'VOLTAGE_SEQUENCE = None  # configから取得。別実験では実際の印加順を明示する。'
    for line in parameters.splitlines()) + '\n'
parameters = '\n'.join(line for line in parameters.splitlines()
                       if not line.startswith('overlap_n = ')) + '\n'
source0 = parameters + '''
dir_path = os.path.dirname(input_folder)

# 共通の処理条件を読み、コピー・STFT・ONBも同じ結果CSVを参照する。
import sys, json, hashlib, math
from pathlib import Path
repo_root = Path(dir_path).resolve().parents[3]
if str(repo_root / 'code') not in sys.path:
    sys.path.insert(0, str(repo_root / 'code'))
from utils.calculation.heatflux_preprocessing import (
    load_processing_config, resolve_heat_flux_csv_path, read_logger_csv,
    summarize_stages, add_heat_flux, detect_voltage_stages,
)
processing_config = load_processing_config(dir_path, repo_root)
shunt_s = processing_config.get('shunt_std_limit_v', shunt_s)
Pt_s = processing_config.get('electrode_std_limit_v', Pt_s)
Setting_Temp = processing_config.get('water_target_c', Setting_Temp)
if VOLTAGE_SEQUENCE is None:
    VOLTAGE_SEQUENCE = processing_config.get('voltage_sequence')
if VOLTAGE_SEQUENCE is None:
    raise ValueError('VOLTAGE_SEQUENCEに実際の印加電圧順（0 Vを含む）を指定してください。段階数からは推定しません。')

STABILITY_WINDOW_SECONDS = processing_config.get('stability_window_seconds', 35.0)
SETTLING_SECONDS = processing_config.get('settling_seconds', 5.0)
save_path = str(resolve_heat_flux_csv_path(dir_path, repo_root).parent)
label_csv = os.path.join(save_path, 'heat_flux_' + os.path.basename(dir_path) + '.csv')
if os.path.exists(label_csv):
    raise FileExistsError('計算済みの熱流束CSVを保護しました。再比較する場合はconfigのresults_run_tagを新しい名前にしてください。')
save_area_path = os.path.join(save_path, '非沸騰領域のグラフ')
save_time_path = os.path.join(save_path, 'time')
save_area_all_path = save_area_path  # 共通の過去グラフも上書きしない。
os.makedirs(save_area_path, exist_ok=True)
os.makedirs(save_time_path, exist_ok=True)

# GL860原本。旧形式ではNoneにすると従来の<実験名>.csvを読む。
LOGGER_CSV_RELATIVE = os.path.join('raw', 'logger', 'GL860_M411L1602_2026-10-07_11-12-55.CSV')
read_csv_file = (os.path.join(dir_path, LOGGER_CSV_RELATIVE) if LOGGER_CSV_RELATIVE
                 else os.path.join(dir_path, os.path.basename(dir_path) + '.csv'))
df = read_logger_csv(read_csv_file)
processing_df0 = df.copy()
processing_df1 = df.copy()  # 水温条件で行を詰めず、時間の欠落を区間判定に残す。
pd.set_option('display.max_rows', 30)
pd.options.display.float_format = '{:.8f}'.format
print('ロガー記録間隔 [s]:', float(np.median(np.diff(df['elapsed_s']))))
print('安定判定時間 [s]:', STABILITY_WINDOW_SECONDS)
print('結果保存先:', save_path)
'''
source1 = '''# 電極電圧の持続的な変化から段階の境界を確認する。
# 印加電圧の値はconfigの実験条件から与え、計測電圧とは区別する。
stage_boundaries = detect_voltage_stages(
    processing_df1,
    min_step_v=processing_config.get('transition_min_electrode_v', 0.04),
    confirm_seconds=processing_config.get('transition_confirm_seconds', 5.0),
    baseline_seconds=processing_config.get('baseline_seconds', 60.0),
)
boundary_report = stage_boundaries.copy()
boundary_report['start_scan'] = [int(df.index[i]) for i in boundary_report['start_pos']]
boundary_report['end_scan'] = [int(df.index[i-1]) for i in boundary_report['end_pos_exclusive']]
boundary_report.to_csv(os.path.join(save_path, 'detected_stage_boundaries.csv'), index=False)
print(boundary_report[['start_scan', 'end_scan']].to_string(index=False))
'''
source2 = '''# 1段階につき、同じ時間区間のシャント電圧・電極電圧から1組の平均を作る。
# 基準を満たさない段階や、印加順との不一致があれば自動補完せず停止する。
df_joint_stable, stable_interval_candidates = summarize_stages(
    processing_df1, VOLTAGE_SEQUENCE,
    window_seconds=STABILITY_WINDOW_SECONDS,
    baseline_seconds=processing_config.get('baseline_seconds', 60.0),
    min_step_v=processing_config.get('transition_min_electrode_v', 0.04),
    confirm_seconds=processing_config.get('transition_confirm_seconds', 5.0),
    settling_seconds=SETTLING_SECONDS,
    water_target_c=Setting_Temp,
    water_tolerance_c=processing_config.get('water_tolerance_c', 1.0),
    shunt_limit_v=shunt_s, electrode_limit_v=Pt_s,
)
shunt_average = df_joint_stable['shunt'].tolist()
Pt_average = df_joint_stable['Pt'].tolist()
voltage_values = df_joint_stable['volt'].to_numpy()
# 従来の水温条件・通電条件の平均を保ち、0 Vは選ばれた区間を加える。
first_power = stage_boundaries.iloc[1]['start_pos']
power_end = stage_boundaries.iloc[-1]['end_pos_exclusive']
powered = df.iloc[int(first_power):int(power_end)]
water_mask = powered['water'].between(Setting_Temp - 1, Setting_Temp + 1) & (powered['shunt'] > 0.001)
baseline = df.loc[df_joint_stable.iloc[0]['start_scan']:df_joint_stable.iloc[0]['end_scan']]
T_bulk = pd.concat([baseline, powered.loc[water_mask]])['water'].mean()
print('T_bulk:', T_bulk)
print(df_joint_stable[['volt', 'start_scan', 'end_scan', 'duration_s', 'coverage_fraction', 'stable_components', 'shunt', 'Pt']].to_string(index=False))
'''
source3 = '''# 採用区間と、採用されなかった安定区間も確認用に保存する。
stage_quality = add_heat_flux(df_joint_stable, R_shunt, d, L)
stage_quality.to_csv(os.path.join(save_path, 'joint_stable_regions_' + os.path.basename(dir_path) + '.csv'), index=False)
stable_interval_candidates.to_csv(os.path.join(save_path, 'stable_interval_candidates.csv'), index=False)
provenance = {
    'logger_csv': str(Path(read_csv_file).resolve()),
    'logger_sha256': hashlib.sha256(Path(read_csv_file).read_bytes()).hexdigest(),
    'processing_code_sha256': hashlib.sha256(Path(sys.modules[summarize_stages.__module__].__file__).read_bytes()).hexdigest(),
    'processing_config': processing_config,
    'voltage_sequence': list(VOLTAGE_SEQUENCE),
    'stability_window_seconds': STABILITY_WINDOW_SECONDS,
    'settling_seconds': SETTLING_SECONDS,
    'shunt_std_limit_v': shunt_s, 'electrode_std_limit_v': Pt_s,
    'R_shunt_ohm': R_shunt, 'diameter_m': d, 'length_m': L,
    'pressure_kpa': P, 'water_target_c': Setting_Temp,
    'T_bulk_c': T_bulk,
    'label_definition': 'Voltage-stage representative; no assumed WAV/logger clock synchronization',
    'resistance_status': 'Notebook value; actual experimental resistance is not yet confirmed',
}
with open(os.path.join(save_path, 'processing_provenance.json'), 'w', encoding='utf-8') as stream:
    json.dump(provenance, stream, ensure_ascii=False, indent=2)
'''
source4 = ''.join(notebook['cells'][4]['source'])
start = source4.index('if VOLTAGE_SEQUENCE is None:')
end = source4.index("df_heat_flux['volt']", start)
source4 = source4[:start] + '''if len(voltage_values) != len(df_graph):
    raise ValueError('印加電圧と熱流束の対応が一致しません。採用区間CSVを確認してください。')
''' + source4[end:]
# Near-zero current offsets do not measure wire resistance or temperature.
source4 = source4.replace('    R_Pt, q = Heat_flux(shunt, Pt)\n',
    "    R_Pt, q = Heat_flux(shunt, Pt)\n    if voltage_values[i] == 0:\n        R_Pt = np.nan  # 無通電時の微小オフセットから抵抗は求めない。\n")
source4 = source4.replace('    R_Pt = Pt / I', '    R_Pt = Pt / I if I != 0 else np.nan')
source4 = source4.replace('    T = Temperature(T_bulk, R_bulk, list_R)',
    '    T = T_bulk if voltage_values[i] == 0 else Temperature(T_bulk, R_bulk, list_R)')
# Existing thermal fit can finish without losing linearity; do not leave an undefined variable.
source4 = source4.replace('prev_intercept = None\n', 'prev_intercept = None\nlost_x = np.nan\n')
source4 = source4.replace('print("直線が失われる1つ前の近似直線の切片:", R_bulk)',
    "if prev_intercept is None:\n    raise ValueError('R_bulk推定に必要な非沸騰候補点が不足しています。')\n"
    "if not np.isfinite(lost_x):\n    raise ValueError('自動直線性喪失候補が得られません。熱流束CSVは保存済みです。温度・沸騰曲線の適用条件を確認してください。')\n"
    'print("直線が失われる1つ前の近似直線の切片:", R_bulk)')
for index, source in enumerate([source0, source1, source2, source3, source4]):
    notebook['cells'][index]['source'] = source.splitlines(keepends=True)
# Clear stale results from the former method; snapshots retain all original outputs.
for cell in notebook['cells']:
    if cell.get('cell_type') == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')

rename_path = Path('code/1.run_rename_files.ipynb')
rename_notebook = json.loads((snapshot_dir / rename_path.name).read_text(encoding='utf-8-sig'))
source = ''.join(rename_notebook['cells'][0]['source'])
start = source.index('# CSVファイルとフォルダのパスを指定')
experiment_root = str(Path('Pool_boiling/Subcooling_20_degrees/0.3/2026.10.07_0.3_1').resolve())
source = source[:start] + '''# 実験フォルダを指定（結果の版は熱流束用configと共通）。
import sys
from pathlib import Path
EXPERIMENT_ROOT = Path(''' + repr(experiment_root) + ''')
repo_root = EXPERIMENT_ROOT.parents[3]
if str(repo_root / 'code') not in sys.path:
    sys.path.insert(0, str(repo_root / 'code'))
from utils.calculation.heatflux_preprocessing import resolve_heat_flux_csv_path
csv_file_path = str(resolve_heat_flux_csv_path(EXPERIMENT_ROOT, repo_root))
if not os.path.isfile(csv_file_path):
    raise FileNotFoundError('先に熱流束v2を実行して採用区間とCSVを確認してください: ' + csv_file_path)
source_path = str(EXPERIMENT_ROOT / '録音データ')
folder_path = str(EXPERIMENT_ROOT / '録音データ_熱流束')
copy_folder(source_path, folder_path)
rename_files(csv_file_path, folder_path)
'''
rename_notebook['cells'][0]['source'] = source.splitlines(keepends=True)
for cell in rename_notebook['cells']:
    if cell.get('cell_type') == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None
rename_path.write_text(json.dumps(rename_notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
