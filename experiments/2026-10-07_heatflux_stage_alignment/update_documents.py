"""Record the completed comparison and link the existing user execution guide."""
import json
from pathlib import Path


def replace_line(path, prefix, replacement):
    path = Path(path)
    lines = path.read_text(encoding='utf-8').splitlines()
    matches = [i for i, line in enumerate(lines) if line.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f'Expected one document line: {path}: {prefix}')
    lines[matches[0]] = replacement
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')


replace_line('docs/research_status.md', '**今日の実測実験',
    '**今日の実測実験の熱流束修正は確認済み、音響生成・ONB学習は未実行**：'
    '`2026.10.07_0.3_1`は従来の熱流束v2→録音コピー・改名→STFT→ONBを使う。'
    '[本人の実行順](2026-10-07_new_experiment_processing.md)、'
    '[段階・時間幅修正の比較](../experiments/2026-10-07_heatflux_stage_alignment/README.md)。'
    '独立抽出23/21段階の停止に対し、明示印加順＋35秒の共通安定窓＋各段階の最長連続領域に修正。'
    '新21段階・録音11本、旧6/18の非等間隔18段階（保存qとの差最大0.1581%）、'
    '熱計算セル1〜8の別保存先での完走、52テストを確認した。'
    '原13ファイル・本人debug・旧ラベルの計15ファイルはSHA-256一致。'
    '通常の新CSVは本人が`stage_time_v1_20261007`へ生成する。'
    '結果版configを改名/STFT/ONBにも共通化し、未生成なら旧版へ切り替えない。'
    'ONBは今回の日内75/25%・clean学習・1秒・3 kHz・既存5モデル、別日設定は6/11＋6/18→今回。'
    '現在ONBは7雑音評価、前処理はcleanだけなので実行時にそろえる。'
    'ONBはノートの観察1.2 Vに対応するqを読む。本人設定R_shunt=1.667e-3は保持し、'
    '実物の抵抗の確認・時計同期・1秒ごとの真値とは区別する。以下の完了済みモデル比較とは別工程。')

p = Path('docs/data_inventory.md')
s = p.read_text(encoding='utf-8')
start = s.index('### 2026.10.07_0.3_1')
end = s.index('\n### ', start + 1)
section = s[start:end]
section = section.replace('原データ整理済み・処理未実行', '原データ保持・熱流束の比較確認済み・音響生成未実行')
lines = section.splitlines()
replacements = {
    '- 未確認:': '- 未確認: 実物のシャント抵抗、低電圧側の独立した操作履歴、録音/ロガーの時計同期、音響切り出し、研究性能。本人設定1.667e-3 Ωと実物の確認を区別する。',
    '- 状態:': '- 状態: 明示印加順と35秒の共通区間で新21段階・録音11本が対応。旧6/18の保存qとの差最大0.1581%、既存Notebookの熱計算セル1〜8を別保存先で確認。本人debug・旧ラベルは保持、音響生成・学習は未実行。',
    '- 設定:': '- 設定: 従来3入口と`configs/datasets/2026.10.07_0.3_1_heatflux_processing.json`。通常CSVは`実験結果2026.10.07_0.3_1/stage_time_v1_20261007/`へ本人が生成する。改名・STFT・ONBも同じ版を参照。',
    '- 原ファイル記録:': '- 原ファイル記録: [配置記録](../experiments/2026-10-07_pool_boiling_data_setup/README.md)、[熱流束比較・原ファイル15件の再照合](../experiments/2026-10-07_heatflux_stage_alignment/README.md)。現在設定の1.2 Vは検算上330.370 kW/m²だが、装置条件を含めた確定値ではなく通常CSVも未生成。旧ONB監査の範囲は保持。現在のONB7雑音とcleanのみの生成条件の整合は手順書を参照。',
}
for i, line in enumerate(lines):
    for prefix, replacement in replacements.items():
        if line.startswith(prefix):
            lines[i] = replacement
p.write_text(s[:start] + '\n'.join(lines) + '\n' + s[end:], encoding='utf-8')

replace_line('docs/document_index.md', '| 今日追加した10/7実験を',
    '| 今日追加した10/7実験を自分で処理する | [従来コードを使う実行手順](2026-10-07_new_experiment_processing.md) | 原13ファイル保持、明示印加順・秒単位の共通区間、共通結果版を改名/STFT/ONBが参照。熱計算は別保存先で確認、音響生成・学習は本人が実行 |\n'
    '| 新実験の熱流束エラーの修正根拠を見る | [段階対応・時間幅の比較](../experiments/2026-10-07_heatflux_stage_alignment/README.md) | 新21/旧18段階、旧qとの差最大0.1581%、15/35/55秒、既存Notebookセル1〜8、52テスト、原15ファイル保持 |')
replace_line('docs/code_map.md', '更新日:',
    '更新日: 2026-10-07。現在の設定・完了runは[研究の現在地](research_status.md)。通常の主実行は[run_ensemble_regression_onb.py](../code/run_ensemble_regression_onb.py)。')
replace_line('docs/code_map.md', '| 10/7新実験の準備 |',
    '| 10/7新実験の準備 | [従来の熱流束Notebook](../code/0.run_auto_heatflux_analysis_v2.ipynb)、[heatflux_preprocessing.py](../code/utils/calculation/heatflux_preprocessing.py)、[実行手順](2026-10-07_new_experiment_processing.md) | GL860/旧CSV、時刻保持、明示印加順・35秒の共通区間。configで結果版を改名/STFT/ONBと共通化。新21/旧18段階を検算済み |')
replace_line('experiments/README.md', '| [10/7 新しい実測実験のONB設定]',
    '| [10/7 熱流束の段階・時間幅修正](2026-10-07_heatflux_stage_alignment/README.md) | 新21/旧18段階、旧qとの差最大0.1581%、15/35/55秒、熱計算セル1〜8、52テスト。原ファイル・過去結果保持、音響生成・学習なし |\n'
    '| [10/7 新しい実測実験のONB設定](2026-10-07_onb_new_day_configuration/README.md) | 日内/別日検証、本NPY指定、ノート1.2 VのCSV閾値。結果版の共通化は後継の熱流束修正記録を参照。実データ学習は未実行 |')
replace_line('experiments/README.md', '| [10/7 新しい実測実験のデータ配置]',
    '| [10/7 新しい実測実験のデータ配置](2026-10-07_pool_boiling_data_setup/README.md) | 原WAV11本・計測CSV・ノートCSVの配置とSHA-256。従来の実行入口を保持。後継の熱流束比較は上段を参照 |')

for name in ['2026-10-07_pool_boiling_data_setup', '2026-10-07_onb_new_day_configuration']:
    p = Path('experiments') / name / 'README.md'
    s = p.read_text(encoding='utf-8')
    first, rest = s.split('\n', 1)
    notice = ('\n\n**後続更新**：[段階対応・時間幅の修正と実データ検算](../2026-10-07_heatflux_stage_alignment/README.md)。'
              '以下は当初の準備記録。現在は新21/旧18段階・35秒窓・熱計算セル1〜8の完走まで確認済み。'
              '本人の現ONBは7雑音評価、生成側はcleanのみなので、[現行手順](../../docs/2026-10-07_new_experiment_processing.md)に沿ってそろえる。'
              '音響生成・モデル学習は未実行。\n')
    p.write_text(first + notice + rest, encoding='utf-8')

p = Path('configs/experiments/2026-10-07_onb_new_day_ready.json')
record = json.loads(p.read_text(encoding='utf-8'))
record['data']['noise_dir_names'] = ['heatflux_no_noise'] + [f'heatflux_reference_SNR={x}' for x in [0,-4,-8,-12,-16,-20]]
record['data']['preprocessing_noise_setting'] = 'Currently clean only; align with ONB conditions before running'
record['heatflux_processing_config'] = 'configs/datasets/2026.10.07_0.3_1_heatflux_processing.json'
record['heatflux_validation_record'] = 'experiments/2026-10-07_heatflux_stage_alignment/README.md'
p.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print('Current docs updated; prior setup records explicitly linked to the later validation.')
