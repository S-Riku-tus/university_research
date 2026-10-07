# データ索引

巨大なデータ本体はGitで管理せず、このファイルで場所・意味・条件を管理する。

更新日: 2026-10-07。現行の研究状態は[現在地](research_status.md)。下のトップレベル容量は9/15時点の棚卸し、データ件数の整合性は個別のデータ監査結果として区別する。

## トップレベル概要

2026-09-15 09:46 JSTの再帰走査。`.git`内部・シンボリックリンク先を除く。bytesの詳細は[棚卸しCSV](audits/2026-09-15_workspace_review/storage_by_top_directory.csv)。Git除外ファイルも含む。

| 場所 | 役割 | ファイル数 | おおよその容量（GiB） |
| --- | --- | ---: | ---: |
| `Pool_boiling/` | 実験データ、生成特徴量、解析結果 | 902,929 | 242.536 |
| `研究進捗報告/` | 週次報告、発表資料、論文、学会資料 | 258 | 0.938 |
| `archive/` | 過去コード・Notebook等 | 71 | 0.363 |
| `water_flow/` | 水流音ノイズ音源 | 4 | 0.039 |
| `experiments/` | 数値・解釈・検証の軽量記録（今回の追加前） | 148 | 0.016 |
| `logs/` | ログ類 | 62 | 0.005 |

全体は903,734ファイル、261,888,717,600 bytes。実行中の出力や今回の追加ファイルで増え得る固定時点の棚卸し。`High_speed_compare/`は[6/22の削除記録](storage_cleanup_2026-06-22.md)があり、今回も存在しない。

拡張子別件数は[storage_by_extension.csv](audits/2026-09-15_workspace_review/storage_by_extension.csv)、実験フォルダ別は[storage_by_area.csv](audits/2026-09-15_workspace_review/storage_by_area.csv)。現行データ102,900配列は全保存配列の一部。過去結果や診断データを含む総数と混同しない。

## データセット記録テンプレート

新しいデータセットや生成特徴量を作ったら、ここに追記する。

```text
## YYYY-MM-DD データセット名

- 場所:
- 元データ:
- 実験条件:
- 前処理:
- chunk:
- maxfreq:
- noise:
- SNR:
- 生成スクリプト:
- 生成物:
- 使用した実験:
- 注意点:
```

## 現行データと過去データ

### 2026.10.07_0.3_1（原データ整理済み・処理未実行）

- 場所: `Pool_boiling/Subcooling_20_degrees/0.3/2026.10.07_0.3_1`
- 原録音: `録音データ/`に1.0〜2.0 Vの11 WAV。192 kHz、mono、16 bit、約60.93〜74.07秒。原名・内容を保持。
- 計測原本: `raw/logger/GL860_M411L1602_2026-10-07_11-12-55.CSV`。GL860、1秒、8171行。CH2シャントはmV、CH3電極電圧はV。
- 実験ノート: `raw/experiment_notes/2026-10-07_06-57-53-export.elabftw.csv`。細線0.3 mm、固定端間38.27 mm、沸騰開始の観察1.2 V。
- 未確定: 今回のシャント抵抗値、自動抽出段階と印加電圧の対応、音響切り出しの確認、ONB熱流束値。抵抗は従来値と異なり未確定なので、従来v2 Notebook冒頭で`R_shunt = None`としている。
- 状態: 保存先を整理し、従来3入口を今回のパスへ設定済み。本人希望により今回専用コード・設定表は撤去。熱流束計算、ラベル付きWAVコピー、スペクトログラム・NPY生成、モデル学習は未実行。
- 設定: `code/0.run_auto_heatflux_analysis_v2.ipynb`、`code/1.run_rename_files.ipynb`、`code/2.run_npy_waterflow_2つhighpass.py`の従来の設定欄。
- 手順: [今回の配置と実行順](2026-10-07_new_experiment_processing.md)。既定生成条件はcleanのみ、60秒/WAV、1秒chunk、22.05 kHz。660配列は予定件数で、出力済み件数ではない。
- 原ファイル記録: [配置記録・SHA-256](../experiments/2026-10-07_pool_boiling_data_setup/README.md)。今回の実験を既存学習設定へ自動追加していない。

### 2025.06.11＋2025.06.18統合実験（2026-09-29作成）

- 場所: `Pool_boiling/Subcooling_20_degrees/0.3/2025.06.11_0.3_2_6.18_0.3_3`
- 元データ: 下記`waterflow_20260817_1s`の6/11・6/18。元フォルダは変更せず独立コピー。
- NPY合計: 226,800ファイル、約42.42 GiB。
- 0.5秒: 151,200ファイル、約28.28 GiB。5周波数×7 noiseの35条件、各条件4,320 chunk、各WAV 120 chunk。
- 1秒: 75,600ファイル、約14.14 GiB。5周波数×7 noiseの35条件、各条件2,160 chunk、各WAV 60 chunk。
- 音響: `録音データ/<source_experiment>/`に元名36 WAV、`録音データ_熱流束/<source_experiment>/`に熱流束名付き36 WAV。出典日別に保持。
- メタデータ: `source_metadata/<source_experiment>/`に計6ファイル。
- provenance: 各`chunk_manifest.csv`へ出典実験、元NPY名、元WAV ID・名前を追加。統合概要は`combined_dataset_manifest.json`。
- ONB: 6/11の221,505.1102と6/18の271,677.6816 W/m²を算術平均し、246,591.3959 W/m²を今回の統合値とする。
- 生成スクリプト: `code/build_combined_0611_0618_dataset.py`。既存コピーはbyte数を照合して再利用する。
- 初回評価: 1秒データで各WAVの60 chunkを45学習／15テストへ分ける`within_wav_chunk`。同じchunkは重複しないが同一WAVは両側に入る。主configの`chunk_seconds`を0.5へ変えると、フォルダ名も自動で0.5秒版へ切り替わる。
- Git管理用設定: `configs/datasets/2026-09-29_combined_0611_0618.yaml`。
- 実装・検証記録: `experiments/2026-09-29_within_wav_chunk_combined/README.md`。

### waterflow_20260817_1s（現行データ）

- 場所: `Pool_boiling/Subcooling_20_degrees/0.3/<実験>/data/npy/waterflow_20260817_1s`
- 状態: 2026-08-17に生成済み。2026-09-03に全105条件の構造整合性を確認済み。
- 使用実験: `2025.06.18_0.3_3`, `2025.07.09_0.3_1`, `2025.06.11_0.3_2`
- 前処理: 500 Hz高域通過（pass loss 1 dB、400 Hz stop attenuation 40 dB）、線形パワーSTFT、224×224、float32。
- chunk: 1秒。
- maxfreq: 3、5、10、15、22.05 kHz。
- noise: `water_flow_125.wav`。全3実験で1つの固定基準RMSを共有し、選択したノイズchunkを目標パワーへ正規化する。
- reference SNR: 無雑音、0、-4、-8、-12、-16、-20 dB。個々の信号に対する実現SNRではない。
- paired design: 同一chunkでは信号間・reference SNR間で同じノイズ断片を使用する。
- provenance: ファイル名と `chunk_manifest.csv` に元WAV ID、ノイズoffset、実現SNR、各パワーを保存する。
- 生成スクリプト: `code/2.run_npy_waterflow_2つhighpass.py`。
- 学習スクリプト: `code/run_ensemble_regression_onb.py`。外側は各WAV内chunk holdoutまたは別日分離、内部performance/tuningは通常chunk KFold（shuffle=True、10/5本人指定）。元WAV IDは出典・対応管理に使用する。
- 生成件数: 06.11が37,800、06.18が37,800、07.09が27,300、合計102,900 `.npy`。
- 整合性: 3実験 × 5周波数 × 7ノイズ条件の全105条件で、`.npy`数と`chunk_manifest.csv`の行数が一致。06.11/06.18は各条件1,080、07.09は各条件780サンプル。
- Git管理用設定: `configs/datasets/waterflow_20260817_1s.yaml`。
- Git管理用監査結果: `experiments/2026-08-17_waterflow_dataset_snapshot/`。条件別実現SNR、paired-noise検査、manifest SHA-256を保存。
- データ本体: 容量が大きく再生成可能なためGit対象外。

### waterflow_20250523_1s（主結果から除外）

- 状態: アーカイブ扱い。今後の学習・主結果には使用しない。
- 除外理由: 20250523アーカイブ版の生成処理に `y_power + scaled_noise` の誤加算があったため。
- 方針: 過去経緯の説明以外では比較表へ混在させない。

## 2026-08-17 旧生成学習データの削除（過去記録）

以下は当時の削除記録であり、現在の`data/npy`一式を削除する指示ではない。同日に生成した現行データは上記を参照する。

- `Pool_boiling/**/data/npy`
- `Pool_boiling/**/data/spectrogram`
- `Pool_boiling/**/data/spectrogram_png`
- `Pool_boiling/**/data/spectrum`
- 対象: 4実験の11ディレクトリ、約2,074,864ファイル、約551.85 GiB。
- 削除理由: 旧ノイズ生成条件、20250523加算バグ版、元WAV相対RMSによる振幅ショートカットを主結果から完全に除外するため。
- 復元性: 完全削除。生WAV、実験CSV、解析結果、研究資料は削除していない。

### water_flow

- 場所: `water_flow/`
- 役割: 水流音ノイズ付与に使う音源。
- よく参照されるファイル: `water_flow_125.wav`

## 注意点

- ファイル名先頭の熱流束値を教師ラベルとして使う処理があるため、リネームは慎重に行う。
- 大量の `.npy` や `.png` はGitに入れない。
- 新しい生成条件を試したら、出力フォルダ名だけでなく、この索引か `configs/datasets/` に条件を残す。
