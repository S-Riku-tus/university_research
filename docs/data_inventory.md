# データ索引

巨大なデータ本体はGitで管理せず、このファイルで場所・意味・条件を管理する。

更新日: 2026-09-15。現行の研究状態は[現在地](research_status.md)。下の容量は今回の棚卸し、データ件数の整合性は別途行われたデータ監査の結果として区別する。

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
- 学習スクリプト: `code/run_ensemble_regression_onb.py`。元WAV ID単位のGroupKFoldを使用する。
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
