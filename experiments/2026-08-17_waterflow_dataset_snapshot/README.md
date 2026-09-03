# waterflow_20260817_1s データ監査

## 目的

`waterflow_20260817_1s`本体と105個の`chunk_manifest.csv`は`Pool_boiling/`以下にあり、容量上Git対象外である。このスナップショットは、ノイズと精度の関係を検証するうえで必要なデータ生成の証拠だけをGitHubから確認できるようにしたものである。

## 監査結果

| 項目 | 結果 |
| --- | ---: |
| 実験数 | 3 |
| 元WAV数 | 49 |
| 周波数上限 | 5種類 |
| ノイズ条件 | 7種類 |
| 条件数 | 105 |
| manifest総行数 | 102,900 |
| `.npy`総数 | 102,900 |
| `.npy`数とmanifest行数の一致 | 全105条件 |
| SNR間のノイズseed・offset・元ノイズpower対応 | 全15実験×周波数組 |
| 固定基準RMS | 0.0004822199855309985 |
| ノイズpower式の最大相対誤差 | 約4.1×10^-13 |

3つの`global_reference_manifest.json`は生成時刻を除く意味内容が一致した。ノイズ強度の決定に熱流束ラベルや個々の音源RMSは使われていない。同一の元WAV・chunkについて、6つのノイズレベル間で同じノイズseed、offset、元ノイズpowerが使われている。

## reference SNRの読み方

フォルダ名の`reference_SNR`は、49元WAVから求めた一つの固定基準RMSに対するノイズレベルであり、各サンプルの実現SNRではない。信号強度は熱流束とともに大きく変わるため、同じreference SNRでも実現SNRは広く分布する。

例として、reference SNR -20 dBにおける実現SNR平均は、06.11で-9.635 dB、06.18で-13.171 dB、07.09で-17.228 dBだった。これは異常ではなく、全サンプルへ同じ絶対ノイズpowerを与え、信号強度差を保持した結果である。条件別の平均・中央値・最小・最大は`condition_audit.csv`に保存している。

## Gitで確認できるファイル

| ファイル | 内容 |
| --- | --- |
| `condition_audit.csv` | 105条件の件数、実現SNR分布、信号・ノイズpower、manifest SHA-256 |
| `paired_noise_audit.csv` | 実験×周波数ごとのpaired-noise整合性 |
| `source_wav_filtered_rms.csv` | 49元WAVの500 Hz high-pass後RMS |
| `dataset_snapshot_manifest.json` | データ全体の件数、合否、元global manifestのSHA-256 |

個々の`.npy`値そのものの完全ハッシュではない。ここで確認しているのは、ファイル数、manifestとの対応、ノイズ設計、provenanceの整合性である。

## 再生成

```powershell
python code/export_waterflow_dataset_snapshot.py `
  --dataset-root "Pool_boiling/Subcooling_20_degrees/0.3/2025.06.11_0.3_2/data/npy/waterflow_20260817_1s" `
  --dataset-root "Pool_boiling/Subcooling_20_degrees/0.3/2025.06.18_0.3_3/data/npy/waterflow_20260817_1s" `
  --dataset-root "Pool_boiling/Subcooling_20_degrees/0.3/2025.07.09_0.3_1/data/npy/waterflow_20260817_1s" `
  --output-dir "experiments/2026-08-17_waterflow_dataset_snapshot"
```
