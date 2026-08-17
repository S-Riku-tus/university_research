# P0ノイズ診断コードの実行手順

作成日: 2026-07-24  
位置づけ: `2026-07-24_current_progress_and_priorities.md` のP0-1～P0-6を、長時間の深層学習より先に検証するための実行メモ

## 1. 今回実装した範囲

現行の水流音データ生成処理とRF評価処理に、次の診断経路を追加した。

| P0 | 実装 | 現在の状態 |
| --- | --- | --- |
| P0-1 条件固定 | 前処理manifest、chunk manifest、診断manifestへ条件・seed・入力パスを保存 | コード実装済み、実データ実行前 |
| P0-2 総パワー1変数 | `total_power_only` の1変数だけを入力するXGBRF回帰 | コード実装済み、実データ実行前 |
| P0-3 水流音のみ | 元の沸騰音を入力へ含めず、元WAVパワーに応じてスケールした水流音だけを保存 | コード実装済み、実データ生成前 |
| P0-4 固定振幅 | 実験内で1個に固定した基準RMSから水流音振幅を決める | コード実装済み、実データ生成前 |
| P0-5 group分割 | `source_wav_id` を保存し、通常KFoldとGroupKFoldを同じRF条件で比較 | コード実装済み、実データ実行前 |
| P0-6 実現SNR | 各chunkの信号・ノイズpowerから実現SNRを計算して保存 | コード実装済み、実データ生成前 |

したがって、現時点で完了したのは「検証可能なコードと保存形式」であり、研究上の判定はまだ完了していない。

## 2. 変更したコード

- `code/2.run_npy_waterflow_2つhighpass.py`
  - `P0_DIAGNOSTIC_MODE = True` を追加した。
  - 1秒、22 kHz、no noiseとSNR -20 dBに限定した。
  - 各元WAVの全時間範囲から、等間隔に5 chunkを選ぶ。
  - 次の3データ系列を、既存の20260722版とは別フォルダへ保存する。
- `code/utils/dataloading/waterflow_preprocessing.py`
  - chunkごとの再現可能な水流音offsetを追加した。
  - source-relativeとfixed-referenceのスケーリングを分離した。
  - `source_wav_id`、`chunk_start_seconds`、要求SNR、実現SNR、各powerを `chunk_manifest.csv` へ保存する。
- `code/run_noise_shortcut_diagnostics.py`
  - P0専用の短時間RF診断ランナー。
  - `chunk_manifest.csv` の `model_input_power`、すなわち1秒波形の平均二乗値1変数とfull spectrogram＋PCAを比較する。
  - 通常KFoldとsource-WAV GroupKFoldを比較する。
- `code/utils/diagnostics/noise_shortcut.py`
  - 上記の読込、分割、RF評価、集約を実装した。

## 3. 生成する3系列

| variant | 入力 | 判定したいこと |
| --- | --- | --- |
| `relative_mixture` | 沸騰音＋元WAV RMS比例の水流音 | 現行方式に近い対照 |
| `relative_noise_only` | 元WAV RMS比例の水流音のみ | 沸騰音なしでも熱流束を予測できるか |
| `fixed_mixture` | 沸騰音＋全WAV共通振幅の水流音 | 元WAV比例スケーリングを除くと精度上昇が消えるか |

固定振幅系列では、各実験内にあるhigh-pass後の元WAV RMS中央値を1個の基準として使う。これは原因切り分け用の固定絶対振幅であり、物理的に校正したSPLではない。また、学習foldだけから算出した値でもないため、最終方式として採用する前に、校正SPLまたは学習fold内基準へ進める必要がある。

## 4. 実行順

リポジトリ直下で次を実行する。

### 4.1 診断データを生成

```powershell
python "code/2.run_npy_waterflow_2つhighpass.py"
```

既存の `waterflow_20260722_1s_y_power` は上書きしない。新しい保存先は、各実験の `data/npy/` 以下にある次の3フォルダである。

- `waterflow_20260724_1s_p0_relative_mixture`
- `waterflow_20260724_1s_p0_relative_noise_only`
- `waterflow_20260724_1s_p0_fixed_mixture`

### 4.2 最初に総パワー1変数だけを実行

```powershell
python code/run_noise_shortcut_diagnostics.py --scalar-only
```

この実行はP0-2とP0-3の最短確認である。水流音のみの `total_power_only` が高いR²なら、元の沸騰音がなくても、スケーリングされた水流音の振幅から熱流束を復元できている。

### 4.3 full spectrogram RFとの比較を実行

```powershell
python code/run_noise_shortcut_diagnostics.py
```

この実行では、各foldの学習標本だけでPCAと熱流束scalerをfitする。総パワーRFとfull spectrogram RFは、同じ外側splitで比較される。

データ生成が一部だけ完了しているときに限り、暫定確認として次を使える。

```powershell
python code/run_noise_shortcut_diagnostics.py --scalar-only --skip-missing
```

`--skip-missing` の結果は全条件比較ではない。欠けた条件は `diagnostic_manifest.json` に保存される。

## 5. 出力

保存先:

`experiments/2026-07-24_noise_shortcut_diagnostic/`

| ファイル | 内容 |
| --- | --- |
| `diagnostic_manifest.json` | 目的、全設定、入力パス、完了・欠損条件、判定基準 |
| `metrics_summary.csv` | 条件・分割・特徴量別のR²、RMSE、MAE、相関の平均と標準偏差 |
| `fold_metrics.csv` | fold単位の指標 |
| `fold_predictions.csv` | 各検証標本の真値、予測値、総パワー |
| `split_assignments.csv` | 各標本が各foldでtrain/validationのどちらか |
| `input_power_summary.csv` | 入力総パワーと熱流束の相関 |
| `realized_snr_summary.csv` | chunk実現SNRの平均、標準偏差、最小、最大 |

各生成データのSNRフォルダには `chunk_manifest.csv` があり、標本単位の詳細を確認できる。

## 6. 判定順

1. `relative_noise_only / total_power_only / kfold` のR²を見る。
2. 同じ条件のGroupKFold R²を見る。
3. `relative_mixture` のno noiseと-20 dBを比較する。
4. `fixed_mixture` のno noiseと-20 dBを比較する。
5. `total_power_only` と `full_spectrogram_pca` の差を見る。
6. `realized_snr_summary.csv` で、要求-20 dBに対するchunkばらつきを見る。

主な読み方は次のとおりである。

| 結果 | 支持される解釈 |
| --- | --- |
| 水流音のみで高R² | 元WAV比例の水流音振幅shortcut |
| 総パワー1変数がfull RFに近い | 時間周波数構造より絶対powerが支配的 |
| KFoldだけ高くGroupKFoldで低下 | 同一元WAVのchunk共有への依存 |
| relativeでは-20 dBで上昇し、fixedでは消失 | 元WAV比例スケーリングが主因 |
| fixedでも-20 dBで上昇 | 正則化効果、帯域形状、別の分割問題を追加調査 |

現状は基本的に1つの元WAVが1つの熱流束水準に対応する。そのためGroupKFoldの低下には、「同じ元WAVのchunk共有を除いた効果」と「検証foldの熱流束水準そのものが学習側にない効果」の両方が含まれる。GroupKFold低下だけをリーク量として断定せず、反復WAVを追加した設計で再確認する。

## 7. 実装検証

2026-07-24時点で、次を確認済みである。

- 変更したPythonファイルの構文確認
- relative/fixedスケールの数値確認
- seed付きnoise offsetの再現性
- chunk manifestの重複更新
- KFold/GroupKFoldと総パワーRFの小規模synthetic test
- 診断jobが3実験×5条件の計15件へ展開されること

実データの生成と15条件の評価はまだ実行していない。出力が得られるまでは、「原因を確認した」ではなく「原因を判定できるコードを実装した」と記載する。
