# 2026-10-07 新実験を従来コードで処理する手順

対象：`Pool_boiling/Subcooling_20_degrees/0.3/2026.10.07_0.3_1`。

本人の希望に合わせ、**従来の3つの入口を使う**。先に作った今回専用のNotebook・Pythonファイル・設定表は撤去した。フォルダ整理は維持し、熱流束計算・録音コピー/改名・音響生成・モデル学習は実行していない。

## 配置

```text
2026.10.07_0.3_1/
├─ 録音データ/                       原WAV 11本
├─ raw/logger/                       GL860計測CSV原本
├─ raw/experiment_notes/             eLabFTWノートCSV原本
├─ 実験結果2026.10.07_0.3_1/        熱流束などの出力先
├─ 録音データ_熱流束/                ラベル付きコピーの出力先
├─ data/npy/                         NPYの出力先
├─ data/spectrogram_png/             PNGの出力先
└─ regression_result/                将来のモデル出力先
```

原ファイルの名前・内容を保持した。全13ファイルの移動前後のサイズ・SHA-256は[配置記録](../experiments/2026-10-07_pool_boiling_data_setup/original_files_manifest.json)で照合済み。

## シャント抵抗は以前から計算に使っていた

従来の`0.run_auto_heatflux_analysis.ipynb`と`0.run_auto_heatflux_analysis_v2.ipynb`には、`Heat_flux()`内に以下があった。

```python
R_shunt = 1.667 * pow(10, -3)
I = shunt / R_shunt
q = I * Pt / (np.pi * d * L)
```

ロガーが測るのはシャント両端の電圧なので、抵抗値で割って電流を求める。その電流と細線の電極電圧から電力を求め、細線の表面積で割って熱流束にする。`R_Pt = Pt / I`にも使い、白金温度・沸騰曲線の計算につながる。

以前は関数内に抵抗値が固定されていたため、毎回入力する必要はなかった。今回新たに必要になった物理量ではない。STFT計算自体には使わないが、スペクトログラム・NPYに付ける熱流束ラベルに影響する。

今回の抵抗は従来値と異なり未確定との本人回答を受け、v2 Notebook冒頭へ`R_shunt = None`として移した。確認できた抵抗を**Ω単位**で入れる。例えば1 mΩなら`0.001`だが、これは単位変換の例で、今回の値ではない。

## 1. 従来の熱流束Notebook

入口：[0.run_auto_heatflux_analysis_v2.ipynb](../code/0.run_auto_heatflux_analysis_v2.ipynb)。カーネルを再起動して冒頭から順に実行する。従来の安定区間の自動抽出・熱流束の式を保持し、CSV読込を旧形式とGL860の両方へ対応させた。

| 設定 | 現在の値・確認内容 |
|---|---|
| `input_folder` | 今回の`録音データ/`を指定済み |
| `LOGGER_CSV_RELATIVE` | 今回の`raw/logger/GL860_M411L1602_2026-10-07_11-12-55.CSV`を指定済み |
| `d` / `L` | 0.0003 m / 0.03827 m。ノートの0.3 mm・38.27 mm |
| `P` | 101.21 kPa。ノートの1012.1 hPaを換算 |
| `Setting_Temp` | 従来の80 ℃。実際の液温に合うか確認 |
| `R_shunt` | `None`。今回の値を入れるまで熱流束計算は停止 |
| `overlap_n`, `shunt_s`, `Pt_s` | 従来値8、1e-4 V、1e-3 Vを維持。抽出結果で確認 |
| `VOLTAGE_SEQUENCE` | `None`なら従来どおり0.0 Vから0.1 V刻み。必要時だけ全段階の電圧リストを指定 |

GL860はCH1水温、CH2シャント[mV]、CH3電極電圧[V]を読み、CH2を1000で割ってVへ変換する。旧CSVの読込も残した。旧実験へ戻すときは条件を戻し、`LOGGER_CSV_RELATIVE = None`にすれば従来の`<実験名>.csv`を読む。

WAVが1.0〜2.0 Vであることと、ロガーの抽出段階が1.0 Vから始まることは同じではない。**熱流束CSVの行と実際の印加電圧の対応を、コピー/改名前に確認する**。0.0〜0.9 Vも抽出されたなら、その行を保持できる。段階抜け等がある場合は`VOLTAGE_SEQUENCE`へ抽出された全段階に対応する電圧を指定する。録音の開始電圧に合わせて無条件に1.0 V開始へ変更しない。

出力は`実験結果2026.10.07_0.3_1/heat_flux_2026.10.07_0.3_1.csv`など。`q`はW/m²。ノートの電流9.82〜18.77 Aは照合用に使える。電源設定電圧を測定電極電圧の代わりにしない。

自動抽出・沸騰曲線の全セルが今回の実データで完走するかは未確認。自動直線性喪失候補はONB確定値とは区別する。ノートの沸騰開始1.2 Vは観察記録として保持している。

## 2. 従来の録音コピー・改名Notebook

入口：[1.run_rename_files.ipynb](../code/1.run_rename_files.ipynb)。熱流束CSVを今回へ変更済み。

原WAVを`録音データ_熱流束/`へコピーし、従来の`index=<CSV行番号>.<熱流束の整数>.wav`へ改名する。NPYの正確なラベルはCSVの`q`から取得する。

既存コピーを削除して作り直す処理は外し、データがある場合は停止する。現在の空フォルダにはコピーできる。対応できなかったコピーも削除せず表示するため、**11本すべてがindex付きへ改名できたことを確認してから**次へ進む。1.4 V・1.5 Vの小文字vはWindows上で扱え、原名を保持する。

## 3. 従来のSTFTスクリプト

入口：[2.run_npy_waterflow_2つhighpass.py](../code/2.run_npy_waterflow_2つhighpass.py)。現在の設定：

```python
SAVE_DATE = 20261007
EXPERIMENT_NAMES = ["2026.10.07_0.3_1"]
RECORDING_DIR_NAME = "録音データ_熱流束"
MAX_FREQ_HZ = [22050]
CHUNK_SECONDS_LIST = [1]
REFERENCE_SNR_DB = [None]
MAX_CHUNKS_PER_SOURCE = None
```

少量確認では同じファイルの`SAVE_DATE = "20261007_preview"`、`MAX_CHUNKS_PER_SOURCE = 1`へ変更し、研究ルートのPowerShellから実行する。

```powershell
python "code/2.run_npy_waterflow_2つhighpass.py"
```

各WAVの最初の1秒、計11 NPYと11 PNGを生成する設定。確認後、`SAVE_DATE = 20261007`、`MAX_CHUNKS_PER_SOURCE = None`へ戻し、同じコマンドで全生成する。プレビューと全生成はそれぞれ`waterflow_20261007_preview_1s`、`waterflow_20261007_1s`へ保存する。

既定は先頭60秒、1秒chunk、500 Hz高域通過、22.05 kHzまで、無雑音。予定件数は660 NPYと660 PNG。既存処理の44.1 kHzへのリサンプリング、224×224、float32、時間×周波数の線形powerを使う。原録音は192 kHzのまま保持する。切り出す範囲を変える場合は本人の音響処理で調整する。

```text
data/npy/waterflow_20261007_1s/maxfreq=22kHz/heatflux_no_noise/
data/spectrogram_png/waterflow_20261007_1s/maxfreq=22kHz/heatflux_no_noise/
```

manifestも保存する。`maxfreq=22kHz`は丸めた表記で、実値は22,050 Hz。同じタグで再実行すると既存生成物へ書き込む従来動作なので、条件を変える場合は`SAVE_DATE`を別タグにする。

無雑音設定でも従来runnerは水流音を読みRMSを計算するが、入力へノイズは加えない。過去の雑音条件との比較は別工程で、新しい評価日の録音から決めたRMSを過去と同一基準として扱わない。

変更前の3入口は[previous_code](../experiments/2026-10-07_pool_boiling_data_setup/previous_code/)に保存した。設定は従来のコード冒頭で行う。ONB閾値の登録とモデル実験への対象追加は未実施。
