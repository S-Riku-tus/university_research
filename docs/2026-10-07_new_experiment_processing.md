# 2026-10-07 新実験を従来コードで処理する手順

対象：`Pool_boiling/Subcooling_20_degrees/0.3/2026.10.07_0.3_1`。

本人の希望に合わせ、**従来の3つの入口を使う**。今回専用の実行コードは撤去した。10/7の追加修正では、熱流束・温度等を別の検証フォルダで計算して比較済み。原録音のコピー/改名・音響生成・モデル学習は実行していない。[修正と比較記録](../experiments/2026-10-07_heatflux_stage_alignment/README.md)。

## 配置

```text
2026.10.07_0.3_1/
├─ 録音データ/                       原WAV 11本
├─ raw/logger/                       GL860計測CSV原本
├─ raw/experiment_notes/             eLabFTWノートCSV原本
├─ 実験結果2026.10.07_0.3_1/        従来と同じ熱流束・温度等の出力先
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

当初、今回の抵抗は未確定との本人回答を受け、v2冒頭へ`R_shunt = None`として移した。その後、本人のNotebookで`1.667e-3`へ設定されており、今回の修正ではその設定値を保持した。設定済みであることと、今回の装置の抵抗値として確認済みであることは別なので、ラベル確定時に照合する。単位はΩ。

## 1. 従来の熱流束Notebook

入口：[0.run_auto_heatflux_analysis_v2.ipynb](../code/0.run_auto_heatflux_analysis_v2.ipynb)。開いている旧内容を保存して上書きせず、変更されたファイルを開き直し、カーネルを再起動して冒頭から順に実行する。熱流束式・電圧標準偏差の閾値は従来どおり。安定判定を点数から秒単位へ変更し、電圧段階ごとに同じ区間から両電圧を平均する。

本人の実行ではシャント23段階・電極21段階となり停止した。今回は1秒ロガーのため、従来の8点窓は7秒になり、旧約5秒ロガーの約35秒より短い。共通窓だけでも35区間になるため、**段階境界と安定区間を分けて処理する**。configの印加順と段階数が一致しなければ停止し、各段階で両条件を満たす最長の連続領域を代表区間にする。温度条件外の行や時間の欠落をまたいで結合しない。

| 設定 | 現在の値・確認内容 |
|---|---|
| `input_folder` | 今回の`録音データ/`を指定済み |
| `LOGGER_CSV_RELATIVE` | 今回の`raw/logger/GL860_M411L1602_2026-10-07_11-12-55.CSV`を指定済み |
| `d` / `L` | 0.0003 m / 0.03827 m。ノートの0.3 mm・38.27 mm |
| `P` | 101.21 kPa。ノートの1012.1 hPaを換算 |
| `Setting_Temp` | configの80 ℃、許容差±1 ℃ |
| `R_shunt` | 本人設定の`1.667e-3`を保持。今回の抵抗値との照合は別途必要 |
| `STABILITY_WINDOW_SECONDS` | configから35秒。旧8点の時間幅が基準 |
| `shunt_s`, `Pt_s` | configから従来の1e-4 V、1e-3 V |
| `SETTLING_SECONDS` | 切替後5秒を除外 |
| `VOLTAGE_SEQUENCE` | `None`ならconfigの明示リストを使用。段階数×0.1では割り当てない |

処理条件は[2026.10.07_0.3_1_heatflux_processing.json](../configs/datasets/2026.10.07_0.3_1_heatflux_processing.json)にある。保存先はconfigによらず従来階層を使う。形状・圧力・シャント抵抗はNotebook冒頭で設定する。configの`water_target_c`・標準偏差閾値はNotebookの同名設定より優先する。通常は現在の設定で実行できる。

GL860はCH1水温、CH2シャント[mV]、CH3電極電圧[V]を読み、CH2を1000で割ってVへ変換する。旧CSVの読込も残した。旧実験へ戻すときは条件を戻し、`LOGGER_CSV_RELATIVE = None`にすれば従来の`<実験名>.csv`を読む。

ロガーの0.0〜0.9 Vも保持し、録音11本は1.0〜2.0 Vへ対応させる。今回は21段階が成立し、旧6/18の段階抜けを含む18段階でも保存qとの差は最大約0.16%だった。低電圧側の0.1刻みはロガーの上昇と整合する設定であり、録音・ノートで直接裏付けられる1.0〜2.0 Vと区別する。段階を飛ばす別実験では実際の印加順を指定する。

出力は従来と同じ`実験結果2026.10.07_0.3_1/heat_flux_2026.10.07_0.3_1.csv`など。`q`はW/m²。追加の`stage_time_v1_20261007`階層は廃止し、本人が生成した結果も内容を保持して従来階層へ移した。[配置修正と本人出力の確認](../experiments/2026-10-07_heatflux_result_layout/README.md)。

再実行時は、既存結果を`experiments/<再実行日>_heatflux_result_history/<実験名>/<時刻>/`へ自動コピーしてから、通常の保存先へ出力する。退避に失敗した場合は出力前に停止する。既存の本人debugはNotebookの書込対象に含めない。

セル4までに次の確認記録、セル5で熱流束CSVが保存される。

- `detected_stage_boundaries.csv`：全段階の切替境界。
- `joint_stable_regions_<実験名>.csv`：採用開始/終了時刻・スキャン、両平均、標準偏差、採用率、段階全体とのq差。
- `stable_interval_candidates.csv`：採用外を含む安定候補。
- `processing_provenance.json`：原CSV/処理コードのSHA-256、形状・抵抗・条件。

熱流束CSVが0.0〜2.0 Vの21行、録音対象が1.0〜2.0 Vの11行になっていることを確認する。ノートの電流9.82〜18.77 Aは照合用に使える。電源設定電圧を測定電極電圧の代わりにしない。

本人の全11セルの実行記録にエラーはなく、熱流束・温度・共振CSVの21段階と原ロガーからの再計算の一致を確認した。原録音11本の改名・STFTラベル対応と、ONBの1.2 V行の読込も確認済み。自動直線性喪失候補はONB確定値とは区別し、ノートの沸騰開始1.2 Vを維持する。ラベルは段階の代表値であり、WAV/ロガーの時計同期を仮定した1秒ごとの瞬時熱流束ではない。温度には1.1→1.2 Vの低下があり、計算一致と物理的原因の確認は区別する。

## 2. 従来の録音コピー・改名Notebook

入口：[1.run_rename_files.ipynb](../code/1.run_rename_files.ipynb)。`EXPERIMENT_ROOT`を今回へ指定済み。従来階層のCSVを読み、通常はパスの手編集は不要。今回の生成済みCSVは配置修正済みなので、熱流束を再計算せずこの段階から進める。

原WAVを`録音データ_熱流束/`へコピーし、従来の`index=<CSV行番号>.<熱流束の整数>.wav`へ改名する。NPYの正確なラベルはCSVの`q`から取得する。

既存コピーを削除して作り直す処理は外し、データがある場合は停止する。現在の空フォルダにはコピーできる。対応できなかったコピーも削除せず表示するため、**11本すべてがindex付きへ改名できたことを確認してから**次へ進む。1.4 V・1.5 Vの小文字vはWindows上で扱え、原名を保持する。

## 3. 従来のSTFTスクリプト

入口：[2.run_npy_waterflow_2つhighpass.py](../code/2.run_npy_waterflow_2つhighpass.py)。現在の設定：

```python
SAVE_DATE = 20261007
EXPERIMENT_NAMES = ["2026.10.07_0.3_1"]
RECORDING_DIR_NAME = "録音データ_熱流束"
MAX_FREQ_HZ = [3000, 22050]
CHUNK_SECONDS_LIST = [1]
REFERENCE_SNR_DB = [None]
MAX_CHUNKS_PER_SOURCE = None
```

少量確認では同じファイルの`SAVE_DATE = "20261007_preview"`、`MAX_CHUNKS_PER_SOURCE = 1`へ変更し、研究ルートのPowerShellから実行する。

```powershell
python "code/2.run_npy_waterflow_2つhighpass.py"
```

各WAVの最初の1秒、各周波数11 NPYと11 PNG、2周波数合計22個ずつを生成する設定。確認後、`SAVE_DATE = 20261007`、`MAX_CHUNKS_PER_SOURCE = None`へ戻し、同じコマンドで全生成する。プレビューと全生成はそれぞれ`waterflow_20261007_preview_1s`、`waterflow_20261007_1s`へ保存する。

既定は先頭60秒、1秒chunk、500 Hz高域通過、3 kHzと22.05 kHz、無雑音。各周波数660 NPYと660 PNG、2周波数合計1320個ずつの予定。現行5モデルONBは3 kHzを使用するため、3 kHzを前処理へ追加した。既存処理の44.1 kHzへのリサンプリング、224×224、float32、時間×周波数の線形powerを使う。原録音は192 kHzのまま保持する。切り出す範囲を変える場合は本人の音響処理で調整する。

```text
data/npy/waterflow_20261007_1s/maxfreq=3kHz/heatflux_no_noise/
data/npy/waterflow_20261007_1s/maxfreq=22kHz/heatflux_no_noise/
data/spectrogram_png/waterflow_20261007_1s/<周波数>/heatflux_no_noise/
```

manifestも保存する。`maxfreq=22kHz`は丸めた表記で、実値は22,050 Hz。同じタグで再実行すると既存生成物へ書き込む従来動作なので、条件を変える場合は`SAVE_DATE`を別タグにする。

無雑音設定でも従来runnerは水流音を読みRMSを計算するが、入力へノイズは加えない。過去の雑音条件との比較は別工程で、新しい評価日の録音から決めたRMSを過去と同一基準として扱わない。

変更前の3入口は[previous_code](../experiments/2026-10-07_pool_boiling_data_setup/previous_code/)に保存した。設定は従来のコード冒頭で行う。

## 4. 今回のデータでONBを検証する

入口：[run_ensemble_regression_onb.py](../code/run_ensemble_regression_onb.py)の`VALIDATION_CONFIG`。今回の本生成先`waterflow_20261007_{chunk_tag}`を登録済み。プレビューだけが存在しても本データの代わりには使わず、指定先の不足として止める。

現在の既定は、**今回の日の各WAV内75%学習・25%評価、clean学習、1秒、3 kHz、5モデル、200 epochs、内部3-fold、seed42**。評価の`noise_dir_names`は現在7条件が有効。既存のモデル・統合方式は保持した。同じWAVを学習・評価で共有する検証であり、別日への転送評価とは分けて扱う。

ノートの「沸騰開始1.2 V」を登録し、起動時に熱流束CSVの`volt=1.2V`の`q`を読む。CSV未生成なら閾値は未確定のまま学習前に停止する。1.2 V行の欠落・重複・非有限値・非正値も停止する。前の測定点、丸めたWAV名、自動直線性喪失候補から閾値を決めない。出典と読み込んだCSVのSHA-256を実行設定へ残す。ONBとして1.2 Vを使うので、熱流束CSVの電圧段階対応を必ず確認する。

準備できたら研究ルートで実行する。

```powershell
python code/run_ensemble_regression_onb.py
```

別日評価を行うときは`learning_policy.evaluation_mode`を`"cross_day"`へ変える。設定済みの`evaluation_settings.cross_day`は、**6/11＋6/18学習 → 今回10/7評価**。対象日のリストは自動で解決される。逆方向などへ変えたい場合は同じ欄の`train_experiments`と`test_experiments`だけを編集する。今回の日を別日テストとして使う場合、その結果を見て学習パラメータや重みを調整する前に条件を固定する。

**現在はNPY生成側が無雑音だけ、ONB側が7条件なので、実行前にそろえる。** 無雑音だけで始める場合は、ONBの`VALIDATION_CONFIG["data"]["noise_dir_names"] = ["heatflux_no_noise"]`にする。7条件で評価する場合は、生成側の`REFERENCE_SNR_DB = [None, 0, -4, -8, -12, -16, -20]`にし、全条件を生成してからONBを実行する。今回の修正では本人のONB設定を変更していない。新しい日から作ったRMS基準は過去の固定基準と同一ではない。

STFTとONBも従来階層の同じ熱流束CSVを参照する。今回のconfigがある実験でCSV未生成の場合、WAV名の整数値へ切り替えず停止する。ラベルを変更してNPYを作り直す場合は、`SAVE_DATE`を別タグにし、ONBの`data_source_dir_by_experiment`をその本生成先に合わせる。既存のラベル付きコピーは自動削除せず、本人が必要に応じて別名保存してから改名Notebookを実行する。

通常の回帰・ONB指標、保存予測、`metrics_by_source_day.csv`、モデル保存と再読込確認は有効のまま。旧36 WAV・全7条件用プロトコルの`main_comparison_report`は今回の日に適用できないため無効にした。旧固定主表の確認を再実行する場合だけ、対応する旧条件へ戻して有効にする。

条件記録：[ONB新実験の準備条件](../configs/experiments/2026-10-07_onb_new_day_ready.json)。これは記録用で、通常実行はコード内の`VALIDATION_CONFIG`を使う。実装確認：[ONB接続の記録](../experiments/2026-10-07_onb_new_day_configuration/README.md)、[熱流束修正の比較記録](../experiments/2026-10-07_heatflux_stage_alignment/README.md)。AI側で実行した熱計算は隔離した検証出力のみ。音響生成・モデル学習は本人が行う。
