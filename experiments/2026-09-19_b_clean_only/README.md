# B：固定cleanモデルによるノイズ谷の識別

## B1 実行可否監査（実行前、2026-09-19）

| 項目 | 完了run 151715での保存状態 | 今回の処理 |
|---|---|---|
| 1秒評価予測・正解・キー | `fold_pred/pred_f1_<SNR>.csv`に保存。A監査で全1080キーとラベルの一致を確認 | B4のmatched比較に使う |
| 統合重み | 各SNRの`ensemble_weights_<SNR>.csv`に保存。cleanはRF .143610、Conformer .681358、AlexNet .175031 | 新しいclean学習で重みを1回だけfitし、4 SNRで共有する |
| RF・CNN学習済み本体 | 完了runの22 kHz・無雑音ディレクトリにモデルファイルなし。`learning_runner.py`は終了時にモデルを解放し永続化しない | **再学習が必要** |
| PCA・目的変数scaler | 同ディレクトリにfit済みオブジェクトなし。`learning_runner.py`で学習集合にfitしてメモリ上で使用 | **再fitが必要**。6/18ではfitしない |
| ノイズ配列と標本対応 | `waterflow_20260817_1s`の6/18各SNRに入力配列と`chunk_manifest.csv`あり。`learning_policy.aligned_indices()`がキー・ラベル・開始時刻・長さを確認 | 4条件の同じ18 WAV・1080秒を評価 |
| `clean_only`実装 | `learning_policy.build_learning_families()`が複数評価ノイズを1学習familyへまとめ、`learning_runner.run_learning_experiments()`の同一モデル・PCA・scaler・重みを4条件に適用 | 現行の主コードを変更せず、[条件書](../../configs/experiments/2026-09-19_b_clean_only_22khz.json)を[実行ラッパー](run_from_condition.py)で適用 |

再学習のため、完了matched runのcleanモデルと**同一の学習済みパラメータ**は保証できない。比較は新run内の固定モデルを主とし、matched runは既存の谷の対照として扱う。現行主コードの編集中の日付設定は条件書で上書きし、結果を別保存先`20260919/b_clean_only`へ分ける。

## B2 実行前に固定した条件

[条件書](../../configs/experiments/2026-09-19_b_clean_only_22khz.json)：学習6/11、評価6/18、22 kHz、200 epochs、seed 42、`wav_kfold`内部検証、`inner_holdout`重み、学習側ピーク高さ閾値`1e-9`、4評価条件は無雑音・−8・−16・−20 dB。全18 WAV×60秒。3単体、前処理、重みを同じclean学習セッション内で固定する。評価指標はR²、RMSE、MAE、近傍RMSE、見逃し、誤報、WAV別二乗誤差。−8/−16/−20は既存matched結果の谷と回復を見て選んだ**探索的条件**である。XAIはこの識別比較に不要なので無効とした。

## B3以降

### B3 固定モデル実行

`run_from_condition.py`で[事前条件書](../../configs/experiments/2026-09-19_b_clean_only_22khz.json)を主コードの設定解決時だけ適用し、主コードの編集中設定は変更しなかった。[実行ログ](run.log)、[原run位置](run_location.txt)。新runは`b_clean_only_xd-t0611-v0618_iw3-nc_s1e-9_e200_163926`、seed 42、200 epochs、学習側の選別1080→986秒、評価側は各1080秒。`completed.json`の`fit_id`とrun hash、3単体の統合重みは4条件で一致する。RF .1774435、Conformer .5424543、AlexNet .2801022。4条件の`experiment_name×source_wav_id×chunk_index`も完全一致し、欠損・重複・ラベル不一致は0。保存統合予測は同じ重みの加重和と一致する。**学習済みモデル・PCA・scaler・重みをSNRごとに再fitしていない**。評価はすべて6/18の18 WAV・1080秒。

### B4 固定モデルの結果

[再計算スクリプト](analyze_fixed_predictions.py)、[全モデル指標](fixed_metrics.csv)、[領域別指標](fixed_region_metrics.csv)、[WAV別二乗誤差差](fixed_paired_wav_sse.csv)。等重みは保存3単体からの事後計算。ONB閾値はこのrunの271,677.6816 W/m²、近傍は±10%。原記録との閾値不一致は[A監査](../2026-09-19_noise_recovery_review/A1_A7_saved_prediction_diagnosis.md)に留保済み。

| 指標・モデル | 無雑音 | −8 dB | −16 dB | −20 dB |
|---|---:|---:|---:|---:|
| RF R² | .850 | .858 | .870 | .856 |
| CNN＋Transformer R² | .917 | .527 | −.276 | −.931 |
| AlexNet R² | .893 | .762 | .721 | .613 |
| 現行重み統合 R² | **.918** | .737 | .442 | .152 |
| 等重み R²（事後） | .915 | .826 | .689 | .526 |
| 現行重み統合 RMSE (kW/m²) | 77.8 | 139.6 | 203.4 | 250.7 |
| 現行重み統合 ONB近傍RMSE (kW/m²; 1 WAV×60秒) | 141.0 | 22.6 | 115.3 | 182.6 |
| 現行重み統合 見逃し / 誤報（秒） | 114 / 0 | 22 / 369 | 0 / 480 | 0 / 480 |

固定モデルの全域R²には**−8 dBの谷と−16/−20 dBでの回復がない**。−8→−16 dBの統合二乗誤差は23.653×10¹² (W/m²)²増え、18 WAV中6本改善・12本悪化。−8→−20 dBは46.848×10¹²増、3本改善・15本悪化。全域悪化の中心はONB前8 WAVの**正方向の残差**で、統合のONB前平均残差は無雑音+5.4、−8 +187.7、−16 +287.7、−20 +353.4 kW/m²。強ノイズでONB前の全480秒が閾値以上となる。近傍RMSEだけ見ると−8で改善するが、全域と誤報は悪化するため単一指標で有効とは判定しない。

**識別できた範囲**：固定clean学習状態を強い水流ノイズへそのまま転送すると、この条件では低熱流束を大きく過大予測する。RFのR²は比較的横ばいだが、CNN＋TransformerのR²は大きく低下し、cleanで付けた重み.542が統合の転送性能を下げる。等重みも−16/−20ではRF単体に届かない。既存matchedの回復を「固定モデルにノイズを足すほど良くなった」とは説明できない。matchedでは各SNRでの再学習・前処理・重み変動とrun/seed差がまだ混ざる。新clean runと旧matched runで学習済みパラメータの同一性はないため、**単一原因を断定しない**。

### B6・B7の分岐

固定モデルに谷が残らなかったため、B7の「谷が残れば」という開始条件は満たさない。Hz帯別SNRと保持／除去摂動による谷の機構探索は、現時点では行わない。B6は残った学習側の不安定性に絞り、[実行前条件書](../../configs/experiments/2026-09-19_b_matched_seed_pair.json)のseed 42/43/44でmatched −8/−16 dBを対応比較する。6/18評価ラベルはepoch・重み選択に使わない。

## B5 既存matched runの学習側監査

代表条件は22 kHz・−8/−16 dBの2条件。完了run 151715の`run_manifest.json`では両方ともseed 42、200 epochs、学習6/11、評価6/18、同じ`waterflow_20260817_1s`とピーク選別閾値`1e-9`。各SNRで別familyを作って学習するため、学習済みパラメータ、PCA、scaler、内部重みは条件間で共有しない。`learning_runner.py`では元WAV単位の内部holdout後、選別された6/11学習集合に目的変数scalerとPCAをfitする。評価日6/18はfitに使わない。乱数はfamilyごとに`seed+fold`へ戻し、Python/NumPy/TensorFlowを設定する。ノイズ配列はデータ生成時の`fixed_global_rms`、同じchunkの水流音区間をSNR間で共有する設計。`configs/datasets/waterflow_20260817_1s.yaml`に記録がある。

保存図`.../heatflux_reference_SNR=-8/loss/loss_conformer_f1_-8.png`、`...=-16/loss/loss_conformer_f1_-16.png`と同条件のAlexNet図を確認した。両モデル・両条件で**training loss**は初期から大きく下がり、終盤は低く平坦に見える。Conformer −8 dBの初期約.064、−16 dBの初期約.124は図のスケールであり、学習に失敗してlossが高止まりした証拠はない。図の実装は`history.history['loss']`だけを描く。`model_training.py`の`model.fit()`に`validation_data`や早期停止はないため、**validation lossとbest epochは既存runには存在せず、過学習や最良学習量を判断できない**。前処理オブジェクトと学習済みモデルも保存されていない。実行順・GPU上の非決定性による学習結果差は、この2図だけでは評価できない。

## B6 matchedの3 seed対応比較

[実行前条件書](../../configs/experiments/2026-09-19_b_matched_seed_pair.json)に従い、seed 42/43/44で−8/−16 dBを各々学習した。3モデル、元WAV分離の内部重み推定、200 epochs、学習選別986秒、評価18 WAV・全1080秒は共通。各seedの2条件は**別fit ID**、同じseed内の設定run hashは共通。予測キー・ラベルは6条件すべて一致し、欠損・重複0。各条件の重みは[全モデル指標](matched_seed_metrics.csv)、[WAV別差](matched_seed_paired_wavs.csv)、[run位置とfit ID](matched_seed_run_locations.csv)へ保存した。実行ログは`matched_seed42.log`、`matched_seed43.log`、`matched_seed44.log`。

| seed | 統合R² −8 → −16 dB | R²差 | CNN＋Transformer R²差 | AlexNet R²差 | 統合近傍RMSE −8 → −16 (kW/m²) | 統合見逃し −8 → −16 |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | .8542 → .8847 | **+.0305** | +.0441 | +.0421 | 192.9 → 172.6 | 110 → 116 |
| 43 | .8688 → .8501 | **−.0188** | +.0191 | **−.0796** | 170.0 → 196.6 | 111 → 125 |
| 44 | .8461 → .8608 | **+.0147** | +.0289 | +.0363 | 198.4 → 193.1 | 120 → 120 |

統合R²差は平均+.0088、範囲−.0188〜+.0305。各SNRの統合R²のseed間幅は−8で.0228、−16で.0346。単一runの谷／回復幅を安定したSNR効果とは扱えない。RFは各seedの−8でR² .8590〜.8599、−16で.8575〜.8587と狭い。一方、−16のAlexNetは.7691〜.8641、統合重みのRF分は.251〜.402と変わる。**CNN＋Transformer単体は3 seedとも−16で改善したが、AlexNetと統合は一貫しない**。等重みのR²差も+.0221、−.0155、+.0130と同じ符号の入れ替わりが残るため、重みの変動だけでは説明できない。旧run 151715のseed 42は統合.8261→.8838で、新たなseed 42の.8542→.8847とも数値が異なる。同じseed名の別実行差の要因は未特定。

| seed | −8→−16の統合二乗誤差減少 合計 | ONB前8 WAV | 近傍1 WAV | 近傍外ONB以上9 WAV | 改善 / 悪化WAV |
|---:|---:|---:|---:|---:|---:|
| 42 | +2.442 | +2.604 | +.445 | −.607 | 8 / 10 |
| 43 | −1.505 | +1.102 | −.585 | −2.021 | 7 / 11 |
| 44 | +1.176 | +1.767 | +.124 | −.715 | 10 / 8 |

二乗誤差和の単位は`10¹² (W/m²)²`で正が改善。**ONB前は3 seedとも改善し、近傍外ONB以上は3 seedとも悪化**する。全域R²が改善したseedでもONB見逃しは改善しない。近傍は1 WAV・60秒であり、独立反復60本ではない。3 seed、同じ学習日・評価日・水流音源から統計的な一般化は主張しない。

[10 epochsごとのtraining loss](matched_seed_training_loss.csv)と[4パネル図](matched_seed_training_loss.png)は、6条件のConformer/AlexNet本学習で終盤lossが低いことを示す。各条件200 epochsは完了。seed 43の−16 dB AlexNetはR² .7691まで下がったが、training lossの最終値は.000792で、高止まりしていない。これで過学習と断定はできず、**元WAV分離のvalidation loss・epoch別の未知WAV成績がないこと**が次の学習側診断の不足である。6/18を用いたbest epoch選択は行っていない。

## B7の適用判定と次の改善方針

B7は「固定モデルでも谷が残る場合」の条件付きタスク。B4で谷が消え、強ノイズほど固定モデルは悪化したため、**B7は開始条件不成立で未実施**。周波数帯の保持／除去や実測帯域SNRを、今回のmatched回復の原因説明として後付けしない。

**ここまでの確認事実**：clean学習済みCNNを未知の強ノイズへ直接転送すると低熱流束を過大予測し、ONB前の誤報が激増する。matchedではCNN＋Transformerの−16 dB成績は3 seedとも−8 dBより高いが、AlexNetと統合の全域差はseedで符号が変わる。ONB以上の二乗誤差は3 seedとも悪化する。元WAV分離のtraining loss以外の学習曲線は存在しない。

**整合する仮説**：ノイズ条件に合わせた学習がclean転送の過大予測を抑え、同時にCNN、とくにAlexNetの別日一般化がseed・実行条件に敏感である可能性が高い。重みも変わるが、等重みで符号の入れ替わりが残る。どの学習要因、入力帯域、GPU非決定性が寄与したかの割合は未識別。

**次の識別比較・採否方針**：まず6/11内で元WAVを分離したvalidationを新設し、CNNのepoch別未知WAV誤差とONB誤りを記録する。epoch・重み・ノイズ学習方針は学習側だけで決め、6/18は固定条件の評価に保つ。雑音が未知の用途にはRFを現時点の頑健性対照とし、cleanのみ学習したCNN統合を基準としない。matchedのR²改善だけで採用せず、近傍・ONB以上・誤報／見逃しと複数seedで判断する。選別の有無は別の問いとしてCの対応比較へ進める。独立日・独立水流音での確認までは一般化範囲を広げない。
