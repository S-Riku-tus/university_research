# 別日分割・通常KFold・学習区間選別の実装とスペクトル解析

**同日後続の変更**：本人の付録図と追加説明により、選別を帯域パワーの分位点からピーク高さへ変更した。[現行の方式・全線形画像・閾値画面・1,649秒の根拠](../2026-09-16_peak_height_selection/README.md)。以下は先行実装時の履歴であり、1,648秒など当時の数値は保持する。

2026-09-16。本人の今回の方針変更と追加回答に対応。**学習6/11+7/9、テスト6/18**。既存runや原データは変更していない。旧計画・草稿は[previous_documents](previous_documents/)に保存した。教授の回答・承認を得た記録ではない。

## 1. 完了範囲

- ONB主configから学習日・テスト日を明示選択できる `explicit_days` を追加。
- 学部コードに合わせ、学習内で通常のchunk KFoldを使う `performance_kfold` を追加。逆誤差重みを内部OOFから決める。
- 音響特徴による**学習だけの選別**を追加。内部fit・最終fitそれぞれで参照閾値を推定。
- 3実験、49 WAV、2,940秒の**全1秒スペクトル画像**を生成。全秒で現行生成データの信号パワーを照合。
- 元のスペクトログラムを代表区間で確認し、全秒PSDの一覧と9候補帯域を数値比較。
- 実RF・CNN＋Transformer・AlexNet、3 kHz・無雑音、選別なし/ありの**2 epochs動作確認**を完了。テスト全1,080秒・18 WAVを両方で評価。
- 学習日重複拒否、内部KFold、選別、学習/PCAからのテスト日除外、ノイズ間モデル共用、完了再開をテスト。最終テスト数は[tests.log](tests.log)。
- [全105条件のmanifest・全画像・実データ出力の検算](verification.json)を保存。

**未実施**：300 epochsでの3モデル本比較、ノイズ条件での改善確認、新しい統合方式の比較、気泡イベント正解に基づく閾値最適化。短い動作確認を研究上の性能改善へ読み替えない。

## 2. 学部コードで実際に行っていたこと

[旧コード](../../code/3.run_ensemble_ROC_100%_analysis.py)の303行付近で `KFold(n_splits=DIVISIONS, shuffle=True, random_state=42)`、325行付近で `kf.split(x)` を使用。`x`は読み込んだ1秒NPY配列なので**chunk単位**。433–446行付近の統合は単体誤差の逆数を正規化する方式。

今回は、同じ通常KFoldを**学習2日の内部**に置く。各chunkに一度ずつ得た内部OOFをpoolして単体R²を計算し、`1 / max(1-R², 1e-6)` を正規化して重みを決める。その後、学習2日の対象区間で最終モデルを学習し、固定した重みで6/18を評価する。学部コードの評価fold自身から重みを決める部分は、別日テストへ持ち込まない。

内部chunk KFoldでは同じWAVがtrain/validationに跨ぐ。今回の確認では選別なし31 WAV、選別あり29 WAVが各内部foldで共有される。この値は学習内での重み決定用であり、未知録音の一般化性能と呼ばない。別日テストとの元WAV・日付の共有はゼロ。KFoldの基本動作は[公式API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)も確認した。

PCA、熱流束scaler、音響閾値は各内部fitだけで決める。最終fitでも学習日だけを使用する。[scikit-learnの漏洩防止の説明](https://scikit-learn.org/stable/common_pitfalls.html)に対応する。ただし既存3日は過去の開発・今回のスペクトル閲覧でも参照済みであり、完全に未閲覧の独立確認データとは称さない。

## 3. スペクトルの生成仕様と保存先

指定された旧 `code/trush_box/Spectrum.py` は、2024データの録音全体を1枚にし、振幅を画像ごとに0–1正規化、上限3 kHz、sr=44110で読み込む処理。絶対強度の閾値比較と現行1秒区間の対応には不向きだったため、原本を保持して[新しい生成コード](../../code/analyze_boiling_spectra.py)を追加した。

| 項目 | 今回の仕様 |
|---|---|
| 音源 | 現行manifestが指す `録音データ_熱流束` の元WAV。付加ノイズなし |
| 前処理 | 現行NPY生成コードの同じロード・resample・500 Hz高域通過を呼び出す |
| 区間 | manifestの開始sample・長さどおり。各1秒・60区間/WAV |
| スペクトル | Welch PSD、Hann 2048点、1024点overlap、44,100 Hz、周波数刻み約21.53 Hz |
| 強度 | digital amplitude²/Hz。画像別の最大値正規化なし。校正されたPa²/HzやdB SPLではない |
| 画像 | 各秒を0.5–5 kHzと0.5–22.05 kHzの2パネル、同じ縦軸で保存 |
| 特徴 | 0.5–1 / 1–2 / 2–3 / 3–5 / 5–10 / 10–15 / 15–22.05 / 1–5 / 2–5 kHzのPSD積分をdB化 |
| 対応検算 | 全2,940秒の平均二乗をmanifestのsignal_chunk_powerと相対許容1e-7で照合 |

PSDを帯域内で積分した**パワーに閾値を置く**。「ある周波数より高い音なら沸騰」という規則ではない。Welchの推定法とdensity単位は[SciPy公式資料](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html)を参照。

画像は各日の `data/spectrum_png/waterflow_20260817_1s/<元WAV ID>/chunk-0000.png` 以下。6/11・6/18は各1,080枚、7/9は780枚。数値PSDは本記録の `spectra_<実験日>.npz`（大きいためGit対象外）、特徴は[spectral_features.csv](spectral_features.csv)。[生成manifest](spectrum_manifest.json)にパス・入力hash・件数を保存。

## 4. 見た図と確認した事実

- [全秒の帯域強度と熱流束](band_power_by_heat_flux.png)：全2,940秒を表示。
- [ONB代表の静かな秒・強い秒と既存スペクトログラム](onb_spectra_and_existing_spectrograms.png)：各日ONB点の2–3 kHz最小/最大の秒を機械的に選択。代表選択は[CSV](onb_quiet_loud_representatives.csv)へ保存。
- 全秒PSD一覧：[6/11](overview_2025.06.11_0.3_2.png)、[6/18](overview_2025.06.18_0.3_3.png)、[7/9](overview_2025.07.09_0.3_1.png)。低熱流束→高熱流束、WAV内は秒順。
- [9候補帯域・各WAVの数値](candidate_band_retention.csv)。test日6/18の列は記述用で、学習閾値推定へ渡さない。

6月の強い秒では2.2–2.4 kHz付近を中心に幅のある増加が見られ、弱い秒との差も大きい。単一binのピークより、**2–3 kHz帯域の積分パワー**を最初の候補とした。3 kHz入力にも22 kHz入力にも存在し、同じ選別集合で帯域比較できる。これは候補の合理的な選択であって、イベント正解に対する最適性の証明ではない。

特に7/9のONB点442.17 kW/m²では、最小/最大秒のPSDがほぼ背景レベルで近い。次点505.10 kW/m²でも同様。571.69、643.52 kW/m²には帯域の強い秒と弱い秒が混在し、720.69 kW/m²以上では全60秒が今回の閾値を超える。

原実験結果txtはONBを4.42e5 W/m²と記載しており、登録値を今回勝手に修正していない。**音響変化と実験上のONBの対応が日によって違うこと**が確認事実。録音と実験点の対応、計測感度、音の伝達、ONB判定方法のどれが理由かは未確定。

## 5. 暫定ルールと残存数

日dの現fit側で、ONB未満の2–3 kHz帯域パワーの99パーセンタイルを閾値Tdとする。

- q < qONB：そのまま保持。
- q ≥ qONB：帯域パワー > Tdの秒を保持、以下を学習から除外。
- 残す秒は元の熱流束ラベルのまま。テストと内部validationは全秒を保持。
- clean原音で判定し、全周波数・付加ノイズ版で同じ元WAV/秒を採用。
- 対象上限はconfigで日別に指定可能。既定は全ONB以上。内部foldごとに背景閾値を再fitする。

| 学習日 | 最終fitの閾値（dB re digital amplitude²） | ONB前の参照秒 | ONB以上の秒 | 除外 | 全域の残存数 |
|---|---:|---:|---:|---:|---:|
| 6/11 | −53.74296 | 600 | 480 | 8 | 1,072 / 1,080 |
| 7/9 | −76.82896 | 360 | 420 | 204 | 576 / 780 |
| 合計 | 日別 | 960 | 900 | 212 | **1,648 / 1,860** |

6/11の最初のONB点は52/60秒を保持。7/9はONBと次点0/60、571.69 kW/m²で17/60、643.52 kW/m²で19/60、それ以降3点は各60/60。元WAV数は31→29（最初の2陽性WAVが全除外）。[WAV別一覧](training_retention_by_wav.csv)、[全採否・閾値](training_selection_preview.json)を保存。

6/18は全1,080秒・18 WAVをテストする。記述上の同日99パーセンタイルは−57.52882 dB、ONB点で超える秒は36/60だが、この値を使ってテストを削ったり学習を調整したりしない。

## 6. この閾値の限界と次の識別比較

1. **熱流束ラベルと音響イベントの有無は異なる。** 静かな秒にも実際の熱流束が存在するので、選別は誤ラベル訂正とは限らない。音の強い区間だけに学習分布を寄せる効果を、全秒の別日テストで評価する。
2. **最適閾値は現状では決められない。** 秒単位の独立した気泡正解がない。99パーセンタイルは背景を超える暫定基準であり、気泡判定の特異度99%を意味しない。同期映像・音響イベントの点検を後で使う。
3. **日ごとの閾値差は約23 dBある。** 一律の絶対値では日差を無視する。背景参照にもONB以前の突発音が入っており、弱いイベントを落とす可能性がある。
4. **1秒平均PSDは短い突発音を薄める。** 今回は1秒学習単位に合わせる。問題が確認された段階で短時間帯域パワーや持続時間を比較する。現時点で追加の補正・サンプリングは行わない。
5. **同じ帯域で選別すると、その帯域の説明性が強く見える場合がある。** 後のXAIでは選別なしとの比較も根拠にする。選別帯域が重要と出ただけで物理原因を証明しない。
6. **既存データの限界**：全3日は過去に参照済み。また付加ノイズの固定基準RMSは既存49 WAV全体から生成済みで、今回再生成していない。モデル/PCA/scaler/選別閾値/統合重みのテスト日分離は実施したが、完全な未閲覧・前向き評価とは区別する。

次は**別日分離＋選別なし**と**同じ別日分離＋選別あり**を、3 kHz・無雑音・同じ3モデル・300 epochs・seed42・性能重みで比較する。全域R²だけでなく、ONB見逃し/誤警報とWAV残差を確認。その結果を読んでから強ノイズ・別方式へ進む。旧日内評価との差だけでは、日を分けた効果と選別の効果を分離できない。

## 7. 動作確認の根拠

最終方式名を保存した実データ確認run：baseline `1d5e04`、selected `7a63a2`。両方とも3 kHz・無雑音、深層は2 epochs、RFは現行300 trees。PCAは100成分、内部3-fold、入力224×224×1。3単体と統合の全テスト予測が有限で、重みを保存内部OOF誤差から再計算して一致を確認。

[verification.json](verification.json)、[baselineの出力](smoke_snapshot/baseline/)、[selectedの出力](smoke_snapshot/selected/)、[実行ログ](real_data_smoke.log)。深層未収束なので、このrunで最良モデルや選別による統合改善を判断しない。RFの値も、今回の短いパイプライン確認の文脈で保持する。

実行中の方式名整理前に出した開発runは保持したが、最終監査は上記2runだけを採用。過去の研究runへ遡って結果を混ぜていない。

## 8. configと再実行

[ONB主設定](../../code/run_ensemble_regression_onb.py)に集約。

```python
"learning_policy": {
    "split_mode": "explicit_days",
    "train_experiments": ["2025.06.11_0.3_2", "2025.07.09_0.3_1"],
    "test_experiments": ["2025.06.18_0.3_3"],
    "internal_validation": "chunk_kfold",
    "training_noise": "matched",
},
"acoustic_selection": {
    "enabled": True,  # Falseにすると選別なしの対照
    "features_csv": "experiments/2026-09-16_day_split_spectral_selection/spectral_features.csv",
    "feature": "band_2000_3000_db",
    "background_quantile": 0.99,
    "margin_db": 0.0,
    "apply_max_heat_flux_by_experiment": {},
},
"ensemble": {
    "enabled_strategy_names": ["performance_kfold"],
    "primary_strategy_name": "performance_kfold",
},
```

別の日をテストにするときは、data.experiment_namesに3日を残し、train_experiments/test_experimentsを互いに重ならないよう変更。未使用日を学習へ自動で入れない。内部fold数はrun.folds。内部を元WAVで分けたい場合には `wav_kfold`（元WAV一覧への通常KFold）も選択できる。

旧日内GroupKFold、LODO、等重み、inner holdout、crossfit各方式は保持。今回の学習選別はperformance_kfoldの内部fitに統合したため、新crossfitと併用する設定は明示エラーにする。将来の方式比較時には、選別条件も同じに保つ対応を追加してから比較する。

最初の本比較では data.max_freq_hz_listを3 kHzだけ、noise_dir_namesを無雑音だけ、explainability.enabledをFalseとし、run.epochs=300を保持して選別False/Trueを順に実行する。主コードの既定は5帯域×7ノイズ・XAI有効のため、そのまま実行すると限定比較ではなく35テスト条件になる。

```powershell
python code/analyze_boiling_spectra.py
python code/summarize_boiling_spectra.py
python code/verify_day_split_pipeline.py  # 専用の2 epochs動作確認
python code/audit_day_split_outputs.py
python -m unittest discover -s tests -v
```

本検証用はconfigを上記の限定範囲へ変更して `python code/run_ensemble_regression_onb.py`。選別設定と特徴CSVのSHA-256はrun hash・manifestに含め、以前の結果の誤再利用を防ぐ。保存先はテスト日の `regression_result/npy/.../<実行日>__days_matched/<帯域>/<ノイズ>/<run>`。内部OOF・分割・選別採否・テスト予測を個別保存する。

実行環境：NumPy 1.22.3、SciPy 1.7.3、scikit-learn 1.6.1、TensorFlow 2.9.1、RTX 4090。現在のオンラインAPIの既定値へ依存せず、窓・分割・seed等を明示した。

後期の計画は[階層タスク](../../docs/research_plan/2026-09-16_second_semester_plan.md)、[修論目次](../../docs/research_plan/2026-09-16_master_thesis_outline.md)、[9/18発表案](../../docs/research_plan/2026-09-18_xai_progress_brief.md)。
