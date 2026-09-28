# 3 kHz固定・matched比較・チューニング・評価方式

記録日: 2026-09-28。本人の最新希望を優先し、3 kHzを主入力、`performance_kfold`を主方式として継続する。等重みは追加学習不要の対照として残す。直前の[双方向3 seed解析](../../experiments/2026-09-28_bidirectional_lowfreq_3seed/README.md)の数値は維持し、そこでの「等重みを主方式にする」という提案を本書で更新する。

## 1. 3 kHzへの固定

主研究の入力上限を3 kHzへ固定することには賛成する。clean_onlyの双方向3 seedでperformanceのnoise平均RMSEは順方向91.9±3.0、逆方向91.1±3.1 kW/m²となった。低周波を主にして、ノイズ学習方針とモデルの調整、説明性に研究資源を集中できる。

ただし「3 kHzが今回の固定モデル転送に有効」と「すべての学習方針で最適」は区別する。3 kHzで切ると信号だけでなくnoiseの入り方・周波数軸の224画素への配分も変わる。matchedなら高周波のnoiseを学習に取り込めるため、帯域間順位が変わる可能性がある。これは未確認である。

今後の本比較は3 kHzのみで進めてよい。既存5/10/15/22 kHz結果は帯域を選んだ根拠として保持する。matchedでも3 kHzが最適だと主張する必要が生じた場合だけ、5 kHzの限定対照を追加する。3 kHzでの学習方針を研究する限り、全帯域の再探索は必要ない。

## 2. matchedで何が変わるか

| 学習方針 | 学習 | 外部評価 | 答える問い |
|---|---|---|---|
| clean_only | cleanでモデル・PCA・scaler・重みを1回fit | 同じfitを各noiseへ適用 | clean学習モデルのnoise変化に対する性能 |
| matched | SNRごとに別のモデル・PCA・scaler・重みをfit | 学習と同じSNRの別WAV/別日 | noise条件に適応した場合の性能 |

22 kHzの既存3 seed比較では、performanceのnoise平均RMSEはmatched 97.73で、clean_onlyより52.63 kW/m²改善した。ただし帯域が異なるので、この改善量を3 kHzへ移せない。3 kHz clean_onlyは既に約91.5であり、matchedが必ずさらに良くなるとは限らない。[既存matched比較](../../experiments/2026-09-26_clean_matched_performance_kfold_3seed/README.md)

次の優先比較は3 kHz、選別なし、150 epochs、内部WAV 3-fold、既存パラメータ、seed 42でのmatched双方向である。clean＋全6 SNRを揃える。cleanは両学習方針で同じ処理なので再現確認用に残し、noiseありの差を適応学習の効果として読む。noise側で複数seedが必要と分かれば43・44を追加する。

matchedが改善した場合は「既知noiseに合わせた学習が有効」、clean_onlyと同等なら「この範囲では追加noise学習の利得が小さい」、悪化した場合は「有用情報の変化・学習の不安定性・日付間の対応の悪さ」を候補として調べる。matched曲線の形から同一モデルのnoise耐性とは言わない。

同じwaterflow素材を使った結果は、その素材・付加方法・SNR条件についての結果である。WAV分離で沸騰音の共有は防げるが、独立した水流音への一般化は別の未検証事項である。

## 3. performance_kfoldを継続する理由と条件

本人の希望どおり主方式として維持する。3 kHz双方向noise平均ではperformance 91.5、等重み92.4であり、performanceの明確な劣位が確認されたわけではない。前回の「0/12の内部―外部最良一致」だけから主方式を外す推奨は強すぎた。

最良単体順位の一致は診断値であり、アンサンブル性能そのものではない。統合には各モデルの誤差量、bias、残差の相関も関わる。内部clean順位が外部cleanへ移らない事実は残すが、noise下のperformanceの良い絶対性能も同時に評価する。

パラメータ変更で単体の適合度・予測bias・OOF誤差が変われば、重みと外部結果も変わり得る。一方、1日の内部検証だけで最適化しても別日の校正差が必ず解消するわけではない。チューニング前後の同じ外側分割で、全域RMSE、ONB近傍RMSE、FPR/Recall、単体との比較、重み順位を揃えて確認する。

## 4. 2つの評価方式

| config | 外側テスト | 内部検証 | 結果が示す範囲 |
|---|---|---|---|
| evaluation_mode=cross_day | train_experimentsと完全に別のtest_experiments | 学習日内の元WAV非共有K-fold | 収録日が変わったときの転送 |
| evaluation_mode=within_day | within_day_experimentの元WAVを一度だけ取り置く | 残りの元WAV内のK-fold | 同日の未使用録音への予測 |

同じ日であってもテストを学習に使わなければ有効な評価である。ただし別日の録音条件・音響応答の変化は試していない。同じ録音を1秒ずつ分けただけだと、サンプルが完全一致しなくても録音の特徴を共有する。今回の日内方式は元WAV単位で分ける。

現データは各熱流束段階の元WAVを分けることになるため、日内評価には未使用の熱流束測定段階への補間/外挿も含まれる。従来の同一WAVのchunkランダム分割とは難しさが異なる。昔の結果と絶対値を直接比較して学習性能の変化と断定しない。

### 日内holdoutの実装

- `test_fraction=0.25`: 元WAV数の25%をテストへ。18 WAVなら端数切上げの5 WAV、残り13 WAV。
- `test_split_seed=42`: 外側テストの抽出を固定。学習の`run.random_seed`を43・44にしてもテストは同じ。
- `test_stratify="onb"`: 確定済みの物理ONB閾値を使い、ONB前/以上のWAV比率をおおむね保つ。特徴量やモデルの予測成績で分割を選ばない。
- `test_stratify="none"`: 単純なWAVランダム分割。ONB層化を行えない用途で明示選択する。
- cleanとnoise版は同じ元WAVを同じ側へ置く。matchedでもnoise間で外側テストは共通。
- 13本の学習側の中で内部3-foldを行い、OOFからperformance重みを計算し、13本で最終fitして5本に1回予測する。
- PCA、scaler、内部重み計算の全てから外側テストを除外する。日内の音響選別は従来同様に無効のみ許可する。
- 保存名は別日`xd`、日内holdout`wh`、従来の日内外側K-fold`wd`で区別する。manifestに実効日付・設定・WAV一覧・テスト領域別数を記録する。
- `evaluation_mode`を省略した旧configでは、従来の日付リスト推論（日内K-fold/別日/leave-one-day-out）を維持する。

実データのmanifestだけを読み、全7 noiseの同一WAV割当を確認した結果:

| 日 | 学習/テストWAV | 学習/テストchunk | テストONB前/以上 | テストONB±10% |
|---|---|---|---|---|
| 6/11 | 13 / 5 | 780 / 300 | 120 / 180 | 0 |
| 6/18 | 13 / 5 | 780 / 300 | 120 / 180 | 60 |

6/11のテスト熱流束は57.59、181.38、427.28、554.20、698.42 kW/m²、6/18は58.39、184.19、271.68、434.02、709.79 kW/m²。ONB近傍WAVは各日に1本しかなく両側に置けない。6/11日内テストの近傍指標は未評価とし、0や良好とは解釈しない。近傍がある分割を成績に合わせて探さず、必要なら事前に定義した別分割/外側WAV K-foldを別実験として追加する。

同一日の外側テストは既に過去runで使ったデータでもあるため、これを新しく取得した未見実験データと呼ばない。新方式の手続き上の学習除外を検証するものである。

## 5. 再チューニング案

まず未調整の基準を日内・別日、clean_only・matchedで揃える。学習方針とパラメータを同時に変えると改善理由が判別できなくなる。

### 固定する条件

- 3 kHz、1秒、224×224×1、既存STFT/data版、水流音、選別なし、PCA 100。
- performance_kfold主方式、等重み対照、内部WAV 3-fold、まずseed 42。
- 外側WAV/実験日の割当を固定し、候補間で共有する。
- 第1段階は150 epochs固定。epochは次段階で100/150/200を比較する。

### 最初の小さな候補集合（提案、未実行）

| モデル | 探索候補 | 固定 |
|---|---|---|
| RF（実装はXGBRFRegressor） | max_depth: 3/4/8、n_estimators: 100/300 → 6候補 | subsample=.6、colsample_bynode=.6、PCA=100 |
| Conformer | lr: .0001/.0003/.001、batch_size: 6/12 → 6候補 | 構造・150 epochs |
| AlexNet | lr: .0001/.0003/.001、batch_size: 6/12 → 6候補 | 構造・150 epochs |

全モデル候補の直積6×6×6を学習せず、単体ごとの6+6+6=18候補を内部3-foldで評価する（1つの学習pool・1条件で54内部fit）。モデルごとに候補を絞り、その組合せでOOF重みを再計算し、最後に外側テストを評価する。現状候補は全て基準として残す。

### 何で候補を選ぶか

第1段階はcleanの内部OOF pooled-chunk RMSEを主基準にする。ONB近傍RMSE、FPR、Recallを併記し、RMSE最小との差2%以内を事前に同等群と定義する。同等群ではFPRが低い候補、同値ならRecallが高い候補を選ぶ。これは提案する選択規則であり、テストの結果を見て変更しない。

clean_onlyのnoise下性能を直接改善したければ、絞った上位候補だけ、内部held-out WAVのclean/−4/−12/−20 dBを予測する。学習自体はcleanのみ。候補ごとに4条件のRMSEを等重み平均して選ぶ。この場合は「noiseを検証時に既知とした頑健性チューニング」であり、noiseを全く見ていない条件とは分ける。performanceの重み式は変更せず、clean OOFから従来どおり算出する。

matchedとの初回比較には同じハイパーパラメータを使い、SNRごとにモデルと重みだけ再fitする。SNR別のハイパーパラメータ最適化は、既知noiseへの追加適応という別の実験にする。

### テストを候補選択に使わない

現行ONBの`models.parameter_sets`は候補の実行ループであり、内部CVで候補を選んで選択済み候補だけをテストする専用チューナーではない。`tuning_summary.csv`も外側評価の結果を含むため、それを最小化するとテストへ合わせた選択になる。

本ターンでは分割切替を実装し、チューニングは設計までとした。探索を開始するときは学習側のみの探索入口を用意し、外側テストを渡さずにパラメータを決定・保存してから、ONBへ固定値を戻す。日内では13 WAVだけ、別日では指定学習日の18 WAVだけで選ぶ。6/11→6/18では6/11だけ、逆方向では6/18だけから独立に選ぶ。両日のテスト結果を合算して最良パラメータを決めた比較は探索的結果であり、独立した別日テストとは呼ばない。

候補選択と性能推定のデータを分ける原則は[scikit-learnのnested CVの説明](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)に対応する。WAV単位のholdout比率はサンプル数でなくグループ数を指定する（[GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html)）。

## 6. 実行順と判断

1. 既存パラメータの3 kHz matchedを6/11→6/18、6/18→6/11、seed 42で実施。既存clean_onlyに対する学習方針の効果を調べる。
2. 日内holdoutを6/11、6/18それぞれでclean_only、seed 42。日付変化がない条件でのnoise応答と内部―外部順位を確認する。
3. 同じ日内テストWAVでmatchedを実施。これで「別日/日内」×「clean_only/matched」の4枠が揃う。日内は学習13 WAV、別日は18 WAVなので、差を日付効果だけに帰属しない。分量差が判断を左右する場合だけ、別日側も同じ13 WAVで学習した対照を追加する。
4. 上の基準を読んでから、目的に合う学習側チューニングを実施。選択した候補を固定しseed 43・44で再現する。学習seedを変えても外側テストは固定する。
5. 2.1–2.5 kHz帯遮蔽とIG等で、モデル・学習方針の違いが予測変化と整合するかを確認する。

1〜3は6 run×7 noise=42条件、すべて3 kHz・seed 42の新規比較。直ちに全条件×3 seed×パラメータ探索へ展開しない。白色noise、7/9、追加帯域は今回の主対象へ自動追加しない。

## 7. configの操作

`code/run_ensemble_regression_onb.py`の`VALIDATION_CONFIG["learning_policy"]`で選択する。

```python
"evaluation_mode": "cross_day",  # 別日。train/test_experimentsを使用
"within_day_experiment": "2025.06.11_0.3_2",
"test_fraction": 0.25,
"test_split_seed": 42,
"test_stratify": "onb",
"training_noise": "clean_only",  # matchedにするとSNRごとに学習
```

日内にするときは`evaluation_mode`を`"within_day"`に変更し、`within_day_experiment`を選ぶ。このとき別日用train/testリストは無視される。日付リストを同時に書き換える必要はない。通常の実行コマンドはリポジトリ直下から`python code/run_ensemble_regression_onb.py`。

実装確認後の主configは元のcross_day・6/11→6/18・clean_only・3 kHz・seed 42・150 epochsを維持した。このままでは既存条件の再実行になるので、次の比較にはまずtraining_noiseをmatchedへ変更する。
