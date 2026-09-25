# ONB run 143020（performance_kfold・3/22 kHz clean）の解析

日付: 2026-09-25。状態: **実行完了・解析済み。ただし本比較ではなく予備結果として扱う。**

## 1. 今回答える問いと対象

対象runは次である。

`Pool_boiling/Subcooling_20_degrees/0.3/2025.06.18_0.3_3/regression_result/npy/ensemble/202609/25/onb_xd-t0611-v0618_iw3-nm_s1e-9_e200_143020`

このrunで確認できる問いは、**6/11のcleanデータを2.1–2.5 kHzピークで選別して学習したとき、学習日内3-fold OOFから求めた`performance_kfold`重みが、別日の6/18 clean評価で単体モデルを補完するか**である。

ノイズ耐性、7 SNRでの傾向、複数seed再現性、選別なしとの差、5-fold方式の本結果はこのrun単独では答えられない。

## 2. 再現条件と完了監査

- 学習日: 2025.06.11、18 WAV、1,080 chunks
- 評価日: 2025.06.18、18 WAV、1,080 chunks
- 周波数上限: 3 kHz、22 kHz
- noise: cleanのみ
- `training_noise`: `matched`。ただしcleanしかないため、実質はclean学習→別日clean評価
- 音響選別: 2.1–2.5 kHz最大Welch PSD `< 1e-9`のONB以降chunkを除外
- 最終学習: 1,080→986 chunks（94 chunks除外）
- seed: 42
- 深層モデル: 200 epochs固定、batch size 12、OOMなし
- 重み用内部検証: 元WAV非共有3-fold。各fold 6 WAV・360 chunksを検証
- 統合: `performance_kfold`のみ。`simple_equal`は正式な比較方式として未実行
- 説明性: fold 1、各モデル5代表chunk
- 完了印: 3 kHz・22 kHzとも存在。fit IDは各条件1件

内部OOFは両周波数とも1,080件、キー重複なし、18 WAV×60 chunksであり、各foldの`shared_source_wavs=0`だった。評価日データは重みfitに使われていない。

一方、事前に作成した本比較条件は、選別なし、22 kHz、5-fold、7 SNR、seed 42–44、`performance_kfold`＋`simple_equal`、XAI無効である。したがって今回のrunは、その本比較条件とは一致しない。

## 3. 確認した性能

評価単位は1秒chunkである。ONB RMSEはONB閾値と同じ熱流束の1 WAV・60 chunksに対する値であり、独立した60実験ではない。

### 3.1 3 kHz

| 方法 | R² | 全域RMSE | 全域MAE | ONB RMSE | ONB Recall |
|---|---:|---:|---:|---:|---:|
| RF | 0.8481 | 106.1 | 86.2 | 187.2 | 0.8267 |
| Conformer | 0.8565 | 103.2 | 74.5 | **51.2** | **0.8917** |
| AlexNet | 0.8774 | 95.3 | 82.6 | 121.0 | 0.8000 |
| performance_kfold | **0.8999** | **86.2** | **70.5** | 96.6 | 0.8283 |

誤差単位はkW/m²。全域では統合が最良単体AlexNetよりRMSEを9.2 kW/m²改善した。一方、ONB点ではConformerより45.4 kW/m²悪化し、Recallも0.0633低い。

### 3.2 22 kHz

| 方法 | R² | 全域RMSE | 全域MAE | ONB RMSE | ONB Recall |
|---|---:|---:|---:|---:|---:|
| RF | 0.8475 | 106.3 | 83.2 | 197.7 | **0.8300** |
| Conformer | 0.9062 | 83.4 | 66.3 | 124.1 | 0.8233 |
| AlexNet | 0.9147 | 79.5 | 64.4 | **96.8** | 0.7783 |
| performance_kfold | **0.9155** | **79.2** | **62.7** | 130.1 | 0.8117 |

全域では統合のAlexNetに対するRMSE改善は0.35 kW/m²、R²改善は0.00076に留まる。ONB点ではAlexNetより33.4 kW/m²悪化した。RFとの比較では全域MAEは20.5 kW/m²改善したが、false negativeは102→113へ11件増えた。

### 3.3 重みと誤差相殺

| 帯域 | OOF R²: RF / Conformer / AlexNet | 重み: RF / Conformer / AlexNet |
|---|---|---|
| 3 kHz | 0.9018 / 0.9272 / 0.8881 | 0.310 / 0.418 / 0.272 |
| 22 kHz | 0.9063 / 0.9447 / 0.9287 | 0.250 / 0.423 / 0.328 |

3 kHzの評価日バイアスはRF −61.3、Conformer +44.3、AlexNet −25.2 kW/m²であり、統合後は−7.3 kW/m²になった。したがって全域改善の主要な説明は、モデル間の符号の異なるバイアスを平均して相殺したことと整合する。

ただし熱流束領域ごとの最良モデルは異なる。3 kHzではONB以上でConformerのRMSEが47.5 kW/m²に対し統合は69.8 kW/m²、22 kHzでもONB以上はConformer 68.6に対し統合76.1 kW/m²だった。全域OOF R²だけから作る1組の重みは、ONB領域を保護する目的関数ではない。

保存予測から事後計算した3モデル等重みRMSEは、3 kHz 87.0、22 kHz 80.0 kW/m²であり、`performance_kfold`の改善はそれぞれ約0.8、0.9 kW/m²だった。この等重み値は診断用の事後計算であり、今回の正式保存方式ではない。現時点では、追加学習を伴う性能重みが等重みに対して明確に優位とはいえない。

## 4. 説明性結果

### 4.1 共通周波数帯マスク

- RFは3 kHzで2–3 kHzを消すとR²が0.528低下し、22 kHzでも2–5 kHzで0.480低下した。RFが2–5 kHz付近へ強く依存するという過去結果と整合する。
- 3 kHzの深層2モデルでも1–3 kHzマスクの影響が大きかった。
- 22 kHzではConformerは5–10 kHzおよび15–22 kHz、AlexNetは1–5 kHzのマスク影響が大きく、モデル間の利用帯域は同一ではなかった。
- ただしゼロマスク後にR²が大幅な負値になる例や、全域RMSEが悪化する一方でONB Recallが改善する例がある。これは分布外入力と閾値方向の移動を含むため、帯域の物理的因果性とはみなさない。

### 4.2 IG

- 3 kHzはConformer 5/5、AlexNet 5/5で主IGが数値収束した。
- 22 kHzはAlexNet 5/5が収束したが、Conformerは3/5のみ。false negative例と低熱流束例は4,096点でも未収束だったため、これらのmapは採用しない。
- BatchNormalization端点差によるCPU fallbackは、Conformer全10例、AlexNetは各帯域1例で使用された。修正後の自動fallbackは実runでも機能した。
- 1%入力摂動に対する絶対mapの平均Pearson相関は、3 kHz Conformer 0.422、AlexNet 0.226、22 kHz Conformer 0.303、AlexNet 0.460だった。高熱流束AlexNet例は特に不安定である。数値収束したことと、微小摂動に対して安定な説明であることは分ける。
- top-layer randomization後との相関も約0.33–0.38残る。現段階ではIG画像の細かな局在を修論の強い物理主張へ使わない。

### 4.3 Grad-CAMとTreeSHAP

- AlexNetのGrad-CAMは両帯域5例ずつ、計10例すべて`UnimplementedError`で失敗した。今回のrunはGrad-CAM完了結果ではない。採用を続けるなら原因を独立に修正・検証する。
- TreeSHAPはRFのPCA空間で完了した。PCA成分寄与であり、そのまま物理周波数帯として命名しない。RFの物理帯域比較には全集合マスクを使う。

## 5. 研究上の解釈

### 確認できたこと

1. `performance_kfold`の元WAV非共有OOF生成と外側評価日未使用は成立した。
2. 3 kHz cleanでは、3モデルの異なるバイアスを統合することで全域誤差が明確に減った。
3. 22 kHz cleanでは最良単体に対する全域利得はごく小さく、等重みに対する利得も両帯域で約1 kW/m²以下だった。
4. 全域R²重みの改善とONB近傍性能は一致せず、両帯域でONB最良単体を悪化させた。
5. RFと深層モデルの帯域感度は異なるが、深層IGの安定性とGrad-CAM完了性は、物理解釈へ進むには不足している。

### 整合する原因仮説

- 全域改善は「異なる特徴を使うから」という一般論だけでなく、評価日に生じた正負のバイアス相殺で説明できる。
- ONB悪化は、ONB領域で良いConformerまたはAlexNetへ十分な重みを集中せず、低値側へ偏るRF等を混ぜたためと整合する。
- 学習日OOFでのモデル順位と評価日・領域別の順位が一致しないため、全域R²から求めた固定重みだけでは別日の局所性能を安定して予測できない可能性がある。

これらは1 seed・clean・選別ありの結果に基づく仮説であり、方式の最終採否ではない。

## 6. 次に行うこと

### 段階0: 今回のrunを予備結果として固定する

今回のrunを`performance_kfold`本比較の完了とは呼ばない。特に、現行主コードをそのまま再実行すると、3-fold・選別あり・cleanのみ・`simple_equal`なしを繰り返す点に注意する。

### 段階1: 主研究質問を固定する

修士研究の中心が「未知雑音による性能低下を統合で抑える」なら、`clean_only`を主比較、`matched`を既知noiseへの適応という補助比較にする。教授と異なる位置づけを決めた場合は、その決定を結果を見る前に記録する。

### 段階2: 本比較条件へ戻す

最低限、次を同時に満たす条件を使う。

1. 音響選別なし
2. 22 kHz
3. 元WAV非共有5-fold（4、4、4、3、3 WAV）
4. `performance_kfold`と`simple_equal`を同じ最終予測で比較
5. XAIは性能 sweep中は無効
6. まずseed 42で監査し、その後42–44を完了
7. clean、0、−4、−8、−12、−16、−20 dB

作成済みのmatched条件は`configs/experiments/2026-09-25_matched_performance_kfold_fixed_epoch.json`で、この条件を満たす。`clean_only`を主比較にする場合は、同じ分割・モデル・seed・評価noiseで`training_noise=clean_only`とした対応条件を先に固定する。

### 段階3: seed 42の完了直後に監査する

- 7/7 SNR完了
- 各内部foldで元WAV共有0
- OOF 1,080件が重複なく1回ずつ予測
- 選別なしで最終学習1,080件
- 5内部fold、全深層fit 200 epochs
- `performance_kfold`と`simple_equal`の両方を保存
- 評価日ラベルが重み・epoch・方式選択に未使用

ここで不一致があればseed 43・44を開始しない。

### 段階4: 3 seed×7 SNRで方式を判断する

全域RMSEだけでなく、次を同時に読む。

- cleanからのRMSE増加量と強noiseでの絶対性能
- RF、最良単体、等重みに対する差
- ONB前、ONB点、ONBより上のRMSE・bias
- Recall、Precision、F1、false negativeの訂正と追加
- seed間の重み幅、モデル順位、残差相関

`performance_kfold`が等重みとの差1%未満に留まる、またはONB悪化が複数seed・SNRで反復する場合は、性能重みを主方式と自動採用しない。まず等重みを基準候補とし、その後に限って同一OOFを共有する`subset_equal_cv`や、学習日内だけで定義するONBを考慮した目的関数の必要性を検討する。

### 段階5: XAIは最終候補へ限定する

性能方式を固定した後、統合が最良単体を訂正したchunkと悪化させたchunkを保存予測から選ぶ。先に残差と重みで誤差を分解し、その後に全集合マスク、RFのTreeSHAP、数値収束かつ安定性を確認したIGを使う。Grad-CAMは全失敗の原因を修正できるまで結果として使用しない。

## 7. 根拠ファイル

- `ensemble_presentation_summary.csv`
- 各帯域の`metrics_summary_no_noise.csv`
- 各帯域の`ensemble_weights_no_noise.csv`
- 各帯域の`internal_validation_fold1.json`
- 各帯域の`training_selection_fold1.json`
- 各帯域の`fold_pred/pred_f1_no_noise.csv`
- 各モデルの`explainability/fold1/*`

解析では元結果を変更せず、保存予測から等重み・領域別bias・WAV別誤差を事後計算した。事後比較値は方式選択の独立証拠とは扱わない。
