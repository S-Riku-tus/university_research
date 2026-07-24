# 説明性出力の読み方・確認順序・2026-07-24時点の診断

作成日: 2026-07-24  
対象コード:

- [説明マップの計算](../../code/utils/explainability/spectrogram_explainers.py)
- [学習・評価処理への統合と指標保存](../../code/utils/explainability/training_integration.py)
- [現在の実行設定](../../code/run_ensemble_regression_onb.py)

## 1. 最初に押さえること

説明画像は、それ単体で「モデルが沸騰現象を理解した」ことを証明しない。次の4種類を順に確認して、初めて研究上の解釈へ進める。

1. **予測自体が妥当か**  
   説明対象の標本で、真値、予測値、誤差、ONB判定を確認する。
2. **説明の数値的一貫性があるか**  
   IG completeness、TreeSHAPの加法性、deletion/insertionを確認する。
3. **説明が安定し、学習済みモデルに依存しているか**  
   input stabilityとrandomization sanity checkを確認する。
4. **物理軸で性能へ影響するか**  
   周波数帯・時間帯マスクで、R²、ONB RMSE、Recall、FNが実際に悪化するか確認する。

見た目の良いヒートマップを先に選ぶのではなく、**妥当な予測 → 数値検査 → 信頼性検査 → 物理軸マスク → 画像解釈**の順に見る。

## 2. 現在の出力全体

各実行条件の `explainability/` 以下に、概ね次のファイルがある。

```text
explainability/
├─ model_group_mask_comparison.csv
├─ top_groups_by_model.csv
├─ group_mask_comparison_*.png
└─ fold1/
   ├─ rf/
   │  ├─ explained_samples.csv
   │  ├─ treeshap_status.csv
   │  ├─ treeshap_pca_summary.csv
   │  ├─ treeshap_pca_values.csv
   │  ├─ group_mask_performance.csv
   │  └─ <代表標本>/group_occlusion_*
   ├─ cnntf_v2_gap/
   │  ├─ explainability_summary.csv
   │  ├─ input_stability.csv
   │  ├─ top_layer_randomization_sanity.csv
   │  ├─ group_mask_performance.csv
   │  └─ <代表標本>/integrated_gradients_*
   └─ alexnet/
      ├─ explainability_summary.csv
      ├─ input_stability.csv
      ├─ top_layer_randomization_sanity.csv
      ├─ group_mask_performance.csv
      └─ <代表標本>/integrated_gradients_* / grad_cam_*
```

表示軸は横が時間、縦が周波数である。内部配列は `(time, frequency)` として扱われ、表示時に転置される。

## 3. まず見る5ファイル

### 3.1 `explained_samples.csv`

説明対象として選ばれた標本の一覧である。

| 列 | 意味 | 確認点 |
| --- | --- | --- |
| `sample_id` | 標本カテゴリとvalidation内index | `near_onb_*`、`false_negative`、`worst_prediction`等を区別 |
| `y_true` | 真の熱流束 | ONBしきい値との位置関係 |
| `y_pred` | 予測熱流束 | 過大・過小予測の方向 |
| `abs_error` | 絶対誤差 | 良い予測と悪い予測の両方を含むか |

現在はfold 1、各モデル最大5標本であり、母集団平均ではない。`near_onb_below`、`near_onb_above`、`false_negative`、低・高熱流束、最大誤差などを意図的に選んだ代表例である。代表画像だけから出現頻度や一般性を判断しない。

### 3.2 `explainability_summary.csv`

各標本・各手法について、説明の加法性、曲線評価、最重要位置をまとめた表である。最初に `y_true`、`y_pred`、`abs_error` を確認し、その後に手法別の列を見る。

### 3.3 `group_mask_performance.csv`

validation標本全体について、特定の時間帯または周波数帯を0へ置換したときの性能変化を保存する。現在の出力で、モデル横断・物理軸比較の中心となる表である。

### 3.4 `input_stability.csv`

入力へ標本標準偏差の1%の微小ノイズを加え、IG絶対値マップがどの程度変わるかを示す。これはno noise / SNR -20 dBというデータ条件の頑健性とは別の、局所的な説明安定性検査である。

### 3.5 `top_layer_randomization_sanity.csv`

最後の学習可能層をランダム化し、説明マップが変わるかを見る。説明が学習済み重みに依存するなら、元マップとの類似度は低下することが期待される。ただし現在は最終層だけの部分検査であり、完全な段階的ランダム化ではない。

## 4. 手法ごとの画像と値

### 4.1 Integrated Gradients

#### `integrated_gradients_signed.png`

予測値とbaseline予測値の差を、入力スペクトログラムの各位置へ符号付きで配分したもの。

- 正の寄与: その位置の入力が、0 baselineに比べて予測熱流束を上げる方向へ寄与
- 負の寄与: 予測熱流束を下げる方向へ寄与
- 色の絶対値: 寄与の強さ

現在の実装では出力スケールを熱流束単位へ戻しているため、IG値の総和も熱流束と同じ単位を持つ。

#### `integrated_gradients_magnitude.png`

符号を落とした絶対値マップで、「上げる・下げる」を区別せず重要度だけを見る。deletion/insertionの順位付けや、安定性・sanityの比較にはこの絶対値が使われる。

重要度が大きくても、正負が隣接して打ち消し合う場合がある。物理解釈ではsigned画像とmagnitude画像を両方見る。

#### completeness関連列

IGの基本的な検査は次式である。

```text
attribution_sum ≈ y_pred - baseline_pred
```

| 列 | 意味 | 良い方向 |
| --- | --- | --- |
| `attribution_sum` | signed IG全要素の合計 | `output_delta_from_baseline`に近い |
| `output_delta_from_baseline` | 元入力の予測－baseline入力の予測 | 正負を含む比較基準 |
| `completeness_error` | `attribution_sum - output_delta` | 0に近い |
| `completeness_relative_error` | `abs(error) / abs(output_delta)` | 0に近い |

普遍的なしきい値はない。現在は実務上の一次スクリーニングとして、相対誤差0.10以下を「概ね確認」、0.05以下を「より良好」としてよい。ただし `output_delta` が0に近いと相対誤差が不安定になるため、その場合は絶対誤差も併記する。

completenessが悪い標本は、IG steps不足、数値誤差、モデルの非線形性、baselineの不適合などを疑い、その説明画像を物理解釈に使わない。

### 4.2 Grad-CAM

AlexNetの最後の畳み込み特徴から、予測に関係する大まかな領域を表示する。

- 0～1に正規化された相対的な局在であり、熱流束単位ではない。
- 解像度はIGより粗い。
- 基本的に「どの領域が活性化したか」を見る補助表示で、正負の寄与分解には使わない。
- IGと完全一致する必要はないが、主要領域が大きく矛盾する場合は理由を調べる。

Grad-CAMだけで周波数帯の因果的重要性を主張せず、共通帯域マスクで裏付ける。

### 4.3 RFのTreeSHAP

RFは100次元のPCA特徴を入力としているため、TreeSHAPは各PCA成分が予測を上げたか下げたかを説明する。

| ファイル | 意味 |
| --- | --- |
| `treeshap_pca_values.csv` | 標本ごとの各PCA成分のSHAP値 |
| `treeshap_pca_summary.csv` | base value、SHAP総和、予測値、加法誤差 |
| `pca_feature_importance.csv` | PCA特徴空間でのRF重要度 |
| `treeshap_status.csv` | 物理軸への直接解釈ができないことを含む状態記録 |

SHAP値の合計とbase valueからRF予測が再構成できるかは確認できる。しかし、PC7が重要だから「7番目の周波数帯が重要」とは言えない。PCA成分は時間・周波数全域の線形結合だからである。

したがって、TreeSHAPはRF内部監査に使い、RFの物理的な周波数・時間主張には `group_mask_performance.csv` を使う。

### 4.4 Group occlusion / group masking

#### 標本単位の `group_occlusion.csv`

帯域または時間帯を0へ置換した際の予測変化を見る。

- 元予測－マスク後予測が正: その領域を消すと予測が下がるため、元予測を上げる方向に使われていた
- 負: その領域を消すと予測が上がるため、元予測を下げる方向に使われていた
- 絶対値: その標本の予測への影響量

#### validation全体の `group_mask_performance.csv`

| 列 | 計算 | 大きい正値の意味 |
| --- | --- | --- |
| `r2_drop` | base R²－masked R² | 消すとR²が低下し、回帰全体に重要 |
| `rmse_all_increase` | masked RMSE－base RMSE | 消すと全体誤差が増える |
| `mae_all_increase` | masked MAE－base MAE | 消すと全体誤差が増える |
| `rmse_onb_increase` | masked ONB RMSE－base ONB RMSE | ONB近傍回帰に重要 |
| `recall_drop` | base Recall－masked Recall | 消すとONB以上の検出率が下がる |
| `f1_drop` | base F1－masked F1 | 二値判定バランスが悪化 |
| `false_negative_increase` | masked FN－base FN | 消すと見逃しが増える |

正の値が大きいほど、その指標に対して重要だったと読む。負の値は、消すことで性能が改善したことを示し、有害な特徴、過学習、交絡の可能性がある。

注意点:

- マスクは介入感度であり、自然現象の因果関係を直接証明しない。
- 0置換は分布外入力になり得る。
- R²低下とRecall低下は尺度が違うので数値の大きさを直接比較しない。
- `top_groups_by_model.csv` はランキングの要約にすぎない。必ずbase値、masked値、標本数を元表で確認する。
- 時間・周波数帯の幅が異なる場合、広い帯域ほど消す情報量が多い。帯域幅を揃えた感度分析も必要である。

## 5. deletion / insertion曲線

### 5.1 deletion

重要度の高い画素から順に0へ置換し、予測がどう変わるかを見る。

`*_deletion_curve.csv` の主な列:

- `masked_fraction`: 消した割合
- `base_pred`: 元入力の予測
- `masked_pred`: マスク後予測
- `delta`: `base_pred - masked_pred`
- `abs_delta`: 変化の絶対値

本当に重要な画素が先に消されれば、少ないマスク率で予測が大きく変わることを期待する。

### 5.2 insertion

0 baselineから始め、重要度の高い画素から元入力へ戻す。

`*_insertion_curve.csv` の主な列:

- `inserted_fraction`: 戻した割合
- `baseline_pred`: 0入力の予測
- `inserted_pred`: 挿入後予測
- `delta_from_baseline`: baseline予測からの回復量
- `remaining_delta_to_original`: 元予測まで残っている差

重要画素が正しく並んでいれば、少ない挿入率で元予測へ近づくことを期待する。

### 5.3 summaryのAUC列

ここでのAUCはROC-AUCではない。予測熱流束をマスク率0～1で積分した面積で、熱流束単位を持つ。

| 列 | 意味 |
| --- | --- |
| `*_prediction_auc` | 実際の予測曲線下面積 |
| `*_linear_auc` | 始点と終点を結ぶ直線の面積 |
| `*_area_between_curve` | prediction AUC－linear AUC |

生の `prediction_auc` は、標本の予測熱流束が高いほど大きくなるため、標本間の説明品質比較には適さない。`area_between_curve` も、元予測がbaselineより高いか低いかで期待符号が変わる。

`output_delta = y_pred - baseline_pred` が正の場合:

- 良いdeletion: 早く低下するためarea between curveは負になりやすい
- 良いinsertion: 早く上昇するためarea between curveは正になりやすい

`output_delta` が負の場合は符号が逆になる。比較用には、今後次の向き付き・正規化値を追加すると分かりやすい。

```text
deletion_score =
  -sign(output_delta) * deletion_area_between_curve / abs(output_delta)

insertion_score =
   sign(output_delta) * insertion_area_between_curve / abs(output_delta)
```

この形なら、概ね正で大きいほど重要画素が先に効いたと読める。ただし、現在のCSVにはこの派生値はまだ保存されていない。

## 6. 安定性とsanity check

### 6.1 `input_stability.csv`

| 列 | 良い方向 | 意味 |
| --- | --- | --- |
| `pearson_abs_map` | 1に近い | 重要度の空間パターンが似る |
| `cosine_abs_map` | 1に近い | ベクトル方向が似る |
| `relative_l1_abs_map` | 0に近い | 重要度総量の差が小さい |

高相関でもrelative L1が大きければ、形は同じでも強度が変わっている。3列をセットで見る。

この検査が良いことは、「入力のごく小さな揺らぎに対して説明が安定」という意味に限られる。未知ノイズや実験日への一般化は別途検証する。

### 6.2 `top_layer_randomization_sanity.csv`

| 列 | 期待する方向 | 意味 |
| --- | --- | --- |
| `pearson_abs_map` | 低下 | 最終層を壊すと空間パターンが変わる |
| `cosine_abs_map` | 低下 | 重要度分布の方向が変わる |
| `relative_l1_abs_map` | 増加 | 元説明との差が大きくなる |

sanity checkでは、stabilityと逆に、類似度が高すぎることが問題になる。最終層をランダム化したのに説明がほぼ同じなら、入力の輪郭やモデル構造だけを反映し、学習済み判断を十分に反映していない可能性がある。

現在は絶対値マップだけを比較し、最終層だけをランダム化している。今後は出力層から前段へ順番にランダム化し、類似度が段階的に低下するかを確認する。

## 7. 2026-07-23出力の実測診断

対象は3実験、22 kHz、no noise / SNR -20 dB、fold 1、各条件・モデル最大5標本である。標本数が少ないため、現時点では診断値である。

### 7.1 IG completeness

| モデル・条件 | 相対誤差平均 | 最大 | 0.10以下 |
| --- | ---: | ---: | ---: |
| AlexNet / no noise | 0.0957 | 0.2898 | 10/15 |
| AlexNet / SNR -20 | 0.0096 | 0.0459 | 15/15 |
| CNN+Transformer / no noise | 0.5061 | 4.808 | 7/15 |
| CNN+Transformer / SNR -20 | 0.0123 | 0.0269 | 15/15 |

SNR -20では両モデルとも良好だが、no noise、とくにCNN+Transformerで不良標本が多い。no noiseのCNN+Transformer説明を先に物理解釈せず、IG steps、baseline、`output_delta`の大きさ、数値スケールを標本別に確認する。

### 7.2 微小入力摂動

IG絶対値マップの平均Pearson相関は全モデル・条件で約0.99996～0.999995、relative L1は約0.011～0.024だった。1%摂動に対する局所安定性は高い。

### 7.3 最終層ランダム化

| モデル・条件 | Pearson相関平均 | おおよその範囲 |
| --- | ---: | ---: |
| AlexNet / no noise | 0.844 | 0.647–0.984 |
| AlexNet / SNR -20 | 0.874 | 0.619–0.997 |
| CNN+Transformer / no noise | 0.925 | 0.757–0.997 |
| CNN+Transformer / SNR -20 | 0.924 | 0.680–0.998 |

relative L1は約0.63～0.96で強度は変化したが、空間形状の相関は高く残った。特にCNN+Transformerは、学習済み最終層への依存性を十分に示せていない。現状の判定は「部分sanity checkで警告あり」である。

### 7.4 周波数帯マスク

3実験のR²低下を平均すると、最大影響帯は次のように変化した。

| 条件 | AlexNet | CNN+Transformer | RF |
| --- | --- | --- | --- |
| no noise | 2–5 kHz、ΔR²約0.615 | 2–5 kHz、約0.468 | 2–5 kHz、約0.415 |
| SNR -20 | 10–15 kHz、約0.624 | 10–15 kHz、約0.383 | 10–15 kHz、約0.294 |

SNR -20ではRecall低下やFN増加も10–15 kHzで大きかった。一方、付与した水流音のパワーも10–15 kHzへ集中している。このため、現在は「沸騰に重要な帯域が10–15 kHz」と結論せず、**ノイズ生成に由来する帯域へモデル依存が移った可能性**として扱う。

## 8. 現時点の総合判定

| 観点 | 判定 | 理由 |
| --- | --- | --- |
| 出力生成 | 完了 | 全モデルの指定手法を代表条件で保存できた |
| IG加法性 | 条件付き | SNR -20は良好、no noiseのCNN+Transformer等に問題 |
| 局所安定性 | 良好 | 1%摂動後の相関が非常に高い |
| モデル依存性 | 未確認・警告 | 最終層ランダム化後も形状相関が高い |
| モデル横断の物理比較 | 実装済み | 共通帯域・時間帯マスクを使用可能 |
| 物理的妥当性 | 未確定 | 強ノイズ時の重要帯域が水流音主帯域と一致 |
| 研究上の説明性完了 | 未完了 | baseline、段階的sanity、fold/実験日再現が必要 |

## 9. 実際の確認手順

1. `explained_samples.csv` で真値、予測値、標本カテゴリを見る。
2. `input_spectrogram.png` で信号・ノイズの見た目と軸を確認する。
3. IGの場合、`explainability_summary.csv` でcompletenessを確認する。
4. completeness不良標本は保留し、良好標本だけsigned/magnitudeを確認する。
5. `input_stability.csv` で微小摂動への安定性を見る。
6. `top_layer_randomization_sanity.csv` で学習済み重みへの依存を確認する。
7. 同一AlexNet標本でIGとGrad-CAMの大まかな局在を比較する。
8. `group_mask_performance.csv` で、強調領域を消すと実際に性能が悪化するか見る。
9. `model_group_mask_comparison.csv` でモデル間を同一指標・同一条件で比較する。
10. no noise / SNR -20、3実験で重要帯域が再現するか確認する。
11. 水流音単体のスペクトルと比較し、ノイズ由来の帯域を除外する。
12. 最後に、説明可能な範囲と未確認事項を文章化する。

## 10. 発表・修論での記述テンプレート

安全な記述は次の形である。

> ○○モデルでは、対象条件の周波数帯マスクによりX–Y kHzを除いたとき、R²がA、ONB RecallがB低下した。このため、同帯域が現在の予測に利用されていることは示された。一方、同帯域は付与ノイズの主成分とも重なるため、沸騰現象固有の特徴であるかは、固定振幅ノイズ、no noise学習からの交差条件評価、未知元WAV評価で今後確認する。

避ける記述:

- 「ヒートマップが赤いので、この帯域が沸騰を示す」
- 「安定性相関が高いので説明は正しい」
- 「Grad-CAMとIGが似ているので物理的に妥当」
- 「deletion AUCが高いので良い」  
  ここでのAUCはROC-AUCではなく、符号とbaselineを含めた解釈が必要である。

## 11. 次に説明性側で行うこと

1. no noiseのCNN+Transformerでcompleteness不良標本を個別確認する。
2. IG stepsを増やした場合とbaselineを変更した場合の改善を比較する。
3. 段階的な層ランダム化を実装し、類似度低下曲線を作る。
4. 代表5標本だけでなく、fold・実験日を跨いだ帯域順位の再現率を集計する。
5. 0マスク以外に、学習平均・局所補間マスクを比較する。
6. ノイズ生成を是正した後、重要帯域が2–5 kHz等へ戻るか再検証する。

この順序で確認すれば、各グラフや値を「きれいな可視化」ではなく、予測根拠に対する反証可能な検査として扱える。
