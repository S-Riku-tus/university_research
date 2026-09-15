# 9月15日の研究整理用スナップショット

> **9/15追加比較後の更新先**: 本書は9/14の6条件と9/15午前の状態の固定記録。最新2帯域比較の数値・IG例数・ONB結果へ読み替えない。 最新の[比較解析](../2026-09-15_onb_frequency_comparison/analysis.md)、[本人の研究仮説](../../docs/research_plan/2026-09-15_master_thesis_hypothesis.md)、[解析手順](../../docs/analysis_workflow.md)を参照する。以下の本文は当時の記録として保持する。

対象: 9/14の完了済み6条件の数値・説明性、および9月のselected-log実行一覧。採取時刻は[snapshot_manifest.json](snapshot_manifest.json)。元runを変更せず読み取り、学習・XAI再計算は行っていない。

## 完了範囲

| 実行系列 | 保存物を確認した範囲 |
|---|---|
| 9/1 | 6条件に旧chunk指標。WAV指標なし |
| 9/2 | 56条件に旧chunk指標、別の1条件は途中出力。既存56条件スナップショットと整合 |
| 9/3 | 105条件にchunk・WAV指標 |
| 9/8 | 3条件にchunk・WAV指標 |
| 9/14 | 6条件にchunk・WAV指標、54 fold-model条件に説明性CSV |
| 9/15 | 09:36開始の1条件にmanifest・途中fold出力。採取時点で完了印・全WAV指標なし |

178件はrun manifestの数。独立実験数や完了条件数ではない。旧runの`completed.json`不在だけを未完了とはせず、旧形式の指標と新形式の完了印を区別している。9/15の実行は後から進み得る。

## 9/14の性能

06.11の18元WAV、元WAV分離3-fold、300 epochs要求、各ノイズで再学習。主値は全OOF WAVの予測中央値から計算したR²。保存された全36モデル条件のR²・RMSE・RecallをWAV予測から再計算し、一致を確認した。

| 周波数 | reference SNR | RF | CNN＋Transformer | AlexNet | 単純平均 | inner holdout |
|---|---|---:|---:|---:|---:|---:|
| 3 kHz | 無雑音 | 0.9149 | **0.9457** | 0.9247 | 0.9369 | 0.9384 |
| 3 kHz | 0 | 0.9170 | **0.9460** | 0.9256 | 0.9377 | 0.9396 |
| 3 kHz | −20 | 0.9189 | **0.9350** | 0.9087 | 0.9298 | 0.9302 |
| 22 kHz | 無雑音 | 0.9139 | **0.9618** | 0.9571 | 0.9544 | 0.9596 |
| 22 kHz | 0 | 0.9133 | **0.9479** | 0.8886 | 0.9324 | 0.9298 |
| 22 kHz | −20 | 0.9214 | **0.9426** | 0.9068 | 0.9349 | 0.9390 |

CNN＋Transformerが6条件すべてで単体最高。各SNRで別モデルを学習した成績なので、同じcleanモデルの未知ノイズ耐性とは解釈しない。

22 kHz・無雑音ではCNN＋Transformer・単純平均・inner holdoutのWAV Recallは1.0。9/8の同日無雑音での「全方式が最初のONB WAVを見逃した」という結果は、9/14のすべての方式には当てはまらない。各条件のONB±10%帯は依然1 WAVである。

## 説明性で確認したこと

- IG: 180例中179例でcompleteness相対誤差が0.05超。0.05は説明用の目安で、事前登録した判定基準ではない。大きな不一致の原因は未確定。
- TreeSHAP: PCA空間90例の再構成絶対誤差は中央値約0.226、最大約2.046 W/m²。内部の数値整合は良好だが、PCA成分をそのまま物理周波数へ対応させない。
- 帯域マスク: RFは3 kHzで9/9 fold条件が2–3 kHz、22 kHzで9/9が2–5 kHzの最大R²低下。深層モデルは最大帯域が分かれる。
- IGの安定性360行、最終層ランダム化180行も保存。微小摂動への安定性と説明の正しさは別の確認事項。

マスクの567行は、3 kHzの9帯域/時間グループと22 kHzの12グループを各27 fold-model条件で計算したもの。R²低下は**fold内のWAV中央値評価**であり、上表の全OOF pooled R²とは集計が異なる。fold・SNR違いを独立した物理反復として数えない。

## 保存ファイル

| ファイル | 内容 |
|---|---|
| [run_inventory.csv](run_inventory.csv) | 9月の178 manifests、作成時刻、hash、出力状態、元パス |
| [wav_median_metrics.csv](wav_median_metrics.csv) | 9/14の36行。現行2統合方式＋単体は`current_strategy=True` |
| [onb_transitions_median.csv](onb_transitions_median.csv) | 1/2 WAV持続条件の遷移72行 |
| [xai_output_inventory.csv](xai_output_inventory.csv) | 54 fold-model条件のCSV存在確認 |
| [ig_diagnostics.csv](ig_diagnostics.csv) | IG 180例の寄与合計、予測差、整合性 |
| [group_mask_performance.csv](group_mask_performance.csv) | 全検証集合への帯域/時間マスク567行 |
| [treeshap_summary.csv](treeshap_summary.csv) | PCA空間の寄与再構成90例 |
| [input_stability.csv](input_stability.csv) / [randomization_sanity.csv](randomization_sanity.csv) | 安定性・最終層ランダム化の診断 |
| [snapshot_manifest.json](snapshot_manifest.json) | 元CSV/JSONのSHA-256と数値検算記録 |
| [representative_figures.json](representative_figures.json) | 代表図3枚の出典・SHA-256 |

9/14の元結果に含まれる`prediction_max`の数値は履歴として保持し、現行方式と区別する。この方式は9/14最終コードで削除済み。保存値を消して過去記録を書き換える必要はない。

## 代表図の読み方

06.11・22 kHz・無雑音・fold1、`near_onb_above_val0180`。真値368,978.105 W/m²。

- [入力スペクトログラム](figures/input_onb.png)
- [CNN＋TransformerのIG magnitude](figures/cnntf_ig_onb.png)
- [AlexNetのIG magnitude](figures/alexnet_ig_onb.png)

入力の約2–3 kHzの離散成分と両IGの局在が似て見える。これは代表画像の観察であり、全域の重要帯域や物理的原因を証明しない。IG図の色は各画像内の正規化値で、モデル間の寄与量の大小を比較できない。数値整合性の問題も併記する。

## 別時点の採取

元runが進んだ後は別の空ディレクトリへ採取し、このスナップショットを上書きしない。

```powershell
python -X utf8 experiments/2026-09-15_research_status_snapshot/collect_snapshot.py --output experiments/YYYY-MM-DD_research_status_snapshot
```

このスクリプトの数値抽出対象は9/14の6条件で固定。9月のrun一覧だけは採取時点を反映する。新しい日付の数値解析が必要なら、対象を明示して別スナップショットを作る。代表図のコピーはスクリプトの対象外で、出典は専用manifestで管理する。
