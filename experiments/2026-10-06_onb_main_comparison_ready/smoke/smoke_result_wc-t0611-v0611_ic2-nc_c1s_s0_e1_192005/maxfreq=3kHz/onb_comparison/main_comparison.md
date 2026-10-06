# ONBの主比較表

評価定義：`onb-fixed-three-five-v1`。誤差・熱流束はkW/m²、率は%で表示。

対象：software smoke or nonstandard configuration; not the standard 150-epoch research result

日別ONB閾値による主表。元3MSEを基準とし、ET4・ET単体も主要対照に含める。
q100/g100は日別に報告し、複数日のq100を平均して全体の検知点とはしない。
連続ROC/PR-AUCは従来通り予測熱流束の生スコア。日別閾値が異なる全体集計では補助的な順位指標として読む。

出力確認：**passed**。性能の改善方向は合否条件に含めない。
研究条件一致・実装確認の区別、学習/評価/再読込の実件数は[verification.json](verification.json)と[run_conditions.json](run_conditions.json)。

## 全体の回帰・ONB判定（各seed内で同じ評価chunk）

| seed | 条件 | モデル | RMSE | MAE | 近傍RMSE | FN | FP | Recall | F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | clean | RandomForest | 115.96 | 88.41 | 143.27 | 3 | 0 | 85.71% | 0.92308 |
| 42 | clean | Conformer | 980.99 | 907.10 | 1160.39 | 0 | 15 | 100.00% | 0.73684 |
| 42 | clean | AlexNet | 322.04 | 250.25 | 188.96 | 21 | 0 | 0.00% | 0.00000 |
| 42 | clean | HGB frequency34 | 37.02 | 29.89 | 24.20 | 2 | 0 | 90.48% | 0.95000 |
| 42 | clean | ExtraTrees frequency34 | 29.18 | 21.66 | 33.35 | 2 | 0 | 90.48% | 0.95000 |
| 42 | clean | Original three + HGB + ExtraTrees | 44.11 | 35.56 | 49.71 | 3 | 0 | 85.71% | 0.92308 |
| 42 | clean | Original three MSE | 112.99 | 85.74 | 101.17 | 3 | 0 | 85.71% | 0.92308 |
| 42 | clean | Original three performance | 119.72 | 90.42 | 123.23 | 3 | 0 | 85.71% | 0.92308 |
| 42 | clean | Original three + HGB | 75.77 | 57.90 | 68.86 | 3 | 0 | 85.71% | 0.92308 |
| 42 | clean | Original three + ExtraTrees | 43.93 | 35.40 | 50.23 | 3 | 0 | 85.71% | 0.92308 |
| 42 | clean | Ensemble simple equal | 187.84 | 161.75 | 156.46 | 0 | 15 | 100.00% | 0.73684 |
| 42 | -20 | RandomForest | 153.52 | 121.08 | 27.98 | 2 | 10 | 90.48% | 0.76000 |
| 42 | -20 | Conformer | 916.79 | 856.64 | 1003.65 | 0 | 15 | 100.00% | 0.73684 |
| 42 | -20 | AlexNet | 320.30 | 247.75 | 161.17 | 21 | 0 | 0.00% | 0.00000 |
| 42 | -20 | HGB frequency34 | 122.52 | 88.46 | 14.53 | 3 | 5 | 85.71% | 0.81818 |
| 42 | -20 | ExtraTrees frequency34 | 84.46 | 61.55 | 79.90 | 3 | 0 | 85.71% | 0.92308 |
| 42 | -20 | Original three + HGB + ExtraTrees | 99.98 | 75.85 | 53.44 | 3 | 0 | 85.71% | 0.92308 |
| 42 | -20 | Original three MSE | 165.36 | 130.49 | 32.90 | 2 | 10 | 90.48% | 0.76000 |
| 42 | -20 | Original three performance | 159.95 | 126.27 | 25.24 | 2 | 10 | 90.48% | 0.76000 |
| 42 | -20 | Original three + HGB | 143.49 | 110.22 | 18.06 | 2 | 10 | 90.48% | 0.76000 |
| 42 | -20 | Original three + ExtraTrees | 98.22 | 74.61 | 56.67 | 3 | 0 | 85.71% | 0.92308 |
| 42 | -20 | Ensemble simple equal | 226.07 | 186.19 | 149.45 | 0 | 15 | 100.00% | 0.73684 |

## 日別q100・g100

| seed | 条件 | 出典日 | モデル | ONB閾値 | q100 | g100 | FN | FP |
|---|---|---|---|---:|---:|---:|---:|---:|
| 42 | clean | 2025.06.11_0.3_2 | ExtraTrees frequency34 | 221.51 | 266.91 | 45.40 | 1 | 0 |
| 42 | clean | 2025.06.18_0.3_3 | ExtraTrees frequency34 | 271.68 | 322.11 | 50.44 | 1 | 0 |
| 42 | clean | 2025.06.11_0.3_2 | Original three + HGB + ExtraTrees | 221.51 | 266.91 | 45.40 | 1 | 0 |
| 42 | clean | 2025.06.18_0.3_3 | Original three + HGB + ExtraTrees | 271.68 | 376.32 | 104.64 | 2 | 0 |
| 42 | clean | 2025.06.11_0.3_2 | Original three MSE | 221.51 | 266.91 | 45.40 | 1 | 0 |
| 42 | clean | 2025.06.18_0.3_3 | Original three MSE | 271.68 | 376.32 | 104.64 | 2 | 0 |
| 42 | clean | 2025.06.11_0.3_2 | Original three + HGB | 221.51 | 266.91 | 45.40 | 1 | 0 |
| 42 | clean | 2025.06.18_0.3_3 | Original three + HGB | 271.68 | 376.32 | 104.64 | 2 | 0 |
| 42 | clean | 2025.06.11_0.3_2 | Original three + ExtraTrees | 221.51 | 266.91 | 45.40 | 1 | 0 |
| 42 | clean | 2025.06.18_0.3_3 | Original three + ExtraTrees | 271.68 | 376.32 | 104.64 | 2 | 0 |
| 42 | -20 | 2025.06.11_0.3_2 | ExtraTrees frequency34 | 221.51 | 266.91 | 45.40 | 1 | 0 |
| 42 | -20 | 2025.06.18_0.3_3 | ExtraTrees frequency34 | 271.68 | 376.32 | 104.64 | 2 | 0 |
| 42 | -20 | 2025.06.11_0.3_2 | Original three + HGB + ExtraTrees | 221.51 | 266.91 | 45.40 | 1 | 0 |
| 42 | -20 | 2025.06.18_0.3_3 | Original three + HGB + ExtraTrees | 271.68 | 376.32 | 104.64 | 2 | 0 |
| 42 | -20 | 2025.06.11_0.3_2 | Original three MSE | 221.51 | 0.00 | -221.51 | 0 | 7 |
| 42 | -20 | 2025.06.18_0.3_3 | Original three MSE | 271.68 | 376.32 | 104.64 | 2 | 3 |
| 42 | -20 | 2025.06.11_0.3_2 | Original three + HGB | 221.51 | 0.00 | -221.51 | 0 | 7 |
| 42 | -20 | 2025.06.18_0.3_3 | Original three + HGB | 271.68 | 376.32 | 104.64 | 2 | 3 |
| 42 | -20 | 2025.06.11_0.3_2 | Original three + ExtraTrees | 221.51 | 266.91 | 45.40 | 1 | 0 |
| 42 | -20 | 2025.06.18_0.3_3 | Original three + ExtraTrees | 271.68 | 376.32 | 104.64 | 2 | 0 |

## 強雑音の絶対誤差とcleanからの劣化量

| seed | モデル | clean RMSE | −20 RMSE | RMSE増加量 |
|---|---|---:|---:|---:|
| 42 | extra_trees | 29.18 | 84.46 | 55.28 |
| 42 | ensemble__original3_hgb_extra_trees | 44.11 | 99.98 | 55.88 |
| 42 | ensemble__original3_mse | 112.99 | 165.36 | 52.37 |
| 42 | ensemble__original3_hgb | 75.77 | 143.49 | 67.71 |
| 42 | ensemble__original3_extra_trees | 43.93 | 98.22 | 54.29 |

## 全条件の出力

- [全7条件・全モデル・日別/全体の主指標](main_metrics.csv)
- [元3MSE・ET4・ET単体との指標差](main_deltas.csv)
- [自身のcleanからの劣化量](noise_degradation.csv)
- [6雑音の平均と最大観測FP](noise_overview.csv)
- [日別q100/g100](q100_by_source_day.csv)
- [clean OOF由来の統合重み](ensemble_weights.csv)

既知WAV内の未使用chunk、反復使用した研究データの記述結果。
seed・雑音加工を独立実験として合算しない。q100は有限標本の到達段階で、時間遅延や将来の100%保証ではない。
従来の平均ONB閾値によるmetrics_summary CSV・散布図は補助出力として別に保存される。
