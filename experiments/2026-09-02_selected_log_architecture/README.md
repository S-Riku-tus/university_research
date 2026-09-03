# 2026-09-02 selected log architecture

## このスナップショットの目的

`Pool_boiling/**/regression_result/` は容量が大きくGit対象外である。このディレクトリには、2026年9月2日に `run_ensemble_regression_onb.py` で実行した結果から、研究上の判断に必要な設定、数値、完了範囲、説明性集計だけを抽出した。

生の重み、全予測、loss図、散布図、個別説明画像はローカルに残し、Gitには追加しない。元の `tuning_summary.csv` のSHA-256は `snapshot_manifest.json` に保存しているため、ローカル結果とこのスナップショットの対応を確認できる。

## 実行範囲

実行は途中で停止した。予定105条件のうち56条件が完了している。

| 実験 | 完了周波数 | ノイズ条件 | 完了数 |
| --- | --- | ---: | ---: |
| `2025.06.18_0.3_3` | 3, 5, 10, 15, 22 kHz | 7 | 35/35 |
| `2025.07.09_0.3_1` | 3, 5, 10 kHz | 7 | 21/35 |
| `2025.06.11_0.3_2` | 未実行 | 0 | 0/35 |

## 主要結果

### 回帰性能

06.18ではCNN+Transformerが35/35条件で単体最高R2だった。周波数条件平均では、ノイズなしのR2 `0.9643`からreference SNR -20 dBの`0.8875`まで低下した。

安全な重み付けであるinner-holdoutは、06.18では単体最高R2に対して平均`+0.0009`、22/35条件で改善したが、07.09では平均`-0.0254`、改善は6/21条件だった。したがって、平均・重み付きアンサンブルによる一般的な回帰性能向上は確認できていない。

07.09では単体最高モデルがCNN+Transformer 13条件、RF 5条件、AlexNet 3条件に分かれた。reference SNR -20 dBの周波数平均R2はRF `0.3953`、CNN+Transformer `0.2380`、simple-equal `0.3741`であり、弱いモデルを含む平均ではRFを上回らなかった。

### ONBしきい値判定

prediction-maxは単体最高R2モデルに比べて、06.18の35/35条件、07.09の21/21条件でRecallを改善した。平均Recall差はそれぞれ`+0.0142`、`+0.0922`だった。一方、R2差は`-0.0215`、`-0.0675`である。

この結果はprediction-maxを「回帰精度向上」ではなく「見逃しを減らす安全側統合」の候補として位置づける根拠になる。ただし、最大値を採ることで陽性判定が増える効果を含むため、固定FPR、固定Precision、検知遅れで公平に再評価する必要がある。

### 説明性

RFの周波数帯マスクでは、06.18と07.09の全完了条件で2--5 kHz帯、3 kHz上限時は2--3 kHz帯が最大のR2低下を生じた。CNN+TransformerとAlexNetの最大寄与帯域は条件により変化した。この一貫性は物理解釈の候補だが、マスクによる分布外入力の影響と実験日再現性を確認するまでは物理的根拠と断定しない。

## 現時点の研究上の判断

- 回帰の主候補はCNN+Transformer。ただし07.09のように音響と熱流束の対応が弱い実験ではRFを含め全モデルが低下する。
- simple-equalとinner-holdoutは回帰の比較対象として残すが、常に単体モデルを上回る手法とは扱わない。
- prediction-maxはONB見逃し低減候補として別目的で評価する。
- `val_fold_legacy`は外側検証foldの正解値を重み計算に使用するため、再現比較専用であり主結果に使用しない。
- 現在のSNR別結果は各SNRで再学習した「条件別の到達性能」であり、clean学習モデルの未知ノイズ頑健性ではない。

## Gitで確認できるファイル

| ファイル | 内容 |
| --- | --- |
| `run_config.yaml` | 実行条件と解釈上の制約 |
| `completion.csv` | 実験別の完了範囲 |
| `metrics_by_condition.csv` | 56条件×7モデルの主要指標 |
| `metrics_by_snr.csv` | 完了周波数間で平均したSNR別指標 |
| `ensemble_comparison.csv` | 条件ごとの単体最高R2モデルと各統合方法との差 |
| `xai_top_frequency_group_by_condition.csv` | 条件・モデル別の最大周波数マスク寄与 |
| `xai_top_frequency_group_counts.csv` | 最大寄与帯域の出現回数 |
| `snapshot_manifest.json` | 元集計のパス、SHA-256、行数 |

`metrics_by_snr.csv`の周波数平均は結果を読みやすくするための記述統計である。同じ音源から作った周波数上限違いを独立反復として統計検定に使用しない。

## スナップショットの再生成

ローカルの結果が存在する環境で、リポジトリルートから次を実行する。

```powershell
python code/export_ensemble_result_snapshot.py `
  --result-root "Pool_boiling/Subcooling_20_degrees/0.3/2025.06.18_0.3_3/regression_result/npy/ensemble/20260902_selected_log_architecture" `
  --result-root "Pool_boiling/Subcooling_20_degrees/0.3/2025.07.09_0.3_1/regression_result/npy/ensemble/20260902_selected_log_architecture" `
  --output-dir "experiments/2026-09-02_selected_log_architecture"
```

再生成後は `snapshot_manifest.json` のSHA-256とGit差分を確認する。
