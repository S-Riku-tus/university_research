# 研究状況調査の結果スナップショット

2026年9月14日に、Git対象外の保存結果から研究の現在地を確認するため抽出した。詳しい解釈は[現在地と次の作業](../../docs/research_plan/2026-09-14_current_state_and_next_steps.md)を参照する。

9月3日開始の `20260903_selected_log_architecture` は3実験日それぞれ35条件、計105条件のWAV評価が存在する。9月8日開始の同名系列は、無雑音・22 kHz・300 epoch設定の3条件が完了している。9月2日の56条件スナップショットとは別の実行である。

| ファイル | 内容 |
|---|---|
| `run_inventory.csv` | 108実行の条件、作成時刻、閾値、元manifestパス |
| `wav_median_metrics.csv` | 753モデル条件のpooled OOF WAV median指標 |
| `onb_transitions_median.csv` | 1 WAV・2 WAV持続条件のONB遷移、1,506行 |
| `metrics_condition_means.csv` | 実行・実験日・方式別の条件間単純平均 |
| `latest_group_mask_performance.csv` | 9/8実行の3日×3fold×3モデル×12マスク、324行 |
| `latest_ig_diagnostics.csv` | 9/8実行のIG診断、90件 |
| `latest_randomization_sanity.csv` | 9/8実行の最終層ランダム化診断、90件 |
| `snapshot_manifest.json` | 読み取った元ファイルのパス・SHA-256、最新18モデル条件の検算記録 |
| `collect_snapshot.py` | 元結果を変更せず抽出・検算するスクリプト |

最新3条件のR²は次のとおり。各WAVのchunk予測中央値を使い、全OOF WAVをpoolして計算する。実験日は2025年、解析日は2026年である。

| 方法 | 06.11（18 WAV） | 06.18（18 WAV） | 07.09（13 WAV） |
|---|---:|---:|---:|
| RF | 0.9139 | 0.9345 | 0.4739 |
| CNN＋Transformer | 0.9570 | 0.9688 | 0.8502 |
| AlexNet | 0.9538 | 0.9671 | 0.7465 |
| simple equal | 0.9508 | 0.9681 | 0.7272 |
| prediction max | 0.9489 | 0.9470 | 0.5742 |
| inner holdout | 0.9527 | 0.9696 | 0.7644 |

最新18モデル条件について、R²・RMSE・Recallを `wav_predictions_*.csv` から再計算し、保存指標との一致を検証した。学習やXAIの再実行は行っていない。コードの一般テストを今回再実行した記録ではない。

9/3結果のinner holdoutとvalidation-fold legacyは `claim_safe=0`。9/8結果のinner holdoutは元WAVを分離した方式である。`claim_safe=1` は保存された重み・分割監査上のフラグであり、未知実験日の妥当性やモデル選択の偏りの不存在まで保証するものではない。

`metrics_condition_means.csv` は周波数・ノイズ条件を単純平均した記述統計。同じ元WAVを再利用した条件を独立反復とは扱わない。旧chunk指標やfold別R²平均とWAV pooled R²を混在させない。

再生成はリポジトリルートから次を実行する。同じスナップショット出力だけを更新し、元runは変更しない。

```powershell
python experiments/2026-09-14_status_audit/collect_snapshot.py
```
