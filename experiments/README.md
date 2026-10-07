# 実験結果・実装記録の入口

更新日: 2026-10-07。現在の研究状態は[研究の現在地](../docs/research_status.md)。ここには、その判断の根拠となる**日付付きの結果・検証記録**を置く。

## 直近の結果と使い分け

| 記録 | 範囲・用途 |
|---|---|
| [10/7 熱流束の従来保存階層への修正](2026-10-07_heatflux_result_layout/README.md) | 本人生成の21段階を再計算、11出力の内容保持移動、改名/STFT/ONBの同一CSV読込、23テスト。温度低下の代表区間選択への敏感性を記録 |
| [10/7 熱流束の段階・時間幅修正](2026-10-07_heatflux_stage_alignment/README.md) | 新21/旧18段階、旧qとの差最大0.1581%、15/35/55秒、熱計算セル1〜8、52テスト。原ファイル・過去結果保持、音響生成・学習なし |
| [10/7 新しい実測実験のONB設定](2026-10-07_onb_new_day_configuration/README.md) | 日内/別日検証、本NPY指定、ノート1.2 VのCSV閾値。保存階層の復帰は上段の最新記録を参照。実データ学習は未実行 |
| [10/7 新しい実測実験のデータ配置](2026-10-07_pool_boiling_data_setup/README.md) | 原WAV11本・計測CSV・ノートCSVの配置とSHA-256。従来の実行入口を保持。後継の熱流束比較は上段を参照 |
| [10/7 最終5の入力依存・配分・入力除去後再学習](2026-10-07_onb_input_and_integration/README.md) | 本人依頼の初め3段階を完了。共通加工37,800予測、元3内部比率の限定対照、同じHGB/ETの16 fit・全保存再読込、固定重み対照・損失分解と採用判断 |
| [10/7 通常5モデル・200 epochs本run解析](2026-10-07_onb_200epoch_result_analysis/README.md) | 全7条件・全18,900再読込・231主指標検算。3/4/5/単体の得失、FN訂正/追加・q100・固定比率・損失分解、日別層化WAV区間と次工程。新学習/推論/重み変更なし |
| [10/6 主表確定・ONB本実行の確認準備](2026-10-06_onb_main_comparison_ready/README.md) | 現行正本・評価プロトコル、保存462行/過去252行照合、通常主表・劣化量・重み・全評価再読込・epoch/batchの自動確認、関連50テストと実main 1 epoch確認。本150 epochsは未実施 |
| [10/6 修論の次工程に向けた5モデル保存予測診断](2026-10-06_thesis_next_steps_analysis/README.md) | 全5共通FN・固定元3比率・配分下限を分け、日別q100の最後の陰性とET相対のMSEを分解。新学習・モデル推論なし |
| [10/6 通常ONBの5モデル化](2026-10-06_onb_five_model_integration/README.md) | モデルutils・共有OOF・3/4/5対照・日別指標・保存再読込・指標差と削除対照。実main 1 epoch確認済み、新通常150 epochsは未実施 |
| [10/6 元3保持の追加モデル比較](2026-10-06_fixed_three_additions/README.md) | 9候補・seed43/44・clean_only・3/4/5の研究比較。元3に対する利得と、4/単体/q100/劣化量の限界を固定 |
| [9/25 ensemble保存日付階層整理](2026-09-25_ensemble_date_layout/README.md) | 保存先を`YYYYMM/DD/条件`へ統一し、日付付き既存結果を衝突検査・件数照合付きで移行 |
| [9/25 performance_kfold主方式・固定epoch条件](2026-09-25_matched_performance_kfold/README.md) | 全18 WAVの5-fold OOF、200 epoch固定、simple equal対照、実行コマンドと監査項目。条件作成済み・未実行 |
| [9/24–25 matchedノイズ別重み本比較](2026-09-24_matched_noise_specific_weights/README.md) | noise別再fitの21条件監査、clean-only対応比較、領域別性能、innerの単一holdout制約、後継performance K-fold比較の根拠 |
| [9/24 clean固定モデル3 seed本比較](2026-09-24_clean_train_noise_inference/README.md) | cleanモデル・重みを7 SNRへ固定転送した結果、谷の消失、noise下の残差相関とONB前過大予測 |
| [9/23・学習状態保存とepoch validation](2026-09-23_step2_training_state/README.md) | 実行順2の実装。元WAV非共有epoch validation、3モデル/PCA/scaler/統合重み保存、再読込一致、同一seed再現性と全SNR配線のスモーク確認。本性能は未実行 |
| [9/23・実行順2〜7の再学習なし監査](2026-09-23_steps2-7_readonly_audit/README.md) | matched重み反実仮想、C除外94秒の全件監査、再学習が必要な残作業の識別 |
| [9/16・ピーク高さによる選別](2026-09-16_peak_height_selection/README.md) | 本人の追加説明に対応。2,300 Hz付近のピーク高さと横線、2,940線形図、閾値操作画面、学習1,649秒・テスト全1,080秒。62テスト・実3モデル2 epochs。本性能と最適閾値は未確認 |
| [9/16・別日分割と音響選別](2026-09-16_day_split_spectral_selection/README.md) | 別日指定と通常KFold性能重みの実装。先行する分位点選別での1,648秒等を保持。現行の選別は上段のピーク高さへ変更 |
| [9/16 IG積分・収束診断の修正](2026-09-16_ig_numerical_fix/README.md) | 数値誤差の再現、経路を保った積分と自動収束確認、52テスト。9/14学習済みモデルのIG再計算とは別 |
| [9/16・9/14限定の原因分析](2026-09-16_sep14_cause_analysis/analysis.md) | 本人の最新指定。学部との差、6条件の残差・重み・統合誤差分解・端点への依存・XAI。9/15 runは含めない |
| [9/16 crossfitアンサンブル追加](2026-09-16_crossfit_ensemble/README.md) | 3方式の追加実装・44テスト。研究実データの新方式での本学習・精度改善は未検証 |
| [9/15・3/22 kHz比較](2026-09-15_onb_frequency_comparison/analysis.md) | **最新の比較解析**。卒論の実図・数表、劣化抑制、ONB、XAI420例、54区間の帯域SNR、216配列再構成、金曜の論点 |
| [9/15・22 kHz全7ノイズ解析](2026-09-15_onb_22khz_noise_sweep/analysis.md) | 先行する22 kHz単独解析。35組の性能検算、ONB見逃し、帯域マスク、IG210例の不整合、図表と発表原稿 |
| [9/15研究整理スナップショット](2026-09-15_research_status_snapshot/README.md) | 9/14の完了6条件の性能・XAI、9月178 manifestsの当時の状態。過去runの比較用 |
| [9/14状況監査](2026-09-14_status_audit/README.md) | 9/3の105条件・9/8の3条件のWAV指標、ONB遷移、XAI |
| [9/14一般化・保存階層の検証](2026-09-14_generalization_and_layout/README.md) | 実装・小規模試験・metadata検査。未知日での本性能の証拠とは別 |
| [9/14最大予測値方式の削除](2026-09-14_remove_prediction_max/README.md) | 現行コードの方式整理と、過去表記の棚卸し |
| [9/14ノイズ曲線見本](2026-09-14_noise_trend_graphs/README.md) | 作図の確認。元は9/3保存結果 |
| [9/8 WAV/ONB評価の実装](2026-09-08_wav_onb_transition_evaluation_implementation.md) | 閾値・元WAV分離inner holdout・WAV/遷移評価の変更 |
| [9/2の56条件](2026-09-02_selected_log_architecture/README.md) | 旧中断スナップショット。最新の完了範囲ではない。旧innerの安全性解釈は更新済み |
| [8/30 log-power構造比較](2026-08-30_log_power_architecture_study.md) | 現行深層モデル構造を採用した根拠。当時の指標単位を保持 |
| [現行データ監査](2026-08-17_waterflow_dataset_snapshot/README.md) | 105条件の件数・実現SNR・paired noise・生成仕様 |
| [7〜8月ノイズ診断](2026-07-24_noise_shortcut_diagnostic/README.md) | データ生成修正の根拠。最終3モデルの一般化結果ではない |

## 過去の流れ

- 6月: RF・3モデル・旧重み付け評価、学習条件調整。
- 6月末〜7月上旬: CNN＋Transformer/Conformerの構造・時間軸・pooling・lr/batch比較。
- 7月: 固定統合、ノイズ/周波数別解析、XAI導入、中間発表準備。
- 8月: ノイズ生成の診断と修正、現行データ監査、log-power構造選定。
- 9月: 全条件出力、元WAVとONB評価の修正、一般化実装、説明性の再確認。

全件の所在は[既存文書一覧CSV](../docs/audits/2026-09-15_workspace_review/document_catalog.csv)。古いデータ・分割・指標の結果は、その当時の証拠として保持し、現行結果の比較表へそのまま混在させない。

## 新しい結果を保存する

[実験サマリーテンプレート](_template/run_summary.md)をコピーして、次を残す。

```text
experiments/YYYY-MM-DD_short-name/
  README.md
  run_config.yaml
  metrics.csv
  snapshot_manifest.json
```


- 収録日と解析日、コード版、データ版、条件、実際の完了範囲を記す。
- 主指標の評価単位、学習/評価分割、重み決定の範囲、ONB正解の出典を記す。
- 数値・元run相対パス・SHA-256を残す。元予測を用いた検算があれば記録する。
- `claim_safe`は保存された分割/重み監査のフラグであり、未知日への一般化やモデル選択の妥当性を保証する値ではない。
- 本文を新しいrunの値へ差し替えず、別スナップショットを作って訂正・更新先をリンクする。

大量の全画像・配列・重みは`Pool_boiling/`に置き、ここには判断を追える軽量な根拠を残す。`export_ensemble_result_snapshot.py`は主に旧chunk集計用なので、WAV主評価と混同せず、抽出器の対象と主張可否の判定範囲を確認する。

文書の更新判断・原資料との対応は[文書整合性監査](../docs/audits/2026-09-15_document_consistency/README.md)、追加結果を読む順序は[解析手順](../docs/analysis_workflow.md)を参照する。文書監査は新しい学習結果ではない。
