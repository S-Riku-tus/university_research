# 修士研究の検証・判断・開発作業の証拠台帳

2026-10-21。[研究全体の再検討](2026-10-21_research_logic_reassessment_user_feedback.md)の第5部を具体化する付表。RQ1＝全域回帰、RQ2＝ONB判定、RQ3＝改善・悪化の説明。章・節番号は同文書の**提案**であり、現LaTeXへ反映済みではない。

分類は、A＝中心的検証、B＝選択理由・妥当性を支える補助、C＝品質保証・運用、D＝探索後に主方式へ採用しなかったもの、E＝未実施候補。Dにも研究上有益な否定結果はある。分類は研究の価値の順位ではなく、今回の論文での役割である。「完了」は記載範囲の完了を意味し、未知日・実機での検証まで含まない。

調査は、実験索引・README・条件・既存解析から必要な原コード・表・予測へ遡った。10/15主系列は今回も原予測から検算したが、過去の全runを再解析したわけではない。古い計画書しか根拠がないものは完了にしない。実行中・対象外runの監視や追加評価は行っていない。

## 1. 目的・初期モデル・旧条件の探索

| ID・区分 | 検証・活動 | 実施理由 | 状態 | 確認できた事実・範囲 | RQ | 論文に必要か・章節 | 推奨配置 | 裏付け |
|---|---|---|---|---|---|---|---|---|
| H01 B | 5月の目的・評価設計 | ONB応用と説明性を研究目標へ接続 | 計画・発表済み。列挙した全実験の実施ではない | 5/22原本は早期・低誤報・説明性・条件外評価を掲げる | 全体 | 目的の検討根拠。1.3 | 内部記録。本文は課題から導く | [原発表](../../研究進捗報告/2026/522（研究計画発表）/研究進捗_柴崎陸_0522.pptx)、[原本抽出](../../experiments/2026-10-21_research_logic_audit/primary_document_extracts.md) |
| H02 B | 6/12相談後整理・6/19年度計画 | 回帰・検知・説明性の関係を明確化 | 整理・原計画あり。相談の逐語録ではない | 原計画は熱流束推定も扱うが、ONB早期検知・物理対応への要求が強い | 全体 | 1.3の整合判断に必要 | 内部記録 | [相談後整理](../../研究進捗報告/2026/612/今後の研究計画_2026-06-12.md)、[年度原本](../../研究進捗報告/2026_AE＿植木研＿8125517＿柴崎陸.docx) |
| H03 D | 6/15 RF・CNN系2種の基準比較 | 単体・単純平均の基準を得る | 当時の基準実行・集計済み | 旧入力・7/9等で回帰・判定を出力。現3モデル・現生成とは異なる | 1,2 | 現主張の直接根拠には不要 | 内部記録 | [基準](../../experiments/2026-06-15_rf_3models_baseline.md) |
| H04 C | AlexNet差替えスモーク | 類似したCNN＋Transformer2種から構成を整理 | 動作確認済み | スモークの低精度を本性能と扱わない | 1 | 通常不要 | 内部記録 | [スモーク](../../experiments/2026-06-15_alexnet_ensemble_smoke.md) |
| H05 D | 6月のlr・batch探索 | 学習不成立と構成差を区別 | 42条件の集計あり | 当時の22 kHz・旧入力の探索。現在のCNN探索とは別 | 1,2 | 調整履歴の説明に必要な場合のみ | 付録の調整履歴 | [探索集計](../../experiments/2026-06-16_parameter_tuning_summary.md) |
| H06 C/D | legacy検証fold重み | 旧結果の再現と評価情報使用の点検 | 旧比較・リーク指摘済み | 外側正解で重みを決める方式は性能主張から除外 | 1,2 | 3.3で学習側選択の理由だけ説明 | 方法の短い注意・内部記録 | [旧比較](../../experiments/2026-06-17_legacy_val_fold_analysis.md)、[126条件監査](../../experiments/2026-07-09_fixed_ensemble_full_grid_analysis.md) |
| H07 D | 軸・pooling・次元・v1/v2・canonical Conformer比較 | 深層モデルの不調と入力軸を切り分け | 複数の実行・比較記録あり。各準備メモ全ての完走は未確認 | 7/2集計にv1/v2・GAP・次元変更の性能差。現在は標準ConformerではなくCNN＋Transformer | 1,3 | 最終構造の説明に要約だけ必要 | 3.2、探索詳細は内部 | [軸仮説](../../experiments/2026-06-29_cnn_transformer_axis_compare.md)、[比較結果](../../experiments/2026-07-02_v1_v2_rf_control_analysis.md)、[構造探索](../../experiments/2026-07-02_conformer_architecture_loop.md) |
| H08 C | EarlyStoppingなしの旧条件再現準備 | 停止規則と構造の交絡を確認 | 設定・解釈計画の記録。本文書単独では成功完了を確認しない | 500 epochs再現条件を記録。改善の事実とはしない | 1 | 不要 | 内部記録 | [再現条件](../../experiments/2026-07-02_v1_no_early_stopping_repro.md) |
| H09 D | v2 GAPのlr/batch・候補確認 | 使用可能な深層基準を得る | 30条件完了の記録あり。後続全グリッド案は別 | 選択条件と他日・他雑音の性能は別問題 | 1 | 現行パラメータの直接証拠ではない | 付録候補・内部 | [探索](../../experiments/2026-07-06_v2gap_lrbs_tuning.md)、[完了後候補確認](../../experiments/2026-07-06_v2gap_5fold_candidates.md) |
| H10 D | 7月126条件・AlexNet84条件・22 kHz再比較 | モデル・帯域・雑音の性能を広く把握 | 当時の集計済み | 126条件統合はlegacy重み。AlexNet探索も旧生成。現主結果へ数値を混ぜない | 1,2 | 広い探索を経た事実は内部判断に有用 | 内部、必要時だけ履歴付録 | [126条件](../../experiments/2026-07-09_fixed_ensemble_full_grid_analysis.md)、[AlexNet](../../experiments/2026-07-10_alexnet_22khz_noise_tuning_analysis.md)、[7/13結果](../../experiments/2026-07-15_20260713_fixed_ensemble_22khz_analysis.md) |

## 2. 雑音・入力・説明計算の基盤

| ID・区分 | 検証・活動 | 実施理由 | 状態 | 確認できた事実・範囲 | RQ | 論文に必要か・章節 | 推奨配置 | 裏付け |
|---|---|---|---|---|---|---|---|---|
| N01 B/C | P0：相対雑音・noise-only・固定負荷対照 | 強雑音時の高精度の情報経路を識別 | 診断済み | 沸騰波形を除いても旧相対付加雑音が振幅手掛かりを持つ。単なるプログラム動作点検を越える評価設計の根拠 | 1,3 | 固定基準の必要性に必要。3.1/3.4 | 本文短節、詳細付録 | [P0原集計](../../experiments/2026-07-24_noise_shortcut_diagnostic/README.md) |
| N02 C | 旧誤加算版の除外・生成監査 | 無効な入力を主結果から除く | 修正・現版生成・構造確認済み | `y_power + scaled_noise`の旧版と相対RMS方式を区別。現版は別データ | 全体 | 生成版の明記は必要 | 3.1、修正履歴は内部 | [データ索引](../data_inventory.md)、[現版監査](../../experiments/2026-08-17_waterflow_dataset_snapshot/README.md) |
| N03 B | 固定基準RMS・paired noise | ラベルによらない共通負荷を与える | 生成・manifest確認済み | 6/11・6/18・7/9計49 WAVの共通基準。主モデル学習36 WAVとは別 | 1,2 | 必須。2.3、3.4 | 本文 | [仕様](../../configs/datasets/waterflow_20260817_1s.yaml) |
| N04 B/D | log-power・構造候補比較 | 微小なraw powerでの最適化・表現を改善 | 8/30探索・採用判断済み | log圧縮と時間tokenのbalanced構造を採用。候補ごとの実施範囲あり | 1,3 | 採用構造説明に必要 | 3.1/3.2、詳細付録 | [構造研究](../../experiments/2026-08-30_log_power_architecture_study.md) |
| N05 D | 9/2 log構造の広条件run | 改訂構造を広条件へ適用 | 105予定中56条件で中断 | 旧inner混在の監査注記あり。完了105とはしない | 1,2 | 現主結果に不要 | 内部記録 | [保存スナップショット](../../experiments/2026-09-02_selected_log_architecture/README.md) |
| N06 B | 9/14–15の3/22 kHz・実現帯域SNR | 雑音曲線・周波数依存を調べる | 指定対象の解析済み | 帯域内負荷・resize・再学習の条件差を確認。旧IG不一致も発見 | 1,3 | 帯域選択の補助。現性能の表に混ぜない | 3.1/5.5、付録 | [3/22 kHz](../../experiments/2026-09-15_onb_frequency_comparison/analysis.md) |
| N07 C | IG積分・経路・決定論GPU修正 | 帰属計算の数値整合を確保 | 実装修正・限定検査済み | 修正前地図を修正済み扱いしない。本学習全地図の採用合格とは別 | 3 | 3.6の採用基準。修正を独立成果にしない | 付録・内部 | [積分](../../experiments/2026-09-16_ig_numerical_fix/README.md)、[経路/ONB監査](../../experiments/2026-09-24_selection_onb_ig_review/README.md)、[GPU](../../experiments/2026-09-25_ig_deterministic_gpu_fix/README.md) |
| N08 C | 学習状態・PCA/scaler・全予測保存 | 同じfitへの雑音評価・説明を可能にする | 実装後、後続本runで保存・再読込済み | 9/23は1 epoch配線確認。9/24以降の本比較と区別 | 全体 | 3.3再現性の短い説明 | 本文短記、詳細内部 | [実装](../../experiments/2026-09-23_step2_training_state/README.md)、[本比較](../../experiments/2026-09-24_clean_train_noise_inference/README.md) |
| N09 C | PCAメモリ配置・保存再読込修正 | RF再現差を抑える | 10/5修正・検証済み | 内部分割変更との新旧差には本修正も含まれる | 1,3 | 新旧性能差を分割だけへ帰属させないため必要 | 付録・方法短記 | [修正](../../experiments/2026-10-05_rf_reload_pca_layout_fix/README.md)、[対応結果](../../experiments/2026-10-06_chunk_clean_baseline_analysis/README.md) |

## 3. 評価分割・学習方針・入力条件の識別比較

| ID・区分 | 検証・活動 | 実施理由 | 状態 | 確認できた事実・範囲 | RQ | 論文に必要か・章節 | 推奨配置 | 裏付け |
|---|---|---|---|---|---|---|---|---|
| V01 B | ONB値・自動直線性喪失候補の区別 | 正解と別の診断点の混同を解く | 9/24採用値確定済み | 原観察根拠はなお未確認。値の採用確認を未完了へ戻さない | 2 | 必須。2.2 | 本文、監査詳細付録 | [確定記録](../../experiments/2026-09-24_selection_onb_ig_review/README.md) |
| V02 B/D | 日指定・双方向別日・ピーク選別 | 日変動と弱音選別を調べる | 実装・双方向評価済み | 選別が開始段階を多く除外。主系列は選別なしへ進展 | 1,2 | 選別不採用の理由 | 付録 | [双方向](../../experiments/2026-09-17_bidirectional_onb_ensemble_xai_analysis/README.md) |
| V03 B | A：matched谷型の標本別診断 | 統合重み・領域・単体変動を分ける | A1–A7解析済み | 等重みでも回復が残り、重みだけでは説明不能 | 1,3 | clean_onlyを選ぶ理由に必要 | 3.4、詳細付録 | [保存予測診断](../../experiments/2026-09-19_noise_recovery_review/README.md) |
| V04 B | B：固定clean対照・matched代表seed | 再学習変動と入力外乱を分ける | 比較済み | 固定cleanの代表4条件は劣化、matched回復の統合差はseedで不一致 | 1,2,3 | 既存の識別比較。再提案不要 | 付録 | [固定clean・B診断](../../experiments/2026-09-19_b_clean_only/README.md)、[seed別指標](../../experiments/2026-09-19_b_clean_only/matched_seed_metrics.csv) |
| V05 D/B | C：選別あり/なし・3 seed | 弱音除外が別日予測を改善するか | 実施・判断済み | ONB前の改善と近傍悪化が併存。選別の標準採用根拠なし | 1,2,3 | 不採用理由の補助 | 付録、必要時5.5 | [選別対照](../../experiments/2026-09-20_c_selection/README.md) |
| V06 B | 9/24固定clean・3 seed×7条件 | 同じfitへの雑音負荷を確かめる | 21条件完了 | 当時22 kHzでは統合が雑音時にRFを下回る。現3 kHz結果へ上書きしない | 1,2 | 学習方針の根拠 | 付録 | [固定転送](../../experiments/2026-09-24_clean_train_noise_inference/README.md) |
| V07 B | matched対clean_only・3 seed | 雑音既知学習と未学習負荷の差 | 6 run・42条件完了 | clean予測一致。matchedのnoise平均でperformanceのRF優位は安定せず | 1,2 | 実施済み対照。最新5 matchedではない | 3.4短記＋付録 | [対応比較](../../experiments/2026-09-26_clean_matched_performance_kfold_3seed/README.md) |
| V08 B/D | 7/9追加・3日leave-one-day-out | データ追加と日一般化を調べる | 3方向完了 | 当時条件で7/9転送が大きく低下。データ追加＝改善ではない | 1,2 | 一般化の限界に必要なら引用 | 5.5/付録 | [3日比較](../../experiments/2026-09-27_leave_one_day_out_clean_analysis/README.md) |
| V09 B | 2日双方向・5帯域/低帯域3 seed | 周波数選択と日方向依存を確認 | 対象条件の完了記録あり | 低帯域が有利な範囲、最良単体への非一律優位を確認 | 1,2 | 3 kHz選択の補助。150 epochs等を明記 | 3.1/付録 | [低帯域](../../experiments/2026-09-28_bidirectional_lowfreq_3seed/README.md)、[逆方向5帯域](../../experiments/2026-09-28_two_day_reverse_5freq_3seed/README.md) |
| V10 B | 元WAV holdoutの日内clean/matched | 同日未知録音と学習雑音の差 | 4 run・28条件完了 | 6/11テストに近傍WAVなし。高い判定値だけで近傍性能を主張不能 | 1,2 | 分割と評価対象の説明 | 付録 | [日内WAV比較](../../experiments/2026-09-29_within_day_clean_matched_seed42/README.md) |
| V11 B | q100と評価段階の監査 | 100%到達点の意味・解像度を確認 | 定義・既存照合済み | 全陽性到達は誤報0を含まない。評価にない段階は測れない | 2 | 必須。3.5 | 本文、旧例は付録 | [定義](../../experiments/2026-09-29_q100_and_pooled_holdout/README.md)、[定義照合完了](2026-10-14_completed_checks_noise_history_and_future_ensembles.md) |
| V12 B | 単日→統合within-WAV chunk | 全段階を学習し既知WAV内の構成差を調べる | 生成・単日/統合比較済み | 各45/15。WAV共有を許す対象への変更であり一般化改善ではない | 1,2 | 必須。2.3/3.3 | 本文の評価範囲、比較詳細付録 | [単日](../../experiments/2026-09-29_within_wav_chunk_single_day_analysis/README.md)、[統合](../../experiments/2026-09-29_within_wav_chunk_combined/README.md) |
| V13 B/D | 7/9同WAV内対照 | 特異な音響―q対応が同日学習でも残るか | 7条件解析済み | 近傍過小予測・高雑音bias変化が残る。6月主学習へ混ぜない | 1,2,3 | 必要なら一般化限界の対照 | 付録 | [7/9](../../experiments/2026-09-29_within_wav_chunk_0709_analysis/README.md) |
| V14 D/B | 0.5秒対1秒 | 入力時間長の利得を確認 | 対応比較済み | 現条件で全域悪化、近傍・誤報との交換。1秒維持 | 1,2 | 選択理由の補助 | 3.1短記、付録 | [時間長比較](../../experiments/2026-09-29_within_wav_chunk_05s_analysis/README.md) |
| V15 D/B | ONB保護付きピーク選別 | 開始層を消さず強雑音誤差を減らす | 2 run・14条件完了 | 強雑音全域改善とclean/近傍悪化、q100不変。主採用せず | 1,2 | 完了した不採用対照 | 付録 | [保護付き結果](../../experiments/2026-09-29_onb_protected_selection_analysis/README.md) |
| V16 B | 3/22 kHzの学習側OOF探索 | 外側を使わずモデルを調整 | 各190候補完了 | RF135/CNN25/Alex30。全域最小近傍の均衡選択も含み、全て純RMSE最小ではない | 1,2 | 公平性の記述に必要。3.3 | 本文要約・探索表付録 | [探索](../../experiments/2026-09-30_3khz_22khz_oof_tuning_analysis/README.md) |
| V17 B | 調整後3/22 kHz外側評価 | clean OOF順位が雑音へ移るか | 7条件ずつ解析済み | 22 kHz clean優位と3 kHz雑音時優位。複数設定差を含む | 1,2 | 帯域選択の根拠 | 3.1、付録 | [外側比較](../../experiments/2026-10-01_3khz_22khz_tuned_outer_comparison/README.md) |
| V18 C/B | 内部WAV/chunk支持と分割監査 | 全段階欠落と予測対象を明確化 | 監査・chunk基準本比較済み | WAV独立と全段階支持は当該データでは両立しない。時間block案は分割監査のみ | 1,2 | 必須。3.3、制約5.5 | 本文短記＋付録 | [支持監査](../../experiments/2026-10-05_onb_endpoint_and_grouped_fold_review/README.md)、[chunk結果](../../experiments/2026-10-06_chunk_clean_baseline_analysis/README.md) |

## 4. 統合・追加モデル・主性能

| ID・区分 | 検証・活動 | 実施理由 | 状態 | 確認できた事実・範囲 | RQ | 論文に必要か・章節 | 推奨配置 | 裏付け |
|---|---|---|---|---|---|---|---|---|
| M01 B | 旧3の残差・予測範囲診断 | 重みだけで補える範囲を見極める | 保存OOF/外側診断済み | 凸結合の限界を示すoracle診断。実装可能な選択器の成績ではない | 3 | 追加モデルの理由 | 3.2/5.2、詳細付録 | [実現可能性](../../experiments/2026-10-03_error_complementarity_feasibility/README.md) |
| M02 D | 帯域SVR・絶対power除去 | 既存3共通失敗を異なる表現で救う | nested学習・評価済み | 訂正と追加誤報・雑音転送崩れ。主採用せず | 1,2,3 | HGB選択までの補助 | 付録・内部 | [OOF予備比較](../../experiments/2026-10-04_training_oof_diversity_diagnosis/README.md) |
| M03 B/D | 基礎10/周波数34/時間46、SVR/HGB | 入力と学習器の寄与を調べる | 予備比較済み | 時間46も順序は保持しない。HGB34を主候補へ | 1,2,3 | 追加入力選択の根拠 | 3.2短記＋付録 | [予備比較](../../experiments/2026-10-06_chunk_complementary_model_pilots/README.md)、[判断集計](../../experiments/2026-10-06_model_adoption_decision/README.md) |
| M04 B | HGB追加seed43/44・失敗/重要度 | 補完と分割感度を確かめる | 150 epochsの追検証済み | clean改善は再現、強雑音単体FPは分割で大きく異なる。現200 epochs3反復ではない | 1,2,3 | 補助再現性 | 付録、5.5短記 | [追検証](../../experiments/2026-10-06_hgb_followup_validation/README.md) |
| M05 D/B | 等重み・performance・MSE・Ridge・多様性・雑音OOF・部分集合 | 利得の取り込み方と代償を比較 | 当時の対照実施済み | Ridge・雑音露出は別条件。多様性項の一例を概念全体の否定にしない | 1,2,3 | MSE採用理由に要約 | 3.3、詳細付録 | [統合対照](../../experiments/2026-10-06_hgb_followup_validation/README.md) |
| M06 B | 元3保持・9追加候補・nested選択 | 入力/学習器を増やす利得を検証 | seed43/44比較済み | ExtraTreesが両seedのclean OOFで選択。RF34/GB/XGB/Ridge/PLS/KRR等も検討済み | 1,2 | モデル選択理由に必要 | 3.2、候補一覧付録 | [9候補](../../experiments/2026-10-06_fixed_three_additions/README.md) |
| M07 B/C | 3/4/5・各メンバー削除・通常組込み | 全5の必要性と実装一致を確認 | 比較・組込み・再構成済み | 元3保持は設計方針。RF小重み、5の全指標最良/全員必須は示さない | 1,2,3 | 3.2/3.3、4.2、5.2 | 本文の構成比較＋付録 | [組込み](../../experiments/2026-10-06_onb_five_model_integration/README.md)、[採用理由](../notes/2026-10-06_five_model_rationale_and_next_steps.md) |
| M08 A | 10/15主系列：5単体＋6統合×7条件 | 全域回帰・判定を同一保存予測で比較 | 全7条件・解析完了、今回再検算 | 元3 MSE比で全7条件の全域RMSE・総FN改善。単体/4構成への一律優位ではない | 1,2 | 中心。4.1–4.5 | 本文 | [主系列](../../experiments/2026-10-15_clean_only_result_analysis/README.md)、[今回検算](../../experiments/2026-10-21_research_logic_audit/verification.json) |
| M09 A | 領域別・日別・FN/FP・開始段階・q100 | 集約改善と境界の違いを明確化 | 保存表・今回照合済み | 近傍30、開始段階8/30→0/30。全域改善と異なる。日別q100の一部改善 | 1,2 | 中心。4.3/4.4 | 本文 | [主指標](../../experiments/2026-10-15_clean_only_result_analysis/pooled_metrics.csv)、[q100](../../experiments/2026-10-15_clean_only_result_analysis/q100_by_source_day.csv) |
| M10 A | 絶対誤差・cleanからの劣化量 | 雑音下性能と劣化抑制を分ける | 10/14分析済み、10/15主系列でも出力済み | 5は元3 MSE比で全6雑音の増加量縮小を示さない。再計算必須の未完了課題ではない | 1 | 中心。4.3 | 本文 | [既存監査](../../experiments/2026-10-14_2025_noise_degradation_review/README.md)、[主系列劣化量](../../experiments/2026-10-15_clean_only_result_analysis/noise_degradation.csv) |
| M11 C | 主表・再読込・統合再構成・表示検算 | 数値と主張の信頼性を確保 | 既存と今回の照合完了 | 品質保証は重要だが独立した精度改善成果ではない | 全体 | 方法に短記 | 付録・監査記録 | [10/15検算](../../experiments/2026-10-15_clean_only_result_analysis/verification.json)、[今回](../../experiments/2026-10-21_research_logic_audit/verification.json) |

## 5. 誤差要因・説明性・入力情報

| ID・区分 | 検証・活動 | 実施理由 | 状態 | 確認できた事実・範囲 | RQ | 論文に必要か・章節 | 推奨配置 | 裏付け |
|---|---|---|---|---|---|---|---|---|
| X01 B/C | TreeSHAP、IG、Grad-CAM、マスク、deletion等 | 構造に合う説明計算を用意 | 実装・過去出力あり。個々の採用判定は別 | RFはPCA成分、深層は入力/中間層で説明対象が異なる | 3 | 3.6に採用理由 | 方法短記・付録 | [実装記録](../../experiments/2026-07-06_explainability_implementation.md)、[選択理由](2026-07-16_model_specific_xai_selection_and_implementation.md) |
| X02 B/D | D1–D3：成功/失敗と説明性の対応 | 説明図の信頼性と改良判断を検証 | 9/22対象範囲終了 | TreeSHAP整合、IG採用不足、深層旧モデル再現不能、帯域除去不採用まで判断済み | 3 | 新手法追加の前に既存限界を示す | 付録、5.4短記 | [D1–D3](../../experiments/2026-09-22_d_explainability/README.md) |
| X03 A | 同方向/逆符号・領域別MSE分解 | 第4章の利得がどこから生じたか説明 | 10/14完了、10/15主系列で再確認 | −20の利得87.15%は3ブロック同方向群。物理因果率ではない | 3 | 中心。5.1/5.2 | 本文 | [接続分析](../../experiments/2026-10-14_error_feature_linkage/README.md)、[主系列方向表](../../experiments/2026-10-15_clean_only_result_analysis/error_direction_contributions.csv) |
| X04 A | 共通gain・同幅帯域減衰・時間シャッフル | 手法固有尺度を避け依存差を調べる | 10/7第1段階完了 | HGB/ETは順序不変、CNN系は時間配置に反応。帯域・強度には要約モデルも敏感 | 3 | 中心。5.3 | 本文 | [第1段階](../../experiments/2026-10-07_onb_input_and_integration/README.md) |
| X05 B | 元3内部比率自由化 | 保持制約の損失を識別 | 第2段階完了 | clean改善、−20追加FN、q100不変。通常方式変更せず | 1,2,3 | 制約の補助対照 | 5.2/付録 | [第2段階](../../experiments/2026-10-07_onb_input_and_integration/README.md) |
| X06 A | 34→power5/相対29の除去・再学習 | 固定モデル感度と保持情報を区別 | 第3段階16 fit完了 | 現学習器・固定条件で全34の回帰優位。powerのみでは強雑音FP増加 | 3 | 中心。5.3 | 本文 | [第3段階](../../experiments/2026-10-07_onb_input_and_integration/README.md) |
| X07 A/B | 入力変更×固定重み/再fit重み | 表現差と配分差の交絡を分ける | 保存予測対照・恒等式検算完了 | 相対29 HGBのq100改善は配分変更を伴う。入力のみの効果ではない | 3 | 5.3の解釈に必要 | 本文または補足表 | [分解表](../../experiments/2026-10-07_onb_input_and_integration/stage3/input_weight_loss_decomposition.csv) |
| X08 A/B | FNの単体判定・現制約の上限診断 | ONB目的統合の前に変更可能範囲を確かめる | 10/15既存分類＋今回軽量照合完了 | clean FN27＝11全陰性＋16一部陽性。ただし固定元3比率・下限下で1件ずつ救済可能な上限は8件。−20開始30件は0 | 2,3 | 5.2、追加計画の根拠 | 本文短記・詳細付録 | [元失敗表](../../experiments/2026-10-15_clean_only_result_analysis/five_failure_cases.csv)、[今回上限](../../experiments/2026-10-21_research_logic_audit/onb_failure_feasibility.csv) |

## 6. 新データ・進行中・候補

| ID・区分 | 検証・活動 | 実施理由 | 状態 | 確認できた事実・範囲 | RQ | 論文に必要か・章節 | 推奨配置 | 裏付け |
|---|---|---|---|---|---|---|---|---|
| F01 C | 2026/10/7原データ配置・熱計算・生成 | 新日の解析入力を整える | 配置/生成/検算済み | 主系列とは別RMS・別ラベル。生成済みを未知日検証済みとしない | 全体 | 主データを拡張しなければ不要 | 内部記録 | [入力監査](../../experiments/2026-10-09_new_experiment_generation_audit/verification.json)、[熱流束対応](../../experiments/2026-10-07_heatflux_stage_alignment/README.md) |
| F02 D/B | 新2026日のwithin-WAV評価 | 新日の既知録音内で現構成を確認 | run114213解析済み。現在の主対象外 | 新日で再学習した評価。旧6月fitの未知日適用ではない。過去の「新日未評価」は履歴 | 1,2 | 今回は中心へ追加しない | 内部、将来採用時付録 | [完了・範囲](../../experiments/2026-10-14_new_day_within_wav_result_analysis/interpretation.json) |
| F03 C | 2024年入力再生成・整合確認 | 旧記録の再利用基盤を整える | 完了記録あり | 入力出力の完成と現方式での学習評価は別 | 全体 | 現主張に不要 | 内部記録 | [生成完了](../../experiments/2026-10-16_2024_generation_complete/verification.json) |
| F04 B（進行中） | CNN3：AlexNet/ResNet50/VGG16 | 同じ3モデル数で構成を対照化 | 実装済み、探索一部完了。外側比較未完了 | 今回17:27 JST保存監査はAlex36/36、ResNet24/36、VGG0/36。停止記録より1候補進展 | 1,2、結果次第3 | 優先補強。4.2 | 完了すれば本文、未完なら欠測のまま省略可 | [設計・再開履歴](2026-10-19_cnn_only_three_comparison_and_writing_policy.md)、[現在保存監査](../../experiments/2026-10-21_research_logic_audit/verification.json) |
| F05 E | 全域MSE対ONB目的統合 | 回帰/判定の交換を識別 | 計画のみ。今回重みfitなし | 制約上の上限診断を先に完了。固定制約では−20開始層を救えない | 1,2,3 | 条件付き補強。成功は完成条件ではない | 採用後3.3/4.4/5.2 | [限定設計](2026-10-21_research_logic_reassessment_user_feedback.md)第7部 |
| F06 E | 未知日・独立水流音・追加ONB反復 | 外部妥当性と独立事例を増やす | 今回未実施 | 既存別日比較はあるが、現5の未知条件性能の証明とは別 | 1,2 | 強い一般化を主張するときのみ必須 | 5.5/6.3展望 | [適用範囲と条件](2026-10-21_research_logic_reassessment_user_feedback.md)第6・7部 |
| F07 E | 入力依存重み・学習条件強化 | 外乱や局所誤りへ適応 | 未実施候補 | 固定重み変更や既存Ridgeと別。真のSNR/qを推論入力にしない | 1,2,3 | 余裕がある場合。中核に不要 | 展望 | [将来方式の区別](2026-10-14_completed_checks_noise_history_and_future_ensembles.md) |
| F08 E | 物理音源を分ける同期観察・条件変更 | 入力依存を物理機構へ結びつける | 現系列での証明未了 | 既存感度/帰属だけで代替できない | 3 | 物理因果を断定する場合のみ必須 | 5.4/6.3 | [説明性限界](../../experiments/2026-10-14_error_feature_linkage/README.md) |
| F09 C | 発表・目次・LaTeX試作・図表検査 | 研究の筋と証拠を文章化 | 原稿は存在。研究方針・提出稿の確定ではない | 6章の大枠は利用可能。目的・RQの再検討結果は未反映 | 全体 | 今後の執筆作業として必須 | 内部運用、本文全章へ順次反映 | [現草稿](../../LaTeX/master_thesis/main.tex)、[残る事項](../thesis/2026-10-19_latex_manuscript_and_remaining_work.md) |

## この台帳からの判断

中心に据えるのはM08–M10とX03–X08である。V群は評価対象を選んだ理由と適用範囲、N群は入力・雑音・再現性の根拠を支える。旧方式・不採用候補の全探索表を本文へ並べる必要はない。

既に終わっている雑音生成診断、matched/clean_only対照、q100定義照合、特徴除去再学習、元3比率自由化、固定重み対照、主表検算を、新しい必須実験として提案しない。追加すべき中心は、**既存の研究成果を目的へ接続する文章と出典の確定**である。CNNとONB目的統合は、それぞれ構成選択と目的関数の問いを補う別の作業として扱う。
