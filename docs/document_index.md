# 文書案内と更新ルール

更新日: 2026-10-06。**現在の状態は一か所で更新し、過去の数値と判断は日付付き記録に残す。**

## 読む順序と各文書の役割

| 目的 | 読む文書 | 更新方法 |
|---|---|---|
| 次の作業を始める | [研究の現在地](research_status.md) | 状態・本人の希望・次の一手を更新する唯一の入口 |
| 新chunk条件の追加モデル・入力・統合の実行結果を見る | [10/6 SVR/HGB予備比較](../experiments/2026-10-06_chunk_complementary_model_pilots/README.md) | 18候補、固定29統合、同じ外側7条件、周波数/時間入力の固定対照、q100/誤報、残る6/11陰性、保存モデルと検算を記録 |
| chunk内部検証clean_onlyの完了結果と次工程を見る | [10/6新旧基準比較](../experiments/2026-10-06_chunk_clean_baseline_analysis/README.md) | 7条件完了、同一540、q100不変、OOF支持と最後の共通陰性、重み/単体差、固定組合せ、候補比較の条件案を記録 |
| RF保存再読込の予測不一致と修正を見る | [10/5 PCA配置の数値互換性修正](../experiments/2026-10-05_rf_reload_pca_layout_fix/README.md) | 本人停止runで差を再現、共通PCAのC配置、変換版2のhash記録、実RF再検証差0、49テストと再実行前の状態を記録 |
| ONB主コードの実行準備と現在の条件を見る | [10/5 clean_only実行準備](../experiments/2026-10-05_chunk_clean_baseline_ready/README.md) | clean_only・学習状態保存、内部分割の実行転送修正、9テスト、7条件のID対応と旧テスト維持、本学習未実施を記録 |
| chunk分割の実装後に進む順序と判断条件を見る | [10/5実装後の次工程](research_plan/2026-10-05_after_chunk_split_next_steps.md) | 固定値clean_only基準、q100の最後の陰性による分岐、旧候補の再検討、入力とモデルの対照、統合・XAI・再現性の順序を提案として整理 |
| 本人指定のシャッフルありchunk分割と実装確認を見る | [10/5内部chunk KFoldへの切替](../experiments/2026-10-05_shuffled_chunk_internal_validation/README.md) | 外側ID維持、内部performance/tuningの切替、46テスト、全段階支持、旧結果識別、未出力の本比較を記録 |
| 全熱流束支持の内部fold・早い100%判定・追加候補の結果を見る | [10/5内部fold・q100・候補比較](../experiments/2026-10-05_onb_endpoint_and_grouped_fold_review/README.md) | WAV分離の限界、順位均衡と既知WAV時間block、q100優先の再評価、ExtraTrees/HGB予備比較、固定統合と次の入力表現を記録 |
| 学習OOFの共通失敗と第4モデル候補の予備比較を見る | [10/4 OOF診断と帯域特徴SVR](../experiments/2026-10-04_training_oof_diversity_diagnosis/README.md) | 内部ONB配置、学習chunkの特徴、nested SVR、訂正と追加誤り、noise転送、絶対power除去の不採用、次の配置対照を固定 |
| 最新課題・説明性の選択理由・追加モデルの役割を確認する | [10/4現行結果と追加モデルの検討](research_plan/2026-10-04_current_challenges_xai_and_model_diversity.md) | clean_only主軸、両系列の共通失敗、逆方向出力と有用な補完の違い、現行log-power IG、XAI品質、追加モデルから統合へ進む条件を整理 |
| 採用条件matchedとclean_onlyの対応結果を見る | [10/3 matched対応解析](../experiments/2026-10-03_tuned_within_wav_matched_comparison/README.md) | 14条件のID・無雑音一致、全域/ONB/q100、RF適応、共通陰性、OOF重みと外側順位、次の比較を固定 |
| matched対照と誤り方を使う統合の次工程を考える | [10/3現在地・次工程](research_plan/2026-10-03_matched_and_error_aware_ensemble_next_steps.md) | 本人の最新希望、輪講論文との接続、共通条件matched、学習OOFでの選択、推論情報・テスト再利用の制約を記録 |
| 既存モデルで誤りを補完できる可能性と限界を見る | [10/3補完可能性診断](../experiments/2026-10-03_error_complementarity_feasibility/README.md) | 外側2条件・clean OOFの誤差符号、凸結合の仮想下限、ONB見逃しの補完余地、OOF/外側の違いを記録 |
| 外側評価を2方式に整理した理由と確認を見る | [10/3評価方式整理](../experiments/2026-10-03_remove_within_day/README.md) | 日内WAV分割の廃止、内部WAV検証の維持、旧設定の扱い、実データ540テストID一致と60テストを記録 |
| 現在の採用条件で誤差・誤判定が多い場所を見る | [10/2採用条件の誤差分解](../experiments/2026-10-02_current_adopted_condition_error_profile/README.md) | 全域とONBの最良の違い、熱流束領域別二乗誤差、段階別陽性数、単体からの訂正/追加、q100不変の意味を記録 |
| 現行の修論目次と次回相談までの準備を見る | [10/2目次・2週間の計画](research_plan/2026-10-02_master_thesis_outline_and_two_week_plan.md) | 学部論文との接続、7章案、章ごとの材料/不足、追加検証の分岐、仮目標10/16の準備工程を記録 |
| 修論に使う主張・採否理由・根拠を蓄積する | [10/2主張・根拠・判断台帳](thesis/2026-10-02_claims_evidence_and_decisions.md) | 本人の目標と確認済み結論、主条件の理由、q100と卒論のCHF正規化指標の違い、説明性の役割・未検証部分を記録 |
| 後期目標・教授の助言・実施状況と現在の優先順位を整理する | [10/2目標・助言・進捗監査](research_plan/2026-10-02_second_semester_goal_progress_and_professor_feedback_audit.md) | 原報告と教授メモの確認範囲、9/18計画の4本柱、noise精度逆転への回答、未完了部分、教授への説明順を記録 |
| 3 kHz採用値の外側結果と22 kHzとの最終比較を見る | [3/22 kHz外側対応比較](../experiments/2026-10-01_3khz_22khz_tuned_outer_comparison/README.md) | 完了性、日別ONB、入力―出力傾向、FPR・Recall・q100、従来3 kHzとの差、3 kHz固定判断を記録 |
| 最大周波数別パラメータ設定と現在の3 kHz条件を見る | [周波数別config実装](../experiments/2026-10-01_frequency_specific_parameter_config/README.md) | 3/22 kHz採用値、自動通常run／OOF探索切替、複数周波数時の動作、検証結果を固定 |
| 22 kHz採用値の外側7 noise結果を見る | [22 kHz外側評価](../experiments/2026-10-01_22khz_tuned_outer_analysis/README.md) | 完了性、日別ONB、FPR、Recall、q100、cleanでの統合効果、noise時の正biasと次のmatched比較を固定 |
| 3/22 kHzのOOFチューニング結果と採用値を見る | [3/22 kHz OOF探索解析](../experiments/2026-09-30_3khz_22khz_oof_tuning_analysis/README.md) | 完了性、2%候補、日別ONB、推奨パラメータ、保存OOF統合、22 kHz外側評価への判断を固定 |
| ONB保護付きピーク選別の本結果と採否を見る | [選別結果解析](../experiments/2026-09-29_onb_protected_selection_analysis/README.md) | 選別なしとの対応比較、日別ONB指標、q100、WAV別誤差、採否判断を固定 |
| ONBを残せるピーク閾値と限界を見る | [ONB保持制約の閾値監査](../experiments/2026-09-29_peak_threshold_preservation/README.md) | 外側学習だけの分布、共通・日別閾値の不成立、ONB保護＋上側1e-9案を固定 |
| 統合データのピーク選別と学習側OOFチューニング実装を見る | [選別・チューニング実装記録](../experiments/2026-09-29_within_wav_selection_and_oof_tuning/README.md) | 元実験日への特徴対応、選別数の分母、外側テスト非使用の18候補探索、実行順を固定 |
| 0.5秒と1秒の比較結果・採用判断を見る | [0.5秒対照解析](../experiments/2026-09-29_within_wav_chunk_05s_analysis/README.md) | 完了性、全域・ONB指標、時間位置の非対応、対応小標本、1秒採用と次工程を固定 |
| 修論で主張する評価範囲と0.5秒・チューニングの順序を見る | [評価範囲と次工程](research_plan/2026-09-29_scope_chunk_length_and_tuning.md) | 未知WAV・未知日の後順位化、7/9の補助扱い、対応時間での0.5秒比較、選択後チューニングを固定 |
| 7/9単日within_wav_chunkの結果を見る | [7/9解析](../experiments/2026-09-29_within_wav_chunk_0709_analysis/README.md) | noise trendの見かけの回復、571.7～720.7 kW/m²の過小予測、q100、統合重み、6月との差を固定 |
| 6/11＋6/18統合within_wav_chunkの結果を見る | [統合within_wav_chunk解析](../experiments/2026-09-29_within_wav_chunk_combined_analysis/README.md) | 単日runとの対応比較、日別ONB、noise trend、低熱流束とONB近傍のトレードオフ、次のseed再現を固定 |
| 6/11・6/18単日within_wav_chunkの結果を見る | [単日within_wav_chunk解析](../experiments/2026-09-29_within_wav_chunk_single_day_analysis/README.md) | 完了性、noise劣化、共通75 chunkでのWAV holdout比較、重み、q100、次の統合runを固定 |
| 6/11＋6/18統合データと各WAV内chunk分割を確認する | [統合データとWAV内chunk holdout](../experiments/2026-09-29_within_wav_chunk_combined/README.md) | 旧方式との違い、データ来歴、平均ONB、分割検証、実行条件を固定 |
| 現在の研究問題・改善対象・実験の重複を確認する | [9/29問題定義と次の優先順位](research_plan/2026-09-29_problem_definition_and_next_priority.md) | clean_only／matchedの主従、改善指標、split追加を後順位にした理由、チューニングの目的を整理 |
| 100%分類可能熱流束と当初の6/11＋6/18 pool案を確認する | [q100監査とpooled holdout設計](../experiments/2026-09-29_q100_and_pooled_holdout/README.md) | 現行140 cellの再集計、FPRとの役割分担、後続方針に置換された当初案を記録 |
| 6/11・6/18日内holdoutのclean_only／matched結果を見る | [日内4 run解析](../experiments/2026-09-29_within_day_clean_matched_seed42/README.md) | 同一WAV比較、日別noise効果、ONBトレードオフ、performance重み、split再現の次工程を固定 |
| 3 kHz matched双方向の結果と次の判断を見る | [matched双方向seed 42解析](../experiments/2026-09-28_3khz_matched_bidirectional_seed42/README.md) | clean_only対応比較、方向依存、ONBトレードオフ、performance重み、次の日内比較を固定 |
| 3 kHz固定、matched追加、チューニング、日内/別日評価を進める | [9/28本人方針と評価設計](research_plan/2026-09-28_3khz_matched_tuning_and_evaluation.md) | performance継続、WAV固定holdoutの設定、チューニング候補と学習側選択、次の6条件を記録 |
| 6/11・6/18双方向、3/5 kHz、3 seedの結論を見る | [双方向低周波3 seed解析](../experiments/2026-09-28_bidirectional_lowfreq_3seed/README.md) | 84条件、方向別性能、3 kHz等重みの採用根拠、performance重み移転、次の帯域遮蔽を固定 |
| 順方向3/5 kHz seed 42結果を見る | [6/11→6/18低周波解析](../experiments/2026-09-28_forward_lowfreq_seed42/README.md) | 低周波優位の方向再現、noiseによるbias相殺、統合の成立範囲、次のseedを固定 |
| 逆方向5周波数×3 seed結果を見る | [6/18→6/11・5周波数解析](../experiments/2026-09-28_two_day_reverse_5freq_3seed/README.md) | 3–5 kHzのnoise耐性、3 kHz統合の成立範囲、22 kHz崩壊、重み移転、次の方向再現を固定 |
| 6/11・6/18だけの双方向結果を見る | [2日双方向clean学習解析](../experiments/2026-09-27_two_day_bidirectional_clean_analysis/README.md) | clean転送、noise方向差、bias相殺、重み移転、次のseed再現を固定 |
| 3日leave-one-day-outと7/9の崩れを確認する | [clean学習・3方向解析](../experiments/2026-09-27_leave_one_day_out_clean_analysis/README.md) | clean転送性能、内部―外部順位、7/9の予測・2.1–2.5 kHzピーク同時変化、次の原因識別を固定 |
| clean_only・matchedの3 seed結論と次の実験を確認する | [performance_kfold 3 seed比較](../experiments/2026-09-26_clean_matched_performance_kfold_3seed/README.md) | 6 run・42条件、seed安定性、重み順位移送、採否基準、追加方式の事後診断を固定 |
| 9/25～26完成のclean_only・matched結果を比較する | [clean_only・matched比較解析](../experiments/2026-09-26_clean_only_vs_matched_analysis/README.md) | 同一6/11学習のnoise方針、performance重み、ONB誤報、7/9追加悪化を分離して記録 |
| 現行の日付分割とperformance_kfold内部検証を確認する | [WAV固定内部検証と日付リスト分割](../experiments/2026-09-25_wav_kfold_and_inferred_day_split/README.md) | 旧2設定の削除、自動判定3規則、部分重複拒否、検証範囲を固定 |
| 9/25完了のperformance_kfold予備runを確認する | [run 143020の解析](../experiments/2026-09-25_onb_run_143020_analysis/README.md) | 3/22 kHz clean、選別あり・3-fold・1 seedの性能、ONBとのトレードオフ、XAI品質、本比較との差を固定 |
| 9/25進捗報告の修正点と次の研究順序を確認する | [9/25報告レビューと段階的研究方針](research_plan/2026-09-25_weekly_report_review_and_next_steps.md) | Aの不足、事実と解釈、clean-only/matchedの研究質問、performance_kfold→選別→説明性の順序を整理 |
| performance_kfoldの次条件・手順を確認する | [主方式・固定200 epoch比較](../experiments/2026-09-25_matched_performance_kfold/README.md) | 5-fold OOF、固定epoch、実行コマンド、実行後監査。条件作成済み・未実行 |
| アンサンブル研究の目的・成功基準・次の検証を確認する | [アンサンブル研究の位置づけと次の検証](research_plan/2026-09-24_ensemble_research_position_and_next_steps.md) | 本人の意図、現行重みの数式、評価基準、matchedのnoise別重み比較から方式改善への分岐を固定 |
| matchedのnoise別重み本比較と次の方式比較を確認する | [matched 7 SNR×3 seed本比較](../experiments/2026-09-24_matched_noise_specific_weights/README.md) | 21条件の完了性、clean-only対応比較、領域別性能、重み順位の別日移送失敗、subset・shrinkage比較への根拠 |
| ONB値・選別閾値・アンサンブル原因・IG修正を確認する | [9/24追跡監査](../experiments/2026-09-24_selection_onb_ig_review/README.md) | 確定ONB、3日分ピーク分布、Cの再解釈、fixed clean統合、log-power IGを固定 |
| clean学習モデルのノイズ転送と統合効果を確認する | [9/24 clean固定モデル3 seed本比較](../experiments/2026-09-24_clean_train_noise_inference/README.md) | 完了性、全SNR性能、領域別誤差、残差相関、次の頑健重み診断を固定 |
| 9/18発表後の議論を確認する | [教授との議論・修論との接続](research_plan/2026-09-18_professor_discussion.md) | 原メモ、解釈、確認事実、次の識別比較を分離 |
| 本人指定の今後の計画を確認する | [最終スライドの計画](research_plan/2026-09-18_presented_master_plan.md) | 9月〜翌2月の工程をテキスト化。名称を「アンサンブル手法の検証」へ |
| 発表後の優先順位・達成条件を確認する | [今後すべきことの提案](research_plan/2026-09-19_research_actions.md) | AI提案。手法の採否、着手・終了条件、修論の最低達成条件 |
| 説明性手法の採用理由を確認する | [XAI手法選択の再検証](notes/2026-09-19_explainability_method_selection.md) | 共通帯域マスク・TreeSHAP・IG・Grad-CAMの利点、他候補、現行runの限界 |
| 次の研究作業を小タスクから始める | [9/18発表後の小タスク](research_plan/2026-09-19_next_small_tasks.md) | 保存予測診断→固定モデルのノイズ試験→選別対照→説明性・執筆。各タスクの入力と終了条件 |
| A〜D後の最新方針と実行順を見る | [A〜D後の解釈と今後の研究方針](research_plan/2026-09-22_after_A-D_interpretation_and_next_steps.md) | 本人見解と確認事実を分離。Bを最優先に、固定clean、matched原因診断、統合方式、C・Dの分岐を段階化 |
| 実行順2〜7のうち再学習なしで進めた範囲を見る | [9/23の再学習なし監査](../experiments/2026-09-23_steps2-7_readonly_audit/README.md) | 主コード未実行。matchedの重み反実仮想、Cの除外94秒、残る作業の学習・推論要否を固定 |
| 実行順2の保存・epoch validation実装を見る | [9/23の学習状態保存・再現性スモーク](../experiments/2026-09-23_step2_training_state/README.md) | 元WAV分離epoch曲線、3モデル/PCA/scaler/重み保存、再読込検証、決定論修正、全SNR配線確認 |
| 説明性の採否と帯域改善判断を見る | [9/22 Dの説明性監査](../experiments/2026-09-22_d_explainability/README.md) | 残差とマスク、TreeSHAP再構成、IG収束・baseline不足、Grad-CAMの層・粗さ、帯域再学習の不採用 |
| 修論本文へ移す作業稿を見る | [背景・方法・A〜D結果の作業稿](thesis/2026-09-22_working_draft_background_methods_results.md) | 背景、実験条件、評価設計、手法選択、結果・限界、主張と根拠の対応 |
| ピーク選別の採否と残る問題を見る | [9/20 Cの対応比較](../experiments/2026-09-20_c_selection/README.md) | 2条件×3 seed、1080評価秒、WAV別・ONB領域別の得失、閾値定義と独立日の限界 |
| ノイズ曲線の谷型・回復を確認する | [9/19保存予測監査](../experiments/2026-09-19_noise_recovery_review/README.md) | 22 kHz・7条件の対応標本・等重み・WAV別誤差と、原因識別の順序 |
| 完成スライドと次工程を確認する | [9/18 PPTXレビュー](../experiments/2026-09-18_final_slides_review/README.md) | 全8枚の表示・原図照合、必要訂正、現在地と次の対照・再現性・独立検証の提案 |
| 9/18完了のONB全35条件を確認する | [run 151715の解析](../experiments/2026-09-18_onb_full_sweep_151715/README.md) | 3–22 kHz・7雑音条件の性能、単体と統合の差、説明性と発表用の原図を固定 |
| 9/17の6/11学習→6/18評価と内部OOFを確認する | [9/17別日結果の解析](../experiments/2026-09-17_onb_crossday_result_analysis/README.md) | WAV単位MSEの定義、crossfit重み、3 kHz性能、ONB誤り、説明性、22 kHz未完了を固定 |
| 9/16別日300 epochs結果が悪い理由を確認する | [別日ONB結果の診断](../experiments/2026-09-16_onb_result_diagnosis/README.md) | 分割、3 kHz低域バイアス、旧内部KFold、IG警告、crossfit対応、次の識別比較 |
| ピークの高さで学習する秒を選ぶ | [9/16ピーク高さの検証](../experiments/2026-09-16_peak_height_selection/README.md)、[操作画面](../experiments/2026-09-16_peak_height_selection/peak_threshold_review.html) | 本人の付録図・追加説明に対応。縦軸の横線、全秒画像、1,649秒、62テスト。最適値と改善は未確定 |
| 別日分割・通常KFold・音響選別 | [9/16実装と全秒スペクトル解析](../experiments/2026-09-16_day_split_spectral_selection/README.md) | 学習6/11+7/9、テスト6/18。2 epochs動作確認と本性能の未検証を区別 |
| 12/24までの後期タスク | [後期計画・WBS](research_plan/2026-09-16_second_semester_plan.md) | 大項目・中項目・小項目、時期、終了条件、判断点 |
| 修論の目次と論理 | [7章の構成案](research_plan/2026-09-16_master_thesis_outline.md) | 研究課題→章→必要な根拠→執筆時期を対応 |
| IGの不一致とコード修正を確認する | [9/16 IG数値修正](../experiments/2026-09-16_ig_numerical_fix/README.md) | 数値計算の修正・診断と、旧学習済みモデルの再計算を区別する |
| 決定論GPUでIGがBatchNorm逆伝播により停止した修正を確認する | [9/25 IG決定論GPU修正](../experiments/2026-09-25_ig_deterministic_gpu_fix/README.md) | IG時だけ非fused BNを使用して復元し、未実装時はCPUへfallback。学習・通常予測・IG数式は不変 |
| ensemble結果の年月・日階層と既存結果の移行を確認する | [9/25 保存日付階層整理](../experiments/2026-09-25_ensemble_date_layout/README.md) | `YYYYMM/DD/条件`への正規化、衝突防止、既存結果の移動対応表 |
| 本人指定の9/14結果から学部との差を調べる | [9/14限定の原因分析](../experiments/2026-09-16_sep14_cause_analysis/analysis.md) | 9/16追加。6条件の共通残差、重み、統合誤差分解、端点依存、説明性。9/15の7条件と混ぜない |
| 年間計画・理想と現実的な成果 | [2026年度計画](research_plan/2026_annual_plan.md) | 締切・実験可能期間・研究判断が変わったとき |
| 長期の研究目的・用語 | [研究コンテキスト](research_context.md) | 目的・定義が変わったとき。毎runの成績は置かない |
| 金曜の発表 | [9/18準備メモ](research_plan/2026-09-18_xai_progress_brief.md) | ファイル名は保持。別日分割・選別・後期計画・目次へ内容更新 |
| 最新2帯域の結果を説明する | [9/15・3/22 kHz比較](../experiments/2026-09-15_onb_frequency_comparison/analysis.md)、[本人の修論仮説](research_plan/2026-09-15_master_thesis_hypothesis.md) | 卒論比較・帯域SNR・XAIの役割と原因分析。本人の理想は仮説として別記 |
| 今週の報告を作る | [SOAP下書き](progress/2026-09-18_weekly_progress_draft.md) | 9/16新方針の実装・解析と後期計画。旧草稿を保存した上で更新 |
| 追加結果を解析する | [解析手順](analysis_workflow.md) | 対象固定→性能→原因候補→XAI→次の比較の判定基準 |
| Cursor内のCodexを効率よく使う | [Codex利用枠の節約運用](codex_token_efficiency.md) | サイドバーでのモデル選択、段階昇格、スレッド分割、対象を絞った探索・確認 |
| 6つのアンサンブル手法を確認する | [6方式の手法と数式](ensemble_methods.md) | 主方式performance K-fold、過去inner、等重み、crossfit 3方式の重み、目的関数、性質を同じ記号で比較 |
| 原資料の位置づけを確認する | [原資料と現在方針の対応](original_document_guide.md) | 学部からの連続性、旧計画・説明手法・原本の扱い |
| 22 kHz単独解析時点を確認する | [9/15・22 kHz解析](../experiments/2026-09-15_onb_22khz_noise_sweep/analysis.md) | 先行結果は保持。発表草稿は現行2帯域へ更新 |
| 数値・コード変更・解釈の根拠 | [experiments](../experiments/README.md) | 日付付きで固定。訂正理由と後継を追記する |
| 方針の変遷 | [進捗索引](progress_index.md) | 日付順に要点と根拠リンクを追記 |
| 実行・保存先 | [コード地図](code_map.md)、[結果の保存名](experiment_result_naming.md) | 現行コードと照合して更新 |
| データの版・量・除外理由 | [データ索引](data_inventory.md)、[configs](../configs/README.md) | 新版・新しい監査ごとに更新 |
| 原資料 | [研究進捗報告](../研究進捗報告/) | 発表・提出した原本はその時点の記録として保持 |

## 情報が食い違ったとき

1. 研究の希望・進め方は、**本人の最新の明示指示**を優先する。
2. 実行可能な設定・モデル・方式は、現行コードで確かめる。
3. 完了範囲・数値は、該当runのmanifest、完了印、予測・指標CSVで確かめる。作成時刻やフォルダ名だけで完了としない。
4. 研究上の解釈は、その数値と同じデータ版・分割・モデル世代・評価単位の資料を使う。
5. 古い「次にやること」やAIの提案は、現在の未完了作業・本人の確定方針へ自動昇格させない。

設定ファイルが新しいこと、CSVが存在すること、コードがテストに通ることは、それぞれ本実験の完了や研究上の妥当性とは別。

## 過去資料の更新先

| 過去の資料・記載 | 現在の扱い・更新先 |
|---|---|
| 6月のRF確立・学習コード立て直し計画 | 実装経緯として保持。基盤の完了は[現在地](research_status.md)で確認 |
| 6〜7月のConformer/v1/v2/AttnPool比較 | 当時の構造・データの比較。現行構造の根拠は[8/30構造比較](../experiments/2026-08-30_log_power_architecture_study.md) |
| 7/24の「P0診断は未実行」 | [8/17診断結論](research_plan/2026-08-17_noise_accuracy_reversal_investigation.md)で更新済み |
| 7月以前の高ノイズ高精度・通常KFold結果 | 生成仕様/元WAV混在の影響を区別。現行頑健性の主結果へ流用しない |
| 8/28・8/30のRF最強、全モデル2–5 kHz、深層立て直し案 | 当時の診断・提案。[9/15・2帯域比較](../experiments/2026-09-15_onb_frequency_comparison/analysis.md)が最新の判断材料。モデル順位・利用帯域は当時と異なる |
| 9/2の56/105条件と「安全なinner holdout」 | 中断時の記録。9/3の105条件が別にある。旧innerの元WAV混在は[9/8監査](../experiments/2026-09-08_wav_onb_transition_evaluation_implementation.md)で判明 |
| 9/14午前の「一般化は未実装」 | 同日の[実装記録](research_plan/2026-09-14_result_layout_and_generalization.md)で更新。性能検証の本実行とは区別 |
| 9/14の「現在6条件」「3方式」 | 当日の試行記録。最新解析は2帯域×7条件・2方式。最大予測値方式は[削除記録](../experiments/2026-09-14_remove_prediction_max/README.md)を参照 |
| 9/8の「06.11/06.18は全方式ONBを見逃す」 | 当該runの結果。9/14の一部条件で変化。[9/15・2帯域比較](../experiments/2026-09-15_onb_frequency_comparison/analysis.md)を参照 |
| 5月時点の容量、High_speed_compareの案内 | 現在値は[9/15全体棚卸し](audits/2026-09-15_workspace_review/README.md)。削除経緯は[6/22記録](storage_cleanup_2026-06-22.md) |

最新の全件索引・抽出範囲・修正理由は[文書整合性監査](audits/2026-09-15_document_consistency/README.md)。過去の棚卸し時点の91 Markdown索引は[document_catalog.csv](audits/2026-09-15_workspace_review/document_catalog.csv)。そのほかの原資料は[Office索引](audits/2026-09-15_workspace_review/office_catalog.csv)、[PDF索引](audits/2026-09-15_workspace_review/pdf_catalog.csv)。これらは同日先行監査の固定記録で、新規文書の自動追随一覧ではない。

## 古い情報を消すか

**日付付きの研究記録・数値・原資料は残す。** 今回は古い情報を消す代わりに、現行の入口を更新し、履歴ページの冒頭に更新先・適用範囲を付けた。

- 残す: 旧データの問題、失敗・中断、モデル選定、過去の提案、否定的な結果、報告に使用したCSV。
- 更新する: 現在地、案内文書、テンプレート、実在しないページへの案内、現行設定と食い違う説明。
- 統合する: 同じ成績の重複要約。`notes/experiment_log.md`は実験索引への入口にした。
- 容量整理を別に検討する: 再生成可能な大量配列・全画像・重複重み。数値の低さや古さだけで生データを消さない。

削除が必要になったときは、対象の実在パス・再生成条件・原資料/報告からの参照・代替保存先を具体化して判断する。軽量Markdownを消しても、大半を占める`Pool_boiling/`の容量問題は解消しない。

## 研究作業を終えるときの更新順序

1. `experiments/YYYY-MM-DD_.../`に条件・完了範囲・数値・解釈・限界・出典を固定する。
2. `research_status.md`の完了事項・課題・次の一手を更新する。
3. `progress_index.md`に、結果や判断がどう変わったかを短く追記する。
4. 長期計画が変わる場合だけ年間計画を更新し、本人指定/提案を区別する。
5. 新しい判断と衝突する旧ページに、更新された判断と後継を記す。本文の過去数値は置き換えない。
6. 発表準備中ならSOAPと発表メモも更新する。旧草稿は監査記録へ保持し、現行草稿に二つの異なる次工程を併記しない。

過去スレッドを参照できない環境でも引き継げるよう、会話で決めたことのうち研究に必要な結論・日付・根拠をここへ残す。全会話ログがこのディレクトリに揃っているとは断定しない。
