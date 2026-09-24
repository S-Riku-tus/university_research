# 文書案内と更新ルール

更新日: 2026-09-24。**現在の状態は一か所で更新し、過去の数値と判断は日付付き記録に残す。**

## 読む順序と各文書の役割

| 目的 | 読む文書 | 更新方法 |
|---|---|---|
| 次の作業を始める | [研究の現在地](research_status.md) | 状態・本人の希望・次の一手を更新する唯一の入口 |
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
| 本人指定の9/14結果から学部との差を調べる | [9/14限定の原因分析](../experiments/2026-09-16_sep14_cause_analysis/analysis.md) | 9/16追加。6条件の共通残差、重み、統合誤差分解、端点依存、説明性。9/15の7条件と混ぜない |
| 年間計画・理想と現実的な成果 | [2026年度計画](research_plan/2026_annual_plan.md) | 締切・実験可能期間・研究判断が変わったとき |
| 長期の研究目的・用語 | [研究コンテキスト](research_context.md) | 目的・定義が変わったとき。毎runの成績は置かない |
| 金曜の発表 | [9/18準備メモ](research_plan/2026-09-18_xai_progress_brief.md) | ファイル名は保持。別日分割・選別・後期計画・目次へ内容更新 |
| 最新2帯域の結果を説明する | [9/15・3/22 kHz比較](../experiments/2026-09-15_onb_frequency_comparison/analysis.md)、[本人の修論仮説](research_plan/2026-09-15_master_thesis_hypothesis.md) | 卒論比較・帯域SNR・XAIの役割と原因分析。本人の理想は仮説として別記 |
| 今週の報告を作る | [SOAP下書き](progress/2026-09-18_weekly_progress_draft.md) | 9/16新方針の実装・解析と後期計画。旧草稿を保存した上で更新 |
| 追加結果を解析する | [解析手順](analysis_workflow.md) | 対象固定→性能→原因候補→XAI→次の比較の判定基準 |
| Cursor内のCodexを効率よく使う | [Codex利用枠の節約運用](codex_token_efficiency.md) | サイドバーでのモデル選択、段階昇格、スレッド分割、対象を絞った探索・確認 |
| 5つのアンサンブル手法を確認する | [5方式の手法と数式](ensemble_methods.md) | 既存2方式と追加3方式の重み、目的関数、性質を同じ記号で比較 |
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
