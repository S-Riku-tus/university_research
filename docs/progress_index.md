# 研究進捗索引

このページは**日付ごとの履歴**。各節の「現在」「次にやること」は当時の記録であり、現在の未完了作業とは限らない。最新の状態は[研究の現在地](research_status.md)、更新関係は[文書案内](document_index.md)を読む。

## 2026-10-06 通常ONBの5モデル化・指標差・教授向け理由を整理

- [通常組込み](../experiments/2026-10-06_onb_five_model_integration/README.md)として追加2をモデルutilsへ実装し、5単体、共有OOFの元3/HGB4/ExtraTrees4/5/等平均、最終保存/再読込、説明性入口を接続。通常ONBの既定を5に変更した。
- 従来の統合日平均ONB閾値を保持し、日別閾値の指標/元3との差/q100を別CSVへ追加。通常指標にもTP/FP/TN/FN/FPR・領域母数を保存。[各指標差](../experiments/2026-10-06_onb_five_model_integration/metric_changes_summary.md)でclean Recall+2.22/+2.86pp、F1+0.01219/+0.01542、Accuracy+1.30/+1.67pp、Precision不変を定量化。
- 保存150 epochsモデルの7条件で新統合を照合し、予測差最大7.96×10⁻¹³ kW/m²。追加2の通常Pipeline28条件/15120予測も一致、252指標/3360差行を保存。削除70条件でモデルの寄与と交換を確認し、[教授向け文書](notes/2026-10-06_five_model_rationale_and_next_steps.md)へ元RFの必要性未確認も記録。
- 実main・実5構造の1 epoch/内部2fold、実入力144の15fitを完走。最終5の再読込、clean/−20の11予測列と各33日別指標行を確認。この値は研究性能ではない。新しい通常150 epochs本runは未実施、設定は150/3fold/seed42。既存150 epochs検証は完了のまま保持。

## 2026-10-06 元3固定の4・5モデル追加と9候補比較を完了

- 本人の「元RF/Conformer/AlexNetは固定して追加する」意図へ主方針を修正。[追加比較](../experiments/2026-10-06_fixed_three_additions/README.md)と[後継判断](../configs/experiments/2026-10-06_fixed_three_additions_decision.json)を保存。元3予測・内部比率を保持し、正の全4/5メンバー統合を実装した。
- 同じseed43/44・clean_onlyで9候補をnested学習、648fit、保存72モデル、87方式×7条件、3654指標行を取得。第5候補は両seedでExtraTrees。元3MSE→HGB＋ExtraTrees5のclean RMSE35.74→24.67／33.89→25.26、FN32→25／28→19、−20 RMSE81.41→76.31／81.41→73.57、全7条件FP0。
- 同じ34特徴RFも元PCA画像RFより改善し、HGB一般の優位と入力の差を区別。ExtraTrees4と5の僅差、seed43のHGB4 q100利得喪失、低熱流束/強雑音近傍悪化、絶対誤差改善と劣化量の違いを記録。LightGBM/CatBoost/TabPFN等の公式調査と未検証を分けた。
- 元4予測配列全件完全一致、保存72モデルの全7条件再読込/hash、分割非共有、保持比率/メンバー数、指標再計算と関連unittest3件を確認。比較用入口は実装/実行済み、通常ONB主runnerは元3のまま。前節のRF置換採用は後継方針で更新済み。

## 2026-10-06 追加人手情報なしでモデル・統合の主採用を判断

- 本人が追加の原録音確認・実験メモ・同期映像から事実は得られず、採用判断もAIに任せると再指定。[採用判断](../experiments/2026-10-06_model_adoption_decision/README.md)へ反映し、手動確認を次工程から外した。
- 周波数34 HGB＋Conformer＋AlexNet、clean OOFのMSE最小化・非負重みを研究上の主方式に採用。元3は基準、Ridgeは最有力統合対照、雑音OOF方式は別条件として保持。4モデルMSEとの全7560予測で二値判定一致、最大差約0.00024 kW/m²を検算した。
- SVR/HGB×3入力の6 RMSE候補を旧seed42表に整理し、新19方式×2 seedの38プロファイルを保存。周波数HGBは主採用、時間HGBは第2候補、現SVRは強雑音の崩れから低順位、ExtraTrees34・直接Ridge/PLSは未検証候補と区別した。
- 新モデル学習は今回なし。比較用入口の主方式は実装・評価済み、通常主runnerは元3の既定のまま。次は通常実行への組込みと、同じ入力での正則化/学習器対照。以下の「本人確認待ち」は当時の記録で、現行の未完了事項ではない。

## 2026-10-06 固定HGBの追加seed・共通入力説明・雑音OOF統合を完了

- 本人が依頼した次の4項目を[追検証](../experiments/2026-10-06_hgb_followup_validation/README.md)として実装・実行。seed43/44で全基礎モデルを対応分割で再fitし、32保存fit、CNN16 fitの150 epochs、clean-fitの全7条件OOF/外側を取得した。
- 同じMSEの追加なし→ありでclean RMSE35.74→27.36／33.89→26.51、FN32→25／28→21、FP0、両seed26/36 WAV改善。q100不変でも他指標の利得が残る例を確認。一方HGB単体のseed43 −20 FP88、seed44 FP0で、強雑音の優位は不安定。固定chunkではHGB乱数4通りの差0、全内部fitに全段階支持を確認した。
- 共通chunk32例への4モデル入力加工、held-foldの特徴群置換、新保存HGBの分布/依存診断、多様性・Ridge・雑音OOFの同方式対照を実施。19方式、1596指標行、外側266集合と日別q100532件、32モデルhash、RF56組の再現、関連unittest6件を確認した。
- 主runner/過去matchedの採用設定を保持し、[本人確認手順](../experiments/2026-10-06_hgb_followup_validation/human_review_instructions.md)、記入表38行・原音34抜粋を準備。次は実験記録と採用時の優先を本人が確認し、必要なら学習側で強雑音の分割感度を限定比較する。同じ学習の本人再実行は不要。

## 2026-10-06 q100以外の追加モデル利得と同方式対照を再評価

- 本人の「q100は重要だが、新モデルを他指標の改善からも評価したい」補足を受け、[回帰・判定・雑音耐性の再評価](../experiments/2026-10-06_model_addition_general_metrics_review/README.md)を実施。既存3にも同じMSE・多様性・Ridgeをfitする7対照を学習側だけで固定した。
- 同じMSE方式の追加なし→ありでclean RMSE32.41→28.46、MAE25.07→19.50、FN35→24、FP1→0。clean6/18 q100は両方式376.32で、追加固有の利得を他指標から確認。6雑音平均RMSE68.40→66.41、−20は78.99→74.17。
- clean28/36 WAVで二乗誤差改善。領域別に−4の低熱流束過大予測、−20近傍悪化を確認し、HGB単体と統合の判定差を分けた。441指標行、領域・WAV別表、探索的paired再標本化と図を保存・検算。元モデル再学習なし、主runner変更なし。
- 次は周波数HGBを固定したpaired再現性と改善/悪化領域の説明。最後のq100失敗の識別を他指標での価値確認の必須条件としない。

## 2026-10-06 追加モデルと入力表現と統合の予備比較

- 本人の1〜4実行依頼と「q100と誤報を併記」の回答を受け、[SVR/HGBの比較](../experiments/2026-10-06_chunk_complementary_model_pilots/README.md)を実装・実行。18候補方針をnested chunk学習し、学習側で固定した29統合・対照を同じ外側540、7雑音条件で評価した。
- 周波数HGB＋MSE重みはclean RMSE33.75→28.46、FN40→24、FP1→0、6/18 q100434.02→376.32。6/11 q100は不変。多様性項はさらに見逃しを減らすがq100を追加改善せず、RMSEは通常MSE重みに劣る。既存2モデル対照も併記した。
- 周波数/時間入力をHGB固定15で分ける対照を追加。時間入力の強雑音誤報増加と設定感度を確認。6/11最後のchunk28/36は全候補OOFで陰性のままで、非負平均の限界が残る。
- 候補・対照271 fit＋Ridge3 fit、58保存モデル。指標1272行・日別到達段階848件と保存予測を検算。主runnerやmatched・原runは変更せず、次は固定候補の確認と残る弱いchunkの識別。本人の予備比較再実行は不要。

## 2026-10-06 chunk内部検証clean_onlyの結果解析

- [新旧対応解析](../experiments/2026-10-06_chunk_clean_baseline_analysis/README.md)で修正後の全7条件完了、同じ外側540/学習1620 ID、最終150 epochs/batch一致、3保存モデルの再読込差0を確認。performanceのclean RMSE35.07→33.75、FN43→40だが、日別q100は14比較セルすべて不変。
- 新OOFの全3共通FNは100→52へ減るが、最後の陰性段階に6/11共通4、6/18共通1が残る。最終CNN系予測は旧結果と全件一致し、重みとRF予測の変化を分解。強雑音のRMSE小幅改善と見逃し増加を区別した。
- 学習側の最後の失敗と既存特徴を照合し、弱いpower/時間変動の傾向を再確認。固定2モデル対照の外側6/18改善とOOF q100不変を分け、[次の小規模候補比較案](../configs/experiments/2026-10-06_chunk_complementary_model_pilot_plan.json)を保存。主コード/原結果変更・新学習・XAIなし。
- 210指標条件・140日別到達段階を別計算で検算し、表・図・再現スクリプトを保存。

## 2026-10-05 本人実行のRF再読込エラーを修正

- 本人の`183837` runが保存RFの予測不一致で停止。[原因検証](../experiments/2026-10-05_rf_reload_pca_layout_fix/README.md)で同じ8学習chunk・同じPCA成分値から配置だけを変え、最大11960.8984375 W/m²差を再現した。
- 共通PCA変換をC連続配置へ統一。実際の保存RFの再読込検証は差0、追加回帰テストを含む49テスト通過。変換版2を実行hashと保存状態へ記録した。
- 元run・保存物は保持し、本学習は再起動していない。元runの内部OOFと最終RFは出力済みだが、外側結果・q100は未取得。次は修正後の主コードを再実行する。

## 2026-10-05 chunk内部検証・clean_only基準の実行準備

- 本人の「実行してよいか、必要なら修正」依頼により、主コードをclean_only・最終学習状態保存/再読込確認ありへ変更。matchedは切替コメントで保持した。
- 実行snapshotへの内部分割設定の転送漏れを修正し、chunk/WAV両設定の反映と条件hashの識別、不正値の学習前拒否を確認。関連9テスト通過。
- [事前確認](../experiments/2026-10-05_chunk_clean_baseline_ready/README.md)で全7条件の2160 ID対応、外側1620/540と旧テストID一致、内部全36 WAV支持を確認。[実効条件](../configs/experiments/2026-10-05_chunk_clean_baseline_ready.json)を保存。本学習・新OOF/重み・新q100は未実施。

## 2026-10-05 chunk分割実装後の次工程を整理

- 本人の依頼により[段階案](research_plan/2026-10-05_after_chunk_split_next_steps.md)と[基準実験の条件案](../configs/experiments/2026-10-05_chunk_kfold_clean_baseline_plan.json)を作成。最初は既存採用値固定のclean_only基準で、新OOF・重みと同じ外側540のq100/g100・誤報を得る。
- 最後の陰性を全モデル共通/単体に補完余地ありへ分け、追加モデル・入力表現と重み・統合の優先順を決める。旧WAV分離のSVR/HGB予備結果を新chunk条件で再検討し、モデルの違いと入力の違いを分けて比較する。
- 基準モデル保存からXAIへ接続し、有用な候補を絞って再現性・別日での適用範囲を確認する。主コードはmatched・モデル非保存のまま。今回の成果は計画文書であり、新学習・新精度結果・設定変更ではない。

## 2026-10-05 本人指定のシャッフルあり内部chunk KFoldを実装

- 本人が外側・内部のchunk分割を選択。[実装記録](../experiments/2026-10-05_shuffled_chunk_internal_validation/README.md)のとおり、外側各WAVのランダム15/60テストは維持し、内部performance重みと候補探索を全学習chunkの通常3-fold、shuffle=True、seed42へ変更した。
- 同じchunk非共有・OOF coverage・fitのみのPCA/学習・外側テスト除外・再開識別等の46テストを通過。実データの外側540 IDは10/1と全件一致し、全内部fitに36 WAV・両日各18段階が残った。
- 主設定`chunk_kfold`、省略時もchunk。旧WAV結果との条件hashと`ic3`保存tokenを分離。研究モデルの再学習と新しい精度・q100比較は未実施。次は分割変更だけの既存3モデル本比較を基準にする。

## 2026-10-05 内部foldの支持・q100主評価・追加候補の予備比較

- 本人の主評価を「全chunk陽性へ到達する最小実測熱流束」へ明確化。[分割と候補の検討記録](../experiments/2026-10-05_onb_endpoint_and_grouped_fold_review/README.md)で保存OOFをq100/g100と誤報により再評価し、前回SVR・固定統合のq100不変を確認した。
- 日別各段階1 WAVでは、WAV完全分離と全fitへの全段階支持は両立しない。日別順位均衡分割でONB近傍WAV2/1/1を残せた。既知WAV時間blockのgap0/1/2対照では各fitに全18段階/日を残せたが、この条件のモデル学習は未実施。
- 共有10特徴のSVR/ExtraTrees/HGBを、WAV分離2分割・nested選択3基準で予備学習。従来分割HGBの6/18 q100は434.02→376.32 kW/m²だが同日FPは1→31。25%追加統合では全候補のq100が両日不変。新モデル単体の補完と統合利用を区別した。
- 28モデル・36予測集合各1620 IDを保存・検算。最後の見逃し段階の弱いchunkを特定し、周波数形状・時間系列・音響事前学習等の候補を一次資料から整理した。外側540・主コード/config・旧3モデルの再学習は対象にしていない。

## 2026-10-04 学習OOF診断と帯域特徴SVRの予備比較

- 本人の「次へ進める」指示により[第2段階と小規模予備比較](../experiments/2026-10-04_training_oof_diversity_diagnosis/README.md)を実施。7種類各1620 OOFを監査し、学習chunkのclean/−20 dBだけから3240入力特徴を算出した。外側予測・入力は使用していない。
- cleanのONB〜1.5倍で共通陰性100／270、統合FN133を確認。両日ONBが同じ内部foldに入り、そのfitに近傍録音がない。6/11 ONBは直前段階より帯域powerが低く、弱い区間へ共通失敗が集中した。配置と入力の仮説を分けて記録。
- 10特徴SVRをnested WAV分離で学習し、共通FN8件を訂正。ただし全域RMSE65.29→78.83、FP1→18、近傍68.35→132.10、−20 dB転送FPR54.07%で採用見送り。絶対power除去の一対照でも改善しなかった。固定25%統合では共通FNを訂正できず、訂正と統合利用を分ける根拠を得た。
- 6個の候補foldモデル、CSV、2図、設定、再現スクリプトを保存し検算した。次の内部ONB配置対照は分割作成・監査済みで、既存3モデル再学習は未実施。主コード/config・原結果は変更していない。

## 2026-10-04 現行課題と説明性選択と追加モデルの検討

- 本人方針としてclean_onlyと対応matchedを保持し、基本検証はclean_onlyを主軸にする。直近は現行結果と3モデルの説明性手法の選択理由の整理。教授向け相談資料の新規作成は必須にしない。
- [課題の再評価と段階案](research_plan/2026-10-04_current_challenges_xai_and_model_diversity.md)を作成。cleanの全モデル陰性23/43、clean_only −20 dBの補完余地とRFのFPR96%、matched −20 dBの共通陰性45/49を区別し、追加モデルで役立つ予測を増やすことと、それを統合で利用することを分けた。
- 現行log-power経路のIG、PCA空間TreeSHAP、補助Grad-CAM、共通Hzマスクの選択理由を整理。9/25別条件のIG収束・摂動不安定性・Grad-CAM失敗を最新主モデルの検証結果と混同しない。最新主runはXAI無効。
- 第2段階のOOF・特徴診断を追加候補の選択へ接続し、候補追加後に単純統合で利点が移るかを確認する順序へ更新。新規学習、コード/config変更、新モデル・新統合の実装は行っていない。

## 2026-10-03 採用条件matchedの完了とclean_only対応比較

- 本人のmatched実行結果を[対応解析](../experiments/2026-10-03_tuned_within_wav_matched_comparison/README.md)へ保存。14条件の学習1620／評価540 ID・採用値一致、無雑音の5方式完全一致、7別fit・学習側WAV検証・テスト不使用を監査した。
- performanceはnoise平均RMSE 69.30→63.19 kW/m²、−20 dBの見逃し64→49へ改善。ただし同条件のONB近傍RMSEは85.94→129.87へ悪化し、−20 dBの全域RMSEは82.46→82.65でほぼ不変。q100は14セル中13不変、6/18の−12 dBのみ改善した。
- −20 dBのmatched RFは全域RMSE 76.17でperformance 82.65を下回る。一方RFのFPは1、performanceは0。統合の見逃し49中45は全3モデル陰性であり、現在の非負重みだけでは訂正できない。
- clean_onlyとmatchedを別の問いとして保持し、次は単体品質・誤り補完・共通失敗の理由を整理する。新統合や物理的原因は未検証。主コード・config・原出力を変更せず、未知WAV/日の追加評価もしていない。

## 2026-10-03 matched対照と誤り方を使う統合の判断

- 本人から、現行clean_onlyにmatchedを追加したいこと、10/2輪講の誤り多様性を自研究に活用したいことが共有された。[研究質問と段階的な次工程](research_plan/2026-10-03_matched_and_error_aware_ensemble_next_steps.md)へ記録した。
- 輪講pptxの関連スライドと原論文§3.2・式8〜10を確認。依存構文解析のsociety entropyを回帰へ直接転用するのでなく、誤差方向・量・領域・同時失敗を学習OOFで扱う考え方を整理した。
- [保存予測の補完可能性診断](../experiments/2026-10-03_error_complementarity_feasibility/README.md)を実施。外側clean/−20各540、学習OOF1620、外側ID対応・内部WAV共有0・同一chunk重複0を確認。−20低熱流束116/120で全モデル過大、clean見逃し23/43は全モデル陰性だった。
- 次は共通条件matched→OOFで誤り補完を診断→必要なら一つの単純統合。新方式の有効性・仮想最良・観察上の補完余地を区別した。今回は主コード・config・学習実行を変更していない。修論準備と評価範囲の本人方針は維持。

## 2026-10-03 外側評価を2方式へ整理

- 本人の方針により、主configの外側方式をwithin_wav_chunkとcross_dayへ整理し、within_dayのWAV holdout・層化・旧日内K-fold経路を削除した。[実装記録](../experiments/2026-10-03_remove_within_day/README.md)。
- 内部の学習側WAV GroupKFold、performance重み、OOF候補探索を維持。関連60テストが成功し、実データのテスト540 chunkは10/1の保存予測と全件一致した。
- 過去結果・提出資料・日付付き条件記録は保持。現行3 kHz・1秒・統合・clean_only・選別なし・採用値は維持し、本学習は起動していない。

## 2026-10-02 現在の採用条件と誤差・誤判定の分布

- 本人の依頼により、現行3 kHz・1秒・統合・clean_only・選別なし・performanceの保存予測を再集計。[条件・領域別解析](../experiments/2026-10-02_current_adopted_condition_error_profile/README.md)。7条件の540 chunk対応、36 WAV、既存RMSE/FP/FN一致を確認した。
- −20 dBの二乗誤差の65.48%は60 kW/m²未満へ集中したが、ONB閾値未満の予測のため誤報にはならなかった。全7条件の見逃しはONB〜1.5倍の6測定段階へ集中し、clean43件、−20 dB64件だった。
- 全域RMSEでの統合優位と検知利得を区別。Conformerに対しcleanは3誤り訂正/18誤り追加、−20 dBは3訂正/1追加。日別q100は不変でも直前の陽性率は低下した。
- これはモデルの特性の分析であり、教授と共有する研究主問題・直近の修論準備方針を変更しない。コード・config・モデル実行は変更していない。

## 2026-10-02 修論目次案と判断理由の記録

- 本人より、再来週の研究進捗で教授へ修論目次案を示すよう求められたと共有された。[現行目次と準備計画](research_plan/2026-10-02_master_thesis_outline_and_two_week_plan.md)を作成。10/16は通常の金曜周期に基づく仮目標であり、正式日時・提出期限ではない。
- 卒論PDFの目次・研究目的・ONB相対位置・結言を確認し、5章構成を条件検討・主比較・要因分析を含む7章案へ展開した。9/16目次と9/22作業稿は保持し、後継案内を追記した。
- [主張・根拠・判断台帳](thesis/2026-10-02_claims_evidence_and_decisions.md)に、主条件の採否理由、noise問題への回答経路、統合の絶対性能と劣化量の区別、q100/g100と卒論のCHF正規化指標の違い、説明性の研究上の役割を記録した。
- 章ごとの材料と不足、目次相談前の4成果物、最終結論に応じた追加検証を分けた。次は現行条件で第1〜3章の執筆と主図の出典整理。新規学習・コード変更は行っていない。

## 2026-10-02 後期目標・教授の助言・進捗の再整理

- 本人の指示により、細かなONB改善案の実行より、教授と共有した大きな問題・助言と後期計画の精査を先に行った。[原資料監査と対応表](research_plan/2026-10-02_second_semester_goal_progress_and_professor_feedback_audit.md)。
- 6/26～9/25原報告、7/24・9/18発表の本文、本人提示の9/18教授メモを照合。原資料の記載、教授の助言、AIの後続提案を区別し、口頭の了承は推定しなかった。
- 後期4本柱のうち評価・選別・統合比較は進展したが、元の疑問への回答の共有、現行条件の説明性による原因分析、修論本文への統合が残ると整理した。
- noise精度逆転は旧生成の振幅経路、matched再学習、bias相殺による部分回復を分ける。ONB直後の見逃しは解析で見つかった候補課題とし、matched等の追加runを直近の必須作業から外した。コード・結果は変更していない。

## 2026-10-01 3 kHz採用値の外側評価と22 kHz対応比較

- [対応解析](../experiments/2026-10-01_3khz_22khz_tuned_outer_comparison/README.md)で、3 kHzの外側540 chunk・7 noise・同一fitを監査し、日別確定ONBで22 kHzおよび従来3 kHzと対応比較した。
- performanceのclean全域RMSEは3 kHz 35.07、22 kHz 29.97 kW/m²だったが、noise平均は69.30対189.55、FPRは.0015対.8911で、主入力を3 kHzに固定する根拠が得られた。
- 3 kHzの強noiseでは低〜ONB近傍予測が約160 kW/m²へ圧縮され、FPR 0を保つ一方、Recallが−20 dBで.797へ低下した。22 kHzは同領域を約471–483 kW/m²へ過大予測して常時陽性化した。
- tuned 3 kHzは従来値に対してnoise平均RMSEを69.64→69.30、強noise78.36→77.13と僅かに改善したが、強noise ONB近傍は94.78→103.54へ悪化し、q100は14/14セル不変だった。次は同じ固定条件で`matched`だけを変え、低FPRとONB直後Recallを両立できるか確認する。

## 2026-10-01 最大周波数別パラメータ設定

- [実装記録](../experiments/2026-10-01_frequency_specific_parameter_config/README.md)のとおり、3 kHzと22 kHzの採用値を`models.parameter_sets.by_max_freq_hz`へ独立登録した。
- 通常runでは複数周波数も各専用値で順番に実行でき、候補リストを増やしたOOF探索では従来どおり1周波数だけを選ぶ。
- 現在は3 kHzだけを有効にし、RF `100/深さ12/0.6/0.6`、Conformer `0.001/12`、AlexNet `0.003/8`、外側540 chunk・7 noiseの通常runになることをdry runで確認した。

## 2026-10-01 22 kHz採用パラメータの外側7 noise評価

- [対応解析](../experiments/2026-10-01_22khz_tuned_outer_analysis/README.md)で、外側学習1620／評価540 chunk、6/11・6/18各270、clean_onlyの同一fit、7/7 noise、選別なし、採用済み22 kHzパラメータを確認した。
- performanceはclean全域RMSE 29.97 kW/m²で等重み34.54、最良単体Conformer 32.30を下回った。等重みとの差のWAV cluster bootstrap 95% CIは[−7.76, −1.67] kW/m²だった。
- 0 dBでperformanceのONB前biasは+162.39 kW/m²、FPRは46.67%へ悪化し、−4 dBで88%、−8 dB以下で100%となった。noise平均RMSEはRF 111.36、等重み170.38、performance 189.55 kW/m²で、clean OOF重みはnoise劣化を防がなかった。
- 強noiseのq100=0は早期検知の改善ではなく常時陽性化なので、q100は日別ONB前FPRと対で報告する。22 kHz採用値は外側結果で選び直さず、次は3 kHzのOOF採用値を固定した同一外側540 chunk・7 noiseを一度実行し、周波数とパラメータ差を分ける。

## 2026-09-30 3/22 kHz学習側OOFチューニング完了

- [対応解析](../experiments/2026-09-30_3khz_22khz_oof_tuning_analysis/README.md)で両周波数各190候補、外側テスト未使用、選別なし、同一1620学習chunkを確認した。
- 推奨performance OOFは22 kHz対3 kHzで全域RMSE 53.50対65.29、ONB近傍65.51対68.35 kW/m²、Recall .866対.859、FPRは両方.0015だった。全域差のWAV cluster bootstrap 95% CIは[−20.58, −2.48] kW/m²、ONB差は[−23.77, 19.52]で、全域優位は支持されたがONB優位は未確定。
- 3 kHz推奨はRF `100/深さ12/0.6/0.6`、Conformer `0.001/12`、AlexNet `0.003/8`。22 kHz推奨はRF `600/深さ6/0.6/0.6`、Conformer `0.0003/8`、AlexNet `0.01/24`。
- clean学習側では22 kHzが有望なため、主コードを22 kHz推奨値の各1要素へ固定した。次は外側7 noiseを一度だけ通常評価する。

## 2026-09-29 ONB保護付きピーク選別の本結果

- [対応解析](../experiments/2026-09-29_onb_protected_selection_analysis/README.md)で、選別なしと`ONB +10%保護＋上側1e-9`の14/14条件、外側540 chunk完全一致、外側学習1620→1556を確認した。
- performanceは−20 dB全域RMSE 88.20→77.69 kW/m²だが、clean 35.33→44.20、強noise平均ONB近傍94.78→116.38、ONB以降59.71→67.90へ悪化。FPR・Recall・q100の主検知指標は改善しなかった。
- 除外は両日のONB保護帯直後4測定点に集中し、低熱流束強noiseの過大予測抑制とONB近傍悪化のトレードオフになった。主条件には採用せず、選別なし1秒で学習側OOFチューニングへ進む。

## 2026-09-29 ONB保持制約によるピーク閾値監査

- [閾値監査](../experiments/2026-09-29_peak_threshold_preservation/README.md)で、外側学習1620秒だけを使い、ONB保持率と上側除外数を共通閾値・日別閾値で比較した。
- ONBを両日80%以上残せる最大共通値`7.4304e-12`はONB 9秒だけを除外し、ONBより上を1秒も除かなかった。6/11のONB分布と上側最小値が重なるため、数値閾値だけでは目的を達成できない。
- ONB +10%以内を必ず保持し、その上だけ`1e-9`を適用する方式を実装した。ONB 90/90保持、上側64秒除外、全体1620→1556となる。
- 内部3-foldでも保護帯除外0、held-out無選別。主コードは全パラメータ候補リストが1要素のため、固定パラメータの通常runを実行できる。

## 2026-09-29 統合within-WAVピーク選別・学習側OOFチューニング実装

- [実装・事前検算](../experiments/2026-09-29_within_wav_selection_and_oof_tuning/README.md)で、統合manifestの元実験日・元WAVを用いる固定ピーク選別と、外側テストを使わないモデル別OOFチューニングを追加した。
- 現行通常設定は1秒・3 kHz・6/11＋6/18統合・clean_only・`1e-9`選別・固定パラメータ。外側学習1620→1470、外側テスト540は無選別となることを事前確認した。
- 以前の6/11単日全学習は1080→986（8.70%）、今回は9.26%でほぼ同率。ONBちょうどでは86/90除外だが、これは全体でなく局所層の割合であり、除外処理が大きく変わったわけではない。
- 実行モデルはRandomForest・Conformer・AlexNetの3つへ固定し、モデル選択configを廃止した。チューニング専用configも持たず、`models.parameter_sets.model_grids`のどれかの候補リストを複数値にすると、モデル別の学習側OOF探索へ自動切替し、全リストを採用値1要素へ戻すと通常runへ戻る。次は選別だけを変えた通常runを先に比較し、採否後に候補を追加して探索し、固定候補を外側テストで一度評価する。本実行結果はまだない。

## 2026-09-29 0.5秒対照完了・1秒採用

- [0.5秒対照解析](../experiments/2026-09-29_within_wav_chunk_05s_analysis/README.md)で7/7条件、学習3,240／テスト1,080 chunk、36 WAV、clean_onlyの1 fit共有を確認した。
- performanceの2日合算RMSEは0.5秒対1秒でclean 40.27対35.33、noise平均80.71対69.64、強noise92.49対78.36、−20 dB 98.65対88.20 kW/m²。日別14/14 noiseセルで0.5秒が悪かった。
- 0.5秒は−20 dBのONB近傍RMSEとRecallを改善したが、ONB前biasとFPRを増やした。q100は両入力長・全noiseで不変であり、早期確実判定の改善はなかった。
- 現行splitは入力長間で時間非対応で、完全共通区間は540中22。小標本の2区間平均比較でも強noiseは0.5秒が悪かった。入力長比較を中心主張にする場合だけ対応splitを再実行し、通常は1秒を固定して学習側OOFチューニングへ進む。

## 2026-09-29 7/9対照と次の0.5秒比較

- [7/9解析](../experiments/2026-09-29_within_wav_chunk_0709_analysis/README.md)で7/7条件、13 WAV、学習585／テスト195 chunk、clean_onlyの1 fit共有を確認した。
- performanceはclean 103.29から−12 dB 239.66 kW/m²まで悪化後−20で219.06へ見かけ上回復したが、負biasの相殺であり6月と同じ単調trendではない。q100は全noiseで720.691 kW/m²、Recallはclean .667、−20 .600だった。
- 571.7～643.5 kW/m²の大幅過小予測と720.7 kW/m²での急変が同日学習でも残り、7/9は6月pool・チューニングへ混ぜない補助対照とした。
- [評価範囲と順序](research_plan/2026-09-29_scope_chunk_length_and_tuning.md)で未知WAV・未知日は直近優先から外し、次は6月統合0.5秒を1秒と対応時間区間で比較してから、採用chunk長だけをチューニングする方針を記録した。

## 2026-09-29 6/11＋6/18統合within_wav_chunk結果

- [対応解析](../experiments/2026-09-29_within_wav_chunk_combined_analysis/README.md)で7/7条件、clean_onlyの1 fit、学習1,620／テスト540 chunk、単日runとの両日270/270対応を確認した。
- performanceの2日合算RMSEは単日別モデルに対しclean 39.57→35.33、noise平均74.77→69.64、強noise87.88→78.36 kW/m²へ改善した。
- 6/18 −20 dBは低熱流束過大予測を抑えて全域116.94→90.92、FPR .042→0となったが、Recall .833→.773、ONB近傍25.85→85.08へ悪化した。q100も早まらず、誤報抑制とONB見逃しのトレードオフが残った。
- performanceは等重みを14/14条件で上回ったが、−20 dBは両日AlexNetが最良。次は外側split 42を固定した学習seed 43・44で再現性を確認し、matched・0.5秒・チューニングを同時に変えない。

## 2026-09-29 6/11・6/18単日within_wav_chunk解析

- [解析記録](../experiments/2026-09-29_within_wav_chunk_single_day_analysis/README.md)で2 run・14/14条件、各WAV 45学習／15テスト、同一chunk重複0を確認した。
- 従来WAV holdoutとの共通75 chunkではclean RMSEが6/11で7.59、6/18で7.98 kW/m²改善したが、同一WAVの別chunkを学習する既知WAV補間の利得を含む。
- 6/18 −20 dBはRMSEとFPRが改善した一方Recallが1.000→.689へ低下し、低熱流束の誤報とONB以上の見逃しのトレードオフが残った。
- performanceは等重みを平均で上回ったが、最良単体への勝利は6/11で5/7、6/18で0/7。統合分割を出典日ごとのseed 42へ修正し、単日runとのテストchunk一致を両日270/270にした。主コードは統合1秒・3 kHz・clean_onlyを次に実行する設定へ変更済み。

## 2026-09-29 6/11＋6/18統合データとWAV内chunk holdout

- [実装記録](../experiments/2026-09-29_within_wav_chunk_combined/README.md)のとおり、全WAVから同率のchunkを外側テストへ分ける`within_wav_chunk`を追加した。変更前の`within_day`はWAV丸ごとholdoutであり、別目的の方式として保持する。
- 6/11＋6/18の0.5秒151,200 NPYと1秒75,600 NPY（計226,800）、元名WAV 36本、熱流束名付きWAV 36本、測定メタデータ6件を、出典を保持して統合実験フォルダへコピーした。統合ONBは両日の算術平均246,591.3959 W/m²。
- configは`evaluation_mode`を先に選び、`evaluation_settings`の`cross_day / within_day / within_wav_chunk`のうち同名欄だけを編集する構成へ整理した。
- seed 42・test 25%では各WAV 45学習／15テストchunk、全体1,620／540、同一chunk重複0。内部`performance_kfold`は学習側だけのWAV GroupKFoldを維持する。
- この評価は同一chunkを再利用しないが、同じWAVの録音条件を共有する。既知WAV内の未使用区間への性能であり、未知WAV・未知日一般化とは区別する。現行初回設定は3 kHz・7 noise・clean_only・150 epochで、学習runは未開始。

## 2026-09-29 q100監査と6/11＋6/18 pool設計

- [q100監査とpooled holdout設計](../experiments/2026-09-29_q100_and_pooled_holdout/README.md)で、現行scatterの100%分類可能熱流束を日内4 runの保存予測から数値化した。
- `q100`は6/11の全70 cellで427.276 kW/m²、6/18の全70 cellで434.018 kW/m²となり、モデル・noise・学習方針を識別しなかった。100%条件と測定熱流束間隔の影響である。
- 6/11 clean_only −20 dBのFPRはRF .492、performance .008、6/18はRF .933、performance .775であり、アンサンブルの誤報抑制は`q100`でなくFPRに現れた。`q100/g100`は卒論との接続を示す補助指標、FPRを未知noise劣化の主指標とする。
- この時点では物理コピーしない`pooled_holdout`を設計したが、後続の本人指示で採用せず、上段の物理統合＋`within_wav_chunk`へ更新した。

## 2026-09-29 3 kHz日内holdout・clean_only／matched

- [日内4 run解析](../experiments/2026-09-29_within_day_clean_matched_seed42/README.md)で4 run・28/28条件、同一テストWAV、clean予測差0、clean_onlyの1 fit共有／matchedの7 fitを確認した。
- performanceのnoise平均RMSEは6/11で50.76→50.22 kW/m²とほぼ同等、6/18で73.25→64.97へ改善した。matchedの利得は日依存で、6/18を学習元にした既存別日結果とも整合した。
- 6/18 −20 dBの全域改善110.36→81.33は最低熱流束WAVの改善が中心。ONB近傍は21.04→146.13、ONB以上は32.64→90.78へ悪化し、全域回帰とONB近傍回帰のトレードオフが残った。
- matchedは強noiseの過大予測を抑え、6/18 −20 dBでFPR .775→0、F1 .756→.827、ROC-AUC .849→.918。ただしONB付近を低く外した。
- noise平均では両日とも等重みがperformanceを僅かに上回り、6/18 matchedの内部最大重みと外側最良単体は0/7一致。当初は外側split seed追加を次案としたが、旧日内GroupKFoldと目的が重なるため直近優先から外した。[問題定義](research_plan/2026-09-29_problem_definition_and_next_priority.md)では、clean_onlyの未知noise耐性を主問題とし、学習側限定チューニングで現パラメータの不足を確認する案へ更新した。

## 2026-09-28 3 kHz matched双方向・seed 42

- [matched双方向解析](../experiments/2026-09-28_3khz_matched_bidirectional_seed42/README.md)で、2方向×7 noiseの14/14条件と各noiseの独立fitを確認した。
- performanceのnoise平均RMSEは順方向でclean_only 92.6→matched 97.5と悪化、逆方向で93.8→76.2と改善し、効果が方向で反転した。
- 双方向matchedはRF 86.8、performance 86.9、等重み86.5 kW/m²。performance固有の上積みは未確認。
- performanceはFPR .113→.001、F1 .885→.912へ改善したが、ONB近傍RMSE 91.2→123.1、ROC-AUC .931→.927。逆方向強noiseのONB前過大予測を抑える校正効果が中心だった。
- matched内部OOF最良と外部最良単体はnoise 12 cell中2一致、平均順位相関−.333。次は両日の日内固定holdoutでclean_only/matchedを比較してからチューニングへ進む。

## 2026-09-28 6/11・6/18双方向、3/5 kHz、3 seed

後続の本人指示で、主方式はperformance_kfoldを継続し、3 kHzに集中する。日内固定テストの切替を実装し関連39テスト成功。[最新方針・matched比較・チューニング設計](research_plan/2026-09-28_3khz_matched_tuning_and_evaluation.md)、[実装記録](../experiments/2026-09-28_within_day_holdout/README.md)。以下は当初の解析時点の推奨を保持する。

- [双方向低周波3 seed解析](../experiments/2026-09-28_bidirectional_lowfreq_3seed/README.md)で、2方向×3 seed×2周波数×7条件の84条件を揃えた。
- noiseあり双方向平均RMSEは3 kHz performance 91.5±2.8、等重み92.4±2.4 kW/m²。3 kHz等重みはRFを37/42条件で上回り、方向間で最も安定した主候補となった。
- 5 kHz等重みは順方向で最良単体に14/21条件で勝ったが、逆方向では3/21。順方向だけの成功で最終方式にしない。
- 内部OOF最良と外部clean最良単体は0/12、平均順位相関−.667。`performance_kfold`は主方式でなく比較・診断方式とする。
- 順方向のnoise改善はclean正biasの相殺を含む。次は追加seedでなく、3 kHz固定で2.1–2.5 kHz帯と隣接帯域の同幅遮蔽を行い、低周波優位の原因を識別する。

## 2026-09-28 6/11→6/18・3/5 kHz seed 42

- [順方向低周波解析](../experiments/2026-09-28_forward_lowfreq_seed42/README.md)で14/14条件を確認し、3/5 kHzの低周波優位が逆方向だけでないことを予備確認した。
- noise平均RMSEは3 kHz performance 92.6・等重み92.9、5 kHz AlexNet 94.4・等重み96.7 kW/m²で、22 kHz統合137～145より良かった。
- 5 kHz等重みはRFに6/7、最良単体に4/7で勝利。−20 dBではAlexNet単体が統合より良く、moderate noiseまでという境界は逆方向と整合した。
- 3/5 kHz deep modelはcleanの正biasをnoiseが相殺してRMSEが改善した。単純なnoise不変性とは解釈しない。
- 次は同条件のseed 43・44を実行し、双方向×3 seedで周波数・方式を判断する。

## 2026-09-28 6/18→6/11・5周波数×3 seed

- [5周波数解析](../experiments/2026-09-28_two_day_reverse_5freq_3seed/README.md)で、seed 42/43/44、3/5/10/15/22 kHz、各7 noiseの105条件を確認した。
- noise平均RMSEは5 kHz RF 90.5±1.5が最小。3 kHzではAlexNet・performance 91.1、等重み92.0 kW/m²で、15/22 kHzより大幅に安定した。
- 3 kHz等重みはRFを20/21条件で上回るが、最良単体勝利11/21・2%以内12/21。moderate noiseでは有効、−16・−20 dBではAlexNet単体が優位だった。
- 22 kHzのnoise平均はRF 128.8、performance 246.5、等重み216.3 kW/m²で、高帯域deep・統合の正biasと全陽性化が3 seedで再現した。
- 内部OOF最良と外部clean最良単体は0/15、平均順位相関−.700。次は順方向の3・5 kHz seed 42で低周波優位を再現確認する。

## 2026-09-27 6/11・6/18のみの双方向clean学習

- [双方向解析](../experiments/2026-09-27_two_day_bidirectional_clean_analysis/README.md)で、6/18 clean学習→6/11評価seed 42の完了性と、既存の逆方向を比較した。
- clean最良RMSEは6/11→6/18でConformer 75.2、6/18→6/11でRF 75.5 kW/m²。2日限定の転送は成立した。
- 等重みはcleanの正負biasを相殺して両方向でperformanceを上回ったが、noiseでは全モデルが同方向へ誤り、RFが両方向で最良となった。
- 逆方向noiseあり平均はRF 135.6、等重み215.8 kW/m²。深層2モデル・両統合はSNR 0 dBからONB前FPR 1となった。
- 内部OOF最大重みは両方向AlexNetだが、外部最良単体との一致は0/2。次は逆方向seed 43・44を同条件で再現する。

## 2026-09-27 clean学習・3日leave-one-day-out

- [3方向解析](../experiments/2026-09-27_leave_one_day_out_clean_analysis/README.md)で、6/11・6/18・7/9のうち2日をclean学習、残る1日を外部評価する比較を揃えた。
- 6/11・6/18評価は転送できたが、7/9評価は最良RFでもR² .105、RMSE 274.4 kW/m²、bias −195.6 kW/m²となり、日付依存が支配的だった。
- 7/9では2.1–2.5 kHzピークと全モデル予測が720.7 kW/m²で同時に急変した。確定ONB 571.7 kW/m²から643.5 kW/m²までは低出力状態として予測された。
- 内部OOF最大重みは3方向ともConformerだが、外部最良単体との一致は0/3。performanceは全3方向で等重みより悪く、最終方式には採用しない。
- 次は追加seedより先に、7/9の波形・スペクトログラム、録音条件、帯域保持/除去による予測差を診断する。

## 2026-09-26 clean_only・matched・performance_kfold 3 seed比較

- [3 seed比較](../experiments/2026-09-26_clean_matched_performance_kfold_3seed/README.md)で、6/11学習→6/18評価、22 kHz、選別なし、150 epochsのmatched／clean_only計6 run・42条件を監査した。
- noiseあり平均RMSEはmatchedでRF 96.24±0.24、performance 97.73±2.29、等重み97.66±1.33 kW/m²。seed 42で見えたperformanceのRF比改善はseed 43・44で安定せず、最良単体2%以内も11/21で基準未達。
- clean_onlyではRFがnoiseあり18/18条件で全域最良。deep・統合のONB近傍RMSE低下は正biasと誤報増加を伴い、強noiseでは全陽性化した。
- 内部最大重みモデルと別日全域最良単体の一致は9/21、順位相関は平均.214。ONB近傍は7/21、.048であり、全域OOF重みはONBと日間順位移送を解決しない。
- 保存済みOOFを用いた追加3方式の事後診断も既存performance・等重み・RFを上回らず、本学習は後順位。次はclean・seed 42・無雑音だけで残るheld-out day方向を揃え、日別校正差を診断する。

## 2026-09-25 performance_kfoldを今後の主方式へ変更

- 本人の理解とコードを照合し、直近runの3-foldは深層2モデルのepoch選択だけ、`inner_holdout`は18 WAV中4 WAVを使う単一分割だったことを確認した。
- 希望する「全WAVを一度ずつ検証側へ回し、全OOF予測を通して1組の重みを決める」方法は`performance_kfold`と一致する。今後の主設定を`performance_kfold`へ変更し、`inner_holdout`は過去run再現用にだけ残した。
- [次条件](../experiments/2026-09-25_matched_performance_kfold/README.md)では`run.epochs=200`を全fitで固定使用し、元WAV非共有5-foldとする。18 WAVは検証4・4・4・3・3本、学習14・14・14・15・15本となる。別validationによるepoch選択機能と評価fold正解を重みに使う旧方式は実装から削除し、主コードには残る5方式を短い説明付きコメントで保持した。条件作成まで完了し、本実行は未着手。
- `subset_equal_cv`、`crossfit_wav_stack`、`crossfit_shrinkage_stack`は共通4-fold OOFを共有し、その後の結合規則だけが異なる。一方、現行`performance_kfold`は別OOF処理なので、主方式の結果確認後に必要なら同一OOF共有へ整理する。

## 2026-09-25 matched・noise別重み7 SNR×3 seed本比較

- [本比較と診断](../experiments/2026-09-24_matched_noise_specific_weights/README.md)を完了。3 seed×7 SNRの全21条件で、noise別の別`fit_id`、学習noiseと評価noiseの一致、`per_training_noise` scope、学習日だけのepoch選択を確認した。
- 7 SNR平均RMSEはRF 96.17、等重み93.54、inner 94.45 kW/m²。ただしnoiseありだけではRF 96.24、等重み96.11、inner 96.98で、cleanを除く優位は安定しない。最良単体2%以内は両方式11/21で暫定14/21基準に未達。
- matchedはclean-onlyに比べ強noiseの全域RMSEを大幅に改善したが、改善の中心はONB前の過大予測抑制で、ONB近傍・以降は過小予測と見逃しが増える。matched内のRF比では統合が近傍・以降RMSEを改善する一方、ONB前RMSEと見逃しは悪化するため、回帰と閾値検知を分けて評価する。
- noiseありでは最大重みモデルと評価日の最良単体が18条件中6条件しか一致せず、順位相関は平均0。現行innerの課題を、単一4-WAV holdout、単体逆MSE相当で相関を扱わないこと、有害モデルを0にできないことへ絞った。当時の次案はsubset・shrinkage比較だったが、同日後続の本人判断により、現在は上段の`performance_kfold`比較を優先する。

## 2026-09-24 アンサンブル研究の判断基準を更新

- [本人の研究上の意図と次の検証](research_plan/2026-09-24_ensemble_research_position_and_next_steps.md)を記録。全条件で最良は要求せず、平均性能、21セル中の最良／実質同等割合、最悪時悪化、ONB前誤報を結果前に固定して評価する。
- 現行`inner_holdout`はclean学習日の元WAV非共有20% holdoutに対するchunk R²から`1/(1-R²)`を計算する。これは同じ標本上では逆MSE重みと等価で、epoch選択用3-fold `rmse_all`とは別処理。
- 本人の追加方針により、全SNR共通の固定頑健重み案は不採用。コードと保存済みmatched 3 seedを監査し、matchedでは既にnoise別にモデル・epoch・重みを再fitしていることを確認した。[監査と次回条件](../experiments/2026-09-24_matched_noise_specific_weights/README.md)を固定し、manifestへ重みscopeを追加した。次は同条件の7 SNR×3 seed matched本比較を行う。

## 2026-09-24 clean学習・固定noise推論3 seed本比較

- [本比較](../experiments/2026-09-24_clean_train_noise_inference/README.md)を完了。6/11 clean学習の同じ保存モデルを6/18の7 SNRへ適用し、3 seedともfit ID固定・再読込差0を確認。
- 等重みとinner holdoutのノイズ曲線は全seedで単調低下し、matchedの谷は消失。cleanでは統合がRFを3/3 seedで上回るが、ノイズあり18条件ではRFが全て最良。
- inner重みは深層2モデルへ80〜87%を配分し、ノイズ下では深層残差相関が最大.987へ上昇。ONB前の同方向過大予測により、Recall改善と同時に誤報が最大480/480秒となる。
- 当初は全SNR共通の頑健重みを次案としたが、同日後続の本人方針により不採用。現在の次工程は本ページ冒頭のnoise別matched比較。

## 2026-09-24 ONB・音響選別・アンサンブル・IG追跡監査

- [追跡監査](../experiments/2026-09-24_selection_onb_ig_review/README.md)で、現行run値を正しいONBとして確定。旧実験txtの値は自動直線性喪失候補でONBではないため3本を削除し、0番ノートブックの表示・出力名も修正した。
- 現行ONBで全2,940秒の2.1–2.5 kHzピークを再集計。Cの`1e-9`は6/11のONB以降94/660秒を除く。追加識別は既存`1e-9`、強い`1e-8`、同数ランダム除外を比較する。
- アンサンブル悪化の直接原因を、未知ノイズ下での深層モデルのONB前同方向過大予測と、clean学習側重みがモデル順位変化へ適応しないことに固定。fixed cleanで谷は消えるが強ノイズではRFを下回る。
- IGへlog-power空間の直線経路を追加して既定化。raw-power方式は再現用に保持。単体・配線テストは成功し、1 epoch実データで数値誤差は改善したが本学習モデルの信頼性判定は未完了。
- CのONB前改善は、弱音のONB以降境界標本を削って共有回帰関数が低値側へ動いた結果と整合する。最大記録熱流束はCHFではない。追加実験と同期映像は取得不可。

## 2026-09-23 実行順2の実装・再現性スモーク

- [学習状態保存・元WAV分離epoch validation](../experiments/2026-09-23_step2_training_state/README.md)を実装。モデル3種、PCA、scaler、統合重み、seed、環境、分割、epoch別の全域・ONB領域別指標を保存し、再読込予測を検証する。
- PCAの微小演算差、RFの固定seed、TensorFlow GPUの非決定性を実データで検出して修正。同一seedの1 epochスモーク2回は、3単体と等重みの6/18全1080予測が完全一致。各保存モデルの再読込差も0 W/m²。
- 同じclean fitを6/18の7 SNRへ適用する配線と、等重み・現行重み・subset equal・shrinkage stackの実データ配線も完了。ただし1 epoch結果は性能結論に使わない。200 epochs×3 seedの本実験、4方式の本比較、保存モデルIGは計算時間が大きいため未実行。

## 2026-09-23 実行順2〜7の再学習なし監査

- [既存成果物だけの監査](../experiments/2026-09-23_steps2-7_readonly_audit/README.md)を実施。ONB主コード、モデル学習・推論、IGは実行していない。
- matched −8/−16 dBは重みを−8固定、−16固定、等重みに替えても、R²差の符号がseed 42/44で正、43で負のまま維持された。谷の見え隠れは統合重みではなく、条件ごとに再学習した単体予測の差が主因。
- Cの除外94秒を全件監査。後日の9/24確認で、221,505 W/m²以降をONBとする現行run値が正しいと確定し、94秒は全てONB以降と訂正した。追加取得・同期映像・独立ノイズは使えない。
- 実行順2〜4、5の残り、6のcrossfit方式、7-Dは学習・推論runが必要。本人の開始指示まで保留。

## 2026-09-22 A〜D後の本人見解と次工程

- [A〜D後の解釈と今後の研究方針](research_plan/2026-09-22_after_A-D_interpretation_and_next_steps.md)を作成。Aは今後の標準診断、Bは最優先、CはONB定義と削除効果の識別、Dは保存モデル上のIG信頼性診断として整理した。
- 固定cleanモデルによる別日ノイズ転送はBで一度実施済み。ただし既存結果では強ノイズ時にRFが統合を上回るため、「統合が劣化を抑える」は未確認の仮説として扱う。matchedは耐雑音性ではなく、条件別適応学習の性能として分離する。
- 次はONB定義・追加取得可否の確認、学習済み状態の保存、学習日内元WAV分離validation、同一条件再現性、選別なし固定cleanの複数seed比較の順。新しい実験結果ではなく、本人の最新方針と識別手順の記録である。

## 2026-09-22 D：残差と説明性の監査・修論作業稿

- [D1〜D3の実験記録](../experiments/2026-09-22_d_explainability/README.md)：run151715の無雑音／−8 dBで、保存失敗秒と全1080秒の帯域マスクをAの残差へ接続。安定した改善帯域はなく、帯域保持／除去再学習は不採用。成功秒補完runはRFのみ再現し、深層2モデルは元予測を再現できなかったため元runの説明に使わない。
- TreeSHAPは10/10秒で再構成合格。IGは20枚中7枚だけ収束し代替baselineなしのため主張から除外。Grad-CAMは最終畳み込み層12×12の正方向補助図へ限定し、追加CAM比較は研究判断を変えないため未実施。
- [A〜Dの修論作業稿](thesis/2026-09-22_working_draft_background_methods_results.md)：背景、実験条件、評価設計、A〜Dの結果・限界、主張と根拠の対応を記述。ONB定義、独立日・同期観察、正式日程、先行研究引用は未確定事項として分離。

## 2026-09-20 C：ピーク選別の対応比較

- [C1〜C4の実験記録・可否判断](../experiments/2026-09-20_c_selection/README.md)：事前条件を固定し、6/11学習→6/18全1080秒評価、22 kHz・無雑音・200 epochsで選別なし／ありをseed 42/43/44で比較。選別で94秒を除外。統合R²差は−.0035、+.0025、+.0016と一貫せず、run閾値のONB前二乗誤差は全seedで減少、近傍1 WAVと近傍外ONB以上は全seedで増加。1e-9選別はこの条件の基準に不採用。現行学習側重みを運用基準として維持し、等重みは固定対照。
- 当時は旧txtとrun閾値の不一致を留保したが、9/24にrun閾値を正しいONBとして確定した。追加の未使用実験日と同期映像は取得できない。

## 2026-09-19 9/18発表完了と教授との議論・今後の計画を記録

- [B1〜B7の適用判定と実験記録](../experiments/2026-09-19_b_clean_only/README.md)：既存matched runの学習済み状態は未保存。cleanを1回学習した固定モデルでは無雑音→−8→−16→−20 dBの統合R²が.918→.737→.442→.152と悪化し、谷型は消失。matched −8/−16 dBの3 seed対応比較では統合R²差が+.0305、−.0188、+.0147と揃わず、ONB以上の二乗誤差は全seedで悪化。training lossだけでは最良epochを判断できない。B7は開始条件不成立。次は学習日内の元WAV分離validationを設ける。
- [A1〜A7の保存予測診断](../experiments/2026-09-19_noise_recovery_review/A1_A7_saved_prediction_diagnosis.md)：無雑音・−8 dBの秒別対応、重み付き誤差分解、ONB領域・WAV別の得失を固定。−8→−16/−20 dBの回復は一部WAVへ集中。runのONB閾値と出典テキストの不一致を記録。既存監査の二乗誤差減少量の桁を訂正。
- [説明性手法の再検証](notes/2026-09-19_explainability_method_selection.md)：共通帯域マスク・TreeSHAP・IG・Grad-CAMの採用理由を原論文・現行実装と照合し、旧メモの時間マスク・IG・評価単位の混同を訂正。
- 同日追記：[モデル内の代替手法比較](notes/2026-09-19_explainability_method_selection.md)でGrad-CAM++・Score-CAM・LayerCAM、木の重要度・KernelSHAP、深層の勾配・SmoothGrad・attention等と比較。通常版が最良と実証済みとは扱わない。[着手用の小タスク](research_plan/2026-09-19_next_small_tasks.md)は未実施作業を入力・成果物・分岐で列挙。
- [ノイズ曲線の追加監査](../experiments/2026-09-19_noise_recovery_review/README.md)：22 kHz・run151715の7条件の保存予測を同じ1080秒で再計算。谷型は等重みでも残り、重みだけを原因とできない。現行データの旧振幅shortcut再発を裏付ける証拠なし。`clean_only`による固定モデル対照を原因識別の次候補とした。

- 本人の発表完了報告を受け、[教授との議論](research_plan/2026-09-18_professor_discussion.md)に原メモ、修論目的との対応、説明性の活用と限界を整理。
- 本人が指定した[最終スライドの計画](research_plan/2026-09-18_presented_master_plan.md)を現行PPTXからテキスト化。9月〜翌2月の工程を優先し、「アンサンブル手法の検証」へ名称変更。正式日程は未確認。
- [次工程と達成条件](research_plan/2026-09-19_research_actions.md)をAI提案として分離。既存予測と学習状態の診断→選別対照→固定モデルの雑音評価、必要時のみ方式・帯域を追加。研究完了と手法の優位を別の条件として定義した。
- 原本・既存結果は保持。追加学習・コード変更は未実施。

## 2026-09-17 6/11学習・6/18評価のcrossfit結果を解析

- 完成している3 kHz runは学習6/11のみ、評価6/18、150 epochs、seed 42。ピーク選別は有効だが1,080秒を全保持し、このrunでは選別効果を評価できない。22 kHzは完了印と性能CSVがなく対象外。
- 内部OOFのWAV単位MSEは、各WAVを学習に含めない予測を作り、60秒の加重予測を中央値へ集約してから、18 WAVの熱流束二乗誤差を平均した重みfit用指標。6/18性能ではない。
- 6/11内部OOFではCNNのMSE 4.114×10^9がRF 7.021×10^9、AlexNet 7.066×10^9より小さく、crossfit重みはCNN中心となった。
- 6/18のWAV中央値ではRF R²=0.842が全域最良。shrinkage統合はR²=0.621で、CNN単体より改善したがRFを超えない。CNNはONB以上R²=0.967の一方、ONB前を平均203.9 kW/m²高く予測した。
- 元WAV分離で内部リークは解消したが、同日未知WAVのOOFだけでは日間校正ずれと別日のモデル順位逆転を重みへ反映できない。
- [条件、内部OOFの数式、性能、ONB、説明性、次の識別比較](../experiments/2026-09-17_onb_crossday_result_analysis/README.md)。

## 2026-09-16 別日300 epochs結果を診断し、内部WAV分離・crossfit対応へ修正

- 6/11+7/9学習1,649秒、6/18テスト全1,080秒の外側分割は正常。二度の読込表示は学習日2日のため。
- 3 kHzでRF R²=0.818、CNN+Transformer=-0.084、AlexNet=0.688、旧統合=0.534。CNNは低熱流束を約45万 W/m²へ持ち上げる一方、高熱流束R²=0.821で、全域未学習ではなく別日低域バイアス。
- 旧内部chunk KFoldは3 foldすべてで29 WAVを共有し、内部最良のCNNへ46.3%を付与。主設定をWAV分離とcrossfit 3方式へ変更し、固定ピーク選別との併用を実装。
- IG警告は4096点上限での品質未収束で例外ではない。旧積分破綻より改善済みだが、未収束mapを確定図に使わない。
- output/evaluation/explainabilityとモデルregistryを`utils/config/onb_defaults.py`へ移動。入力ログもパス・件数・shapeを明示。
- [結果、原因段階、修正、次の識別比較](../experiments/2026-09-16_onb_result_diagnosis/README.md)。

## 2026-09-16 追加説明を受け、ピーク高さでの選別へ修正

- 本人の意図は周波数帯の面積ではなく、1,000・1,500・2,300 Hz付近の山の高さに閾値を設けて弱い前後を含めることと確認した。
- 3帯域を全秒で比較し、現行は2,100–2,500 Hz内の最大PSDに横線1e-9を置く。日別分位点は使わない。これは探索的候補で、最適値と確定していない。
- 2,940秒の線形図、49時系列、横線を変えられる画面を作成。7/9・643.52 kW/m²で横線10/1/0.3により5/13/16秒が残る。学習2日で1,649秒、テスト1,080秒は全保持。
- 62テスト、全105 manifest、実3モデル2 epochsの動作と保存予測を検算。本性能の改善は未確認。発表案・SOAPも現行方式へ更新した。
- [実装・図・検証の根拠](../experiments/2026-09-16_peak_height_selection/README.md)。以下の分位点ルールは先行時点の履歴。

## 2026-09-16 別日評価・通常KFold・音響選別へ方針変更

- 本人の追加指定で学習6/11+7/9、テスト6/18。学部コードは1秒配列への通常KFoldと確認し、学習内の性能重み決定に採用。テスト日をモデル・前処理・重みのfitから分離した。
- 49 WAV・2,940秒の全スペクトルを出力・照合。2–3 kHz帯域パワーのONB前99パーセンタイルを暫定条件とし、学習1,860→1,648秒、テスト1,080秒は全保持。7/9の最初の2陽性WAVが全除外となるため、実験上のONBと音響変化の対応は未確定と記録。
- 3モデルで選別なし/ありの2 epochs動作確認を実施。本学習の性能改善と新方式比較は未実施。新方式はコードを保持し、基準結果を読むまで保留。
- 12/24頃に研究終了、1月執筆・1月末発表という本人の見通しに合わせ、後期WBS・修論構成・発表案・SOAPを更新。教授の承認・正式締切は未確認。
- [実装・解析・検算](../experiments/2026-09-16_day_split_spectral_selection/README.md)、[後期計画](research_plan/2026-09-16_second_semester_plan.md)、[修論目次](research_plan/2026-09-16_master_thesis_outline.md)。

## 2026-09-16 IGの数値積分と収束診断を修正

- ゼロbaseline・log-powerモデルに対する旧64分割積分の大きな誤差を解析解で再現。raw直線経路を保った積分点配置とGauss–Legendre積分、合計/mapの収束確認を追加した。
- 点数上限4096、勾配batch8。未収束を明示し、旧出力と新診断を区別。単位逆変換、正負の寄与、相互作用、保存、補助診断等の専用8テストと既存44テストが成功。
- 9/14の学習済み重みはrun配下に未発見。旧説明画像は保持し再計算していない。実入力＋ランダム初期化の現行構造による数値検証を別記。
- [変更・検証・次回の使い方](../experiments/2026-09-16_ig_numerical_fix/README.md)。

## 2026-09-16 本人指定の9/14結果に限定して学部との差を原因分析

- 06.11・3/22 kHz×無雑音/0/−20のみ。9/15 runは対象外。18 WAV・1,080 chunksの対応と全5方式の中央値を再検算。
- 共通残差、実際の統合予測の二乗誤差分解、重みと外側成績の不一致、学習ラベル範囲外の端点を確認。22 kHzのinner回復はほぼ無加熱1 WAVの寄与が全体減少の187.6%、他17本は合計悪化。
- 全集合マスクとIG179/180例の不整合から、入力感度と物理的な説明を区別。2024への現行適用、年×処理の対照比較は未実行の検証案。
- [詳細・図・再現コード・CSV](../experiments/2026-09-16_sep14_cause_analysis/analysis.md)。既存9/15報告案の数値はこの6条件に置き換えていない。

## 2026-09-16 crossfitアンサンブル3方式を選択式で追加

- 本人の添付した2件の提案を現行コード・標本数・中央値の評価順序と照合し、subset選択、WAV損失の重み最適化、等重みへの縮小を追加した。
- 共通の元WAV分離inner OOFを外側学習側だけで生成し、clean_onlyは各評価ノイズへ同じ重みを適用する。既定の2方式・主表示は維持。
- CPUで44テスト成功。4学習方針の分割、保存、再開、旧予測との一致、小型Kerasの実学習を確認。実データ300 epochsでの性能改善は未検証。
- [実装・検証記録](../experiments/2026-09-16_crossfit_ensemble/README.md)、[5方式の手法と数式](ensemble_methods.md)、[未実行の代表比較案](../configs/experiments/2026-09-16_crossfit_ensemble.yaml)。

## 2026-09-15 比較解析後の文書全体の整合性を更新

- 本人の依頼により、Markdown・PDF・Word・PowerPoint計322文書を棚卸しし、抽出可能な本文を検索、現在の方針に関係する記述を照合した。画像中心PDF2件と過去の全図の目視未確認は監査に明記。
- 研究の中心を、学部の観察を基礎としたアンサンブルの劣化抑制、その成立条件と帯域差の理由の検証として統一。希望する結論と確認済みの事実を分けた。
- 現在地・年間計画・金曜のSOAPと発表案・手法メモを2帯域比較へ更新。旧計画のRF優位、旧ノイズ原因、未実装・未完了の記述には適用時点と後継を追記した。
- 既存データの限定比較と、新しいONB実験の準備を並行する案へ整理。新規学習は起動せず、本人が対象外とした進行中runは採用していない。
- [再利用する解析手順](analysis_workflow.md)、[原資料と現在の対応](original_document_guide.md)、[確認範囲・変更記録・検証](audits/2026-09-15_document_consistency/README.md)。

## 2026-09-15 追加3 kHz・22 kHzの比較、本人の修論仮説を記録

- 本人の追加回答に従い、完了済み06.11・3/22 kHz各7ノイズだけを比較。進行中runは対象外。
- 3 kHzのCNNは無雑音R² 0.9558→−20で0.9194。innerは5/7で最良単体を上回り、−20で0.9234。22 kHzにも劣化と統合効果はあるが、非単調性が大きい。
- 卒論PDFのFig. 4.2.2/4.2.3、Table 4.2.1/4.2.2を実読。AUC定義・chunk集計を揃えた比較も作成。回帰の統合改善とONB改善は一致せず、3 kHzの統合は全条件で最初のONB点を見逃す。
- 18 WAV×3区間の波形診断、216保存配列の再構成照合。2–3 kHzの帯域内SNRが高いこととRFの依存は整合。224列へのresizeも帯域比較に含まれる。22 kHz回復のモデル内部原因は未確定。
- 3 kHzのIG210例も不整合。両帯域で420/420。手法の説明対象、マスク/積分/ランダム化の限界と次の比較を整理。
- [分析・図表・検算](../experiments/2026-09-15_onb_frequency_comparison/analysis.md)、[本人の理想の結論と背景](research_plan/2026-09-15_master_thesis_hypothesis.md)、[金曜準備](research_plan/2026-09-18_xai_progress_brief.md)。

## 2026-09-15 22 kHz・全7ノイズの完了後解析

- 本人の終了連絡を受け、06.11・22 kHzの7/7完了を確認。3 kHzは別途実行中で集計対象外。
- 35組のWAV性能を検算。CNN＋Transformerは5/7で単体最高、単純平均は4/7、inner holdoutは2/7で最良単体を上回った。最大改善は−8のΔR²=+0.010985。
- 全35組でWAV ROC-AUC=1.0だが32組は最初のONB点を見逃す。近傍1 WAVの制約を併記。
- RFの最大マスク帯域は2–5 kHzが21/21。IG210例すべてで数値不一致。log変換と積分法の組合せを原因候補として補助数値例を作成し、モデル全体の原因確定とは区別した。
- [解析・図表・検算](../experiments/2026-09-15_onb_22khz_noise_sweep/analysis.md)、[発表準備](research_plan/2026-09-18_xai_progress_brief.md)、[SOAP下書き](progress/2026-09-18_weekly_progress_draft.md)。原学習コード・結果は変更なし。

## 2026-09-15 研究全体の入口・年間計画・発表準備を更新

- 本人の希望: 9/18（金）は説明性指標の出力・解釈・結果整理を発表。ONB近傍の検知、新実験、同時計測は発表後。
- 現行コードは06.11・22 kHz・7ノイズ、2統合方式。9/14の保存済み6条件や70条件への拡張計画と区別した。9/15開始runは採取時点で途中。
- 9/14の6条件を検算し、CNN＋Transformerが全条件で単体最高R²。IGは180例中179例で相対不一致0.05超、RFの主要マスク帯域は一貫。数値整合と物理解釈を分けた。
- 参照: [現在地](research_status.md)、[年間計画](research_plan/2026_annual_plan.md)、[発表準備](research_plan/2026-09-18_xai_progress_brief.md)、[結果スナップショット](../experiments/2026-09-15_research_status_snapshot/README.md)、[整理の監査記録](audits/2026-09-15_workspace_review/README.md)。
- 10:16追記: 既に進行していた9/15runで無雑音1条件が完了、SNR 0は途中。固定スナップショットの09:52時点と区別して[追跡記録](audits/2026-09-15_workspace_review/run_state_followup_1016.json)を保存した。

## 2026-09-14 保存階層の変更・一般化評価の実装

- 保存先を周波数／ノイズの順へ変更し、6条件を移行。比較図36図を再作成した。
- configから実験日分割と学習ノイズを選択する4方針を実装し、実データ70条件のmetadataで分割を検証。本学習は未起動。
- 参照: [実装・運用・結果の説明](research_plan/2026-09-14_result_layout_and_generalization.md)、[検証記録](../experiments/2026-09-14_generalization_and_layout/README.md)。

## 2026-09-14 ノイズ曲線追加・今週の発表方針

- 参照: [実行方針と一般化評価の設計案](research_plan/2026-09-14_execution_policy_and_generalization_design.md)、[グラフ見本](../experiments/2026-09-14_noise_trend_graphs/README.md)。
- これまでのデータ・モデル・評価基盤の5段階を完了済みとして扱う。今週は06.11/06.18の2実験日×5周波数×7ノイズ＝70条件で、ノイズ別精度と説明性出力を示す。
- ONB本体へR²・連続ROC-AUC・二値化後AUCのノイズ曲線を追加。chunkとWAVの両方を保存し、コメントを日本語化した。
- この時点では一般化評価は設計案のみだった。同日の後続作業で実装済み（上の実装記録を参照）。本学習の起動と実装は区別する。

## 2026-09-14 現在地と次の作業の監査

- 参照: [research_plan/2026-09-14_current_state_and_next_steps.md](research_plan/2026-09-14_current_state_and_next_steps.md)、[結果スナップショット](../experiments/2026-09-14_status_audit/README.md)。
- 9/2の56条件より新しい9/3の105条件と9/8の3条件を確認。元WAV・修正ONB閾値による評価は出力済み。
- 最新の無雑音22 kHzではCNN＋Transformerが3日とも単体最高R²。06.11/06.18は全方式がONBを1測定点遅れて判定し、全域R²とONB捕捉は別課題。
- IG診断90件でcompleteness相対誤差が0.05を超え、物理的解釈前の数値整合性確認が必要。旧inner holdoutは主張用から除外し、修正後方式と分ける。
- 次の推奨: 最新結果の固定、ONB失敗例と近傍データ設計、XAI整合性、別日・未知ノイズ評価、帯域再学習、その後の統合方法の最終比較。

## 2026-07-24 P0ノイズ診断コード

- 参照: [research_plan/2026-07-24_p0_noise_diagnostic_runbook.md](research_plan/2026-07-24_p0_noise_diagnostic_runbook.md)
- 実装: 総パワー1変数、水流音のみ、固定振幅、source-WAV GroupKFold、chunk実現SNRを同じ診断経路へ追加した。
- 現在地: コードとsynthetic testは完了。実データ生成・15条件のRF評価は未実行であり、ノイズ逆転の原因判定はこれから行う。

## 2026-07-24 中間発表後の進捗監査・説明性出力・ノイズ逆転調査

- 参照: [research_plan/2026-07-24_current_progress_and_priorities.md](research_plan/2026-07-24_current_progress_and_priorities.md), [research_plan/2026-07-24_explainability_output_guide.md](research_plan/2026-07-24_explainability_output_guide.md), [research_plan/2026-07-24_noise_accuracy_paradox_investigation.md](research_plan/2026-07-24_noise_accuracy_paradox_investigation.md), [../研究進捗報告/2026/724（中間発表）/スライド内容案_0724.md](../研究進捗報告/2026/724（中間発表）/スライド内容案_0724.md)
- 今回の主題: 中間発表の年間計画、現行コード、7/23保存結果、説明性出力、`archive/`の旧前処理を突き合わせ、「実装完了」と「研究上の検証完了」を分けて現在地を再評価した。
- 得られた結果: RF・3モデル統合・モデル別XAIの実行基盤は前倒しで整ったが、アンサンブルはRF単体を総合的に上回っていない。IGは局所安定性が高い一方、no noiseのCNN+Transformerでcompleteness不良があり、最終層ランダム化後もマップ形状相関が高く、物理解釈は未完了。
- ノイズ調査: 現行処理では各元WAVのパワーに比例して水流音振幅を決め、正規化なしの線形powerを入力する。SNR -20では全モデルの最大影響帯が水流音主帯域の10–15 kHzへ移った。archive旧正常系には標本別0–1正規化があり、過去と現在の精度傾向が逆転した有力な差と考えられる。archiveの別世代コードには `y_power + scaled_noise` の不具合も確認した。
- 次にやること: 大規模再学習より先に、総パワーのみ、スケーリング済み水流音のみ、固定絶対振幅ノイズ、log-power、元WAV group分割、chunk実現SNRをRFで診断する。その後、説明性sanity、ONB早期検知定義、3モデル・アンサンブルの再検証へ進む。

## 2026-06-26 各モデルの説明性手法の当たりづけ

- 参照: [research_plan/2026-06-26_xai_method_selection.md](research_plan/2026-06-26_xai_method_selection.md), [6/26報告原本](../研究進捗報告/2026/626/研究進捗報告_柴崎陸_6月26日.docx)
- 今回の主題: 6/19で主軸に置いた「回帰タスクの説明性付与（③）」に向け、各モデルにどの説明性手法が適合するかを先行調査し、現時点の当たりを記録した。実装は未着手。
- 当たり: RF(木モデル)=TreeSHAP（回帰出力に直接）＋帯域/時間マスク、AlexNet=Grad-CAM/IG、CNN+Transformer=Attention(Rollout)+IG+パッチ/帯域マスク。共通軸は周波数帯マスクのΔR2でモデル横断比較。
- 解釈・課題: RFのTreeSHAPは特徴が解釈可能であって初めて物理解釈になるため、②PCA→周波数帯×時間ビン特徴への置換が③説明性付与の前提（②→③の順序依存）。CNN系は入力(224,224,1)を保持しており説明性は付けやすい。
- 並行作業: 6/18新データの`.npy`化、`waterflow_20260622_1s_1`の45,360個確認、DLチューニング環境整備（チューニング本体は⑤＝最後の上積みに位置づけ深追いしない）。
- 確定版（3モデル＋選定理由＋評価指標）: [research_plan/2026-06-26_xai_method_and_metric_selection.md](research_plan/2026-06-26_xai_method_and_metric_selection.md)。原理を「加法的特徴属性（公理を満たす属性法）を各構造に最適化」で統一し、RF=TreeSHAP / AlexNet=IG / Conformer(CNN+Tf v1)=IG に確定。検証は層B（Deletion/Insertion AUC・sanity check・SHAP安定性・周波数帯マスクΔR2）で同一土俵採点。選定理由は①構造整合②回帰の数値分解③定番性④代替案棄却の4段で言語化。
- 次にやること: 説明性手法の当たりを先生に確認→確定、①集計定義の明文化と回帰グラフ精査、②PCA中身確認と解釈可能特徴の試作、整い次第③でRFのTreeSHAP試行。

## 2026-06-19 発表後の研究方針更新

- 参照: [research_plan/2026-06-19_after_presentation.md](research_plan/2026-06-19_after_presentation.md), [../研究進捗報告/2026/619（研究計画発表）/スライド内容案_0619.md](../研究進捗報告/2026/619（研究計画発表）/スライド内容案_0619.md)
- 今回の主題: 6/19研究計画発表後の教授コメントを踏まえ、研究の重心を「どれか1つのモデルを主軸に固定すること」ではなく、「CNN+Transformer、AlexNet、RandomForestをそれぞれチューニングし、性能と説明性まで整えて比較すること」へ修正。
- 実施したこと: 6/19スライド内容案に発表後フィードバックと改訂方針を追記し、今後の計画を「集計定義 -> DL系チューニング -> 3モデルの性能比較 -> モデル別説明性 -> 早期検知評価 -> アンサンブル」の流れに再整理。
- 得られた結果: RandomForestは現時点で高性能だが、CNN+TransformerやAlexNetは入力前処理・学習率・early stoppingなどの設定が十分でない可能性があるため、現状の結果だけで優劣を固定しない方針にした。
- 解釈・課題: 研究の主戦場は分類ではなく熱流束回帰であり、説明性も各モデルの回帰出力に対して確認する。RandomForestではPCAの説明性阻害やno_noiseよりノイズあり条件が高性能になる違和感を別途確認する。
- 次にやること: 集計定義の明文化、CNN+Transformer/AlexNetのチューニング、3モデルの散布図・残差・熱流束帯別誤差、PCA有無ablation、ノイズ付与方法の確認、モデル別説明性評価の試作。

## 2026-06-15 現在状態と次工程

- 参照: [research_plan/2026-06-15_current_state_and_next_steps.md](research_plan/2026-06-15_current_state_and_next_steps.md)
- 主題: RF単体・旧3モデル実行の完了整理、`cnntf_v2` からAlexNetへの差し替え、保存日付修正、fold予測保存の追加。
- 次にやること: `RandomForest + CNN+Tf (AttnPool) + AlexNet` を本番設定で実行し、保存済み予測から重み付けやONB近傍評価を再学習なしで比較する。

## 2026-06-15 パラメータチューニング

- 参照: [research_plan/2026-06-15_parameter_tuning.md](research_plan/2026-06-15_parameter_tuning.md)
- 主題: `MODEL_SPECS` と学習率・バッチサイズの責務を分離し、`PARAMETER_SETS` でモデル別チューニングを管理する。RF単体探索で選んだRFパラメータを `RF_FIXED_PARAMS` として固定し、現在は `cnntf_v1` と `alexnet` の42条件を検証する。

このファイルは、過去の進捗報告をCodexが素早く把握するための索引です。原本は `研究進捗報告/` に残し、ここには要約だけを追記します。

## 2026-05-22 発表後コメント整理

- 今回の主題: 研究タイトルに含まれる「アンサンブル学習」と、発表で強調した「ONB近傍の物理的に妥当な特徴を見る」という説明の関係を再整理。
- 先生からの指摘: アンサンブル学習とタイトルにあるが、今回の説明ではアンサンブルを行わないようにも聞こえる、研究目的は結局何か、今後どの優先順位で検証するのかを明確にする必要がある。
- 整理した方針: アンサンブル学習は外さず、単体モデルに依存した不安定性を下げ、ONB近傍検知の精度・頑健性を評価するための主要な検証対象として位置づける。
- 解釈・課題: 「精度の高いアンサンブルを作る」だけでは修士研究として弱く、アンサンブルで改善した結果が沸騰音の時間・周波数特徴に基づくかを説明性評価とマスク実験で確認する必要がある。
- 当時の次の作業: [5/22計画](research_plan/2026-05-22_after_presentation.md)に沿って、評価指標の固定、中心コードの評価条件整理、基準実験、アンサンブル比較、物理的妥当性確認の順で進める。

## 2026-05-21 研究進捗報告

- 今週の主題: 今週の作業結果ではなく、修士研究として何を明らかにするかを再定義。
- 実施したこと: 学部研究との差分、ONB近傍の離散的な沸騰音特徴、STFT/CWT/窓幅変更、AlexNet/Conformer、XAI・マスク実験の位置づけを整理。
- 得られた結果: 修士研究の芯を「精度が高いモデル探し」ではなく、「ONB近傍の沸騰音特徴を物理的に妥当な根拠として使えている入力表現・モデルを明らかにすること」と整理。
- 解釈・課題: Attention map単体では説明性が不十分なため、Attention Rollout/Flow、Integrated Gradients、Transformer Attribution、RISE、時間・周波数マスク、sanity checkを組み合わせる必要がある。
- 次にやること: 明日の発表で研究方針と教授コメントへの対応を示し、既存データでGrad-CAM、Integrated Gradients、Attention Rollout、RISE、マスク実験、時間シャッフルのプロトタイプを作成する。

## 2026-05-15 研究進捗報告

- 修士論文の結論の方向性を検討。
- 学部研究は「スペクトログラムとCNN回帰による沸騰検知の基本的成立性の確認」と整理。
- 修士研究は「沸騰開始点近傍の離散的特徴を、入力表現やモデルがどう捉えるかを解析し、強ノイズ下・条件変化下でも物理的に妥当な根拠に基づいて早期検知できるかを検証」と考えている。
- 実験データ量として、サブクール度、細線条件、ONB近傍刻み幅などの条件設計を検討中。
- Attention mapの説明性に限界があるため、時間方向シャッフルによる依存性確認、Grad-CAM、RISEなどを検討。
- 先週は実作業が少なかったため、O/Aは大きな更新なし。

## 2026-05-08 研究進捗報告

- データの流れを整理。
- 実験データは過去実験データを一旦使用。
- 前処理はスペクトログラム中心。
- モデル構成は CNN / CNN+Transformer / RandomForest が以前の流れ。
- 研究計画時の議論として、「AIの必要性」「目的が手段化していないか」「物理に即した判断ができるか」を再検討。
- 今後は、モデルが沸騰現象の特徴をどう捉えているかを調べる。

## 更新ルール

新しい報告書を作成したら、次の形式で追記する。

```text
## YYYY-MM-DD 研究進捗報告

- 今週の主題:
- 実施したこと:
- 得られた結果:
- 解釈・課題:
- 次にやること:
```
