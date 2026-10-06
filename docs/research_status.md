# 研究の現在地と次にすること

更新日: **2026-10-06（通常ONBへ5モデルを組込み、指標差・役割・削除対照を確認）**。分位点ルール時点の状態は[変更前の記録](../experiments/2026-09-16_peak_height_selection/previous_documents/research_status.md)に保存した。

**本人の最新指示：従来のRF＋Conformer＋AlexNetは固定し、4・5モデルへの追加を主軸にする。** 前回のRFを外す3モデル採用は、この希望の主方針から外す。追加の人手情報は得られない前提を維持し、モデル調査・比較・判断はAI側で進める。[元3固定の追加比較](../experiments/2026-10-06_fixed_three_additions/README.md)、[後継判断](../configs/experiments/2026-10-06_fixed_three_additions_decision.json)。元モデルの予測と元3内の比率を保持し、元3ブロックと追加モデルの配分を決める。

**追加比較は実行済み**：同じ保存seed43/44で周波数34の正則化HGB・ExtraTrees・RF・通常GB・XGBoost・直接Ridge・PLS・Kernel Ridge、時間46 HGBの9候補をnested clean学習。648fit、保存72モデル、87統合/単体×7条件×2seed、3654指標行を取得した。元3と基準HGBのOOF/外側配列は全件完全一致、再学習なし。時間46の旧clean2160入力との一致、保存72モデルの全7条件再読込、正の4/5メンバー・保持比率・指標再計算、関連unittest3件を確認。

**研究上の主候補は元3＋周波数HGB＋ExtraTreesの5モデル**：clean OOFで第5候補を選ぶと両seedでExtraTrees。元3にも同じMSEをfitした基準に対し、clean RMSE35.74→24.67／33.89→25.26、MAE26.70→17.24／24.12→17.16、FN32→25／28→19、全7条件FP0。−20 RMSE81.41→76.31／81.41→73.57、6雑音平均69.39→65.63／67.49→63.41。元3の従来performance内部比率を保つ対照でもclean RMSE36.04→25.05／35.30→25.74へ改善した。

**5モデルの追加利得には限界もある**：元3＋ExtraTrees4との差はclean RMSE0.07/0.11、FN1/1で小さく、WAV単位の記述的区間は0を含む。seed43 clean6/18 q100はHGB4の322.11から5では376.32へ戻る。低熱流束cleanとseed44−20近傍は悪化。強雑音の絶対誤差は改善してもcleanからのRMSE増加量は縮小していない。元3＋ExtraTrees4を僅差の最有力対照、元3＋HGB4をq100対照として保持する。元3総量25%以上・追加各5%以上は今回の保持制約であり、最適比率の断定ではない。

**HGBがRF一般より良いとはしない**：同じ34特徴RFのclean RMSE29.93/29.97は元PCA画像RF70.97/69.60より良く、−20 FPも0/0。同じ34特徴のHGBはclean28.33/28.88だが−20は106.37/72.41・FP88/0で順位が分割に依存する。ExtraTreesはclean25.76/26.68・−20 RMSE74.72/73.28・FP0/0。直接Ridge/PLS・時間HGBの強雑音失敗も確認。[モデル調査](../experiments/2026-10-06_fixed_three_additions/model_research.md)はLightGBM/CatBoost/TabPFN等の未検証候補と実測9候補を区別する。

**通常ONBへ5モデルを組込み・動作確認済み**：[実装と検算](../experiments/2026-10-06_onb_five_model_integration/README.md)。HGB/ExtraTreesをモデルutilsとregistryへ置き、生power→34特徴の保存Pipelineを学習・推論・再読込・共通周波数maskへ接続した。通常の`run_ensemble_regression_onb.py`は元3＋HGB＋ExtraTreesが既定、同じ1回の内部OOFで元3MSE/performance・HGB4・ExtraTrees4・5・5等平均を出す。従来の統合日平均ONB閾値は保持し、日別閾値の全指標・差・日別q100も別CSVへ追加。TP/FP/TN/FN/FPRと領域母数も通常指標に保存する。

**各指標差とモデル役割を確定**：[指標差](../experiments/2026-10-06_onb_five_model_integration/metric_changes_summary.md)。clean Recall89.84→92.06%／91.11→93.97%、F1 0.94649→0.95868／0.95349→0.96890、Accuracy94.07→95.37%／94.81→96.48%、Precision100%とFP0は不変。新統合の7条件照合は最大差7.96×10⁻¹³ kW/m²、追加2Pipelineの28条件15120予測も一致。保存予測のモデル削除70条件を取得し、Conformer/ExtraTreesの有用な寄与、AlexNetの回帰/FN交換、RFを外すとRMSEがわずかに下がることも確認。[教授向け採用理由と段階案](notes/2026-10-06_five_model_rationale_and_next_steps.md)は5モデル全ての必須性を断定しない。

**実装確認と本実行を分ける**：実際のONB mainを元5構造・1 epoch/内部2fold・実入力144で実行し、15fit、最終5保存/再読込、clean/−20の11予測列と各33日別指標行を確認した。この値を研究性能に加えない。通常設定は150 epochs/3fold/seed42で、新しい通常150 epochs本runは今回未実施。次は標準保存形式の本runを1回確認し、3/4/5の交換を読む。既存の150 epochs候補検証を未実施として再学習しない。追加モデル数・全grid・gateを自動追加せず、物理的音源の同定・未知日検証・追加人手確認を進行条件にしない。

以下は元3固定の最新指示を受ける前の採用判断・追検証記録。RF置換の主採用と当時の次工程は上段で更新済み。

**本人の最新指示を固定**：原録音の追加人手確認・実験メモ・同期映像から新しい事実は得られない前提で進める。それらを本人の宿題や進行条件へ戻さない。採用判断もAI側が引き受ける。[候補・統合の採用判断](../experiments/2026-10-06_model_adoption_decision/README.md)を作成し、**周波数34特徴HGB＋Conformer＋AlexNet、clean OOFのMSE最小化・非負重み**を研究上の主方式として採用した。元のRF＋Conformer＋AlexNetは基準対照、clean Ridgeは最有力対照、雑音OOF方式は雑音へ露出する別条件の対照として残す。新しい数値的な誤報許容値を設けたのではなく、繰り返しの回帰/FN改善、q100、FP抑制、clean_onlyの研究目的を合わせた判断である。

**本人依頼の4項目の計算・比較・文書化を完了**：[固定HGB候補の追検証](../experiments/2026-10-06_hgb_followup_validation/README.md)。seed43/44で外側・内部chunk splitと全モデルを同じ条件で作り直し、基礎32 fit、CNN16 fitの150 epochs完走、clean-fit雑音OOFと19方式×7条件の評価を取得した。同じMSE統合の既存3→HGB追加でclean RMSE35.74→27.36／33.89→26.51、MAE26.70→19.83／24.12→18.74、FN32→25／28→21、FP0を維持。両seedで26/36 WAVの二乗誤差が改善。seed44ではq100が変わらなくても利得が残り、q100のみを採否条件にしない本人方針を継続する。

**新課題は強雑音の分割感度**：HGB単体の−20はseed43 RMSE106.37・FN30・FP88、seed44 RMSE72.41・FN53・FP0。seed43の誤報は6/11に集中し、clean-MSE統合はFP0に抑えるが全域RMSE81.41→90.02へ悪化、seed44は81.41→73.83へ改善した。学習chunkを固定してHGB乱数だけを4通り変えた予測差は0。全内部fitに全36 WAV・全段階があるため、ONB学習段階の欠落とは異なる不足。雑音時の特徴分布のずれは確認したが、誤報の物理的原因と特定分岐の同定は未確認。

**説明と統合の比較も実施済み**：[共通chunk説明](../experiments/2026-10-06_hgb_followup_validation/explanation_findings.md)。HGBはパワー/比率/周波数形状への依存が強く、時間順序に不変。4モデルの共通入力加工32例、保存HGBの追加seed診断を実行した。多様性項は両seedでclean FN/q100の追加改善なし。clean Ridgeは回帰・見逃しをさらに改善するがseed44 FP1。雑音OOFを使う固定重みはHGB比率を約63–64%→23–25%へ下げ、seed43の悪化を縮小するがcleanと見逃しの交換がある。基礎モデルがclean固定でも統合は雑音へ露出しているので、純粋なclean_onlyと分ける。RFはPCAスレッド数への感度が残り、今回の固定2スレッドでは全56保存予測組を再現した。元主runnerの採用設定・過去matchedは変更していない。

**次は主方式の通常実行への組込みと、学習側の限定対照**：主採用の3モデルMSEは比較用入口で実装・評価済み。4モデルMSEのRF重み0を受け、全7560予測を照合して二値判定一致、最大差約0.00024 kW/m²を確認した。通常ONB主runnerの既定はまだ元の3モデルであり、採用文書を書いただけで設定が切り替わったとは扱わない。新たな候補比較はまず周波数HGBの正則化、同じ34特徴のExtraTrees・直接Ridge/PLS、次に時間HGBの再現確認。現在のSVRは強雑音の崩れから主候補から外す。38行の表と34録音抜粋は既存診断資料として保持し、本人の追加確認は不要。未知日検証や物理的音源の同定を今回の進行条件にしない。

以下は追検証前の同日記録。最新の完了状態と新しい強雑音の制約は上段を優先する。

**本人の補足：q100は重要だがモデル追加の唯一の採否条件ではない**：[他指標と公平な対照による再評価](../experiments/2026-10-06_model_addition_general_metrics_review/README.md)。回帰RMSE/MAE、RecallとFP、雑音下性能も追加モデルの中心評価として扱う。既存3にも同じMSE重みをfitする対照を追加し、HGB統合のclean RMSE32.41→28.46、MAE25.07→19.50、FN35→24、FP1→0を確認。clean6/18 q100は既存3の重み調整だけでも376.32になるため、追加固有の恩恵は主に回帰と判定改善として説明する。0/−4の6/18 q100には434.02→376.32の追加利得がある。

**次の優先は有望候補の再現性と改善/悪化領域の説明**：HGB単体、既存3 MSE、Conformer＋HGB、Conformer＋AlexNet＋HGBを固定対照として本比較へつなぐ。RFは追加あり重みほぼ0で、4モデル化だけでなく置換・少数モデル化を検討する。6雑音平均RMSE68.40→66.41、−20は78.99→74.17だが、−4全域RMSEと強雑音近傍は悪化。HGB単体の−20 FN49に対し統合61なので、候補価値と統合価値を分ける。既存モデルへの同方式7対照・441指標行を解析済み、新しい元モデル学習や主runner変更はしていない。残るq100最後のchunkの識別は並行課題であり、他指標の候補価値を確認する前提にしない。

以下の同日予備比較・q100中心の次工程は当時の記録。最新の評価方針と優先順位は上段を優先する。

**本人依頼の1〜4を実行、周波数HGBに有用な補完**：[追加モデル・入力・統合比較](../experiments/2026-10-06_chunk_complementary_model_pilots/README.md)。新chunk条件でSVR/HGB×10/周波数34/時間46特徴×3選択方針の18候補をnested学習し、学習側で固定した29統合・対照方式を同じ外側540の全7条件へ適用した。周波数HGBはclean OOF共通FN52の22を訂正。MSE重み統合の外側cleanはRMSE33.75→28.46、FN40→24、FP1→0、6/18 q100434.02→376.32、6/11は368.98不変。6雑音平均RMSE68.74→66.41、−20は80.67→74.17、FP0を維持するがq100は不変、近傍94.75→112.27に悪化。既存2深層モデル対照のclean6/18 q100322.11も併記し、外側q100だけで追加必要性を断定しない。

**多様性の利得と残る失敗を分けた**：相関罰則付き重みは通常MSE重みよりclean FN24→23、−20 FN61→58だがq100同じ、clean RMSE28.46→28.94。輪講のsociety entropyの直接再現ではない。学習側6/11の315.47 chunk28/36は今回候補すべて陰性で、既存も含む非負平均では救えない。HGB固定15対照では周波数追加の−20 RMSE98.32→75.00、時間追加はFP1→92で、入力と設定の感度が新課題になった。切片付きRidgeでも6/11 q100不変。q100と誤報を併記する本人回答を反映し、誤報許容値は未設定。

**次は固定候補の確認と6/11の弱いchunkの識別**：周波数HGB31＋MSE重みを本比較候補、多様性重みを対照として保持する。最後の共通陰性を元波形/時間順序を持つ入力で調べ、強雑音重みを選ぶなら既存3モデルのclean-fit雑音OOFを新たに得る。matched OOFで代用しない。主runnerは既存3モデルのまま、追加モデルと統合は保存済みの予備比較実装。予備学習・評価は完了しており、本人の再実行は不要。指標1272行・到達段階848件の検算、保存予測再現を通過。現時点の開発比較であり、採用確定と独立な再現性確認を区別する。

以下の同日「次は候補比較」「新学習なし」は基準解析を終えた時点の記録。最新の実施結果は上段を優先する。

**修正後clean_onlyが全7条件完了、q100は不変**：[対応解析](../experiments/2026-10-06_chunk_clean_baseline_analysis/README.md)。10/5の`192410`と10/1旧clean_onlyは同じ外側540・学習1620 ID、同じ採用値、選別なし。新3最終モデルの保存再読込差0、Conformer150 epochs/batch12とAlexNet150/batch8完走。performanceはclean RMSE35.07→33.75、FN43→40、FP1→1、6雑音平均RMSE69.30→68.74。一方−20 dBはRMSE82.46→80.67でもFN64→65、近傍85.94→94.75へ悪化。日別q100は全7条件で6/11=368.98、6/18=434.02 kW/m²、14比較セルすべて不変。内部分割とPCA数値互換性修正を含む比較で、分割だけの因果効果とは断定しない。

**内部支持を改善しても最後の共通陰性が残る**：新clean OOFのRMSE65.29→36.24、FN133→104、全3共通FN100→52。ただし検証課題が未知WAVから既知WAVへ変わった差でもある。最後の陰性は6/11の315.47に4/45（全4共通）、6/18の376.32に2/45（共通1）。同じWAVの成功chunkよりpowerと時間変動が弱い傾向が残る。最終Conformer/AlexNet予測は全7条件で旧結果と完全一致し、RF予測と重みが変化。重みはRF21.99→15.57%、Conformer42.59→49.13%、AlexNet35.42→35.30%。

**次は新chunk条件の追加モデル比較、既存組合せも対照**：[条件案](../configs/experiments/2026-10-06_chunk_complementary_model_pilot_plan.json)。学習側同じ10特徴のSVR/HGBを小さく比較し、最後の共通陰性の訂正・高い段階の追加陰性・日別q100/g100と誤報を読む。必要なら1秒内の周波数形状/時間分布を一つ追加する。外側cleanの6/18はConformer＋AlexNet等平均でq100=322.11になるが、学習OOFのq100は両日不変なので採用確定に使わない。clean_only −20のRFはFP215/225で、逆方向出力だけを補完とはしない。matched保持、全再探索/未知日/blockを自動追加しない。今回の解析では主設定・モデルを変更せず、新学習/XAIは未実施。

以下の10/5の未実行・停止状態は当時の記録。最新の完了結果と次工程は上段を優先する。

**本人実行はRF保存で停止、原因修正済み**：[エラー対応記録](../experiments/2026-10-05_rf_reload_pca_layout_fix/README.md)。`183837` runは内部OOFと最終RFを得た後、再読込検証で停止した。PCA成分のメモリ配置による丸め境界の差を同じ8学習chunkで再現し、報告された最大11960.8984 W/m²差と一致。共通変換をC連続配置へ統一し、保存RFを使った再検証は差0で通過。関連49テスト通過。`pca_transform_version=2`を実行hash/保存状態へ記録し、元runは保持した。本学習は再起動しておらず、外側予測・新q100は未取得。次は修正後の主コードを再実行する。新旧差には数値互換性修正も含まれるため、分割だけの効果と断定しない。

**外側・内部ともchunk分割を本人が選択**：[実装・確認記録](../experiments/2026-10-05_shuffled_chunk_internal_validation/README.md)。既知WAV内の未使用chunkに評価としての価値を認め、別日/未知WAVの結果と分けて読む方針。外側は各WAVの60から15をランダムにテストへ回す既存方式を維持。内部performance重みとtraining_oof探索は、外側学習全chunkの通常KFold、shuffle=True、seed42、3-foldへ変更した。各熱流束の割合を揃える層化や時間blockは今回の方式に含めない。主設定は`run.internal_validation_split="chunk_kfold"`、明示WAV対照だけ`wav_kfold`を残す。

**分割と実装を確認、精度改善は未検証**：関連46テスト通過。実データでは外側1620/540と10/1テストID全件一致、内部は1080/540を3回、同じchunk共有0、全36 WAV・両日各18段階が全fitに残る。ONB段階の学習chunk数は6/11が27/36/27、6/18が29/33/28。PCA・scaler・選別はfitのみ、外側テストは内部検証から除外。旧結果の誤再開を防ぐ条件hashと`ic3`保存tokenを追加した。この実装確認時点では研究3モデルの再学習・新OOF重み・外側q100結果は未出力だった。後続の本人実行の状況は上段を参照する。

**本人依頼の実行準備を完了した時点の記録**：[確認記録](../experiments/2026-10-05_chunk_clean_baseline_ready/README.md)。主コードをclean_only・最終学習状態保存/再読込確認ありへ変更し、matchedは切替コメントで保持。実行snapshotへの内部分割設定の転送漏れも修正した。関連9テスト通過、実データ7条件各2160 IDの対応、外側1620/540と旧テストID一致、内部全36 WAV支持を確認済み。この準備後の本人実行と保存エラー対応は上段を参照する。

**次は分割変更だけの本比較を基準にする**：[実装後の段階案](research_plan/2026-10-05_after_chunk_split_next_steps.md)と[実効条件記録](../configs/experiments/2026-10-05_chunk_clean_baseline_ready.json)に沿い、clean_only主軸・旧採用値固定で、このchunk条件の既存3モデルOOF・重み・最終予測を得て、同じ外側540の旧結果と日別q100/g100・誤報を読む。最終モデルは以前から全1620を学習しており、今回変わるのは主に内部一時モデルによる重み推定。最後の陰性が全3モデル共通なら追加モデル/入力表現、単体で補えるなら重み/統合を先に比べる。旧WAV分離の候補結果を新条件の不採用理由に直結させない。matchedは保持。先行提案のWAV均衡・時間blockを必須工程として自動実行しない。以下の「次工程」は先行解析時点の記録であり、最新方針は本段を優先する。

**本人の主評価をq100へ明確化**：[10/5分割・到達段階・候補比較](../experiments/2026-10-05_onb_endpoint_and_grouped_fold_review/README.md)で、「ONB近傍RMSEより、全chunkをONB以上と判定できる最小実測熱流束」を中心に据えた。日別q100/g100とFP/FPRを併記し、誤報許容値を本人の方針として仮定しない。10/4のSVRは見逃しを減らしてもq100は両日不変。旧統合学習OOFの最後の陰性段階は6/11の315.47（5/45）と6/18の376.32（3/45）kW/m²で、ONB段階だけの改善ではq100が動かない。100%は有限標本の全陽性であり、時間遅延や将来保証と区別する。

**全段階支持とWAV分離を区別して監査**：日別各段階1 WAVのため、WAV完全分離で各fitに全段階を残すことはできない。一方、日別熱流束順位の均衡分割は各fitにONB近傍WAV2/1/1を残せた。既知WAVの離れた時間blockを検証する別案では、gap0/1/2の全foldに両日18段階を残せた。同じWAV共有は意図的、同じchunk共有は0。時間block案は分割監査だけでモデル未学習、時間的独立性も未証明。現行外側within_wav_chunkに内部選択の目的を合わせる選択肢として記録し、主設定は切り替えていない。

**SVR以外の軽量候補を追加検証**：同じ10特徴のExtraTrees・HistGradientBoostingを、従来/順位均衡WAV分割とRMSE/q100優先/FP優先のnested選択で予備学習。従来分割HGBは6/18 q100を434.02→376.32へ早めたが、その日のFPは1→31（RMSE選択）、6/11 q100は不変。旧統合75%＋候補25%では全候補・全選択基準のq100が両日不変。分割を均衡化するだけでも改善は保証されなかった。28 foldモデルと36予測集合各1620 IDを保存・検算し、前回SVRの再現も確認。外側540は使わず、旧3モデル再学習・新重みfit・主コード/config変更はしていない。

**次工程**：既知WAV時間blockとWAV分離で内部選択の問いを明記し、周波数形状または時間変動を残す入力を一つ増やした軽量対照から候補を絞る。弱い1秒の最後の陰性を補えるか、その上へ新たな陰性を作らないかをq100と誤報で読む。SVR全般や追加モデル一般を不採用とはしない。主3モデルの分割感度、時間block条件の学習結果、入力表現の有効性、統合後の外側q100改善は未検証。clean_only主軸・matched保持を継続する。

以下の10/4以前の「次」は当時の記録であり、最新の次工程は上段を参照する。

**第2段階と候補の予備比較を実施**：[学習OOF・入力特徴・SVRの解析](../experiments/2026-10-04_training_oof_diversity_diagnosis/README.md)で、7種類各1620 OOFと、clean/−20 dBの学習chunkだけを診断した。cleanのONB〜1.5倍270 chunkで共通陰性100、統合FN133。両日ONB録音が同じ内部foldへ入り、そのfitに近傍録音0という配置を確認。6/11のONBは直前段階より帯域powerが低く、弱い区間に共通失敗が集中した。帯域10特徴SVRをnested WAV分離で試すと共通FN8件を訂正したが、全域RMSE65.29→78.83、FP1→18、近傍RMSE68.35→132.10で、−20 dB転送のFPRも54.07%。絶対power除去対照も改善せず、今回の候補は採用を見送る。旧統合75%＋SVR25%はFN133→123、FP1→4で共通FN訂正0。これらは外側540結果ではなく学習側の探索的診断。新重みfit・主コード/config変更はしていない。

**次は内部ONB配置の感度を分ける**：[対照分割](../configs/experiments/2026-10-04_internal_onb_fold_sensitivity.json)を作成・監査し、2録音のfold交換で各fitに近傍録音1/1/2を残し、WAV分離と外側1620/540を保てることを確認した。モデル再学習は未実施で、配置が原因と確定したわけではない。主条件cleanの採用値を固定した配置対照で重み・共通失敗への影響を分け、その後に必要な特徴・モデルを選ぶ。最新既存runは学習済み状態非保存のため、今回はそのXAIを再計算していない。SVRの予備foldモデル6個は新規保存・再読込を確認済み。

**本人方針と同日先行の課題整理**：[現行結果・説明性・追加モデルの検討](research_plan/2026-10-04_current_challenges_xai_and_model_diversity.md)に、10/3までの結果を反映した。clean_onlyと対応matchedは両方保持し、基本検証はclean_onlyを主軸とする。直近はclean_only結果と3モデルの説明性手法の選択理由を整理し、教授向け相談資料の新規作成を必須にしない。追加モデルは有用な予測を増やす候補として検討するが、逆方向の出力だけでは不十分。学習OOFの残差・共通失敗と必要な特徴診断から候補を絞り、追加後に予測が統合で活きるか確認する。最新主runのXAIは無効で、旧runのIG収束改善・不安定性・Grad-CAM失敗を最新モデルの検証へ流用しない。現行コード/configはmatched・XAI無効のまま。この課題整理時点では新規学習をしておらず、同日後続のSVR予備学習は上段に記録した。

**最新：同条件matchedの実行と分析が完了**：[対応解析](../experiments/2026-10-03_tuned_within_wav_matched_comparison/README.md)で、10/3のmatchedと10/1のclean_onlyを比較した。両系列7/7条件、同じ学習1620／テスト540 chunk、同じ採用値、選別なし、無雑音5方式の予測完全一致、matchedの7別fit・学習側WAV3-foldを確認。performanceのnoise平均RMSEは69.30→63.19 kW/m²、−20 dBのRecallは79.68→84.44%へ改善したが、強noise平均RMSEは77.13→76.69でほぼ不変、ONB近傍は103.54→123.75へ悪化した。日別q100は14セル中13不変、6/18の−12 dBだけ434.02→376.32へ改善。−20 dBのRF単体はRMSE 148.72→76.17、FP 216→1へ改善し、matchedの統合82.65を下回る一方、統合はFPR 0を維持した。残る統合の見逃し49件中45件は全3モデル陰性で、現在の非負平均の重み変更だけでは救えない。単体品質と誤り補完を分ける根拠として記録した。現行configはmatchedであり、今回コード・config・原出力は変更せず、派生解析と文書だけを追加した。

**候補選択の診断は上段で実施済み**：clean_onlyは未学習noise耐性、matchedは再学習込みの適応性能として保持する。学習OOFの残差と代表条件の入力特徴から帯域特徴SVRを絞り、予備比較では採用を見送った。新統合を自動的な必須実験にせず、次は上段の内部配置対照を優先する。現行条件の物理的原因・追加モデル一般の有効性・独立な新統合利得は未検証。修論準備は並行し、未知WAV/日を必須runへ戻さない。10/4の[段階案](research_plan/2026-10-04_current_challenges_xai_and_model_diversity.md)の未実施記載は作成時点の状態であり、同日後続の結果は上段の解析を参照する。

**外側within_dayを廃止（10/3本人方針）**：[実装・確認記録](../experiments/2026-10-03_remove_within_day/README.md)のとおり、主configの外側選択肢を`within_wav_chunk`と`cross_day`へ整理し、WAV単位日内holdout・ONB層化・旧日内K-foldの実行経路を削除した。現有データは前者、新実験データ等で別日評価を行う場合は後者を使う。内部の学習側WAV GroupKFold、performance重み、OOFチューニングは維持する。関連60テストと実データの分割を確認し、学習1620／テスト540、両日各270のテストIDは10/1保存予測と全件一致した。採用値・学習方針・結果保存先の形式は維持し、本学習は起動していない。過去結果・条件記録は保持。修論準備の優先順位は以下を継続し、未知日評価を新たな必須runとして追加しない。

**採用条件の誤差分布を確認（10/2本人依頼）**：[領域別解析](../experiments/2026-10-02_current_adopted_condition_error_profile/README.md)に、現行3 kHz・1秒・統合・選別なし・clean_only・performanceの誤差を記録した。−20 dBでは二乗誤差の65.48%が60 kW/m²未満へ集中する一方、ONB前誤報は0/225。全7条件の見逃しはONB〜1.5倍の6段階に集中し、clean43/315、−20 dB64/315だった。全域RMSEで最良でも、cleanのRecallはConformer91.11%に対し統合86.35%と低い。「最適」は現行の雑音耐性を含む採用方針であり全指標の最良ではない。今回の特性分析によって、以下の教授への相談・修論準備を別の主問題へ置き換えない。モデル実行・条件変更なし。

**直近は修論目次案と判断理由の文書化（本人の最新依頼）**：[現行目次・2週間の計画](research_plan/2026-10-02_master_thesis_outline_and_two_week_plan.md)と[主張・根拠・判断台帳](thesis/2026-10-02_claims_evidence_and_decisions.md)を作成した。本人より再来週の研究進捗で教授へ目次案を示すよう求められたと共有された。通常の金曜周期なら10/16を仮の準備目標とし、正式日時は未確認。学部論文の5章構成とONB相対位置の式を原PDFで確認し、条件検討・主比較・要因分析を分けた7章案へ対応づけた。現在書ける結果と、特徴補完による原因説明という未検証目標を区別する。次は目次・暫定結論を本人の意図と照合し、第1〜3章の現行条件版と主図の出典整理を進める。最新系列の残差分析は要因説明に必要だが、matched・未知日・再探索を目次準備の必須条件にしない。複数seed・限定した特徴検証は、最終主張に必要な水準を明確にして選ぶ。教授の了承・本文完成・追加検証完了は未記録。コード・config・既存runは変更していない。

**直近は後期計画と教授への説明の整理を優先（本人の最新指示）**：[目標・助言・進捗の再監査](research_plan/2026-10-02_second_semester_goal_progress_and_professor_feedback_audit.md)で、6/26～9/25の原報告、7/24・9/18の発表本文、本人提示の9/18教授メモ、既存解析を照合した。「低FPRを保ちONB直後の見逃しを改善」はAIが10/1結果から導いた候補課題であり、教授と共有・合意済みの主問題ではない。次の優先順位は、元のnoise精度逆転の疑問にどこまで答えたかを整理し、9/18後期計画の4本柱へ成果と残りを対応づけ、主評価範囲・成果指標・説明性の活用を教授へ相談できる形にすること。評価・選別・単体/統合比較は大きく進んだが、現行主条件の原因説明と修論への統合は未完了。旧生成の振幅経路の修正、matchedと固定cleanの区別、bias相殺による見かけの回復を分けて報告する。教授の口頭返答がない資料から了承を推定しない。以下の10/1以前の次工程は当時の提案として保持し、matched・seed等の追加runを直近の必須作業として自動再開しない。onbの3 kHz・1秒・選別なし・統合within_wav_chunk・clean_only設定と既存結果は維持した。

**3 kHz採用値の外側通常run完了・主帯域を3 kHzに固定（最新）**：[対応解析](../experiments/2026-10-01_3khz_22khz_tuned_outer_comparison/README.md)で、7/7 noise、同一fit、外側学習1620／評価540 chunkを監査した。performanceはclean全域RMSE 35.07 kW/m²で22 kHzの29.97より5.10悪い一方、noise平均は69.30対189.55、強noise平均は77.13対240.08、FPRはnoise平均.0015対.8911、強noise0対1.0で、未学習水流noiseへの頑健性は3 kHzが明確に高い。3 kHzの強noiseでは低〜ONB近傍予測が約160 kW/m²へ圧縮され、FPRを抑える一方でRecallがclean .863から−20 dB .797へ低下した。22 kHzは−20 dBでONB前予測が約471–483 kW/m²となる常時陽性化であり、Recall 1・q100早期化を改善とは扱わない。3 kHzチューニングは従来値に対しnoise平均RMSE 69.64→69.30、強noise78.36→77.13と小幅で、強noiseONB近傍は94.78→103.54へ悪化、q100は14/14セル不変だったため、頑健性は主に帯域差として解釈する。主条件は3 kHz、1秒、選別なし、統合`within_wav_chunk`、performanceへ固定し、次は同じ採用値で`matched`だけを変え、低FPRを保ったままONB直後の見逃しを減らせるか確認する。最終表の前にseed 43・44を再現性確認として追加する。

**22 kHz採用値の外側通常run完了**：[対応解析](../experiments/2026-10-01_22khz_tuned_outer_analysis/README.md)で、選別なし・1秒・6/11＋6/18統合・`within_wav_chunk`・clean_only・外側学習1620／評価540 chunk・7 noise・同一fitを監査した。performanceはcleanで全域RMSE 29.97 kW/m²と等重み34.54、Conformer 32.30、AlexNet 39.11、RF 72.95を下回り、等重みとの差のWAV cluster bootstrap 95% CIも[−7.76, −1.67] kW/m²だった。一方、noise時はONB前biasがclean −8.55から0 dB +162.39、−4 dB +187.23 kW/m²へ反転し、FPRは0.44%から46.67%、88.00%、−8 dB以下100%へ崩れた。noise平均RMSEはRF 111.36、等重み170.38、performance 189.55 kW/m²で、clean OOF重み（RF .1575、Conformer .3942、AlexNet .4483）はnoise耐性を与えなかった。q100は強noiseで0になるが、これは早期検知の改善でなく常時陽性化なのでFPRと必ず併記する。外側結果で22 kHzパラメータを選び直さない。後続の3 kHz外側比較は上段の対応解析で完了した。

**最大周波数別パラメータ設定を実装・3 kHz通常run準備済み（最新）**：[実装記録](../experiments/2026-10-01_frequency_specific_parameter_config/README.md)のとおり、`models.parameter_sets.by_max_freq_hz`へ3 kHzと22 kHzの採用値を独立登録した。選択周波数ごとに専用値を解決し、全リスト1要素なら通常run、複数候補ならその周波数単独の学習側OOF探索へ自動切替する。現在は`data.max_freq_hz_list`で3 kHzだけを有効にし、RF `100/深さ12/0.6/0.6`、Conformer `0.001/12`、AlexNet `0.003/8`による同一外側540 chunk・7 noiseの通常runを実行できる。

**3/22 kHz学習側OOF探索完了・採用値決定**：[対応解析](../experiments/2026-09-30_3khz_22khz_oof_tuning_analysis/README.md)で、両周波数各190候補、外側学習1620 chunk、未使用外側テスト540 chunk、選別なしを監査した。主`performance_kfold`の推奨値は、3 kHzがRF `100/深さ12/0.6/0.6`、Conformer `lr=0.001, batch=12`、AlexNet `lr=0.003, batch=8`、22 kHzがRF `600/深さ6/0.6/0.6`、Conformer `lr=0.0003, batch=8`、AlexNet `lr=0.01, batch=24`。保存OOF統合では22 kHz対3 kHzで全域RMSE 53.50対65.29、ONB近傍65.51対68.35 kW/m²、Recall .866対.859、FPRは両方.0015。全域差のWAV cluster bootstrap 95% CIは[−20.58, −2.48] kW/m²だが、ONB差は[−23.77, 19.52]で未確定。22 kHz採用値の外側7 noise評価は上段で完了した。5/10/15 kHz探索は現時点で追加しない。

**選別なし3 kHz主条件の学習側OOFパラメータ探索を準備（完了済みの条件記録）**：[条件記録](../configs/experiments/2026-09-29_3khz_training_oof_parameter_search.json)のとおり、1秒・統合6/11＋6/18・clean_only・within_wav_chunk・seed 42・選別なしを固定した。候補はRandomForest 135、Conformer 25、AlexNet 30の計190条件で、3モデル間の直積にはしない。結果と採用判断は上記9/30解析へ引き継いだ。

**ONB保護付き2.1–2.5 kHzピーク選別は主条件へ採用しない（最新）**：[対応解析](../experiments/2026-09-29_onb_protected_selection_analysis/README.md)で、選別なしと`ONB +10%保護＋上側1e-9`を同じ外側540 chunk・seed 42で比較した。performanceの−20 dB全域RMSEは88.20→77.69 kW/m²へ改善したが、cleanは35.33→44.20、強noise平均ONB近傍は94.78→116.38、ONB以降は59.71→67.90へ悪化した。強noise平均FPRは両方0、Recallは.807→.808、q100は日別14/14セルで不変。除外64 chunkは保護帯直後の4測定点だけで、改善は低熱流束強noise、悪化はONB近傍・直後へ集中した。この選別は帯域ピーク依存を示すアブレーションとして保持し、主条件は選別なし1秒へ戻す。次は選別なし条件だけで学習側OOFパラメータ探索を行い、全域RMSE 2%以内の候補からONB指標も見て選ぶ。

**修論の評価範囲と次工程（本人方針、最新）**：[範囲と実行順](research_plan/2026-09-29_scope_chunk_length_and_tuning.md)のとおり、未知WAV・未知実験日一般化は現時点の必須主張・直近実験から外す。主対象は6/11＋6/18の既知WAV内未使用chunkと、clean学習から未学習の付加水流noiseへの劣化とする。未知WAV・未知日は限界として明記し、新しい実験データを取得できた場合に固定済み最終モデルの独立評価として再開する。0.5秒対照の結果から主入力は1秒に固定する。ピーク分布監査により、ONBを守りながら上側を選別できる共通数値閾値は存在しないと判断し、「日別ONB +10%以内を保護し、その上だけ`1e-9`」を実装した。次は固定パラメータの通常runとして選別なしrunと比較し、採否後に1秒だけを学習側OOFで小規模チューニングする。外側7 noiseを候補選択へ使わず、最終候補を通常runで一度評価する。

**ONB保持を制約にピーク閾値を再監査・保護規則を実装（最新）**：[閾値監査と実装](../experiments/2026-09-29_peak_threshold_preservation/README.md)で外側学習1620秒の分布を確認した。両日ONBを80%以上残せる最大共通閾値`7.4304e-12`では、除外9秒が全てONBちょうどで、ONBより上は0秒だった。6/11ではONB中央値`7.8396e-12`と上側最小`7.8642e-12`が重なり、上側を1秒でも除くとONB保持が半数以下になる。そこで`ONB +10%保護＋上側1e-9`を実装した。外側学習はONB 90/90秒を保持し、上側64秒だけを除外して1620→1556秒。内部3-foldでも保護帯除外0、held-outは無選別。現行主コードはこの条件、全パラメータ候補リスト1要素、XAI無効で通常runを実行できる。

**統合within-WAVピーク選別と学習側OOFチューニングを実装（最新）**：[実装・事前検算](../experiments/2026-09-29_within_wav_selection_and_oof_tuning/README.md)のとおり、統合manifestの元実験日・元WAVを用いてピーク特徴と日別ONBを参照し、固定`peak_height`を外側学習と内部inner-fitだけに適用できるようにした。実行モデルはRandomForest・Conformer・AlexNetの3つに固定し、モデル選択configは持たない。チューニング専用configも廃止し、`models.parameter_sets.model_grids`の全リストが1要素なら通常評価、どれかが複数要素なら外側学習のWAV非共有OOF探索へ自動切替する。探索はモデル内の候補だけを組み合わせ、3モデル間の直積にはせず、外側テストを読まずに各モデルを独立評価する。採用後は各リストを採用値1要素へ戻すことで、通常runへ自動的に戻る。現行設定は1秒・3 kHz・統合6/11＋6/18・clean_only・seed 42・`ONB +10%保護＋上側1e-9`・固定パラメータで、外側学習1620→1556秒、外側テスト540秒は無選別。選別あり本学習とチューニング本実行は未実施。

**6/11＋6/18統合・0.5秒対照完了（最新）**：[解析記録](../experiments/2026-09-29_within_wav_chunk_05s_analysis/README.md)で7/7条件、clean_onlyの1 fit共有、学習3,240／テスト1,080 chunk、36 WAVを確認した。performanceの2日合算RMSEは0.5秒対1秒でclean 40.27対35.33、noise平均80.71対69.64、強noise92.49対78.36、−20 dB 98.65対88.20 kW/m²となり、日別14/14 noiseセルで0.5秒が悪かった。−20 dBではONB近傍RMSEが46.83対69.06、Recallが.817対.800と改善した反面、ONB前biasが118.16対94.82、FPRが.044対0へ悪化し、低熱流束の上方移動によるトレードオフだった。q100は全noiseで両入力長同一。現行splitでは共通時間区間が540中22だけだが、その2区間平均比較でも−12～−20 dBは0.5秒が12.0～19.1 kW/m²悪かった。主入力は1秒とし、0.5秒専用チューニングは行わない。

**7/9単日within_wav_chunk完了（補助対照）**：[解析記録](../experiments/2026-09-29_within_wav_chunk_0709_analysis/README.md)で7/7条件、clean_onlyの1 fit、13 WAV、学習585／テスト195 chunkを確認した。performance RMSEはclean 103.29→−12 dB 239.66まで悪化後、−16 229.72、−20 219.06へ見かけ上回復したが、biasが−129.43→−108.93→−90.75 kW/m²へ戻る相殺であり6月と同じ単調trendではない。cleanでも571.694／643.517 kW/m²を−145.81／−194.56 kW/m²過小予測し、720.691では+22.73へ急変した。q100は全noise 720.691、FPR 0だがRecallはclean .667、−20 .600。performanceは等重み・事後最良単体に各1/7しか勝たず、7/9は6月pool・チューニングへ混ぜず、720.7 kW/m²で音響応答が切り替わる異なる系列の補助結果とする。

**6/11＋6/18統合within_wav_chunk完了（最新）**：[対応解析](../experiments/2026-09-29_within_wav_chunk_combined_analysis/README.md)で7/7条件、clean_onlyの1 fit共有、学習1,620／テスト540 chunk、単日runとの各日270/270一致を確認した。performanceの2日合算RMSEは、各日を別々に学習したモデルに対しclean 39.57→35.33、noise平均74.77→69.64、強noise87.88→78.36 kW/m²へ改善した。利得は6/18の低熱流束と深層モデルに偏り、6/18 −20 dBは全域116.94→90.92、FPR .042→0だが、Recall .833→.773、ONB近傍25.85→85.08へ悪化した。2日強noise全体でもRecall .825→.807、ONB近傍78.42→94.78であり、誤報抑制とONB付近の過小予測がトレードオフになった。q100は6/11で不変、6/18 cleanは376.320→434.018 kW/m²へ遅れた。performanceは等重みを14/14、事後最良単体を9/14条件で上回ったが、−20 dBは両日AlexNetが最良。次は外側split 42を固定して学習seed 43・44を再現し、全域利得とONB悪化が安定か確認する。平均ONBで保存された判定指標ではなく日別確定ONBの後処理を主とし、matched・0.5秒・チューニングを同時に変更しない。

**6/11・6/18単日within_wav_chunk完了（最新）**：[解析記録](../experiments/2026-09-29_within_wav_chunk_single_day_analysis/README.md)で2 run・14/14条件、各日810学習／270テストchunk、18 WAVを両側で共有、同一chunk重複0、clean_onlyの1 fit共有を確認した。performanceのclean／noise平均RMSEは6/11が42.22／64.35、6/18が36.73／83.81 kW/m²。従来WAV holdoutと共通の75 chunkではcleanが両日とも約8 kW/m²改善したが、これは評価WAVの別45 chunkと同じラベルを学習した既知WAV補間の効果を含む。6/18 −20 dBはRMSE 107.66→86.75、FPR .700→.033に改善する一方、Recall 1.000→.689へ低下した。低熱流束の正biasは残り、全域ではclean→−20 dBが6/11で42.22→85.65、6/18で36.73→116.94。performanceは等重みより平均0.53／1.47 kW/m²良いが、外側最良単体への勝利は5/7／0/7だった。統合時は出典日ごとにseed 42を独立適用するよう修正し、単日runとのテストchunk一致を両日270/270にした。主コードは6/11＋6/18統合・1秒・3 kHz・clean_only・seed 42へ設定済みである。0.5秒、matched、追加seedは同時に変えない。

**6/11＋6/18統合データとWAV内chunk holdoutを実装（最新）**：[実装記録](../experiments/2026-09-29_within_wav_chunk_combined/README.md)のとおり、変更前の`within_day`が元WAV丸ごとholdoutだったことを確認し、本人の想定どおり全WAVのchunkを学習・テストへ分ける`within_wav_chunk`を別方式として追加した。同一chunkの重複はないが、同じWAV固有の録音条件は両側で共有するため、これは未知WAV一般化ではなく既知WAV内の未使用時間区間への評価である。6/11＋6/18の0.5秒151,200 NPYと1秒75,600 NPY（合計226,800、約42.42 GiB）、元名WAV 36本、熱流束名付きWAV 36本、測定メタデータ6件を出典情報付きで新しい統合実験フォルダへコピーした。`learning_policy`は`evaluation_mode`を先に選び、`evaluation_settings`の同名欄だけを編集する構成へ整理した。統合ONBは本人指定により両日確定値の平均246.591 kW/m²。現行設定は1秒、3 kHz、7 noise、clean_only、各WAV 45学習／15テストchunk、外側seed 42、内部3-fold WAV GroupKFold、150 epochである。1秒実データ上で学習1,620／テスト540、36 WAVすべてが両側、同一chunk重複0を確認した。学習runはまだ開始していない。初回はこのclean_only設定を実行し、平均ONBによる統合値に加えて日別FPR・ONB近傍誤差も後処理で確認する。

**100%分類可能熱流束と2日pool案を整理（前段）**：[q100監査とpooled holdout設計](../experiments/2026-09-29_q100_and_pooled_holdout/README.md)で、日内4 runの保存予測を再集計した。現行`q100`は6/11の全70 cellで427.276 kW/m²、6/18の全70 cellで434.018 kW/m²となり、100%条件と測定熱流束間隔のためモデル・noise・方針差を識別しなかった。一方、clean_only −20 dBのONB前FPRは6/11でRF .492／performance .008、6/18でRF .933／performance .775であり、誤報抑制はFPRに現れた。`q100/g100`は卒論との接続を示す補助指標として追加し、FPRを未知noise劣化の主指標とする。同記録の物理コピーしない`pooled_holdout`は後続の本人指示で採用せず、上段の統合実装へ更新した。

**3 kHz日内holdout・clean_only／matched完了（最新）**：[日内4 run解析](../experiments/2026-09-29_within_day_clean_matched_seed42/README.md)で4 run・28/28条件、方針間の同一テストWAV、clean予測差0、clean_onlyの1 fit共有とmatchedの7 fitを確認した。performanceのnoise平均RMSEは6/11で50.76→50.22 kW/m²とほぼ同等、6/18で73.25→64.97へ改善した。ただし6/18 −20 dBの全域改善110.36→81.33は58.39 kW/m² WAVの219.80→35.44が中心で、ONB近傍は21.04→146.13、ONB以上は32.64→90.78へ悪化した。matchedはFPR .775→0、F1 .756→.827、ROC-AUC .849→.918と固定閾値判定を改善した一方、ONB付近を低く外す。noise平均は両日とも等重みがperformanceより0.2～0.5 kW/m²良く、6/18 matchedの内部最大重みと外側最良単体は0/7一致、順位相関−.571。主方式performanceは維持するが等重み・RF対照を外さない。test split seed 43・44は厳密には未実施だが、旧日内GroupKFoldと目的が重なるため直近優先から外した。[問題定義と次の優先順位](research_plan/2026-09-29_problem_definition_and_next_priority.md)では、clean_onlyの未知noise耐性を主問題、matchedを既知noise適応の診断対照とする案を推奨する。次は外側テストを使わない学習側限定チューニングで現パラメータの不足を確認し、固定holdoutでstrong noiseの全域RMSE・FPRを下げながらONB以上・Recallを維持できるかを判定する。

**3 kHz matched双方向・seed 42完了（別日比較）**：[matched双方向解析](../experiments/2026-09-28_3khz_matched_bidirectional_seed42/README.md)で14/14条件、方向ごとに7 noise＝7 fit、clean_onlyとのclean予測差0を確認した。performanceのnoise平均RMSEは6/11→6/18でclean_only 92.6→matched 97.5へ悪化、6/18→6/11で93.8→76.2へ改善し、学習方針の効果は方向依存だった。双方向ではRF 86.8、performance 86.9、等重み86.5 kW/m²で、現パラメータのperformance固有利得はない。matched performanceはFPR .113→.001、F1 .885→.912へ改善した一方、ONB近傍RMSE 91.2→123.1、ROC-AUC .931→.927で、主効果は逆方向強noiseのONB前過大予測を抑える校正移動だった。内部OOF最良と外部最良単体はnoise 12 cell中2一致、順位相関−.333。主入力3 kHz、主方式`performance_kfold`、等重み・RF対照という本人方針を維持する。後続の日内比較は上段の9/29記録へ反映済み。[判断と操作](research_plan/2026-09-28_3khz_matched_tuning_and_evaluation.md)、[日内実装記録](../experiments/2026-09-28_within_day_holdout/README.md)。

**6/11・6/18双方向、3/5 kHz、3 seed完了**：[双方向低周波3 seed解析](../experiments/2026-09-28_bidirectional_lowfreq_3seed/README.md)で84/84条件を揃えた。noiseあり双方向平均RMSEは3 kHz performance 91.5±2.8、等重み92.4±2.4 kW/m²で、等重みはRFを37/42条件で上回った。5 kHz等重みは順方向で最良単体に14/21条件で勝ったが、逆方向では3/21であり方向依存だった。内部OOF最良と外部clean最良単体は12 cell中0一致、平均順位相関−.667のため、`performance_kfold`を主方式とせず、3 kHz等重みを主候補、performanceを比較方式、3 kHz RFを同一入力の単体基準とする。順方向deep modelのnoise改善はclean正biasの相殺を含むためnoise不変とは解釈しない。次は追加seedでなく、clean・−4・強noiseを対象に2.1–2.5 kHz帯と隣接帯域の同幅遮蔽を行い、3 kHz優位の原因と予測変化を識別する。

**6/18 clean学習→6/11評価・5周波数×3 seed完了**：[5周波数解析](../experiments/2026-09-28_two_day_reverse_5freq_3seed/README.md)で、seed 42/43/44、3/5/10/15/22 kHz、各7 noiseの105条件を監査した。noiseあり平均RMSEは5 kHz RF 90.5±1.5が全方式中最小、3 kHzはAlexNet 91.1±6.7、performance 91.1±3.1、等重み92.0±3.0 kW/m²で近かった。22 kHzはRF 128.8±5.9に対しperformance 246.5±20.6、等重み216.3±6.8で、seed 42の高帯域noise崩壊が再現した。3 kHz等重みはRFを20/21条件で上回るが、最良単体勝利11/21・2%以内12/21で暫定14/21基準に未達。SNR 0・−4では3/3 seedで最良単体を上回る一方、−16・−20では0/3でAlexNet単体が優位。内部OOF最良と外部clean最良単体は15周波数×seed cell中0一致、平均順位相関−.700で、performanceを主方式にしない。次は順方向6/11→6/18の3・5 kHzをseed 42で確認し、低周波優位の方向再現後に追加seedと説明性へ進む。

**6/11・6/18のみの双方向転送を確認**：[双方向解析](../experiments/2026-09-27_two_day_bidirectional_clean_analysis/README.md)で、今回の6/18 clean学習→6/11評価seed 42と、既存の6/11→6/18 seed 42を対応比較した。clean最良RMSEは順方向Conformer 75.2、逆方向RF 75.5 kW/m²で、2日限定の日付間転送は成立した。等重みは正負biasを相殺し、両方向でperformanceより良かった。内部OOF最大重みは両方向AlexNetだが外部最良はConformer・RFで一致0/2。noiseあり6条件平均RMSEは順方向RF 101.1・等重み137.1、逆方向RF 135.6・等重み215.8 kW/m²。逆方向ではSNR 0 dBから深層2モデル・両統合のONB前FPRが1となり、深層残差相関もclean .894から−20 dB .997へ上昇した。主対象を6月2日に限定可能だが、clean転送成立と未知noise耐性を分け、後者はRFが基準となる。次は逆方向clean_onlyを同じ7条件のままseed 43・44で再現する。

**clean学習・3日leave-one-day-out完了**：[3方向解析](../experiments/2026-09-27_leave_one_day_out_clean_analysis/README.md)で、2日学習→残る1日評価をseed 42、22 kHz、選別なし、150 epochsで揃えた。clean全域RMSEは、6/18評価でRF 115.1・performance 120.5・等重み108.9、7/9評価でRF 274.4・performance 371.7・等重み353.2、6/11評価でRF 115.2・performance 116.3・等重み105.2 kW/m²だった。6/11・6/18への転送は成立したが、7/9では全モデルが大きく負bias化した。7/9の2.1–2.5 kHzピークと全モデル予測は720.7 kW/m²で同時に急変し、確定ONB 571.7 kW/m²より後まで低出力状態として扱われた。内部OOF最大重みは3方向すべてConformer、外部最良単体はRF・RF・AlexNetで一致0/3、performanceは3/3方向で等重みより悪かった。次はseed・fold追加ではなく、7/9の571.7・643.5・720.7 kW/m²を中心に波形・スペクトログラム・帯域PSD、録音条件、帯域除去/保持時の予測差を確認し、日付ドメインシフトの原因を識別する。

**clean_only・matched 3 seed比較完了**：[3 seed解析](../experiments/2026-09-26_clean_matched_performance_kfold_3seed/README.md)で、6/11学習→6/18評価、22 kHz、選別なし、150 epochs、seed 42/43/44の計6 run・42条件を確認した。noiseあり平均RMSEはmatchedでRF 96.24±0.24、performance 97.73±2.29、等重み97.66±1.33 kW/m²となり、seed 42で見えたperformanceのRF比改善は安定しなかった。最良単体2%以内はperformance・等重みとも11/21で暫定14/21基準に未達、ONB近傍の最良単体超えは両方式0/21。clean_onlyではRFがnoiseあり18/18条件で全域最良、deep・統合は強noiseで誤報が急増した。内部最大重みと6/18全域最良単体の一致は9/21、平均順位相関.214で、順位移送が主な制約。保存OOFによる追加3方式の事後診断にも改善の見込みがなく、本学習は後順位とした。次は追加noise学習ではなく、clean・seed 42だけで残る2つの日方向を評価し、既存の6/11＋7/9→6/18と合わせて日間移送を診断する。

**clean_only・matched新run解析**：[9/26比較解析](../experiments/2026-09-26_clean_only_vs_matched_analysis/README.md)で、22 kHz・選別なし・150 epochs・seed 42の3系列を確認した。同じ6/11学習ではclean予測が完全一致し、noiseあり平均RMSEはperformance統合がmatched 95.09、clean_only 145.26 kW/m²。matchedのRF比改善は0.87 kW/m²と小さく、ONB近傍では7/7条件で最良単体を超えなかった。clean_onlyはRFがnoiseあり6/6条件で全域最良で、deep・統合は強noiseで全陽性化した。6/11＋7/9 clean学習は全モデルの全域性能を悪化させ、内部重みがRFを.076まで下げた一方、6/18ではRFが7/7条件で最良だった。次は6/11→6/18のmatched/clean_onlyをseed 43・44で再現確認し、7/9追加seedより日別校正差の診断を優先する。

**検証・分割設定を単純化**：[9/25実装記録](../experiments/2026-09-25_wav_kfold_and_inferred_day_split/README.md)で、`performance_kfold`の内部検証を常に元WAV非共有K-foldへ固定し、設定から`internal_validation`を削除した。`split_mode`も削除し、学習日・評価日のリスト関係から、同一1日=`within_day`、同一複数日=`leave_one_day_out`、完全分離=`cross_day`を導出する。一部重複は曖昧さとリーク防止のため停止する。現行6/11学習→6/18評価は`cross_day`・外側1-foldとして読込確認済み。本学習は起動していない。

**performance_kfold予備run完了**：[run 143020の解析](../experiments/2026-09-25_onb_run_143020_analysis/README.md)で、6/11学習→6/18評価、3/22 kHz clean、seed 42を確認した。3 kHzでは統合が全域RMSEを最良単体より9.2 kW/m²改善した一方、ONB RMSEは最良Conformerより45.4 kW/m²悪化した。22 kHzの全域改善はAlexNet比0.35 kW/m²に留まり、ONB RMSEは33.4 kW/m²悪化した。全域の正負bias相殺には有効だが、全域R²重みはONBを保護しない。IGは3 kHzで主10例全収束、22 kHz Conformerは3/5のみ収束、Grad-CAMは10/10失敗した。今回のrunは選別`1e-9`あり、内部3-fold、cleanのみ、1 seed、等重みなしであり、作成済みの選別なし・5-fold・7 SNR・3 seed本比較とは異なるため予備結果として扱う。次はclean-only/matchedの主従を固定し、XAIを外した本比較をseed 42で監査してから残りseedへ進む。

**ensemble結果の保存日付階層を整理**：[9/25保存日付階層整理](../experiments/2026-09-25_ensemble_date_layout/README.md)で、今後のONB結果を`regression_result/npy/ensemble/YYYYMM/DD/条件/`へ保存するよう変更した。日付付き既存55系列（389,580ファイル、約34.27 GB）も同形式へ移動し、全系列で移動前後のファイル数・総byte数が一致した。日付情報のない旧系列は推測で移動していない。

**決定論GPUでのIG停止を修正**：[9/25 IG決定論GPU修正](../experiments/2026-09-25_ig_deterministic_gpu_fix/README.md)で、TensorFlow 2.9.1の決定論GPUに推論モードFused BatchNormの逆伝播がないため、ConformerのIGで本runが停止する問題を修正した。IG時だけBatchNormalizationを非fusedカーネルへ一時切替し、切替前後の端点予測を照合して必ず元へ復元する。別の未実装GPU演算時だけCPUへ自動fallbackする。学習・通常予測・重み・IGのbaseline／経路／積分／収束判定は変更していない。現行Conformer・AlexNetの224×224決定論GPUスモークで両方完走し、端点予測差・IG前後の通常予測差はいずれも0、全75テスト成功。これはIGの実行可能性の修正であり、本run各標本の数値収束を保証するものではない。9/25 13:15開始の失敗runはfold予測確定前でモデル保存もなく、完成結果として使用・再開しない。

**9/25進捗報告後の判断保留**：本人作成の9/25 SOAPでは、意図する主方式`performance_kfold`の本結果を見る前にアンサンブルの結論を書かない方針とした。教授向け文書では過去`inner_holdout`をコード不具合とは表現せず、目的とする方式と異なる過去比較として扱う。次の全条件runの前に、主研究質問をclean学習固定モデルの未知noise耐性（`clean_only`）とするか、noise別学習による既知条件への適応（`matched`）とするかを固定する。前者を主とする場合は、作成済みmatched条件を先に実行せず、同条件のclean-only比較を主とする。ピーク選別、アンサンブル方式、学習noise方針は同時に変更しない。詳細は[9/25報告レビューと段階的研究方針](research_plan/2026-09-25_weekly_report_review_and_next_steps.md)。

**matched・noise別重み本比較完了**：[3 seed本比較と重み移送診断](../experiments/2026-09-24_matched_noise_specific_weights/README.md)で、6/11学習→6/18評価、音響選別なし、22 kHz、7 SNRを確認した。3 runとも7/7条件完了、seed内7 `fit_id`は全て別で、`training_noise_dir`と評価noiseが一致し、重みscopeは`per_training_noise`である。7 SNR平均RMSEはRF 96.17、等重み93.54、inner 94.45 kW/m²で両統合が良いが、noiseありだけではRF 96.24、等重み96.11、inner 96.98で、等重みは実質同等、innerは悪化した。最良単体2%以内は両方式11/21で暫定14/21基準に未達。最大重みモデルと評価日の最良単体は全21条件中9、noiseあり18条件中6しか一致せず、noiseありの重み順位と評価日順位の平均相関は0であった。当該runの3-foldはepoch選択用で、重み用inner holdoutは各seed・noiseにつき18 WAV中4 WAVを使う1回だった。単一holdout、残差相関を扱わない逆MSE相当式、有害モデルを0にできないことがinnerの内部制約であり、学習日から評価日への一般化は手法固有の欠点ではなく別日評価全体の前提として区別する。9/25の本人判断により、今後は`inner_holdout`を主設定から外し、200 epoch固定・元WAV非共有5-foldの`performance_kfold`を主方式とする。別validationによるepoch選択機能と、評価fold正解で重みを作る旧方式は実装から削除した。主コードでは実装済みの残り5方式を短い説明付きコメントとして保持する。[次条件と監査項目](../experiments/2026-09-25_matched_performance_kfold/README.md)は作成済み、実行は未着手である。その後、必要なら同一OOFを共有して`subset_equal_cv`等の結合規則を比較し、一方式固定後に6/11+6/18学習→7/9評価へ進む。追加実験・同期映像は取得不可である。

**clean学習・固定noise推論本比較完了**：[3 seed本比較](../experiments/2026-09-24_clean_train_noise_inference/README.md)で、6/11 clean学習済みの同じモデルを6/18の7 SNRへ適用した。全runで7/7条件完了、seed内fit ID同一、再読込予測差0。等重み・inner holdoutのR²は全seedでノイズ強度に伴い単調低下し、matchedで見えた谷は消失した。cleanでは両統合がRFを3/3 seedで上回ったが、ノイズあり6 SNR×3 seedではRFが全18条件で最良。平均R²はRFがclean〜−16 dBで.876〜.879、−20 dBで.766、等重みは.918→.457、inner holdoutは.915→.282。inner重みはRF .129〜.202に対し深層合計.798〜.871で、深層2モデルの残差相関はclean .882から−20 dB .987へ上がり、ONB前の同方向過大予測が統合悪化を生んだ。6/18上の0.01刻み診断では全21 seed×SNRでRFより悪化しない固定凸結合はRF単体だけだったが、これは事後診断で採用重みではない。これは固定clean耐性の診断結果であり、matchedでも同じ重みを共有すべきという結論ではない。

**アンサンブル研究の判断方針（本人の最新見解を反映）**：[位置づけ・評価基準・次の検証](research_plan/2026-09-24_ensemble_research_position_and_next_steps.md)に固定した。重みは将来入力へ共通する定数にせず、学習実験データから毎回求める。`matched`はnoise別familyでモデル・PCA・scaler・重みを独立fitし、`clean_only`だけがcleanの状態を評価noise間で共有する。epochは全条件で`run.epochs`の指定値を使用する。manifestの`ensemble_weight_scope`と検査により、将来matchedでnoise間共有が起きれば停止する。全SNR共通の固定頑健重み案は不採用。上段の本比較により、次の問題はnoise別再計算の有無ではなく、学習日内の単体順位が評価日へ移らない場合にも安定する重み推定へ絞られた。

**9/24追跡監査（本人確認を反映）**：[ONB・選別・アンサンブル・IGの照合](../experiments/2026-09-24_selection_onb_ig_review/README.md)で、現行コード値（6/11=221,505、6/18=271,678、7/9=571,694 W/m²）を正しいONBとして確定した。0番ノートブックがONBと記した368,978／376,320／442,169 W/m²は抵抗―熱流束の自動直線性喪失候補であり、ONBではない。誤値txt 3本を削除し、ノートブックは今後この候補をONBと呼ばず別名の診断txtへ出すよう修正した。現行ONBで再集計すると`1e-9`選別は6/11のONB以降94/660秒を除き、意図どおりONB以降だけを対象とする。CのONB前改善は、弱音のONB以降境界標本を削ったことで共有回帰関数が低値側へ動いた結果と整合し、近傍RMSEは全モデルで悪化した。次の閾値識別は既存`1e-9`に`1e-8`と同数ランダム除外を加える。最大記録熱流束はCHFではない。追加実験・同期映像は取得不可。IGはlog-power経路で数値誤差を改善したが、本学習モデル・複数baselineでの採否は未確認。

## 9/18発表後の現在方針

本人より、現状のスライドと研究状況で発表を終えたと報告された。[教授との議論・修論との接続](research_plan/2026-09-18_professor_discussion.md)に原メモと解釈を分けて保存。今後の計画は本人指定により[現行PPTX最終ページの計画](research_plan/2026-09-18_presented_master_plan.md)を基準とする。項目名は**「アンサンブル手法の検証」**。説明性と各作業の目的・採用理由・達成条件を明確にする。

最終ページは9月〜翌2月の工程。旧文書の1月末発表は確定締切として扱わない。正式日程は未確認。[今後すべきこと・達成条件](research_plan/2026-09-19_research_actions.md)はAI提案として別記した。当初の提案は既存予測で統合の成功／失敗を領域別に整理し、学習曲線を確認して、選別対照の条件書を作ることだった。新手法・帯域細分化は問題と採否判断が明確になった場合に限定する。

**追加監査（同日）**：[現行runのノイズ回復](../experiments/2026-09-19_noise_recovery_review/README.md)で22 kHzの1080評価秒を全SNRで照合。−8→−16 dBの統合R²は.8261→.8838、等重みでも−8→−20 dBで.8373→.8751。曲線の谷は統合重みだけでは説明できず、各SNRで再学習するmatchedの影響と入力／日差を分ける必要がある。ノイズ曲線の原因解明には`clean_only`を選別A/Bより先に実施してよい。[説明性手法の選択理由と修正](notes/2026-09-19_explainability_method_selection.md)も保存。今回の追加監査は**既存予測の再計算のみ**で、追加学習・主コード変更はしていない。

**次の着手単位（同日追記）**：[9/18発表後の小タスク](research_plan/2026-09-19_next_small_tasks.md)に、保存予測の誤差分解、固定モデルのノイズ比較、選別対照、説明性の採否、修論記述を入力・成果物・終了条件付きで分解した。説明性メモには同一モデル内の代替手法（Grad-CAM++等）との比較を追加。これらは計画であり、新たな学習・手法比較の実施結果ではない。

**A1〜A7完了（同日追記）**：[保存予測の標本別診断](../experiments/2026-09-19_noise_recovery_review/A1_A7_saved_prediction_diagnosis.md)で無雑音・−8 dBの2160秒を対応付け、重み、残差、ONB誤り、領域・WAV別の得失、等重みを記録した。−8 dBの統合悪化にはCNN＋TransformerとAlexNetのONB前・近傍誤差が寄与し、−8→−16 dBの回復は一部WAVに集中する。次の識別比較は`clean_only`固定モデル。なお当時留保したONB不一致は9/24に現行run値を正として解消した。再学習なし。従前監査READMEの二乗誤差減少量は`10^9`から元CSVに合う`10^12`へ訂正した。

**B1〜B6完了（同日追記）**：[固定モデル・複数seedの診断](../experiments/2026-09-19_b_clean_only/README.md)でclean学習1回の4条件転送とmatched −8/−16 dBの3 seed再学習を実施。固定モデルで谷は消失し、matched統合の回復はseedで符号が変わった。B7は開始条件不成立。次は学習日内の元WAV分離validationと、選別の有無を問うCの対応比較。

**C1〜C3完了（9/20追記、9/24解釈更新）**：[選別なし／ありの3 seed対応比較](../experiments/2026-09-20_c_selection/README.md)を6/11学習→6/18全1080秒評価、22 kHz・無雑音・200 epochsで実施。1e-9選別は正しい6/11 ONB以降の弱音94秒を除き1080→986秒とした。ONB前誤差は全seedで改善したが、近傍1 WAV・近傍外ONB以上の誤差は全seedで悪化。全域R²差は−.0035、+.0025、+.0016と一貫せず、**この条件の基準には選別を採用しない**。これはONB前を直接削除した結果ではなく、弱音境界標本の削除で共有回帰関数が低値側へ動くトレードオフと解釈する。追加の未使用独立日と同期映像は取得できない。

**D1〜D4実施（9/22追記）**：[残差と説明性の対応・信頼性監査](../experiments/2026-09-22_d_explainability/README.md)でrun151715の無雑音／−8 dBを点検。AlexNetの1–2 kHzゼロマスクは両条件で極端に悪化し、CNN＋Transformerの−8 dB 256–512 Hz改善は無雑音で反転したため、帯域加工・再学習は不採用。TreeSHAPは10/10代表秒を最大0.523 W/m²誤差で再構成。IGは20枚中7枚のみ収束し代替baselineもなく主張から除外、Grad-CAMは`alex_conv5`の12×12正方向補助図に限定した。元runの深層成功秒は重み未保存で再現不能だったため限界として固定。背景・実験条件・評価設計とA〜Dの結果・限界を[修論作業稿](thesis/2026-09-22_working_draft_background_methods_results.md)へ接続した。

**A〜D後の本人見解と次工程（9/22追記）**：[最新方針](research_plan/2026-09-22_after_A-D_interpretation_and_next_steps.md)に、Aを今後の標準診断、Bを最優先、CをONB定義と削除効果の識別、Dを保存モデル上の信頼性診断として整理した。固定cleanの別日ノイズ転送は既にBで実施済みだが、強ノイズ時はRFが統合を上回るため、統合による劣化抑制は未確認の仮説とする。次は前提確認と保存・validation基盤を整え、選別なし固定cleanの複数seed比較、matchedの原因診断、必要な統合方式比較の順に進む。

**実行順2〜7の再学習なし監査（9/23追記、9/24訂正）**：[監査記録](../experiments/2026-09-23_steps2-7_readonly_audit/README.md)で、2〜7のうち既存成果物だけで進められる範囲を確認した。主コード、学習、推論、IGは実行していない。matched −8/−16 dBへ条件別・固定・等重みを事後適用しても、R²差はseed 42/44で正、43で負のまま変わらず、谷の符号不安定性は重みより単体モデルの再学習差が主因と判断した。Cの除外94秒は221,505で60秒、266,908で25秒、315,466 W/m²で9秒で、9/24に確定した正しいONBでは全てONB以降である。追加の実験日・同期映像はなく、独立水流ノイズも当面用意しない。

**実行順2の実装・スモーク確認（9/23追記）**：[実装・検証記録](../experiments/2026-09-23_step2_training_state/README.md)のとおり、6/11内の元WAV非共有epoch validation、モデル・PCA・scaler・統合重み・seed・環境の保存、再読込予測検証を現行主コードへ追加した。TensorFlowの厳密決定論を妨げていた位置埋め込みの疎更新を同値な密更新へ変え、RFの固定seedとPCA微小演算差も修正。6/11 clean学習→6/18 clean評価の1 epochスモーク2回は3単体・等重みの全1080予測が完全一致し、保存後再読込差も0。clean fitを6/18の7 SNRへ流す配線と、実行順6の4方式アンサンブルも1 epochで完了した。ただし全て動作確認であり、200 epochs×3 seedの実行順3・4、5の学習側診断、6の本比較、7-Dは未実行。本実験設定は[`2026-09-23_steps2-4_full.json`](../configs/experiments/2026-09-23_steps2-4_full.json)に固定した。

**本実験条件（9/24、完了）**：[`2026-09-24_clean_train_noise_inference.json`](../configs/experiments/2026-09-24_clean_train_noise_inference.json)で、音響選別なし、6/11 clean学習、6/18の7 SNR推論、元WAV非共有validation、200 epochs、seed 42/43/44を実行済み。結果は本ページ冒頭と[実験記録](../experiments/2026-09-24_clean_train_noise_inference/README.md)を参照する。

**9/17の評価方針更新**：現行実行コードから元WAVへの予測集約・ONB遷移の追加評価と「主評価」の選択を外し、1秒chunkの通常指標を全モデル・有効な統合方式について確認する。元WAV情報は学習/検証の分割で同じ録音を跨がせないために使う。現行`inner_holdout`と`performance_kfold`の重み用内部スコアもchunk単位へ変更した。以下のWAV中央値・crossfit成績は変更前に保存した過去runの記録であり、新実行の評価定義とは混同しない。

## 9/18完了のONB全条件run

以下のスライドレビューは発表前の版への記録。現行PPTXはハッシュが異なり、旧版の表示上の指摘を現行版の未修正事項とは断定しない。発表原本は保持する。

**完成スライドの確認**：[全8枚のレビューと次工程案](../experiments/2026-09-18_final_slides_review/README.md)。方法→回帰→説明性→計画の構成は成立し、5–7枚目の5画像はrun151715の元図と画素が一致。3枚目の実験日重複、4枚目の選別と物理ラベルの区別、5枚目の外側評価回数、6・8枚目の画面外要素は修正が必要。PPTX原本は編集していない。次工程として22 kHz・無雑音・200 epochsを基準に選別なし／ありを同じ複数seedで比較し、既存予測の統合誤差分析、固定モデルの雑音転送、独立日検証へ進む案を提示した。提案段階であり、追加学習は未実行。以下の9/16時点の3 kHz・300 epochs案や過去選別数を、最新runの未完了作業・条件と混同しない。

[run 151715の解析](../experiments/2026-09-18_onb_full_sweep_151715/README.md)。6/11学習・6/18全1080秒評価、学習986/1080秒にピーク選別、3/5/10/15/22 kHz × 無雑音と6 SNRの35条件が完了。22 kHz・無雑音の統合R²=.911、MAE=67.9 kW/m²、連続ROC-AUC=.974で、この条件の単体モデルを上回る。全35条件では統合の最良単体超えがR² 15条件、MAE 22条件、連続ROC-AUC 8条件で、ONB近傍RMSEは0条件。前回の22 kHz単独runと同じ名目条件でもCNN結果が変わり、安定性は未確認。ゼロマスクのモデル別感度差は確認できたが、IGは22 kHz・無雑音でConformer 1/5例、AlexNet 0/5例しか数値収束していない。今回の発表では100%分類可能熱流束の散布図を扱わず、R²・連続AUCと説明性の条件付き結果を中心にする。

**9/18の本人指示**により、1秒内の時間区間マスクと時間感度profile・最大感度時刻の出力を現行コードから削除。周波数帯マスクの評価指標と、RandomForestのTreeSHAP、CNNのIG・Grad-CAMおよび信頼性診断は残す。旧runの時間区間出力は履歴として保存するが、発表の考察には使わない。マスクではAlexNetのRecall上昇と同時にR²・F1が悪化する例があり、単一指標で有効帯域を断定しない。モデル別の追加確認は[run解析](../experiments/2026-09-18_onb_full_sweep_151715/README.md)に記録。

## 9/17保存結果の確認

- [9/17別日結果の解析](../experiments/2026-09-17_onb_crossday_result_analysis/README.md)を追加。完成runの実条件は学習6/11のみ、評価6/18、3 kHz、無雑音、150 epochs、seed 42。ピーク選別による除外はなく、22 kHzは未完了。
- 6/11内部OOFのWAV単位MSEは重みfit用の同日未知WAV指標で、6/18性能ではない。内部ではCNNが最良だったため重みはCNN中心となったが、6/18全域ではRFのWAV中央値R²=0.842が最良。主shrinkage統合は0.621でRFを超えなかった。
- CNNはONB以上R²=0.967だがONB前を平均203.9 kW/m²高く予測する。元WAV分離は内部リークを解消した一方、日間校正ずれは残る。

## 本人が今回決めたこと

- **学習日：6/11・7/9、テスト日：6/18**。ONB主コードのconfigから変更できるようにする。
- 当初は学部コードと同じ1秒通常KFoldを指定したが、300 epochs結果で全内部foldに同じ29 WAVが共有され、CNNへ最大重みを与えたまま別日で破綻した。**現在は元WAV分離KFold/crossfitを使う**。当該runの履歴は変更しない。
- ONB以上の熱流束ラベルについて、**特定周波数付近の山の高さに横線を引き、弱い前後をどこまで学習に含めるか決めたい**。帯域パワーの分位点ではない。ラベルは変えず、データ不足への対策は保留。
- 過去に検討した`subset_equal_cv / crossfit_wav_stack / crossfit_shrinkage_stack`は元WAV集約を重み学習に使う別方式で、現行の実行設定では有効にしていない。実データで最良方式は未確定。
- **9/18（金）発表は完了**。発表後の議論と計画は本ページ冒頭を参照する。
- 以前の12/24頃研究終了・1月末発表の見通しより、本人が今回指定した**最終スライドの9月〜翌2月計画**を優先する。公式締切は未確認。

## 今回できたこと

最新の性能・分割・IG・実装修正の根拠は[9/16別日結果の診断](../experiments/2026-09-16_onb_result_diagnosis/README.md)。選別自体は[ピーク高さの実装・検証](../experiments/2026-09-16_peak_height_selection/README.md)と[操作画面](../experiments/2026-09-16_peak_height_selection/peak_threshold_review.html)で確認できる。

| 項目 | 確認した状態 | 未確認のこと |
|---|---|---|
| 別日指定 | explicit_daysで学習6/11+7/9、テスト6/18。300 epochs runでも学習1,649秒・テスト全1,080秒を確認 | 新しい独立日での再現性 |
| 内部検証・統合 | 旧chunk KFoldは全foldで29 WAV共有と判明。現在はWAV分離KFoldとcrossfit 3方式へ修正 | 新3方式の実データ成績 |
| 1秒スペクトル | **49 WAV・2,940秒の線形PSD画像、49時系列、3帯域のピーク高さを出力**。全値と105 manifestを照合 | 気泡由来かどうかの秒単位正解 |
| 学習選別 | clean音の**2,100–2,500 Hz内の最大PSD ≥ 1e-9**。図の縦軸×10⁻⁹で横線1。学習のONB以上だけに固定値を適用 | 1は探索的候補。閾値の最適性、十分な学習後の改善は未確認 |
| 選別の量 | 学習**1,860→1,649秒**。6/11は全保持、7/9で211秒を除外。テスト6/18の1,080秒・18 WAVは全て保持 | データ不足対策は本人指定で保留 |
| 検証 | 3/22 kHz・無雑音・300 epochs完了runを診断。3 kHz CNN R²=-0.084、RF=0.818、旧統合=0.534。22 kHz CNN=0.804 | 選別なし対照、7/9追加効果、複数seed、新方式比較 |
| 文書 | 後期WBS、修論7章、9/18発表案とSOAPを更新 | 教授の承認、正式な提出日、完成PPTX |

**スペクトルの重要な事実（9/24訂正）**：7/9の正しいONBは571.69 kW/m²である。現行`1e-9`ではONB以降300秒中91秒を除外する。旧442.17 kW/m²は自動直線性喪失候補でONBではない。日ごとに弱音群の比率が異なるため、単一絶対閾値の一般化は追加比較で確認する。

付録のPowerと今回のWelch PSDは尺度が異なるため数値を直接流用しない。1e-9は今回の図で確認する横線。学習日の低熱流束領域と強い/弱い山を比較した暫定値で、6/18の予測精度から選んでいない。テスト日のスペクトル自体は依頼どおり確認・出力済み。

音の弱い秒にも物理的な熱流束は存在する。今回の選別は「誤ラベル修正」ではなく、学習集合を音響的に選ぶ仮説。テスト全秒を評価し、強い音だけのテスト成績へ置き換えない。

## 現行設定と次の一手

[ONB実行コード](../code/run_ensemble_regression_onb.py)の learning_policy / acoustic_selection / ensemble を使用する。現行の主コード既定値は学習6/11+6/18、テスト7/9であるが、本比較の再現条件は各`configs/experiments/*.json`を正とする。`acoustic_selection`はピーク高さ閾値だけを指定し、`None`なら選別なし。固定的な選別条件・output/explainability・モデルregistryは`utils/config/onb_defaults.py`へ置く。元WAV集約の評価設定は削除済み。保存先は`<解析日>/onb_<学習・評価日>_<内部検証・学習ノイズ>_<選別閾値>_<epoch>_[parameter番号_]<HHMMSS>/...`とし、日付直下を最大52文字に制限しながら主要条件と実行時刻を読めるようにしている。

2026-09-16に[実験日指定を整理](../experiments/2026-09-16_onb_experiment_day_audit/README.md)。現在の`explicit_days`は学習日・テスト日から対象3日を自動算出する。設定変更と単体テストの記録であり、300 epochs本比較の実行・性能検証ではない。

**直近の判断は[A〜D後の解釈と今後の研究方針](research_plan/2026-09-22_after_A-D_interpretation_and_next_steps.md)を参照する。** A〜Dを実施し、1e-9選別・帯域加工・現行IG主張は採用せず、Bのノイズ・統合問題を最優先とする。まず教授とONB閾値の定義、新しい独立日・独立水流音・音声同期映像の取得可否を確認する。並行して、6/11内の元WAV分離validation、学習済みモデル・前処理・全秒予測の保存、同一条件再現性を整える。その後、選別なし固定cleanの複数seed転送、−8/−16 dB matchedの原因診断、必要な統合方式比較へ進む。6/18は既に研究判断に使っており、最終的な一般化主張には新しい独立日を必要とする。

### 2026-09-19 ノイズ谷の原因識別で確認した範囲

- 保存予測の標本別診断では、既存matched runの−8→−16 dBに見える回復を確認した。ただし回復はWAV・領域によって異なり、等重みでも谷型は残る。ONBは9/24に現行run値を正として確定した。
- [clean学習固定の4条件](../experiments/2026-09-19_b_clean_only/README.md)では統合R²が無雑音.918、−8 .737、−16 .442、−20 .152と一方向に悪化した。強い雑音でONB前480秒すべてが誤報となり、RF単体のR²は.850〜.870。既存matched runの学習済みモデル・PCA・scalerは保存されていなかったため、新しくcleanを1回学習して4条件へ固定適用した対照である。
- matchedの−8/−16 dBをseed 42/43/44で再学習すると、統合R²差（−16−−8）は+.0305、−.0188、+.0147。CNN＋Transformer単体の差は3 seedとも正だが、AlexNetと統合は一貫しない。ONB前の二乗誤差は3 seedとも改善し、近傍外ONB以上は3 seedとも悪化した。全域R²が良くてもONB見逃しは改善しない。
- training lossだけでは過学習や最良epochを判断できない。次は6/11内で元WAVを分離したvalidationにより、CNNのepoch別未知WAV誤差・ONB誤りを記録する。epoch・重み・学習方針は学習側で決め、6/18は固定評価とする。雑音が未知の用途ではRFを頑健性の対照とし、clean学習CNN統合の直接転送を基準にしない。B7の帯域摂動は固定モデルで谷が消えたため開始条件不成立。独立日・独立水流音まで一般化主張を広げない。

新しい実行では1秒chunkのR²/RMSE/MAE、ONB近傍誤差、見逃し/誤警報など通常指標をすべて読む。内部OOF・別日テスト・秒単位の音響イベントを分け、音響イベントの真値がないことも明記する。

- [後期計画・階層タスク・判断点](research_plan/2026-09-16_second_semester_plan.md)
- [修論目次・各章に必要な証拠](research_plan/2026-09-16_master_thesis_outline.md)
- [年度計画](research_plan/2026_annual_plan.md)
- [9/18発表案](research_plan/2026-09-18_xai_progress_brief.md)
- [9/18 SOAP下書き](progress/2026-09-18_weekly_progress_draft.md)

## 引き継ぐ目的と過去成果

学部のアンサンブルによるノイズ劣化抑制を、成立条件と理由まで発展させる目的は維持する。[本人の仮説](research_plan/2026-09-15_master_thesis_hypothesis.md)。

- [9/14限定の原因分析](../experiments/2026-09-16_sep14_cause_analysis/analysis.md)：本人指定の06.11・6条件。9/15を混ぜないという当時の対象範囲を保持。
- [9/15・2帯域比較](../experiments/2026-09-15_onb_frequency_comparison/analysis.md)：旧日内評価の履歴。新しい別日・選別後結果ではない。
- [IG数値修正](../experiments/2026-09-16_ig_numerical_fix/README.md)：積分・収束診断を修正済み。旧9/14学習済みモデルは未発見で旧画像再計算は未実施。ランダム初期化の検証ではCNNが収束、AlexNetは上限でも厳しい許容値に未到達。
- [crossfit3方式](../experiments/2026-09-16_crossfit_ensemble/README.md)：ピーク選別との併用を追加し主設定で有効化。合成/小型Keras検証済み、実データ性能は未確認。
- 9/3の105条件や途中runを追加解析・再実行していない。

実装済み・出力済み・研究上の検証済みを分ける。[文書案内](document_index.md)。
