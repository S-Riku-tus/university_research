# 固定HGB候補・追加seed・共通入力説明・統合方式の確認

2026-10-06。本人が依頼した、評価軸の整理、固定候補の再現性、成功/失敗の説明、次の統合方式の検討に対応する。前回の[他指標と公平な対照による再評価](../2026-10-06_model_addition_general_metrics_review/README.md)を受けた追検証であり、q100の改善をモデル追加の唯一の採用条件にしない。

**4項目の計算・比較・文書化を完了した。HGB追加によるcleanの回帰・見逃し改善は追加2 seedでも再現した。一方、強雑音下では学習chunk分割によりHGB単体の誤報が大きく変わり、雑音耐性を一律の採用理由にはできない。** HGBは有望候補として保持し、主runnerの採用確定と未知日一般性の検証は今回の完了範囲とは区別する。

## 1. 比較条件と実装

条件は[固定設定](../../configs/experiments/2026-10-06_hgb_followup_validation.json)、入口は[run_hgb_complementarity_validation.py](../../code/run_hgb_complementarity_validation.py)。新候補用の独立した実行入口を実装した。既存主runnerの採用モデル切替や、保存済みmatchedの再学習はこの比較に含めない。

| 項目 | 条件 |
|---|---|
| 録音 | 2025/6/11＋6/18、各18熱流束段階、計36 WAV |
| 入力 | 3 kHz、1秒、224×224、選別なし、既存の7雑音条件 |
| 外側 | WAVごとにshuffleして45 chunk学習・15 chunk評価、計1620/540 |
| 内部 | 外側学習1620の3-fold chunk KFold、shuffle=True |
| 追加seed | 43と44。外側・内部・モデル学習seedを共に変え、全モデルを再fit |
| 既存モデル | 保存基準runのRF（XGBRF）・Conformer・AlexNetと採用値、PCA100変換版2 |
| CNN学習 | 各150 epochsを完走、Conformer batch12、AlexNet batch8 |
| HGB | 周波数34特徴、leaf31、150 iterations、lr0.05、minleaf20、L2=1、early stoppingなし |
| 学習方針 | 全基礎モデルをcleanでfit。同じモデルをcleanと6雑音へ適用 |
| 保存 | 各内部foldと最終モデル、PCA、y scaler、損失、分割、予測、再読込検証 |
| 統合決定 | 学習側OOFだけから決め、外側正解を重み学習へ渡さない |

34特徴は[acoustic_summary_features.py](../../code/utils/dataloading/acoustic_summary_features.py)へ実装し、以前の学習1620例と数値が全件一致することを確認した。特徴抽出は標本ごとの固定変換で、真の熱流束・日付・clean参照・SNRを入力しない。PCAとラベルscalerは各fit側だけで学習する。

seed42は、入力と候補を選んだ開発上の参考結果として残す。HGBの内部選択値が31/15/15だったこと、旧RFの数値計算条件との違いもあるため、新しい固定31・2スレッドのseed43/44と一括して「seedだけの対照」としない。seed43/44間と、各seed内の追加なし/ありの対応比較を主に読む。

各seedの評価は**既知WAV内の未使用chunk**であり、同一chunkの学習・評価共有はない。同じ録音からの隣接区間・同じ2日の繰り返し分割であるため、540 chunkを独立録音540件とは数えない。既存データ全体を候補開発で使ってきた後の再標本化であり、未知日や一度も研究判断に使っていない盲検データによる独立検証とも区別する。

## 2. 比較する統合と、雑音を学習に使う範囲

個別4モデルに加え、従来performance、等平均、既存3/HGB追加4/Conformer＋HGB/Conformer＋AlexNet＋HGBのMSE重み、正の残差相関への罰則を加えた重み、Ridgeを比較する。MSE重みは非負かつ総和1、Ridgeは標準化＋alpha1、切片あり。多様性項は学習OOFの残差相関を使う固定対照であり、輪講論文のsociety entropyを直接再現したものではない。

**純粋なclean_only比較**では、基礎モデルと重み・metaモデルをすべてcleanから決める。**雑音OOFを使う統合比較**では、基礎モデルはclean固定のまま、内部held chunkへ雑音を加えた予測を作り、clean＋6雑音を同じ重みで統合学習へ使う。後者では雑音を統合学習に見せており、未知雑音への純粋なclean_only転送の結果とは呼ばない。各雑音で基礎モデルを再学習するmatchedとも異なる。

推論時に真のSNRや熱流束で重みを切り替えず、すべての入力へ一つの固定統合を適用する。内部OOFへ当てはめた統合の指標はfit診断であり、統合自体の未使用評価は外側540で読む。

Ridgeの効果を新候補追加の効果と区別するため、既存3にもclean/雑音OOFの同じRidgeをfitする[補足対照](../../configs/experiments/2026-10-06_hgb_followup_ridge_controls.json)を追加した。これはseed43途中結果を見た後に決めた対照であり、事前固定された新しい独立検証とは表現しない。HGB・基礎学習の条件を変えず、補足対照も学習OOFだけから決める。

## 3. 追加seedの結果

### 3.1 同じMSE統合の追加なし/あり

各seed内で同じ外側540を比較する。熱流束・誤差の単位はkW/m²。陽性315、陰性225。

| seed | 方式 | clean RMSE | MAE | Recall | FN | FP |
|---|---|---:|---:|---:|---:|---:|
| 43 | 既存3 MSE | 35.74 | 26.70 | 89.84% | 32 | 0 |
| 43 | HGB追加 MSE | **27.36** | **19.83** | **92.06%** | **25** | 0 |
| 44 | 既存3 MSE | 33.89 | 24.12 | 91.11% | 28 | 0 |
| 44 | HGB追加 MSE | **26.51** | **18.74** | **93.33%** | **21** | 0 |

両seedで26/36 WAVの平均二乗誤差が改善した。既存3の見逃しをseed43で8例訂正・1例追加、seed44で9例訂正・2例追加し、差引きどちらも7例減った。3基礎モデルがすべて陰性だった陽性21/19例のうち、HGB単体は7/6例を陽性にした。単に高い値を出すだけでなく、cleanでは有用な訂正を確認した。

改善は主にONB以上と高熱流束にある。閾値の1.2倍以上270 chunkのRMSEはseed43で42.92→27.94、seed44で40.67→27.24。一方、60 kW/m²未満120 chunkは23.98→27.63、27.05→28.50へ悪化した。全域指標が改善しても全領域が改善するとは言えない。

根拠：[replication_summary.csv](replication_summary.csv)、[paired_wav_errors.csv](paired_wav_errors.csv)、[paired_region_errors.csv](paired_region_errors.csv)、[error_complementarity.csv](error_complementarity.csv)。

### 3.2 強雑音では改善が反転する

| seed | 方式 | −20 dB RMSE | FN | FP |
|---|---|---:|---:|---:|
| 43 | 既存3 clean-MSE | 81.41 | 62 | 0 |
| 43 | HGB単体 | 106.37 | 30 | **88** |
| 43 | HGB追加 clean-MSE | 90.02 | 57 | 0 |
| 44 | 既存3 clean-MSE | 81.41 | 63 | 0 |
| 44 | HGB単体 | 72.41 | 53 | 0 |
| 44 | HGB追加 clean-MSE | 73.83 | 58 | 0 |

seed43のHGB誤報88例はすべて6/11のONB前に出た。同日の陰性120例に対して73.33%。見逃しやq100の良さだけを評価すると、この代償を見落とす。HGB追加MSEでは両seedの全7条件でFP0を維持したが、seed43の低熱流束−20 RMSEは133.97→169.96へ悪化し、全域誤差も増えた。FP0は熱流束回帰が正確であることを保証しない。

6雑音条件の平均RMSEは既存3→HGB追加で、seed43が69.39→75.96、seed44が67.49→63.68。両seedの平均だけでは不安定さが隠れるため、個別結果を採否の根拠にする。図は平均線と個別seedの破線を併記した。[比較図PNG](replication_comparison.png)、[PDF](replication_comparison.pdf)。

### 3.3 q100を併記すると分かること

q100は「その段階以降の評価chunkがすべて陽性だった最初の熱流束段階」。有限の評価集合に対する到達段階であり、検知確率100%や録音内の瞬間的気泡発生時刻ではない。

| seed | 条件 | 方式 | 6/11 q100 | 6/18 q100 | 2日合計FP |
|---|---|---|---:|---:|---:|
| 43 | clean | 既存3 MSE | 315.47 | 376.32 | 0 |
| 43 | clean | HGB追加 MSE | 315.47 | **322.11** | 0 |
| 44 | clean | 既存3 MSE | 315.47 | 434.02 | 0 |
| 44 | clean | HGB追加 MSE | 315.47 | 434.02 | 0 |
| 43 | −20 | HGB単体 | 315.47 | 376.32 | **88** |
| 43/44 | −20 | HGB追加 MSE | 368.98 | 434.02 | 0 |

seed44ではq100が変わらなくても回帰誤差とFNが改善した。本人の「他指標からも追加モデルの価値を評価する」という方針を支持する実例である。一方、seed43の単体HGBの早いq100は誤報88と対で読む。分割で評価対象chunkが変わるため、seedをまたいだq100差を単独でモデル改善とは呼ばない。

### 3.4 多様性・Ridge・雑音OOF統合

| 方式 | seed43 clean RMSE / FN / FP | seed44 clean RMSE / FN / FP | −20 RMSE（43 / 44） |
|---|---|---|---|
| HGB追加 clean-MSE | 27.36 / 25 / 0 | 26.51 / 21 / 0 | 90.02 / 73.83 |
| 同＋多様性項 | 28.60 / 25 / 0 | 26.59 / 21 / 0 | 91.10 / 73.32 |
| 同＋clean Ridge | 25.99 / 22 / 0 | 25.85 / 16 / **1** | 86.80 / 73.07 |
| HGB追加 雑音OOF-MSE | 28.79 / 25 / 0 | 31.08 / 25 / 0 | 83.99 / 77.00 |
| HGB追加 雑音OOF-Ridge | 30.54 / 28 / 0 | 32.01 / 30 / 0 | 78.02 / 76.89 |

**多様性項**：今回の固定罰則ではclean FN・q100が追加改善せず、RMSEは両seedでわずかに悪化した。HGB比率は約63–64%から約72%へ増えた。多様性を使えば自動的に有用な補完が増えるわけではない。この一つの残差相関罰則の結果から、あらゆる多様性指標を否定もしない。

**切片付きRidge**：非負平均の外へ出る統合を実際に比較できた。既存3にも同じclean Ridgeをfitした対照に対して、HGB追加はRMSE32.92→25.99、31.56→25.85、FN29→22、29→16。候補追加の利得はRidgeでも残る。一方、seed44ではclean FP0→1で、q100はMSEと同じ。回帰・見逃し改善と誤報を併記して採否を考える。

**雑音OOFによるMSE重み**：HGB比率はcleanの約63–64%から約23–25%へ下がり、seed43の−20 RMSE90.02→83.99、6雑音平均75.96→69.31へ戻った。ただし同じ雑音OOFを使った既存3対照はseed43の−20 RMSE80.53、6雑音平均69.01で、HGB追加を全域雑音誤差の優位と主張できない。seed44では既存3雑音OOF-MSEに対し81.87→77.00、68.38→65.39へ改善する。

**雑音OOF-Ridge**：HGB追加の−20 RMSE78.02/76.89はclean重みより安定したが、FN64/65と見逃しが増えた。既存3にも同じ雑音OOF-Ridgeを適用するとRMSE77.94/78.17、FN62/66だった。seed43ではHGBを加える追加利得がなく、統合方式を変えた利得と候補の利得を混同しない。

**少数モデル化**：clean-MSEではRF重みは両seedとも0で、Conformer＋AlexNet＋HGBに絞ると4モデル結果と数値誤差の範囲で一致した。Conformer＋HGBの2モデルはclean RMSE25.81/27.02、FN22/19だが、seed43の−20 FP12。RFを含む候補poolと、実際の推論に使うモデル数を分けて検討できる。

根拠：[weights.csv](weights.csv)、[同方式の対応差](paired_method_deltas.csv)。方式・係数の学習はOOFだけで行ったが、これらの外側結果を見て今後採用方式を決める作業は開発判断になる。

## 4. 説明性と変動の切り分け

[共通chunkの説明解析](explanation_findings.md)に、内部held-foldの特徴群置換、4モデルに対する同じ入力加工、32代表例、数値再現の条件を記録した。[代表図](representative_spectrograms.png)も保存した。重要度・入力加工の反応を、物理的な気泡音の同定とは区別する。

HGBについては、学習chunk分割42/43/44×HGB乱数42/43/44/45の12小規模fitを追加し、分割感度とモデル乱数感度を分けた。同じfit chunkなら4種類のHGB乱数でcleanと−20 dBの予測差は0だった。一方、分割を変えると強雑音の予測が大きく変化した。[hgb_split_vs_model_seed.csv](hgb_split_vs_model_seed.csv)、[hgb_rng_invariance.csv](hgb_rng_invariance.csv)。この診断は候補の再探索や最良seedの選択には使用しない。

新HGBの保存状態でも同じ代表32例の減衰・時間並び替えを比較した。選択clean8例で2.1–2.5 kHz減衰の平均変化はseed43で−189.90、seed44で−133.16 kW/m²、時間並び替えは全例0。一方、−20で1.7–2.1 kHzを弱めた反応は両seedで大きさが違った。[hgb_replication_dependency.csv](hgb_replication_dependency.csv)。代表例の一部は新seedの学習に入っており、行ごとにフラグを残した。これは共通入力の依存診断で、未使用評価の代わりにしない。

強雑音の入力分布も調べた。seed43の−20誤報88例中86例で周波数entropyがclean学習最小値未満だったが、正しく陰性だった137例中135例も同じ範囲外だった。seed44の正しい陰性225例でも218例が範囲外だった。**入力分布のずれは確認できるが、「entropyが学習範囲外なら誤報する」という単一の説明や切替規則では区別できない。** [hgb_replication_input_shift.csv](hgb_replication_input_shift.csv)。

内部fitは全seed・全foldで全36 WAVを含み、各熱流束段階の学習支持がある。今回の強雑音変動を「内部foldにONB段階がなかったこと」で説明できない。確認事実は学習chunk構成への感度であり、特徴分布のずれに対する木の分岐の変化が原因である可能性は仮説として残る。特定特徴の物理的原因や、木のどの分岐が誤報を作ったかまでは確定していない。

## 5. 本人が後で行う部分

[確認手順](human_review_instructions.md)、[記入表38行](human_review_sheet.csv)、[原録音34種類の抜粋](audio_review/README.md)を用意した。原録音の異常と実験メモ・同期観測の所在を確認できる範囲で記入する。モデル結果から動画の存在や気泡の発生を推定して埋めてはいない。

回帰・見逃し・誤報・q100のどれを採用判断で優先するかは、結果を見て本人が決める。現在の「q100と誤報を併記する」方針を継続し、誤報の許容値をAI側で勝手に置かない。新しい実験や未知日評価は、今回の比較を完了するための前提にしない。

## 6. 再現入口と保存物

通常は今回の完了結果を読み、全学習を再実行する必要はない。同じ条件で中断から再開する場合の順序は次のとおり。学習はGPUを使い、設定hash一致の完了fitを再利用する。

```powershell
python code/run_hgb_complementarity_validation.py --phase train
python code/run_hgb_complementarity_validation.py --phase integrate
python experiments/2026-10-06_hgb_followup_validation/add_ridge_controls.py
python code/run_hgb_complementarity_validation.py --phase analyze
python experiments/2026-10-06_hgb_followup_validation/summarize_replications.py
```

主な保存物はseed別の`manifest.json`、`base_predictions.npz`、`frozen_integration.json`、各fold/finalのモデル、`metrics.csv`、`outer_predictions.csv`、`weights.csv`。説明・音声確認は同じフォルダの対応表を使う。環境は[environment.json](environment.json)へ記録した。モデル・キャッシュは大きいため、文書を読む際に一括で開く必要はない。

**確認結果**：基礎モデル32 fitを保存・再読込確認し、そのうちCNN16 fitは150 epochs/batch12・8を完走した。19方式×2 seed×7条件の外側266指標集合と日別q100532件を保存予測から別計算で検算、分割と32モデルのhashも確認した。関連unittestは特徴4件・統合2件の計6件が通過。[verification.json](verification.json)。RFは固定2スレッドで全56組の保存540予測を再現し、先頭8件との比較も差0だった。[rf_replication_reload_audit.json](rf_replication_reload_audit.json)。任意の数値環境での一致とは区別する。

## 7. 事実・仮説・次の識別比較

**確認事実**：HGB追加のclean RMSE/MAE/FN改善は2 seedで再現し、q100不変でも価値が残る。強雑音ではHGB単体の誤報と全域回帰が不安定。雑音OOF重みはHGBへの依存を減らすが、cleanや見逃しと交換になる。今回の多様性項は通常MSEに対する安定した追加利得を示さなかった。

**解釈・原因仮説**：周波数要約を使うHGBは既存画像モデルのclean誤りを補完する。一方、cleanと異なる周波数分布に対し、学習chunkによって木の分岐と出力が変わり、逆方向出力が有用な訂正から誤報へ変わる可能性がある。統合の役割には単体の利得を取り込むだけでなく、単体の不安定な出力を抑えることもある。

**次の順序**：

1. 今回の比較と説明を教授へ提示し、本人側では記入表の優先例から録音・実験記録を確認する。cleanの回帰改善、誤報0のMSE、誤報1を伴うRidgeの見逃し改善を分けて相談する。
2. 主候補はHGBを含むclean-MSEとし、Ridge・雑音OOF統合を目的別の対照として保持する。全条件に対して一方式の採用を確定するには、許容する誤報と回帰/判定の優先を本人が決める。
3. さらに計算を進めるなら、強雑音の分割感度を次の具体的な改善対象とする。学習側だけの限定比較でleaf数・minleaf・L2の正則化、または同じ34特徴で異なる学習器を対照にする。候補を増やす目的を、cleanの精度か雑音下の不安定さかに先に定める。今回の最良seedや外側失敗だけへ合わせた設定変更はしない。
4. 未知WAV・未知日の一般性を主張する段階で、その評価を別に追加する。今回の繰り返しchunk比較を独立日の検証に読み替えない。q100最後の少数chunkの時間構造は並行課題として残す。

4項目を進めるために必要な計算は今回完了しており、本人に同じ学習の再実行を依頼する部分はない。3・4は結果から具体化した後続提案で、すでに実施した追検証を未完了扱いしない。
