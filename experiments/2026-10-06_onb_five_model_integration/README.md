# 5モデルの通常ONB組込み・指標差・採用理由

2026-10-06。本人は、見逃し減少に伴う各指標の変化を知り、追加2モデルを通常のモデルutilsへ置いてONBから実行できるようにし、元3も含めた採用理由を教授に説明できる状態にすることを依頼した。

**通常ONBの既定を元RF＋Conformer＋AlexNet＋HGB＋ExtraTreesへ変更し、モデル・学習OOF・3/4/5統合・指標・保存/再読込・説明性の入口を接続した。実際のmainを元構造のまま1 epochで完走し、保存済み150 epochsモデルでも7条件の統合と全指標を再検算した。** 今回、新しい通常150 epochs本runは実行していない。既存の150 epochs研究結果と、今回の実装確認を区別する。

数値の詳細：[各指標の変化](metric_changes_summary.md)。教授向けのモデルの実体・理由・削除対照・次工程：[説明文書](../../docs/notes/2026-10-06_five_model_rationale_and_next_steps.md)。前段の9候補学習：[元3固定の追加比較](../2026-10-06_fixed_three_additions/README.md)。

## 1. 現在のONBを実行すると何が出るか

```powershell
python code/run_ensemble_regression_onb.py
```

設定は[VALIDATION_CONFIG](../../code/run_ensemble_regression_onb.py)。3 kHz・1秒・選別なし・clean_only・shuffle chunk KFold3・seed42、Conformer150 epochs/batch12、AlexNet150/batch8。元3の構造と採用ハイパーパラメータを保持し、HGBとExtraTreesを同じ分割で学習する。PCA/RFの再読込に関わる数値環境を揃えるためCPU2スレッドを設定/記録する。

同じ5単体の学習OOFを共有し、次の6方式を一度に出す。元3や4の対照のために深層モデルをもう一度学習しない。

| result key | 使うモデルと用途 |
|---|---|
| `ensemble__original3_hgb_extra_trees` | 元3＋HGB＋ExtraTrees。主方式、全5つが正の重み |
| `ensemble__original3_mse` | 元3だけ。同じMSE方式の公平な基準 |
| `ensemble__original3_performance` | 元3だけ。従来の単体逆誤差比率の基準 |
| `ensemble__original3_hgb` | 元3＋HGB4。q100の利点も見る対照 |
| `ensemble__original3_extra_trees` | 元3＋ExtraTrees4。5との差が小さい強い対照 |
| `ensemble__simple_equal` | 5等平均。追加数だけの効果との区別 |

外側未使用chunkはOOF作成・重み推定へ渡さない。統合方式は[fixed_core_stacking.py](../../code/utils/ensemble/fixed_core_stacking.py)と[カタログ](../../code/utils/ensemble/strategy_catalog.py)へ置いた。元3のclean-MSE内部比率を固定し、元3総量25%以上・追加各5%以上で配分をfitする。将来のrunで元3比率が0になることを防ぐ1e−6の数値floorも明示/保存した。43/44の既存比率は正なので、以前の結果を数値誤差の範囲で再現した。

HGB/ExtraTreesは[acoustic_regression.py](../../code/utils/models/regression/acoustic_regression.py)に実装し、[RegressionModelMaker](../../code/utils/models/regression/base_regression.py)と[registry](../../code/utils/config/onb_defaults.py)から呼ぶ。生power→固定34特徴→学習器のsklearn Pipelineを保存する。従来RFだけが画像PCAを使い、追加2つをPCA入力で誤って学習/推論しない。現在の34特徴は3 kHz・1 channel専用で、別帯域への誤流用を設定検査で拒否する。

## 2. 指標の確認先と閾値の違い

| 出力 | 見るもの |
|---|---|
| `metrics_summary_<SNR>.csv` | 従来の実験閾値によるR²/RMSE/MAE/ROC・PR/Recall/F1等。今回TP/FP/TN/FN/FPRと領域母数も追加 |
| `metrics_by_source_day.csv` | 日別閾値の各日と、`pooled_source_day_thresholds`の2日合算。FN/FP・各指標・日別q100/g100を含む |
| `metrics_source_day_deltas.csv` | 同じfold/日別評価単位の元3MSEとの差。率はpercentage_point_changeも保存 |
| `fold_pred/pred_f1_<SNR>.csv` | 5単体＋6統合の11予測列、元WAV/chunkの対応 |
| `ensemble_training_oof_fit_f1.json` | 重み、元3内部比率、保持制約、OOF fit診断、分割情報 |
| `ensemble_training_oof_f1.csv` | 学習側held予測と正解。外側評価と区別 |
| `fitted_state/fold1/` | 最終5モデル、scaler、RF用PCA、各統合の重み、環境・hash・再読込検証 |

統合データに対する従来ONB閾値は、本人が以前指定した2日の平均246.59 kW/m²。先ほどのFN32→25/28→19は各日の221.51/271.68による評価だった。平均閾値を無言で変更せず、両方の評価を保存する。先ほどの数値に対応するのは日別閾値CSVの合算行である。既定seed42の通常本runは新しい分割なので、43/44の表と完全一致する値を保証しない。

すべての出力熱流束/誤差のコード単位はW/m²。説明表はkW/m²へ換算した。日別閾値は評価の正解定義に使い、推論時のモデル入力・重み切替には使わない。

## 3. 保存結果の再検算

[validate_saved_integration.py](validate_saved_integration.py)で、新しい通常統合を保存済みseed43/44の5単体OOFへfitし、同じ540×7条件へ適用した。元3MSE/performance、HGB4、ExtraTrees4、5の旧予測と照合し、最大差7.96×10⁻¹³ kW/m²。追加2の通常Pipelineを保存済み学習器で構成し、ModelTrainerと通常保存コードで再読込した28条件・15,120予測は最大差2.33×10⁻¹⁰ W/m²だった。[検算JSON](saved_integration_verification.json)。

通常のsource-day指標関数でも252行を再計算し、RMSE/MAE/R²・近傍・Recall/Precision/F1・FN/FP・連続ROC/PR・日別q100を旧結果へ照合した。全率にはAccuracyも追加して3360差行を保存。[指標CSV](metrics_recomputed.csv)、[差CSV](metric_changes.csv)。

cleanのRecallは89.84→92.06%／91.11→93.97%、F1は0.94649→0.95868／0.95349→0.96890、Accuracyは94.07→95.37%／94.81→96.48%。Precision100%・FP0は不変。RMSE/MAEや連続AUCの変化量はFN件数だけで一意に決まらないので、別に保存予測から計算した。

モデルを1つずつ外し、残る元モデル群の比率/追加配分をclean OOFで再fitする70条件も取得した。[削除対照](model_removal_ablation.csv)。元RFを外すとclean RMSEが0.20/0.04低下するため、全5つが精度上必須とはしない。Conformer除去はclean RMSE+1.88/+2.12、FN25→29/19→30、ExtraTrees除去は−20 RMSE+16.70/+0.52。採用理由は[教授向け文書](../../docs/notes/2026-10-06_five_model_rationale_and_next_steps.md)で役割と限界に分けた。

## 4. 通常mainの動作確認

[smoke_main.py](smoke_main.py)から実際の`onb.main()`を呼び、元構造の5モデルを1 epoch・内部2foldで学習した。36 WAVから4chunkずつの144入力、108 fit/36外側、cleanと−20。15基礎fit、最終5保存モデルと再読込を通過し、各条件11予測列と日別指標33行を確認した。[動作確認JSON](main_smoke_verification.json)。描画はこの動作確認では省略した。

この1 epochの数値を研究性能表へ加えない。通常設定は150 epochs/3foldのまま。標準の条件保存先がsave_baseの隣に作られる仕様を使い、動作確認の保存先も通常のmanifestから解決して検算した。実装確認の再計算を理由に元研究runを上書きしない。

関連60テストがすべて通過した。[確認範囲](related_tests.json)は通常入口・registry・原3/追加2の保存・保持重み・分割/再開・OOF探索・雑音図・XAIを含む。5単体＋6統合の11系列が増えるため、通常棒グラフを系列数に応じた幅/文字サイズへ調整し、実出力の11系列を描画して確認した。描画用確認値も1 epochのソフトウェア確認に限定する。[実効設定の記録](../../configs/experiments/2026-10-06_onb_five_model_integration.json)。

追加2の説明性は、生入力の共通周波数帯maskを通常入口から使えるようにした。ExtraTreesの重要度はPCA成分として誤表示せず34特徴名で保存する。HGBは学習器重要度を必ず持つとは扱わない。通常のXAI既定は引き続きdisabledで、今回新しい母集団XAIは実行していない。raw-summary maskの数値経路は関連テストで確認した。

## 5. 実装済みと今後

5モデル通常実行、同じOOFによる3/4/5対照、両閾値の指標、保存/再読込、説明性対応は実装・動作確認済み。現在の既定でONBを起動すればこれらを出せる。過去3モデルの完成結果とは条件hash/モデルキー/実行時刻で区別する。

次は通常150 epochs本runを1回確認し、標準保存形式でも3/4/5の交換を記録する。既存の150 epochs候補検証を未実施として再学習する工程には戻さない。モデル追加の全gridや新しいgateは、低熱流束・強雑音近傍・q100の不足から必要な問いを決めてから検討する。本人への追加人手確認・物理的音源の同定・未知日検証を今回の進行条件にしない。

研究上の採用は本人の元3保持方針に合う5モデルであり、ExtraTrees4を残す。5対4の差が小さいこと、RF必要性の未確認、既知WAVの開発比較であることを先生へ説明する資料にも含める。
