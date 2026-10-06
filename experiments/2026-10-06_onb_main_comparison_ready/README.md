# 主張・評価・主表の確定と、通常ONB本実行の確認準備

実施日：2026-10-06。本人の「1をまず終え、その後2を実行して確認できるようONBを修正」の依頼に対応した。過去のモデル・結果は保持し、新しい通常150 epochs本学習は起動していない。

## 1. 第1段階は完了

[修論の現行正本](../../docs/thesis/2026-10-06_main_claims_and_evaluation.md)に、採用する主張・留保・比較相手・主条件・指標の主従を確定した。[機械可読プロトコルv1](../../configs/experiments/2026-10-06_onb_main_comparison_protocol.json)を、保存結果と次の通常runの共通定義にした。

- 主構成：元3＋HGB＋ExtraTrees。基準は元3MSE、主要対照は元3＋ET4とET単体。HGB4、HGB単体、従来元3performance、5等平均も表示する。
- 主ONB評価：日別閾値221505.1102／271677.6816 W/m²。日別q100/g100とFP/FPRを中心に、FN/Recallと領域別回帰を併記。
- 従来の平均閾値246591.3959による表・散布図は補助出力として保持。ROC/PR-AUCは従来の生予測熱流束を使う補助的順位指標と明記。
- 全域MSEでのOOF統合と、q100という評価目的を区別。強雑音の絶対誤差とcleanからの劣化量、5対4・単体の不成立を主表へ含める。

[保存seed43・44の主比較表](saved_main_tables/main_comparison.md)は、5単体＋6統合×7条件×日別2＋全体×2seedの462指標行。同じ[共通reporter](../../code/utils/calculation/onb_comparison_report.py)から生成した。元3・ET4・ET単体との差、日別q100/g100、劣化量、重みを別CSVへ保存。過去の通常統合検算252行との一致と、入力参照・SHA-256は[saved_main_tables/verification.json](saved_main_tables/verification.json)。新学習・モデル推論0。

再生成：`python experiments/2026-10-06_onb_main_comparison_ready/build_saved_main_tables.py`

## 2. 通常ONBの実行方法と既定条件

研究ルートから、既存のTensorFlow/CUDA環境で実行する。

```powershell
python code/run_ensemble_regression_onb.py
```

既定は150 epochs、内部chunk KFold 3fold、seed42、CPU threads2、3 kHz、1秒、選別なし、clean_only、全7条件。5単体の内部15fit＋最終5fitで、6つの統合を同じOOFから求める。統合ごとにCNNを再学習しない。XAIは既定で無効のまま。

冒頭の`VALIDATION_CONFIG.output`で、主比較レポートと全評価予測の再読込確認を有効にした。主比較レポートは現行clean_only用。matchedへ切り替えた場合は既存の条件別評価・再読込を継続し、この固定clean主表の収集は行わない。

## 3. 実行時に自動で確認すること

学習前に、設定した全条件の入力が存在すること、出典日ONB、ノイズ間の評価ID・正解一致、学習/評価の非共有を確認する。標準条件では36 WAV・各日18、各WAV60入力・45学習/15評価、1620/540を照合する。今回実データで[7条件の事前確認](normal_run_preflight.json)が通った。

学習後は、従来の8例probeに加え、保存した各モデル・PCA/scalerを再読込し、**全5モデル×7条件×540＝18,900予測**を元予測と照合する。モデルは元の学習モデルを解放してから再読込し、ノイズ入力は条件ごとに読み込む。追加学習は行わない。比較許容差は既存と同じ`rtol=1e-6`、`atol=1e-3 W/m²`で、最大差・平均差・件数を保存する。

終了時に、予測CSVと日別指標の再計算一致、保存重みからの6統合予測の再構成、元3内比率・最低配分・参加数、同一clean fit、OOF非共有・全件1回のcoverage、内部/最終CNNの実epoch・実batch sizeを確認する。

**性能の改善方向は合否条件にしない。** epoch未完了は`incomplete_epochs`、batch size変更や非標準設定は標準本run完了と区別する。指標が改善しなかった結果も同じ規則で保存する。

## 4. 実行後に読む場所

終了時にコンソールへ、次の主表の実パスを表示する。

`実行別の保存フォルダ/maxfreq=3kHz/onb_comparison/main_comparison.md`

| ファイル | 確認する内容 |
|---|---|
| `main_comparison.md` | clean/−20の単体・3/4/5、日別q100/g100、絶対誤差と劣化量の主表 |
| `main_metrics.csv` | 全7条件、全11方式、日別/全体、母数、回帰、ONB判定、bias、AUC |
| `main_deltas.csv` | 元3MSE、ET4、ET単体との差。率はpp、誤差・熱流束はkW/m² |
| `noise_degradation.csv` | 各方式自身のcleanからの全域・近傍RMSE増加量とFN/FP差 |
| `noise_overview.csv` | 6条件の平均RMSEと観測最大FP。完全な6条件平均かも記録 |
| `q100_by_source_day.csv` | 日別ONB閾値、q100/g100。全体q100の平均値は作らない |
| `ensemble_weights.csv` | clean OOF由来の各配分、元3比率と追加モデル参加数 |
| `fitted_reload_checks.csv` | 5モデル×7条件の再読込差・件数・合否 |
| `ensemble_reconstruction_checks.csv` | 保存重みから再構成した6統合×7条件の差・合否 |
| `fit_epoch_audit.csv` | 内部/最終CNNの要求・実epoch、要求・実batch、メモリ停止 |
| `verification.json` | 標準150 epochs本runを完了したか、実装確認か、確認件数と元ファイル参照 |
| `run_conditions.json` / `evaluation_protocol.json` | 実行した条件と主表の定義。事前確認も実行snapshot内に保存 |

本runの確認では`verification.json`の`status="passed"`に加え、`normal_150_epoch_main_run_completed=true`、`epochs_complete=true`、`requested_batch_sizes_completed=true`を読む。標準設定との一致はepoch・分割・seed・モデルパラメータ・PCA・noise条件も含めて判断する。

各ノイズフォルダ内の従来CSV・予測・散布図・R²/AUC図・重み・学習状態は引き続き保存する。主表は共通日別評価であり、平均閾値の散布図とは判定基準が異なる。完了済みrunの主表生成を再実行する場合は保存ファイルから集計し、学習を増やさない。

実行hashには新しい全評価再読込・主表の定義を含めた。従来のprobeだけを確認した完了runを、新しい確認まで満たしたrunとして再開しない。

## 5. 確認結果と未実施

- [関連テスト](related_tests.json)：8関連モジュール、50件通過。日別/平均閾値の区別、欠落比較・重複行の拒否、kW変換・劣化量、全条件再読込と予測不一致拒否、既存学習・保存・OOFの互換性を確認。
- 実行識別の追加1件と影響する既存チェックを含む[3モジュール12件](identity_tests.json)も通過。重複を除く確認対象は51件。
- [実mainの確認](main_smoke_verification.json)：既存実入力144、36 WAV、clean/−20、実5構造、1 epoch、内部2fold、15fit。5単体＋6統合、日別66行、全360評価予測の再読込、保存重みの再構成、内部4＋最終2のCNN epoch記録、主表一式を確認。通常の作図は抑止したソフトウェア検証であり、この数値を研究性能へ加えない。
- 新しい通常150 epochs本runは未実施。今回完了したのは主張・主表の確定と、本runから必要な結果・実行確認を取得する実装・準備確認である。

実main確認の再実行：`python experiments/2026-10-06_onb_main_comparison_ready/smoke_main.py`。旧スモークの原ファイル・結果を保持し、新しい保存先へ出力する。
