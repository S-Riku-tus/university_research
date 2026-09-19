# モデル・説明性の参照メモ

**2026-09-19選択理由の再検証**：[手法ごとの採用理由、他候補との比較、現行runの限界](2026-09-19_explainability_method_selection.md)を追加。**現行の説明性出力は周波数帯マスクのみ（時間区間マスクは廃止）**。旧IGの不一致と、数値修正後の9/18runにも残る未収束を区別する。現行の単体モデルは[onb_defaults.py](../../code/utils/config/onb_defaults.py)の`randomforest`・`conformer`・`alexnet`。結果表示名の「Conformer」を厳密な標準Conformer構造と同一視しない。

**2026-09-16追記**: IGの積分を修正し、寄与合計とmapの収束を検査する実装へ更新した。[変更内容・使い方](../../experiments/2026-09-16_ig_numerical_fix/README.md)。以下で言及する9/14・9/15の旧IGの不整合は当時の出力に関する事実で、修正済みコードによる再計算を意味しない。

更新日: 2026-09-19。成績と採用判断は[現在地](../research_status.md)、構造の詳細は[コード地図](../code_map.md)に集約する。

| 現行キー | 実体 | 説明性 |
|---|---|---|
| `randomforest` | XGBRFRegressor、flatten→PCA（学習側でfit） | PCA空間TreeSHAP、周波数帯マスク |
| `conformer` | log-power、CNN、時間tokenのTransformer、GAP | IG、周波数帯マスク |
| `alexnet` | log-power、AlexNet系CNN、大きい全結合ヘッド | IG、Grad-CAM、周波数帯マスク |

「Conformer」「v1/v2」「AttnPool」は過去資料の世代名を含む。現行のキーと構造を[モデル定義](../../code/utils/models/regression/base_regression.py)で確認する。VGG16・ResNet50・WaveNet・SELDnet等は、存在するコード/過去候補と現在の有効モデルを区別する。

現行の統合は`simple_equal`と`inner_holdout`。`val_fold_legacy`は外側正解を使う診断・再現用で主張不可。`prediction_max`は9/14最終変更で削除済み。[方式カタログ](../../code/utils/ensemble/strategy_catalog.py)が実装上の正本。

`inner_holdout`も世代を区別する。9/8より前のchunkランダム分割は元WAV混在があり、旧結果では主張用から除外される。現行の元WAV非重複holdoutはnested stackingではない。

説明性画像の生成と、その数値整合・物理的妥当性は別に確認する。TreeSHAPをPCAから元画素へ単純に戻して物理寄与と呼ばない。IG/マスクの**発表後の解釈**は[手法選択の再検証](2026-09-19_explainability_method_selection.md)へ。発表前の解釈は[発表準備メモ](../research_plan/2026-09-18_xai_progress_brief.md)に保持する。

## 説明性で答える問いと、現在の使い方

| 手法 | 説明対象 | 数値・解釈で確認すること |
|---|---|---|
| TreeSHAP | RFのPCA成分が予測に与える正負の寄与 | 期待値＋寄与和の再構成。PCA成分を元画素・物理周波数へ直結しない |
| IG | 深層モデルのbaselineから入力への回帰出力差 | 寄与総和と予測差の一致、積分経路・点数・baseline。9/18runでも未収束例が多い |
| Grad-CAM | AlexNet最終畳み込み特徴の粗い局在 | 正負の寄与分解ではない。細かい周波数・気泡時刻を同定しない |
| 局所occlusion | 代表chunkの周波数帯を置換したときの予測変化 | 代表例の選び方、予測の符号、マスクの重なりと非加法性 |
| 集合マスク | 検証集合の周波数帯置換による性能変化 | 9/18runは全1080秒のchunk評価。ゼロ置換の分布変化を含む |
| deletion/insertion | 重要画素を消す/戻す順による回帰出力曲線 | 出力単位の面積で、分類ROC-AUCではない。符号・baseline・ランダム順位との比較が必要 |
| 安定性・ランダム化 | 小入力摂動へのmap安定性、学習パラメータへの依存 | 絶対値mapの類似だけでIG整合性・物理妥当性を保証しない。現行は最終層だけの部分診断 |

現在の固有手法は単体モデルのchunk回帰出力を説明する。WAV中央値・アンサンブル全体・ONBの物理的発生原因を直接分解しているわけではない。Attention Rollout、RISE、Guided Grad-CAM、CA-LIGは旧候補・論文紹介と現行採用を分ける。旧runのIG不一致とWAV評価は[9/18run](../../experiments/2026-09-18_onb_full_sweep_151715/README.md)の結果へ無条件に移さない。

最新の数値と帯域依存は[2帯域解析の§6](../../experiments/2026-09-15_onb_frequency_comparison/analysis.md)、検証を進める方法は[解析手順](../analysis_workflow.md)。手法を「採用済み」であることと、「説明として数値的・物理的に検証済み」であることを区別する。
