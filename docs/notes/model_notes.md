# モデル・説明性の参照メモ

更新日: 2026-09-15。成績と採用判断は[現在地](../research_status.md)、構造の詳細は[コード地図](../code_map.md)に集約する。

| 現行キー | 実体 | 説明性 |
|---|---|---|
| `rf` | XGBRFRegressor、flatten→PCA（学習側でfit） | PCA空間TreeSHAP、帯域/時間マスク |
| `cnntf_v2_gap` | log-power、CNN、時間tokenのTransformer、GAP | IG、帯域/時間マスク |
| `alexnet` | log-power、AlexNet系CNN、大きい全結合ヘッド | IG、Grad-CAM、帯域/時間マスク |

「Conformer」「v1/v2」「AttnPool」は過去資料の世代名を含む。現行のキーと構造を[モデル定義](../../code/utils/models/regression/base_regression.py)で確認する。VGG16・ResNet50・WaveNet・SELDnet等は、存在するコード/過去候補と現在の有効モデルを区別する。

現行の統合は`simple_equal`と`inner_holdout`。`val_fold_legacy`は外側正解を使う診断・再現用で主張不可。`prediction_max`は9/14最終変更で削除済み。[方式カタログ](../../code/utils/ensemble/strategy_catalog.py)が実装上の正本。

`inner_holdout`も世代を区別する。9/8より前のchunkランダム分割は元WAV混在があり、旧結果では主張用から除外される。現行の元WAV非重複holdoutはnested stackingではない。

説明性画像の生成と、その数値整合・物理的妥当性は別に確認する。TreeSHAPをPCAから元画素へ単純に戻して物理寄与と呼ばない。IG/マスクの最新の解釈は[発表準備メモ](../research_plan/2026-09-18_xai_progress_brief.md)へ。
