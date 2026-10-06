# 元の3モデルを保持するための追加候補調査

2026-10-06。本人の最新指示は、RF・Conformer・AlexNetを固定し、その上にモデルを追加して回帰とONB評価を改善すること。旧判断の「RFの重みが0なので置換」は、この目的の主方式から外し、比較用診断として残す。モデルを固定することと、そのモデルへの絶対重みを固定することを区別する。今回は元3の予測と内部重み比率を保持し、元3ブロックと追加モデルの配分だけを学習する。

## 今回の比較へ進めた候補

| 系統 | 候補・入力 | 調査から立てる仮説 | 実際に識別する比較 |
|---|---|---|---|
| 木を平均 | ExtraTrees・周波数34 | ランダムな分岐を平均する方法は、HGBと異なる誤りを作る可能性 | 元3＋HGB＋ExtraTrees、追加FN/FPと残差相関 |
| 木を平均 | XGBRF・周波数34 | 元RFの弱さにはPCA画像入力が関与する可能性 | 元RFの予測を保持し、同じ周波数入力のRFとHGB単体を対応比較 |
| 逐次残差補正 | HGB正則化・周波数34 | leafサイズ・L2・leaf数の限定変更で分割感度が変わる可能性 | 基準HGB31/20/L2=1と、fit内で選ぶ限定3設定 |
| 逐次残差補正 | 通常GradientBoosting・周波数34 | 小標本ではbinningなしの分岐が有効な可能性 | 同じ34特徴、depth2/3、外側と強雑音へ固定転送 |
| 逐次残差補正 | XGBoost回帰・周波数34 | HGBと異なる木成長・標本/特徴抽出・正則化が補完する可能性 | 元3＋HGB＋XGBoost、clean OOF選択と全7条件 |
| 線形 | 直接Ridge・周波数34 | 木の分岐と異なる連続的な関係を補完できる可能性 | 単体と追加統合、強雑音での外挿失敗も確認 |
| 低次元線形 | PLS・周波数34 | 相関する多数の帯域要約を少数の教師あり成分へまとめられる可能性 | 成分4/8/12をfit内で選択、HGBとの相殺/共通失敗 |
| 非線形kernel | RBF Kernel Ridge・周波数34 | SVRとは異なる二乗損失・正則化による滑らかな補完の可能性 | gamma/alpha限定3設定。入力外で学習平均へ戻す標準化ラベル |
| 別の入力 | HGB・時間46 | 周波数形状に加え時間power分布が有用な補完を持つ可能性 | 同じseed43/44で再学習。時間順序を持つ特徴ではない |

ここに挙げた物理的役割・雑音耐性・精度向上は仮説であり、調査したモデル仕様だけから確定しない。今回の実行結果は[実験記録](README.md)へ別に固定する。

## 公式仕様から確認したこと

- ExtraTreesは多数のランダム化した回帰木を平均する。木の構造、最小leaf、使用特徴数を指定できる。[scikit-learn 1.6 ExtraTrees](https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)
- HGBはleaf数・最小leaf・L2で複雑さを制限できる。今回は導入済み1.6.1のAPIへ合わせ、early stoppingを使わず、同じ150 iterationsを維持する。[HGB 1.6](https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)
- 通常GBは各段階で残差方向を補正する。公式仕様はHGBの速度上の利点を中規模以上（標本数10,000以上）について説明しており、今回のfitは720〜1620 chunk。少数標本の非binning対照を実測する理由があるが、通常GBの優位を保証しない。[GradientBoosting 1.6](https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
- XGBoostはdepth、最小child、L2、標本/特徴抽出で正則化できる。元モデルのXGBRFは多数の木を同じboosting段階で作る方式であり、追加XGBRegressorの逐次boostingと区別する。[XGBoost公式パラメータ](https://xgboost.readthedocs.io/en/stable/parameter.html)、[XGBoost RF](https://xgboost.readthedocs.io/en/stable/tutorials/rf.html)
- RidgeはL2を持つ線形回帰、PLSは教師あり成分を持つ回帰、Kernel Ridgeはkernel空間の二乗損失回帰。今回の直接Ridgeは既存モデル出力を入力にするRidge統合とは別の基礎モデル。[Ridge](https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.Ridge.html)、[PLS](https://scikit-learn.org/1.6/modules/generated/sklearn.cross_decomposition.PLSRegression.html)、[Kernel Ridge](https://scikit-learn.org/1.6/modules/generated/sklearn.kernel_ridge.KernelRidge.html)

## 調査したが今回の計算へ追加しなかった候補

| 候補 | 今回の扱いと根拠 |
|---|---|
| LightGBM | leaf-wise boostingの正則化候補。公式はnum_leaves/min_data_in_leafなどで過適合を抑える方法を示す。同系統のHGBとXGBを先に実測し、同じ目的で依存を増やす優先度は下げた。環境未導入、未検証。[公式調整指針](https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html) |
| CatBoost | Ordered boostingは候補だが、今回はカテゴリ特徴のない固定34連続特徴。公式にvalidationを使うiteration選択がある。使う場合も外側を停止条件へ渡さない。環境未導入、未検証。[原論文](https://arxiv.org/abs/1706.09516)、[公式調整](https://catboost.ai/docs/en/concepts/parameter-tuning) |
| TabPFN | 小規模表データの回帰を扱うfoundation modelとして調査。論文の他データでの性能から本研究の雑音転送・録音反復への優位は導けない。追加実行環境・重み取得を要するため今回未検証。[原論文 Nature 2025](https://www.nature.com/articles/s41586-024-08328-6) |
| Gaussian process | 小標本・滑らかな関係の候補だが、今回の役割はKernel Ridgeで先に対照を取る。予測不確かさをそのままONBの正しさへ置換しない。未検証。[GP公式](https://scikit-learn.org/1.6/modules/gaussian_process.html) |
| 小型時系列NN・ONB直接分類・入力依存gate | 既存Conformer/AlexNetには時間入力があり、新構造の追加だけで不足が消えるとは限らない。今回の4/5モデル結果で残る誤りを具体化してから設計する。熱流束/SNRを知るgateや、分類確率100%とq100の同一視は使わない。 |

導入されていない候補を性能で不採用と決めたわけではない。今回、同一条件で9候補方針を比較し、追加に役立つ誤り方を実際に取得することを優先した。既存環境へライブラリ更新・新規依存を加えない。

## 統合と評価の設計

主比較は元3のperformance内部比率を固定する。公平な重み調整対照として、保存済み元3 MSE比率を固定する比較も実施する。4モデルは元3ブロック＋HGB、5モデルは元3ブロック＋HGB＋追加候補。

元3ブロック25%以上・追加各5%以上は、全メンバーを実際に保持して比較するための今回の明示的な設計制約であり、外側から選んだ値や最適な普遍比率ではない。固定4=75/25、固定5=50/25/25、4/5等平均も併記する。第5候補はclean OOF統合fitのMSEで選び、外側で良かったモデルを後から選ばない。OOF統合fitの値は独立な評価値とは呼ばない。

前処理・教師あり成分・ラベルscaler・ハイパーパラメータ選択は各fit内に限定する。学習はcleanのみ。雑音のOOF/外側への予測を保存するが、雑音を候補・統合の選択へ渡さない。元の3モデルと基準HGBの保存予測をそのまま使用するため、今回の利得を既存モデルの学習変更と混同しない。

RMSE/MAE/R²と、ONB前・近傍・以後の誤差、ROC-AUC/PR-AUC、Recall/F1/FN/FP、日別q100/g100、WAVごとの対応差を比較する。36録音からの既知WAV未使用chunk評価であり、540 chunkを独立録音540件として扱わない。同じ既存データを研究開発に反復使用しているため、独立な未知日検証とも呼ばない。
