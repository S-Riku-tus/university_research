# 元WAV分離crossfitアンサンブルの追加・検証記録

日付: 2026-09-16。**実装・合成データでの検証済み。研究実データの新方式での本学習・性能確認は未実施。**

## 本人の依頼と判断

本人が添付した2件の生成AI提案を読み、現行ONB実装・9/15の3/22 kHz比較・研究仮説と照合した。単体より統合で悪化する場合を減らすため、既存2方式を維持して `subset_equal_cv / crossfit_wav_stack / crossfit_shrinkage_stack` を選択式で追加した。

根拠は、現行inner holdoutが12外側学習WAV中3 WAVから重みを決めていること、単体誤差の逆数が誤差相殺を直接最適化しないこと、WAV中央値とモデル間加重平均の順序が交換できないこと。最後の点は過去の [誤差分解CSV](../2026-09-15_onb_frequency_comparison/comparison/ensemble_error_decomposition.csv)にも差が記録されている。

新方式では同じ外側学習WAVから4-fold inner OOFを作る。候補評価・最適化ともchunk統合後のWAV中央値を目的とし、等重み縮小は事前固定λ=0.1の罰則で実装する。非凸目的なので、多始点の数値解を大域最適解とは呼ばない。詳細な [5方式の手法と数式](../../docs/ensemble_methods.md)。

## 変更箇所

- `code/utils/ensemble/crossfit_stacking.py`: 元WAV分離inner分割、部分集合選択、WAV目的の制約最適化、縮小、診断。
- `ensemble_runtime.py / strategy_catalog.py / strategy_comparison.py`: 選択時だけの共通inner学習、重み適用、監査保存。
- `code/utils/experiment/learning_runner.py`: 学習側のfit結果を各評価recorderへ渡す。clean_onlyではノイズ間で同一のfitを共有。
- `code/run_ensemble_regression_onb.py`: 新3方式のコメント例のみ。実行される設定・処理のASTは変更前と一致。

## 確認結果

`python -m unittest discover -s tests` をCPU指定で実行し、**44テストすべて成功**。うち新規10テスト。CPU指定はテスト用プロセスの環境変数のみで、研究コードのGPU設定は変更していない。先行するGPU試験は空きメモリ確保に失敗し、プロセスが終了したため、GPU実行の確認済みとは扱わない。

確認した内容:

- 12 WAV・chunk数不均等でもinner学習/予測WAVが交わらず、全chunkが一度だけ予測される。
- 有害なモデルの除外、単体への選択、逆方向の誤差の相殺が合成例で動く。
- 中央値の順序に差が生じる合成例で、実際のchunk→WAV順序の損失を最小化する。
- 同じWAVのchunk列全体を反復してもそのWAVを重く数えない。
- 縮小で等重みに近づき、熱流束単位を一律変更しても同じ重みになる。
- 非有限予測・矛盾するWAV真値・未fitの重み・別foldの重みを拒否する。最適化失敗時は有効な部分集合候補へ戻し、診断に記録する。
- 日内/別日 × matched/clean_onlyの4方針で、データ読込・PCA/scaler・inner/outer予測・CSV/JSON/WAV/ONB指標・ノイズ図用集計・完了再開を通した。
- 新3方式を併用してもinner OOFは共通4回だけ。clean_onlyは学習/PCAにノイズ評価データを使わず、全評価ノイズで同じ重み・inner診断になる。
- 小規模Ridgeによる4方針の既存単体・既存2方式の予測列は、新方式併用時と旧方式のみの別実行で完全一致。
- 実際の小型Kerasモデル2つを各4 inner foldで1 epoch学習し、PCAなしの経路、OOF全件、学習履歴が正常。
- 新方式選択でhashが変わり、旧選択では新しい設定項目も追加fitも生じない。`git diff --check`も成功。

## 未検証と次の工程

実RF/CNN+Transformer/AlexNet・300 epochsでの新方式の成績は未確認。単体より優れること、ノイズ耐性やONB検知が改善することは、今回の合成テストからは結論づけない。現在の主実行の既定選択は従来の2方式のまま。

次は [代表条件書](../../configs/experiments/2026-09-16_crossfit_ensemble.yaml)のように、同一モデル・外側分割で既存2方式と新3方式の実データ比較を行う。これは未実行の提案で、進行中runを今回の結果として追加採用していない。全ノイズ同時最適化、XAI gating、ONB校正、多表現expertの追加は別設計とする。
