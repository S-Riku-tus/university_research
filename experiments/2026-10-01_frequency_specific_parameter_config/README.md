# 最大周波数別パラメータ設定

## 目的

3 kHzと22 kHzの学習側OOF探索で採用値が異なったため、選択した最大周波数に対応するパラメータだけを通常run／OOF探索へ渡す。完了済み22 kHzを再実行せず、次の3 kHz外側評価を同じ主コードから実行できる状態にする。

## 実装

- `VALIDATION_CONFIG.models.parameter_sets`を`max_freq_active_model_grid`へ変更し、`by_max_freq_hz`の下に周波数別の3モデル候補を置いた。
- `data.max_freq_hz_list`で選択された周波数ごとに候補リストを解決する。通常runで複数周波数を選んだ場合も、各周波数は専用パラメータで別々に実行される。
- 各周波数の全候補リストが1要素なら通常run、どれかが複数要素なら従来どおりモデル別の学習側OOF探索へ自動切替する。
- OOF探索時は出力衝突と評価範囲の混在を避けるため、`max_freq_hz_list`を1周波数だけに制限する。通常runでは複数周波数を選べる。
- run manifestの`models.active_max_freq_hz`と`models.parameter_sets`には、実際にそのfamilyで使用した周波数と解決済みパラメータだけを保存する。

## 登録した採用値

| 最大周波数 | RandomForest | Conformer | AlexNet |
|---|---|---|---|
| 3 kHz | `n_estimators=100, max_depth=12, subsample=0.6, colsample_bynode=0.6` | `lr=0.001, batch=12` | `lr=0.003, batch=8` |
| 22 kHz | `n_estimators=600, max_depth=6, subsample=0.6, colsample_bynode=0.6` | `lr=0.0003, batch=8` | `lr=0.01, batch=24` |

## 現在の実行条件

`data.max_freq_hz_list`は3 kHzだけを有効にしてある。7 noise、1秒、6/11＋6/18統合、`within_wav_chunk`、clean_only、選別なしなどは直前の22 kHz外側評価から変更していない。したがって、現状の主コードを実行すると、3 kHz採用値による外側540 chunk・7 noiseの通常runになる。

22 kHzを再実行する場合は3 kHzをコメントアウトして22 kHzを有効にする。両方を同じ実行系列で通常runする場合は両方を有効にできる。

## 確認

- 変更4ファイルの構文コンパイル成功。
- パラメータ計画、学習方針、dataset jobに関係する29テスト成功。
- 実データplanは3 kHzの7/7 datasetを検出した。
- 学習関数を差し替えたdry runで、3 kHz jobだけに3 kHz採用値が渡ることを確認した。
