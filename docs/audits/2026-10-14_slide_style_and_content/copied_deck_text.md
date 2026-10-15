## Slide 1 / chars=55 / paragraphs=6 / pictures=0 / image_area_sum=0.00
アンサンブル学習に基づく音響識別による
プール沸騰の発生検知
植木研究室　　柴崎陸
2026年10月16日
1
研究進捗

## Slide 2 / chars=85 / paragraphs=7 / pictures=1 / image_area_sum=0.00
Tokyo University of Science
2
今回の発表内容
01
02
03
04
学習・評価設計と音響選別
回帰・ONB判定と統合結果の分析
説明性
後期の研究計画

## Slide 3 / chars=103 / paragraphs=12 / pictures=1 / image_area_sum=0.00
3
Tokyo University of Science
1. 学習・評価設計
学習・推論時のデータを別日の実験で得た音響データとした
例1
6/11 学習
→
→
6/18 評価
6/11, 18 学習
6/11 評価
例2

## Slide 4 / chars=121 / paragraphs=11 / pictures=3 / image_area_sum=0.40
Tokyo University of Science
4
1. 音響選別
ONB直後からそれ以降である閾値を超えたもののみ沸騰データとした
沸騰データとして残す
未沸騰データとして削除
閾値：
2.1~2.5kHzで
1×10^-9
×10^-8
×10^-9

## Slide 5 / chars=161 / paragraphs=12 / pictures=2 / image_area_sum=0.29
Tokyo University of Science
5
入力ノイズなしの検証では，アンサンブルが最も精度が高くなった
2. 熱流束回帰
実行条件
学習データ：
6/11実験 1080個（未沸騰フィルタ後986個）
テストデータ：
6/18実験　1080個
エポック数：200
Fold数：3
その他各モデルのパラメータは適切なものを設定した

## Slide 6 / chars=74 / paragraphs=4 / pictures=2 / image_area_sum=0.59
Tokyo University of Science
6
マスクによる精度低下の影響が大きい周波数帯はモデルごとに異なる
3. 説明性（周波数帯マスク）

## Slide 7 / chars=114 / paragraphs=8 / pictures=4 / image_area_sum=0.47
Tokyo University of Science
7
同じ入力でも寄与分布は異なる
3. 説明性（Integrated Gradient）
※22 kHz・0 dB、ONB時点の同一データを入力
AlexNet
Conformer
入力

## Slide 8 / chars=363 / paragraphs=5 / pictures=1 / image_area_sum=0.00
Tokyo University of Science
9
4. 後期の研究計画
今ここ
今ここ
大項目
小項目
9月
10月
11月
12月
1月
2月
① 評価設計・学習
データの確定
実験別に学習／テストデータを分割の実装
沸騰後のデータで閾値を設定・スクリーニング実装
上記実装でアンサンブル検証（全条件）
② アンサンブル手法
の確立
【
必要なら
】
単体モデルを下回る条件・要因を調査・整理
【
必要なら
】
必要に応じて新たなアンサンブル手法を調査・実装
【
必要なら
】
複数手法の比較でアンサンブル方法を確定
③ 説明性を用いた要
因分析・検証
説明性指標で特徴利用を可視化・比較
各指標を踏まえて学習・推論パイプラインに活用できるか模索・実装・考察
④ 修論執筆・発表準
備
背景・実験方法（など書ける範囲から進める）
結果・考察・結論の整理
論文執筆完了・発表資料作成・発表練習など
