# 実行順2：学習状態保存・元WAV分離epoch validation

実施日：2026-09-23。

## 目的と範囲

6/18の評価結果を見てepochや重みを選ばないように、6/11内だけを元WAV非共有のfoldへ分けてepoch別予測を保存する。また、同じ学習済み状態を全SNR・説明性計算へ再利用できるように、モデル、PCA、目的変数scaler、統合重み、seed、環境、分割を保存する。

今回の性能値は1 epochの**配線確認**であり、本性能の結論には使わない。200 epochs・3 seedの本実験は未実行。

## 実装したこと

- `training_validation`設定を追加。学習日だけの`wav_kfold`で、指定epochごとに未知WAV予測を記録する。
- epochごとに全域RMSE/R²/MAE/符号付き残差、ONB前・近傍・以後のRMSEとbias、見逃し・誤報を保存する。
- 最小の学習日内RMSEでConformerとAlexNetのepochを個別決定し、そのepochを最終fitと内部統合fitへ渡す。
- RF、Conformer、AlexNet、PCA、scaler、統合重みを保存し、SHA-256、ライブラリ版、実際のseed、学習元WAV、選択epochをmanifestへ記録する。
- 保存直後に別インスタンスへ再読込し、同じprobe予測との一致を検査する。
- PCA出力の約1e-9の演算差でRFの木分岐が変わったため、PCA変換を同一経路に統一し小数6桁で固定した。
- TensorFlowの厳密決定論を有効化。位置埋め込みは全位置を常に使うため、数式と重み形状を保ったまま疎な`Embedding`参照から密な学習可能行列へ変更した。旧保存重みを新実装へ読み込めることも確認した。
- RFの`random_state=42`固定を外し、実際のrun/fold seedを渡してmanifestへ記録するようにした。

## 実データ確認

条件は6/11 clean学習、6/18 clean評価、22 kHz、選別なし、seed 42、元WAV分離2 fold、1 epoch。最終コードで完了した2 runは次のとおり。

- `.../20260923/step2_smoke_xd-t0611-v0618_iw2-nc_s0_e1_213015/`
- `.../20260923/step2_smoke_xd-t0611-v0618_iw2-nc_s0_e1_213103/`

内部validationは6/11の18 WAVを各fold 9/9 WAV、540/540秒へ分け、共有WAVは両foldとも0。スモークなので選択epochはConformer/AlexNetとも1。

保存物の再読込予測差はRF、Conformer、AlexNetの全てで最大0 W/m²。2 run間の6/18全1080秒予測も、3単体と等重み統合の全てで最大差0 W/m²、配列完全一致だった。数値は[`same_seed_smoke_comparison.csv`](same_seed_smoke_comparison.csv)に固定した。

最終runのartifact manifestでは、RF・Conformer・AlexNetの実seedはいずれも43（run seed 42＋外側fold 1）、PCA固定桁は6、学習元WAVは18本、決定論演算は有効と記録された。

## 全SNR配線確認

同じclean fitを6/18の無雑音、0、−4、−8、−12、−16、−20 dBへ適用する1 epochスモークも完了した。

- 設定：`configs/experiments/2026-09-23_step4_noise_curve_smoke.json`
- 出力：`.../20260923/step4_noise__xd-t0611-v0618_iw2-nc_s0_e1_212547/`
- 完了：7条件の予測、指標、R²/ROC-AUC曲線、保存状態と再読込検証

1 epochでは深層モデルが未学習なので、この曲線の良否や谷の有無は研究結果として解釈しない。

## 4方式アンサンブルの配線確認

実行順6についても、`simple_equal`、`inner_holdout`、`subset_equal_cv`、`crossfit_shrinkage_stack`を同じ実データで1 epochだけ実行し、4方式の学習側重み、予測、比較表、crossfit監査が保存されることを確認した。

- 設定：`configs/experiments/2026-09-23_step6_ensemble_smoke.json`
- 出力：`.../20260923/step6_ensemb_xd-t0611-v0618_iw2-nc_s0_e1_213758/`
- 完了：`completed.json`、`ensemble_crossfit_fit_f1.json`、4方式の重み・指標

1 epochでは深層モデルが未学習なので、方式の順位と重みは採用判断に使わない。本比較は実行順4の200 epochs×3 seedが完了した後に行う。

## 検証と残作業

- `python -m unittest discover -s tests -p 'test_*.py'`：最終コードで73件成功。
- 200 epochs×学習日内3 fold×3 seedは数十分〜数時間規模になり、短時間作業ではないため開始していない。
- 本実験用設定は`configs/experiments/2026-09-23_steps2-4_full.json`。seed 42/43/44は個別に明示して実行する。
- 実行順5の保存結果監査と7-Cは[再学習なし監査](../2026-09-23_steps2-7_readonly_audit/README.md)で完了済み。5の学習曲線側、6の本比較、7-Dの保存モデルIGは本学習後に行う。

本実験を開始するときのコマンドは次の3本。各runの完了を確認してから次へ進む。

```powershell
python experiments/2026-09-19_b_clean_only/run_from_condition.py --condition configs/experiments/2026-09-23_steps2-4_full.json --seed 42
python experiments/2026-09-19_b_clean_only/run_from_condition.py --condition configs/experiments/2026-09-23_steps2-4_full.json --seed 43
python experiments/2026-09-19_b_clean_only/run_from_condition.py --condition configs/experiments/2026-09-23_steps2-4_full.json --seed 44
```
