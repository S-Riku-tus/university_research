# 決定論GPUでのIG BatchNormalization勾配修正

日付: 2026-09-25。

## 発生した問題

ONB主コードの3 kHz・無雑音条件で、ConformerのIntegrated Gradients（IG）計算中に次の例外で停止した。

```text
UnimplementedError: A deterministic GPU implementation of fused batch-norm
backprop, when training is disabled, is not currently available.
[Op:FusedBatchNormGradV3]
```

環境はTensorFlow 2.9.1、CUDA 11.3、cuDNN 8.2.1、RTX 4090。主コードは学習再現性のため`TF_DETERMINISTIC_OPS=1`と`enable_op_determinism()`を使用する。IGは推論モード`training=False`のモデルに入力勾配を求めるため、Fused BatchNormalizationの決定論的GPU逆伝播が必要になるが、このTensorFlow版には実装がない。

最小再現では、同じFused BatchNormalization勾配がGPUで同例外、CPUで成功した。また、IG計算中だけ`BatchNormalization.fused=False`としたGPU勾配は成功し、簡易モデルの切替前後の推論値差は0だった。

## 修正方針

学習、通常予測、モデル重み、IGのbaseline・経路・Gauss–Legendre積分・収束判定は変更しない。

1. IG計算中だけ、対象モデルに含まれる`BatchNormalization(fused=True)`を一時的に`fused=False`へ切り替える。
2. IGの基準点・対象点における推論値を切替前後で比較し、`rtol=1e-5, atol=1e-6`を外れる場合は計算を拒否する。
3. 計算成功・例外にかかわらず、`finally`で全BatchNormalizationの`fused`属性を元へ戻す。
4. GPU上の未実装演算が生じた場合、または非fused化した端点予測が厳密な許容範囲を外れた場合は、元のfused設定を保持したまま同じIGをCPUで自動再試行する。
5. 実際の勾配device、非fused化した層数、端点予測差、CPU fallbackの有無と元のGPU例外をdiagnosticsへ保存する。

この変更は説明計算用の演算カーネルだけを切り替える。学習済みパラメータ、moving mean/variance、通常の評価予測、アンサンブル処理は変更しない。

## 設定

`DEFAULT_EXPLAINABILITY_CONFIG`へ次を追加した。

```python
"ig_device": "auto",
"ig_nonfused_batchnorm": True,
"ig_cpu_fallback": True,
```

通常は決定論GPU上の非fused BatchNormalizationを使う。CPU fallbackは未実装GPU演算が残る場合、または非fused化した端点予測が厳密な等価性判定を外れた場合に使用する。

## 実runで確認した追加事象と対応

現行ONB runのConformerで、非fused化前後の端点予測の最大絶対差が`1.3113e-05`となり、固定判定`rtol=1e-5, atol=1e-6`を外れて停止した。これは学習エラーではなく、IG用GPUカーネル切替の等価性チェックによる安全停止である。

この差を許容するよう閾値は緩めていない。等価性チェックを外れた場合は、IG計算中に変更したBatchNormalizationをまず元へ戻し、元のfused設定のモデルをCPU上で再計算するよう修正した。したがって、学習、通常予測、評価値、重み、baseline、積分経路および収束判定は変更しない。diagnosticsには`cpu_fallback_reason=nonfused_batchnorm_endpoint_mismatch`と元の例外を保存する。

## 検証項目

- 既存IG数値テストが変わらず通る。
- BatchNormalizationを含むモデルでIGが有限値を返す。
- IG後に`fused=True`が復元される。
- IG前後の通常予測が変化しない。
- 実際のConformer構造を決定論GPU上で説明したとき、`FusedBatchNormGradV3`例外が再発しない。
- コード構文と関連テストを確認する。

IGが実行完了することと、completeness・map安定性により研究上採用できることは別である。未収束の場合は従来どおりdiagnosticsへ記録し、収束を強制する正規化は行わない。

## 検証結果

- `tests/test_integrated_gradients.py`: 11件成功。既存の解析解、raw/log-power経路、符号、completeness、保存診断、stability・sanity checkを維持し、端点差から元のBN設定によるCPU再試行へ移る経路も確認した。
- 全テスト: 79件成功。
- 現行Conformer構造・224×224入力・決定論GPU: IG完走、勾配deviceはGPU、非fused化2層、切替端点予測差0、IG前後の通常予測差0、全IG値finite、計算後に全層を復元。
- 現行AlexNet構造・224×224入力・決定論GPU: IG完走、勾配deviceはGPU、非fused化2層、切替端点予測差0、IG前後の通常予測差0、全IG値finite、計算後に全層を復元。
- fallback強制試験: fused BNのGPU例外を記録後、CPUで同じIGを再計算し、全値finiteで完了した。
- 対象4 Pythonファイルの構文確認と`git diff --check`に合格した。

実構造スモークは経路の完走を短時間で確認するため、積分点を2→4とし、緩い収束許容値を使った。これは本runの説明結果や研究上の収束実績ではない。本runでは既定の64→最大4096点と既定許容値を使用し、標本ごとに採否を判定する。
