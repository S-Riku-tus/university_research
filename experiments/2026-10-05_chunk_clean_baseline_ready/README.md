# chunk内部検証・clean_only基準の実行準備

確認日: 2026-10-05。本人の「現在のONBコードで実行してよいか、必要なら修正」の依頼に対応した。**設定・事前確認まで完了。本学習と新しい精度結果は未実施。**

同日後続：この準備後の本人実行はRFの保存再読込検証で停止した。[原因と修正](../2026-10-05_rf_reload_pca_layout_fix/README.md)を参照する。以下の9テストと事前確認は実行前の記録であり、保存後のPCA配置による数値差を覆っていなかった。

- [主実行コード](../../code/run_ensemble_regression_onb.py)を`training_noise="clean_only"`へ変更し、matchedを切替用コメントで残した。
- 最終学習モデル・PCA・target scaler・重みの保存と再読込検証を、主実行の`output`で明示的に有効にした。共通の軽量保存既定は変更していない。状態はclean条件の`fitted_state/fold1/`に置き、他の雑音条件から同じ状態を参照する。
- `validation_config_snapshot()`が`internal_validation_split`を実行設定へ転送していなかった箇所を修正した。chunkでは既定値によって動いていたが、WAV対照の選択が伝わらなかった。両方式の転送・条件hashの識別と、不正値の学習前拒否を確認した。
- 3 kHz・1秒・選別なし・150 epochs・PCA100・既存3モデル採用値固定・内部3-fold/seed42・外側各WAV25%/seed42・performance/等重みを維持。パラメータ探索とXAIは無効。
- 実データの全7条件各2160 IDとラベル・時間区間の対応、各条件の外側1620/540、旧10/1テストID一致、内部各1080/540・全36 WAV支持・chunk非共有・OOF全件1回を確認。各条件の1配列は224×224で有限値だった。全配列を一括で読み込んだ確認ではない。
- 関連9テスト通過。学習はテスト用小型モデルだけで、研究3モデルの150 epochsを実行したものではない。

[実効条件記録](../../configs/experiments/2026-10-05_chunk_clean_baseline_ready.json)、[確認結果](verification.json)、[次工程](../../docs/research_plan/2026-10-05_after_chunk_split_next_steps.md)。日付と実行IDは本学習起動時に生成される。日別q100/g100は完了後の保存予測から解析し、改善の有無を判断する。

研究ルートから実行する場合:

```powershell
python -X utf8 code/run_ensemble_regression_onb.py
```

実装確認に使用したテスト:

```powershell
python -X utf8 -c "import sys,unittest; sys.path.insert(0,'tests'); names=['test_onb_entrypoint','test_onb_defaults','test_training_state_persistence','test_within_wav_chunk.WithinWavChunkTest.test_runner_excludes_test_from_inner_cv_preprocessing_and_fit']; result=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromNames(names)); sys.exit(0 if result.wasSuccessful() else 1)"
```
