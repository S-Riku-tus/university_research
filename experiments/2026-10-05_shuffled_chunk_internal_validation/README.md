# シャッフルありchunk分割への切替

実装・確認日: 2026-10-05。本人の「既知WAV内の未使用chunkにも評価としての価値があり、外側・内部ともchunk分割にしたい。学部と同じシャッフルありでよい」という指示に対応した。[条件書](../../configs/experiments/2026-10-05_shuffled_chunk_internal_validation.json)。

## 変更したこと

- 外側`within_wav_chunk`は実装済み。各WAVの60 chunkからseed42で15個をランダムにテストへ取り分け、45個を学習へ残す。処理・割合・テストIDは維持した。
- 内部`performance_kfold`と`training_oof`候補探索は、外側学習chunk全体の通常KFoldへ切り替えた。`shuffle=True`、seed42、3-fold。各WAVの件数を揃える層化や時間blockではない。
- [主設定](../../code/run_ensemble_regression_onb.py)は`run.internal_validation_split="chunk_kfold"`。省略時も同じ方式になる。`wav_kfold`は明示した過去条件対照として残したが、通常実行の既定ではない。
- 同じchunkのfit/validation共有を防ぎ、各chunkのOOF検証は1回。重複した元chunk IDはエラーにする。PCA・scaler・音響選別は各fitだけで学習する。
- 単体のpooled chunk OOF R²と逆誤差重みの計算式は維持した。q100優先の重み最適化や新統合を追加した変更ではない。
- 内部方式を実効条件hashへ含め、旧WAV分割の完了結果をchunk分割として再利用しない。新保存先の内部方式tokenは`ic3`、明示WAV対照は`iw3`。元の結果は変更していない。

別日評価`cross_day`は引き続き実験日を外側で分離する。その場合も、指定した学習日内のperformance/tuning内部検証はchunk KFoldになる。現在無効の`inner_holdout`・WAV型crossfit方式は、それぞれの既存分割・目的を維持しており、今回自動採用していない。

## 実データの分割監査

現行主設定をASTから読み、10/1 clean_onlyの2160サンプルのmanifestから分割だけを再現した。配列・予測の読込、研究モデルの学習はしていない。[監査結果](audit.json)。

外側学習1620、テスト540、36 WAV。テスト540の添字は10/1保存結果と全件一致し、同じchunkの共有は0。全内部fit/validationは外側テストのchunkを含まず、1620 chunkを1回ずつ検証した。

| 内部fold | 学習chunk | 検証chunk | 学習に残るWAV | 各WAVの学習chunk数の範囲 | 同じchunk共有 |
|---|---:|---:|---:|---:|---:|
| 1 | 1080 | 540 | 36/36 | 23～37 | 0 |
| 2 | 1080 | 540 | 36/36 | 26～36 | 0 |
| 3 | 1080 | 540 | 36/36 | 24～37 | 0 |

両日それぞれ18熱流束段階が全foldの学習に残る。ONB段階の学習/検証chunk数は次のとおり。

| 日 | fold1 | fold2 | fold3 |
|---|---|---|---|
| 6/11 ONB | 27 / 18 | 36 / 9 | 27 / 18 |
| 6/18 ONB | 29 / 16 | 33 / 12 | 28 / 17 |

[fold要約](internal_fold_summary.csv)、[熱流束支持](heat_flux_support.csv)、[内部ID](internal_chunk_membership.csv)。通常KFoldは段階ごとの比率を保証しない。全段階が残ったのは今回の本数・seed・分割で確認した事実であり、少数chunkや選別ありの条件でも必ず成立する保証ではない。

同じWAVは学習/検証に共有されるが、同じchunkは共有されない。評価の位置づけは既知WAV内の未使用区間であり、別日・未知WAVの性能と区別する。前段の[時間block・WAV均衡案](../2026-10-05_onb_endpoint_and_grouped_fold_review/README.md)は別の候補比較として保存し、今回の選択へ自動追加しない。

## 実装の確認

関連46テストを通過した。対象は、分割の非共有・seed再現性・OOF coverage・PCA/学習からの外側テスト除外、通常重み計算と候補探索への適用、少数WAVでもchunk数が十分な場合の動作、旧完了結果との識別、clean_only/matchedの分割対応・再開・学習状態保存である。テスト用小型モデルの学習は研究モデルの学習と区別する。

```powershell
python -X utf8 -c "import sys,unittest; sys.path.insert(0,'tests'); names=['test_day_split_acoustic_selection','test_within_wav_chunk','test_training_oof_tuning','test_learning_policy','test_onb_defaults','test_training_state_persistence']; result=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromNames(names)); sys.exit(0 if result.wasSuccessful() else 1)"
python -X utf8 experiments/2026-10-05_shuffled_chunk_internal_validation/audit_splits.py
```

## 次に確認すること

実装・設定・分割監査は完了した。旧3モデルの新しい内部OOF、重み、最終モデル、外側予測はまだ出力していない。内部学習に全段階が残ることを、回帰精度・q100改善の検証済み成果とはしない。

次の本比較は、基本方針のclean_onlyを主軸にこのchunk分割で既存3モデルを再学習し、同じ外側540 chunkの旧結果と日別q100/g100・誤報・回帰誤差を比較する。新候補や入力表現を増やす前に、分割変更だけの結果を基準として固定する。matchedは対応する補助比較として保持する。
