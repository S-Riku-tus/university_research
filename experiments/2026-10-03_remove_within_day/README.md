# 外側評価の2方式への整理と日内WAV分割の削除

実施日：2026-10-03。本人の方針に合わせ、onbで選択する外側評価を`within_wav_chunk`と`cross_day`に整理した。外側のWAV単位日内holdoutと旧日内K-foldを削除し、学習側だけの内部WAV検証は維持した。過去の結果・条件記録・提出資料は削除していない。

## 評価方針

| 外側方式 | 使用場面 | 設定する欄 | 評価できる範囲 |
|---|---|---|---|
| within_wav_chunk | 現有データを使う主評価 | evaluation_settings.within_wav_chunkのexperiment、test_fraction、test_split_seed | 単日/統合フォルダの各WAV内の未使用chunk。同じchunkは共有しないが同じ録音条件は共有する |
| cross_day | 新実験データ等で別日評価を行う場合 | evaluation_settings.cross_dayのtrain_experiments、test_experiments | 学習とテストの日リストを完全分離した評価 |

WAV単位の日内評価には「同じ日の未知WAV」の性能を調べる役割があった。方法自体が不適切という理由ではなく、本人の現行研究範囲では独立した主評価として使わないため削除した。未知WAV・未知日は主結果の必須検証へ自動追加しない。新実験を用いる場合にcross_dayを選ぶ余地を残す。

## 変更箇所

- [onb設定](../../code/run_ensemble_regression_onb.py)：within_day欄・説明・外側fold分岐を削除。現在のwithin_wav_chunk、3 kHz、1秒、統合、clean_only、選別なし、採用パラメータは維持した。
- [分割処理](../../code/utils/experiment/learning_policy.py)：WAV holdout、ONB層化holdout、暗黙の日内GroupKFoldを削除。廃止方式を指定するとエラーになり、別方式へ自動置換しない。
- 同ファイルの実効設定キーは`within_day_experiment`から`within_wav_chunk_experiment`へ変更し、不要な`test_stratify`を廃止した。主configの`evaluation_settings.within_wav_chunk.experiment`という書き方は変わらない。
- [学習runner](../../code/utils/experiment/learning_runner.py)・[OOFチューニング](../../code/utils/experiment/training_oof_tuning.py)：外側を固定1分割に統一。内部検証のWAV分離・PCA/scalerの学習範囲・候補選択は維持した。
- [保存先](../../code/utils/experiment/result_paths.py)：廃止方式の命名分岐を削除。現行のwc/xd名は維持した。
- [noise trend](../../code/utils/plotting/noise_trend_plots.py)：廃止方式名ではなく保存された外側分割情報から集計を判断。現在のchunk holdout図の説明は既知WAV内の未使用chunkを表す文言へ変更した。CSV/画像の保存先は維持した。
- テストを[WAV内chunk分割のテスト](../../tests/test_within_wav_chunk.py)へ整理。旧方式の動作テストは削除し、旧指定の拒否と現在方式の再現性を検査する。

方式省略時の既存cross-day/leave-one-day-out互換は今回の削除対象と切り離して保持する。主configの選択肢は2方式であり、複数日を統合してchunk評価するときは統合フォルダをwithin_wav_chunkへ指定する。

## 確認結果

- 関連8ファイルの計60テストが成功：within_wav_chunk 8、learning_policy 14、experiment_day_resolution 5、training_oof_tuning 7、noise_trend_plots 6、crossfit_stacking 10、onb_defaults 3、day_split_acoustic_selection 7。
- 削除前の合成18 WAV×60 chunkのseed 42分割をSHA-256で固定し、変更後も一致することを検査した。各日のテストは270 chunk、統合は540 chunkで単日runとの出典別対応を保つ。
- 実際の統合データのchunk_manifestを現行configで分割し、10/1採用値runの保存済みclean予測と元WAV/chunk IDを照合した。学習1620／テスト540、6/11と6/18各270で、全540テストIDが一致した。NPY配列の読み込みや本モデル再学習は行っていない。
- 内部検証・PCA・最終学習へ外側テストchunkが入らないこと、clean_only/matched間の分割一致、完了runの再開、cross_day学習日分離、OOF候補選択で外側テスト配列を使わないことを確認した。
- 統合主条件の保存名は`onb_wc-t0611-v0611_iw3-nc_c1s_s0_e150_170655`と同じ形式になることを確認した。

通常の150 epochs・3モデルの本runは実行していない。変更がないことを確認したのは入力分割と処理経路であり、新しい学習結果の数値一致まで検証したとはしない。

## 過去記録の扱い

過去のwithin_day結果や日付付きconfigは履歴として保持した。廃止方式の設定は現行コードでは実行できない。古い平坦なwithin_wav_chunk条件JSONを流用する場合も、実験指定キーを`within_wav_chunk_experiment`へ変更し、`test_stratify`を外す必要がある。

実効設定キーの整理によって新旧の設定hashは異なり得る。過去の実行IDを使って旧manifestへ強制再開せず、新しい実行として扱う。既存保存先の変更・移動・上書きは行っていない。
