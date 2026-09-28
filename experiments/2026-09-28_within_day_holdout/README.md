# ONBの日内固定テスト切替

本人の2026-09-28指示により、主方式performance_kfold・主帯域3 kHzを維持し、別日評価と1実験日内の固定テスト評価をconfigで選択可能にした。

- 設計・研究判断: [3 kHz、matched、チューニング、評価方式](../../docs/research_plan/2026-09-28_3khz_matched_tuning_and_evaluation.md)
- 新規比較の条件記録: [6シナリオ](../../configs/experiments/2026-09-28_3khz_evaluation_matrix.json)。条件記録のみで未実行。
- 実装: `learning_policy.evaluation_mode`を`cross_day`または`within_day`へ。日内では`within_day_experiment`だけから実効日付を設定。
- 日内テスト: 元WAV25%、seed 42、ONB前後層化。残りで従来の内部WAV K-foldと重みfitを実施。学習seedから外側split seedを独立化。
- 保存: 外側分割は1回、`wh`パス、manifestにWAV・実効方針・評価領域別chunk数を記録。
- 互換性: 新キーがない旧条件は日付リストから従来の分割を推論。既存日内外側K-foldやleave-one-day-outを保持。

## 検証済み

- 新テスト4件、既存learning_policy 14件、experiment_day_resolution 4件、crossfit_stacking 10件、day_split_acoustic_selection 7件、合計39件成功。
- 小型データでclean_only/matchedの両方を実行し、テストWAVが内部OOF・PCA fit・最終fitへ入らないこと、noise間の同じテスト割当、fit共有/独立、完了再開時の学習省略、保存を検証した。
- 実データのNPY配列は学習せずmanifestのみ照合。6/11・6/18各7 noiseにおいて同じ5テストWAVを割当、13学習WAVと非共有。
- 各日780学習chunk・300テストchunk。テストONB前120/以上180。ONB±10%のテストchunkは6/11で0、6/18で60。日内近傍精度の解釈にこの非対称を残す。
- 主エントリの読込と実効configを確認。新日内条件の150 epoch本学習、matched追加本学習、パラメータ探索は未実行。

主configの実行方針はcross_day、6/11→6/18、clean_only、3 kHz、seed 42を維持した。次の新規比較を実行する際はtraining_noiseをmatchedへ変更する。専用の学習側限定チューニング入口は今回実装していない。
