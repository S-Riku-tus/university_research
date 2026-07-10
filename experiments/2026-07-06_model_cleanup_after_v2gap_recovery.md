# 2026-07-06 model cleanup after v2 GAP recovery

## Reason

The CNN+Transformer v2 GAP architecture recovered regression performance on
`2025.06.18_0.3_3` while the recently tested `cnntf_v1` and Conformer variants
did not. To reduce confusion in future tuning, the active model registry was
trimmed to the models that still have a clear role.

## Kept

- `rf`: strongest baseline and fixed ensemble candidate.
- `alexnet`: fixed CNN baseline candidate.
- `cnntf_v2_gap`: current CNN+Transformer candidate; latest partial tuning
  reached about `R2 = 0.85`.

## Removed From Active Code

- `cnntf_v1`: removed from `MODEL_SPECS` and deleted from
  `RegressionModelMaker`; recent tuning showed collapsed regression R2.
- `conformer`: removed from `MODEL_SPECS` and deleted from
  `RegressionModelMaker`; recent architecture tuning did not recover the
  desired regression accuracy.
- `cnntf_v2_attn`: removed from `MODEL_SPECS`; keep the underlying
  `cnn_transformer_v2(pooling="attention")` option available only inside the
  shared v2 function for manual experiments.

## Compatibility Edits

Legacy scripts that called `cnn_transformer_v1()` now call
`cnn_transformer_v2()` so they do not fail immediately after the cleanup.

## Remaining Cleanup Candidate

`efficientnet_transformer_v1()` remains in `base_regression.py` but is not in
the active registry. It can be removed later if older exploratory scripts do not
need it.
