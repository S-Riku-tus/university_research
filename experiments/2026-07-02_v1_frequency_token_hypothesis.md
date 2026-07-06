# 2026-07-02 v1 Frequency-Token Hypothesis

## Background

The current `.npy` files are saved as `(time_frame, frequency_bin)`.
This is confirmed by `code/2.run_npy_waterflow_2つhighpass.py`:

- `calc_stft()` appends one power spectrum per frame.
- `amplitude_cropped = amplitude[:, :max_k + 1]`.
- `np.save(..., resized_amplitude)` saves this array directly.

The PNG spectrogram path rotates the array for visualization with
`np.rot90(resized_amplitude, k=3)`, so the visible PNG orientation should not
be used as the model-axis definition.

## Important Axis Point

`cnn_transformer_v1()` historically assumes input axes are
`(frequency_bin, time_frame, channel)` and splits axis 1 into patches.

With the current `.npy` axes `(time_frame, frequency_bin, channel)`, axis 1 is
actually the frequency axis. Therefore, v1's Transformer tokens are frequency
bands, not time patches.

This means the old v1 result may have benefited from frequency-band token
processing, even if it was originally interpreted as time-axis processing.

## Current Code Setting

`code/run_ensemble_regression_onb.py` is now configured for a narrow
`cnntf_v1` single-model check on the restricted current data condition:

- experiments: `2025.06.18_0.3_3`, `2025.07.09_0.3_1`,
  `2025.06.11_0.3_2`
- max frequency: `maxfreq=22kHz`
- noise: `heatflux_no_noise`
- result date suffix: `_v1freq`

Parameter sets:

| parameter_set | purpose |
|---|---|
| `old_v1_reg` | historical CNN+Transformer regression-best condition, `lr=0.01`, `batch_size=24` |
| `old_v1_onb` | historical ONB/ROC-favorable condition, `lr=0.0001`, `batch_size=128` |
| `freq_d256` | same v1 frequency-band tokenization, but with larger Transformer capacity, `model_dim=256`, `ff_dim=2048`, `4` blocks |

## Interpretation Plan

1. If `old_v1_reg` recovers high R2/F1, the old high result was likely tied to
   the historical v1 frequency-band token behavior.
2. If `freq_d256` improves over `old_v1_reg`, the frequency-token direction is
   useful and the larger Transformer capacity matters.
3. If all v1-frequency conditions remain low while RF remains high, the main
   issue is likely not just axis direction. Training/calibration or model
   mismatch should be investigated next.

## Result

Run directory suffix: `20260702_v1freq_full`

All three parameter sets finished on the three restricted experiment dates.

| parameter_set | mean R2 | mean RMSE all | mean ONB RMSE | mean ROC-AUC cont | mean PR-AUC cont | mean Acc | mean F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `freq_d256` | -10.5998 | 731918.3 | 689182.7 | 0.8146 | 0.8891 | 0.5458 | 0.5461 |
| `old_v1_reg` | -11.5896 | 917463.7 | 920097.7 | 0.5022 | 0.6740 | 0.5357 | 0.6280 |
| `old_v1_onb` | -14.8742 | 929611.8 | 914244.4 | 0.9342 | 0.9606 | 0.5589 | 0.6518 |

Fold prediction calibration showed unstable absolute prediction scale:

- `old_v1_reg`: mean predicted positive rate was about `0.889` while the true
  positive rate was about `0.547`; mean prediction bias was about `+759473`.
- `old_v1_onb`: mean predicted positive rate was about `0.816`; mean bias was
  about `+798999`.
- `freq_d256`: mean predicted positive rate was closer at about `0.690`, but
  still had large fold-to-fold scale flips; mean bias was about `+106004`.

## Conclusion

The frequency-band-token hypothesis is useful for explaining what v1 was
actually doing, but this run did not recover the old high performance. The old
v1-like conditions do not restore 0.9-class accuracy on the current restricted
data. The next issue is likely training/calibration stability or model mismatch,
not just whether the Transformer sequence is time or frequency.
