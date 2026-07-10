# 2026-07-09 noise and max-frequency effect analysis

## Source

Analyzed completed full-grid result:

`Pool_boiling/Subcooling_20_degrees/0.3/<experiment>/regression_result/npy/ensemble/20260706_fixed_ensemble_full_grid/tuning_summary.csv`

Models analyzed:

- `rf`
- `cnntf_v2_gap`
- `alexnet`

Rows analyzed:

- `126` conditions per model
- `378` model-condition rows total

## Overall

| model | mean R2 | min R2 | max R2 | mean ONB RMSE | mean F1 |
|---|---:|---:|---:|---:|---:|
| rf | 0.9706 | 0.9131 | 0.9961 | 42917.0 | 0.9775 |
| cnntf_v2_gap | 0.8420 | 0.3900 | 0.9206 | 91268.4 | 0.8523 |
| alexnet | 0.6934 | -5.2494 | 0.9437 | 120903.8 | 0.8525 |

## Noise effect on R2

| noise | alexnet | cnntf_v2_gap | rf |
|---|---:|---:|---:|
| no_noise | 0.7001 | 0.8286 | 0.9548 |
| SNR=0 | 0.3080 | 0.7893 | 0.9741 |
| SNR=-4 | 0.4528 | 0.8389 | 0.9739 |
| SNR=-8 | 0.8562 | 0.8369 | 0.9739 |
| SNR=-12 | 0.7438 | 0.8605 | 0.9735 |
| SNR=-16 | 0.8800 | 0.8693 | 0.9723 |
| SNR=-20 | 0.9128 | 0.8705 | 0.9715 |

R2 range by noise:

- AlexNet: `0.6048`
- cnntf_v2_gap: `0.0812`
- RF: `0.0193`

## Max-frequency effect on R2

| max frequency | alexnet | cnntf_v2_gap | rf |
|---|---:|---:|---:|
| 2 kHz | 0.4204 | 0.8259 | 0.9744 |
| 3 kHz | 0.8145 | 0.8569 | 0.9784 |
| 5 kHz | 0.3787 | 0.8527 | 0.9776 |
| 10 kHz | 0.7774 | 0.8313 | 0.9677 |
| 15 kHz | 0.8868 | 0.8382 | 0.9620 |
| 22 kHz | 0.8826 | 0.8470 | 0.9633 |

R2 range by max frequency:

- AlexNet: `0.5081`
- cnntf_v2_gap: `0.0310`
- RF: `0.0164`

## Interpretation

RF is highly stable across both noise and max-frequency changes. Its worst mean
R2 condition is still above `0.95` when aggregated by noise/frequency group, and
the condition-level minimum is `0.9131`.

`cnntf_v2_gap` is moderately affected. Noise changes matter more than
max-frequency changes, but the model does not collapse in the same way as
AlexNet.

AlexNet is highly sensitive to both noise and max-frequency settings. The
biggest failures are concentrated in `SNR=0`, `SNR=-4`, `2kHz`, and `5kHz`.
However, AlexNet can work well under `SNR=-16/-20` and `15/22kHz`, so the issue
is not simply that AlexNet cannot learn this task.

## Next

Before the final fixed ensemble run, AlexNet should be tuned on stress
conditions containing:

- `2kHz`, `5kHz`, `15kHz`, `22kHz`
- `no_noise`, `SNR=0`, `SNR=-4`, `SNR=-20`

The current AlexNet stress-tune configuration in `run_ensemble_regression_onb.py`
is consistent with this result.
