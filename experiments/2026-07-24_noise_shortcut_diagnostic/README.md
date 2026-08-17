# Noise / accuracy reversal diagnostic

## Outcome

The current inverse noise/accuracy result is dominated by a label-dependent amplitude shortcut.

With source-relative SNR scaling, the injected water-flow-only power is proportional to the source-WAV power. A raw linear-power input therefore exposes heat-flux-correlated source amplitude even after the boiling waveform is removed.

Key P0 result, ordinary KFold and raw-spectrogram PCA:

| Experiment | Clean | Relative −20 | Relative noise-only −20 | Fixed −20 |
|---|---:|---:|---:|---:|
| 2025.06.18 | 0.8699 | 0.9068 | 0.9085 | 0.7860 |
| 2025.07.09 | 0.4692 | 0.6311 | 0.6098 | 0.4219 |
| 2025.06.11 | 0.8717 | 0.8824 | 0.8837 | 0.8627 |

The noise-only result nearly matches the mixture result. When the same arrays are converted to the bachelor's magnitude + per-sample min-max representation, relative −20 dB R² becomes negative in all three experiments.

The final controlled run uses one global reference RMS, exactly equal noise power for every label, and the same five water-flow chunks for every source and SNR. In clean-trained GroupKFold transfer, the three-experiment mean R² changes as follows:

| Feature | Clean | 0 | −4 | −8 | −12 | −16 | −20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Total power only | 0.668 | 0.631 | 0.570 | 0.434 | 0.360 | 0.013 | −0.087 |
| Raw power PCA | 0.595 | 0.540 | 0.543 | 0.531 | 0.579 | 0.587 | 0.574 |
| Old-style magnitude min-max PCA | 0.600 | 0.081 | 0.107 | −0.008 | −0.140 | −0.212 | −0.261 |

See the full Japanese report: [2026-08-17_noise_accuracy_reversal_investigation.md](../../docs/research_plan/2026-08-17_noise_accuracy_reversal_investigation.md).

## Reproduction

```powershell
python -X utf8 "code/2.run_npy_waterflow_2つhighpass.py"
python -X utf8 code/run_noise_shortcut_diagnostics.py
python -X utf8 code/run_existing_noise_curve_diagnostics.py
python -X utf8 code/run_controlled_noise_curve_diagnostics.py
python -X utf8 tests/test_noise_shortcut_diagnostics.py
```

`run_controlled_noise_curve_diagnostics.py` generates its spectrogram subset in memory and writes only metrics and a manifest.

## Output guide

- `diagnostic_manifest.json`: P0 configuration
- `metrics_summary.csv`: P0 per-experiment summary
- `input_power_summary.csv`: input-power/heat-flux correlation
- `realized_snr_summary.csv`: realized per-chunk SNR
- `leave_one_experiment_out_scalar.csv`: cross-day scalar shortcut test
- `existing_curve_metrics_summary.csv`: seven-SNR curve from existing 20260722 arrays
- `controlled_noise_manifest.json`: label-independent final control
- `controlled_retrained_metrics_summary.csv`: separately retrained control
- `controlled_clean_transfer_summary.csv`: clean-trained robustness control
- `fold_predictions.csv` and `split_assignments.csv`: sample-level audit trail

