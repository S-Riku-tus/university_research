"""Read-only audit of the 22 kHz, 151715 saved predictions.

Writes small derived summaries next to this script; never changes source runs.
"""

import csv
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / (
    "Pool_boiling/Subcooling_20_degrees/0.3/2025.06.18_0.3_3/"
    "regression_result/npy/ensemble/20260917/"
    "onb_xd-t0611-v0618_iw3-nm_s1e-9_e200_151715/maxfreq=22kHz"
)
OUT = Path(__file__).resolve().parent
SNR = ("no_noise", "0", "-4", "-8", "-12", "-16", "-20")
MODELS = ("randomforest", "conformer", "alexnet", "ensemble__inner_holdout")
ONB = 271677.6816


def source_dir(snr):
    return RUN / ("heatflux_no_noise" if snr == "no_noise" else f"heatflux_reference_SNR={snr}")


def read_predictions(snr):
    name = "no_noise" if snr == "no_noise" else snr
    path = source_dir(snr) / "fold_pred" / f"pred_f1_{name}.csv"
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    indexed = {(r["experiment_name"], r["source_wav_id"], int(r["chunk_index"])): r for r in rows}
    assert len(indexed) == len(rows) == 1080, (snr, len(rows), len(indexed))
    return indexed


def predict(row, model):
    if model == "equal_offline":
        return sum(float(row[m]) for m in MODELS[:3]) / 3
    return float(row[model])


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    data = {snr: read_predictions(snr) for snr in SNR}
    keys = set(data[SNR[0]])
    for snr, records in data.items():
        assert set(records) == keys, f"Unpaired evaluation samples: {snr}"
        for key in keys:
            assert math.isclose(float(records[key]["y_true"]), float(data[SNR[0]][key]["y_true"]), abs_tol=1e-6)
    truth = [float(row["y_true"]) for row in data[SNR[0]].values()]
    mean_truth = sum(truth) / len(truth)
    tss = sum((y - mean_truth) ** 2 for y in truth)
    metric_rows = []
    for snr in SNR:
        records = list(data[snr].values())
        for model in (*MODELS, "equal_offline"):
            sse = sum((predict(r, model) - float(r["y_true"])) ** 2 for r in records)
            metric_rows.append({
                "reference_snr": snr, "model": model, "n_chunks": len(records),
                "n_wavs": len({r["source_wav_id"] for r in records}),
                "r2": f"{1 - sse / tss:.9f}",
                "rmse_kw_m2": f"{math.sqrt(sse / len(records)) / 1000:.6f}",
                "sse_w_m2_squared": f"{sse:.6f}",
            })
    write_csv(OUT / "r2_from_saved_predictions.csv", list(metric_rows[0]), metric_rows)

    changes = []
    for before, after in (("-8", "-16"), ("-8", "-20"), ("-12", "-16")):
        for model in (*MODELS, "equal_offline"):
            wav_delta = defaultdict(float)
            region_delta = defaultdict(float)
            for key, left in data[before].items():
                right = data[after][key]
                y = float(left["y_true"])
                delta = (predict(left, model) - y) ** 2 - (predict(right, model) - y) ** 2
                wav_delta[key[1]] += delta
                region_delta["below_onb" if y < ONB else "onb_or_above"] += delta
            for wav, delta in sorted(wav_delta.items()):
                changes.append({
                    "before": before, "after": after, "model": model,
                    "source_wav_id": wav, "region": (
                        "below_onb" if float(next(r["y_true"] for r in data[before].values() if r["source_wav_id"] == wav)) < ONB
                        else "onb_or_above"
                    ),
                    "sse_reduction_w_m2_squared": f"{delta:.6f}",
                    "total_reduction_w_m2_squared": f"{sum(wav_delta.values()):.6f}",
                    "n_wavs_improved": sum(x > 0 for x in wav_delta.values()),
                    "n_wavs_worsened": sum(x < 0 for x in wav_delta.values()),
                    "below_onb_reduction_w_m2_squared": f"{region_delta['below_onb']:.6f}",
                    "onb_or_above_reduction_w_m2_squared": f"{region_delta['onb_or_above']:.6f}",
                })
    write_csv(OUT / "paired_wav_sse_changes.csv", list(changes[0]), changes)
    print(f"Verified 1080 paired chunks from 18 WAVs across {len(SNR)} reference-SNR conditions.")
    print("Wrote R2/RMSE and paired-WAV SSE summaries. Source run unchanged.")


if __name__ == "__main__":
    main()
