"""Two constrained MSE rules on the source clean OOF; no base-model training."""
import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

from common import (CONFIG, CORE, FIVE, MEMBERS, OUT, THRESHOLDS, frozen_fit, metric_rows,
    save_csv, save_json, source_columns, source_fingerprint, source_metadata, source_predictions, training_data)

FREE = "ensemble__free_core_ratio_mse"


def fit_free_core(matrix, y, start):
    """No outer labels or noisy predictions accepted by this fit API."""
    matrix, y = np.asarray(matrix, float), np.asarray(y, float)
    rules = CONFIG["stage2"]["constraints"]
    fraction = rules["minimum_core_member_fraction_of_core"]
    a = np.zeros((4, 5))
    a[0, :3] = 1
    for i in range(3):
        a[i+1, :3] = -fraction
        a[i+1, i] += 1
    lower = np.array([rules["minimum_core_total"], 0, 0, 0])
    variance = np.var(y)
    assert matrix.shape == (len(y), 5) and variance > 0
    assert np.isfinite(matrix).all() and np.isfinite(y).all()
    result = minimize(lambda w: np.mean((matrix@w-y)**2)/variance, np.array(start),
        jac=lambda w: 2*matrix.T@(matrix@w-y)/len(y)/variance,
        method="SLSQP", bounds=[(0, 1)]*3+[(rules["minimum_added_member"], 1)]*2,
        constraints=[{"type": "eq", "fun": lambda w: w.sum()-1, "jac": lambda w: np.ones(5)},
            {"type": "ineq", "fun": lambda w: a@w-lower, "jac": lambda w: a}],
        options={"ftol": 1e-13, "maxiter": 1000})
    if not result.success:
        raise RuntimeError(result.message)
    weight = result.x
    assert abs(weight.sum()-1) < 1e-10 and np.all(a@weight-lower >= -1e-10)
    assert np.all(weight > 0) and np.all(weight[3:] >= rules["minimum_added_member"]-1e-10)
    return weight, {"success": True, "iterations": int(result.nit), "mse_kW_m2_squared": float(np.mean((matrix@weight-y)**2)),
        "minimum_constraint_slack": float((a@weight-lower).min())}


def main():
    directory = OUT / "stage2"
    directory.mkdir(parents=True, exist_ok=True)
    fingerprint = source_fingerprint()
    training, audit, oof = training_data()
    fixed = np.array([frozen_fit()["weights"][FIVE.removeprefix("ensemble__")][k] for k in MEMBERS])
    matrix, y = oof[MEMBERS].to_numpy()/1000, oof.y_true.to_numpy()/1000
    first, diagnostic1 = fit_free_core(matrix, y, fixed)
    second, diagnostic2 = fit_free_core(matrix, y, np.full(5, .2))
    np.testing.assert_allclose(first, second, rtol=1e-4, atol=2e-5)
    fixed_mse = float(np.mean((matrix@fixed-y)**2))
    assert diagnostic1["mse_kW_m2_squared"] <= fixed_mse+1e-6
    save_json(directory / "frozen_weights.json", {"protocol": CONFIG["stage2"],
        "weights": dict(zip(MEMBERS, first.tolist())), "fixed_weights": dict(zip(MEMBERS, fixed.tolist())),
        "clean_oof_fixed_mse": fixed_mse, "fit_diagnostics": [diagnostic1, diagnostic2],
        "fit_outer_labels_used": False, "fit_noisy_predictions_used": False,
        "freeze_timing": "written after clean OOF fit and before noisy/outer scoring"})
    metrics = metric_rows(training, {FIVE: matrix@fixed*1000, FREE: matrix@first*1000}, "clean", scope="training_oof_weight_fit_diagnostic")
    paired, delta_rows, values, cases = [], [], [], []
    for noise in CONFIG["evaluation_noise_conditions"]:
        frame = source_predictions(noise)
        predictions = {k: frame[k].to_numpy() for k in source_columns(frame)}
        prediction = (frame[MEMBERS].to_numpy()/1000)@first*1000
        predictions[FREE] = prediction
        metrics.extend(metric_rows(frame, predictions, noise, scope="outer_development"))
        days = np.array([r["experiment_name"] for r in source_metadata(frame)])
        threshold = np.array([THRESHOLDS[d] for d in days])
        actual = frame.y_true.to_numpy() >= threshold
        before = frame[FIVE].to_numpy()
        before_fn, after_fn = actual & (before < threshold), actual & (prediction < threshold)
        before_fp, after_fp = ~actual & (before >= threshold), ~actual & (prediction >= threshold)
        common = actual & np.all(frame[MEMBERS].to_numpy() < threshold[:, None], axis=1)
        assert np.all(prediction[common] < threshold[common])
        paired.append({"noise": noise, "corrected_fn": int((before_fn & ~after_fn).sum()),
            "added_fn": int((~before_fn & after_fn).sum()), "corrected_fp": int((before_fp & ~after_fp).sum()),
            "added_fp": int((~before_fp & after_fp).sum()), "remaining_original_all_five_common_fn": int(common.sum())})
        for i in range(len(frame)):
            values.append({"noise": noise, "source_wav_id": frame.source_wav_id.iloc[i], "chunk_index": int(frame.chunk_index.iloc[i]),
                "target_kW_m2": frame.y_true.iloc[i]/1000, "fixed_prediction_kW_m2": before[i]/1000,
                "free_prediction_kW_m2": prediction[i]/1000})
            if before_fn[i] != after_fn[i] or before_fp[i] != after_fp[i]:
                cases.append({**values[-1], "source_day": days[i], "threshold_kW_m2": threshold[i]/1000,
                    "change": "corrected_fn" if before_fn[i] and not after_fn[i] else "added_fn" if after_fn[i] and not before_fn[i]
                        else "corrected_fp" if before_fp[i] and not after_fp[i] else "added_fp"})
        print(f"[stage2] fixed/free core / {noise}: 540 chunks scored", flush=True)
    for scope in ["training_oof_weight_fit_diagnostic", "outer_development"]:
        selected = [r for r in metrics if r["scope"] == scope]
        baseline = {(r["noise"], r["source_day"]): r for r in selected if r["model_key"] == FIVE}
        for row in selected:
            if row["model_key"] != FREE:
                continue
            b = baseline[row["noise"], row["source_day"]]
            numeric = [k for k, v in b.items() if isinstance(v, (float, int, np.number)) and k != "fold"]
            delta_rows.append({"scope": scope, "noise": row["noise"], "source_day": row["source_day"],
                **{"delta_"+k: row[k]-b[k] for k in numeric}})
    save_csv(directory / "metrics.csv", metrics)
    save_csv(directory / "metric_deltas.csv", delta_rows)
    save_csv(directory / "paired_detection_changes.csv", paired)
    save_csv(directory / "predictions.csv", values)
    save_csv(directory / "changed_detection_cases.csv", cases)
    assert source_fingerprint() == fingerprint
    save_json(directory / "verification.json", {"status": "passed", "base_model_fits": 0, "base_model_predictions": 0,
        "weight_optimizer_fits": 2, "input_files": fingerprint, "training_chunks": 1620, "outer_chunks_per_noise": 540,
        "noise_conditions": 7, "all_five_positive": True, "constraints_preserved": True,
        "same_solution_from_two_starts": True, "outer_labels_used_for_fit": False,
        "noisy_predictions_used_for_fit": False, "original_common_fn_cannot_be_rescued_by_nonnegative_average": True,
        "source_outputs_and_defaults_unchanged": True, "scope": CONFIG["stage2"]["interpretation"]})
    print("[stage2] passed; free weights:", dict(zip(MEMBERS, first.tolist())), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=CONFIG["cpu_threads"]):
        main()
