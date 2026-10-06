"""Describe fixed five-model errors; no fitting, prediction, or selection.

Inputs are the immutable saved arrays and clean-OOF-fitted weights from the
fixed-three additions study. Units are kW/m^2 and (kW/m^2)^2. Training OOF
rows diagnose the ensemble's fitting data, not independent evaluation.
"""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments/2026-10-06_fixed_three_additions"
PAIRED = ROOT / "experiments/2026-10-06_hgb_followup_validation"
OUT = Path(__file__).resolve().parent
THRESHOLDS = {"20250611": 221.5051102, "20250618": 271.6776816}
MEMBERS = ["randomforest", "conformer", "alexnet", "hgb", "extra_trees"]
MAIN = "existing3_clean_mse__selected5"
CORE = "existing3_clean_mse"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_csv(name, rows):
    with (OUT / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    missed, losses, error_terms, examples, endpoints, inputs = [], [], [], [], [], []
    maximum_reconstruction_error = 0.0
    metric_path = SOURCE / "metrics.csv"
    with metric_path.open(encoding="utf-8-sig") as stream:
        recorded_q100 = {(int(r["seed"]), r["noise"], r["day"]): float(r["q100"])
                         for r in csv.DictReader(stream) if r["method"] == MAIN and r["day"] in THRESHOLDS}
    inputs.append({"path": str(metric_path.relative_to(ROOT)),
                   "sha256": hashlib.sha256(metric_path.read_bytes()).hexdigest()})
    for seed in [43, 44]:
        folder = SOURCE / f"seed{seed}"
        paths = [folder / "predictions.npz", folder / "outer_integrated.npz",
                 folder / "frozen_integration.json", PAIRED / f"seed{seed}/manifest.json"]
        inputs.extend({"path": str(path.relative_to(ROOT)),
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                      for path in paths)
        manifest = read_json(paths[3])
        frozen = {r["method"]: np.asarray(r["weights"], dtype=float)
                  for r in read_json(paths[2])["methods"]}
        with np.load(paths[0]) as saved:
            keys, noises = saved["keys"].tolist(), saved["noises"].tolist()
            selected = [keys.index(k) for k in MEMBERS]
            arrays = {"outer": saved["outer"].copy(), "training_oof_fit_diagnostic": saved["oof"].copy()}
            indices = {"outer": saved["test_indices"].copy(),
                       "training_oof_fit_diagnostic": saved["train_indices"].copy()}
        with np.load(paths[1]) as saved:
            prior = saved["predictions"][:, :, saved["methods"].tolist().index(MAIN)].copy()
            np.testing.assert_array_equal(saved["indices"], indices["outer"])
        weights = frozen[MAIN][selected]
        core_weights = frozen[CORE][selected[:3]]
        np.testing.assert_allclose(weights.sum(), 1, atol=1e-12)
        assert np.all(weights > 0) and np.isclose(core_weights.sum(), 1)
        for scope, array in arrays.items():
            metadata = [manifest["samples"][int(i)] for i in indices[scope]]
            y = np.asarray([float(r["heat_flux"])/1000 for r in metadata])
            days = np.asarray([r["source_wav_id"][:8] for r in metadata])
            threshold = np.asarray([THRESHOLDS[d] for d in days])
            actual = y >= threshold
            for n, noise in enumerate(noises):
                matrix = array[n][:, selected]
                prediction = matrix @ weights
                if scope == "outer":
                    maximum_reconstruction_error = max(maximum_reconstruction_error,
                        float(np.max(np.abs(prediction-prior[n]))))
                    np.testing.assert_allclose(prediction, prior[n], rtol=1e-12, atol=1e-10)
                negative = matrix < threshold[:, None]
                fn = actual & (prediction < threshold)
                common = actual & np.all(negative, axis=1)
                # Pointwise maximum under the locked original-three proportions
                # and core >= .25, HGB >= .05, ExtraTrees >= .05. Each row can
                # use different weights: this is an oracle diagnostic, not a
                # feasible common weight vector or an evaluation candidate.
                blocks = np.column_stack([matrix[:, :3] @ core_weights, matrix[:, 3:]])
                upper = blocks @ np.asarray([.25, .05, .05]) + .65*blocks.max(axis=1)
                cannot_under_protocol = fn & ~common & (upper < threshold)
                locked_core_unreachable = cannot_under_protocol & (blocks.max(axis=1) < threshold)
                lower_bounds_unreachable = cannot_under_protocol & ~locked_core_unreachable
                pointwise_possible = fn & ~common & ~cannot_under_protocol
                assert np.all(common <= fn)
                assert int(fn.sum()) == int(common.sum()+cannot_under_protocol.sum()+pointwise_possible.sum())
                for day in ["two_day", *THRESHOLDS]:
                    mask = np.ones(len(y), dtype=bool) if day == "two_day" else days == day
                    missed.append({"seed": seed, "scope": scope, "noise": noise, "day": day,
                        "chunks": int(mask.sum()), "positive_chunks": int((actual & mask).sum()),
                        "fn": int((fn & mask).sum()),
                        "fp": int((~actual & (prediction >= threshold) & mask).sum()),
                        "all_five_common_fn": int((common & mask).sum()),
                        "some_member_positive_but_protocol_upper_below_threshold": int((cannot_under_protocol & mask).sum()),
                        "locked_core_all_three_blocks_below_threshold": int((locked_core_unreachable & mask).sum()),
                        "positive_block_but_lower_bounds_prevent_recovery": int((lower_bounds_unreachable & mask).sum()),
                        "pointwise_protocol_possible_fn": int((pointwise_possible & mask).sum())})
                if scope != "outer":
                    continue
                for day in THRESHOLDS:
                    mask = days == day
                    day_fn = fn & mask
                    last_flux = float(y[day_fn].max()) if np.any(day_fn) else np.nan
                    last = day_fn & (y == last_flux)
                    reached = [v for v in np.unique(y[mask]) if np.all(prediction[mask & (y >= v)] >= THRESHOLDS[day])]
                    endpoint = float(min(reached)) if reached else np.nan
                    np.testing.assert_allclose(endpoint, recorded_q100[seed, noise, day], atol=1e-8, equal_nan=True)
                    endpoints.append({"seed": seed, "noise": noise, "day": day,
                        "q100_kW_m2": endpoint,
                        "last_false_negative_flux_kW_m2": last_flux,
                        "fn_at_last_flux": int(last.sum()),
                        "all_five_common_at_last_flux": int((last & common).sum()),
                        "locked_core_unreachable_at_last_flux": int((last & locked_core_unreachable).sum()),
                        "lower_bounds_unreachable_at_last_flux": int((last & lower_bounds_unreachable).sum()),
                        "pointwise_possible_at_last_flux": int((last & pointwise_possible).sum())})
                for i in np.flatnonzero(fn):
                    examples.append({"seed": seed, "noise": noise,
                        "source_wav_id": metadata[i]["source_wav_id"],
                        "chunk_index": metadata[i]["chunk_index"], "target_kW_m2": y[i],
                        "threshold_kW_m2": threshold[i], "five_prediction_kW_m2": prediction[i],
                        "protocol_pointwise_upper_kW_m2": upper[i],
                        "failure_type": "all_five_common" if common[i] else
                            "protocol_unreachable" if cannot_under_protocol[i] else "pointwise_possible",
                        "positive_members": "|".join(k for j, k in enumerate(MEMBERS) if not negative[i, j])})
                regions = {"all": np.ones(len(y), dtype=bool), "pre_onb": ~actual,
                           "near_onb_10pct": np.abs(y-threshold) <= .1*threshold,
                           "post_onb": actual}
                for region, mask in regions.items():
                    errors = matrix[mask] - y[mask, None]
                    products = errors.T @ errors / mask.sum()
                    contributions = weights[:, None]*weights[None, :]*products
                    diagonal = float(np.trace(contributions))
                    cross = float(contributions.sum()-diagonal)
                    mse = float(np.mean((prediction[mask]-y[mask])**2))
                    np.testing.assert_allclose(diagonal+cross, mse, rtol=1e-12, atol=1e-8)
                    et_error = errors[:, -1]
                    adjustment = prediction[mask]-matrix[mask, -1]
                    alignment = float(2*np.mean(et_error*adjustment))
                    adjustment_cost = float(np.mean(adjustment**2))
                    et_mse = float(np.mean(et_error**2))
                    np.testing.assert_allclose(alignment+adjustment_cost, mse-et_mse, rtol=1e-10, atol=1e-8)
                    weighted_single = float(weights @ np.diag(products))
                    diversity = float(np.mean(np.sum(weights*(matrix[mask]-prediction[mask, None])**2, axis=1)))
                    np.testing.assert_allclose(weighted_single-diversity, mse, rtol=1e-12, atol=1e-8)
                    losses.append({"seed": seed, "noise": noise, "region": region, "chunks": int(mask.sum()),
                        "five_rmse_kW_m2": float(np.sqrt(mse)), "extra_trees_rmse_kW_m2": float(np.sqrt(et_mse)),
                        "five_minus_extra_trees_mse": mse-et_mse,
                        "et_error_adjustment_alignment": alignment, "adjustment_squared_cost": adjustment_cost,
                        "weighted_diagonal_error": diagonal, "weighted_cross_error": cross,
                        "weighted_single_mse": weighted_single, "weighted_prediction_dispersion": diversity,
                        "five_bias_kW_m2": float(np.mean(prediction[mask]-y[mask])),
                        "extra_trees_bias_kW_m2": float(np.mean(et_error))})
                    for a in range(len(MEMBERS)):
                        for b in range(a, len(MEMBERS)):
                            factor = 1 if a == b else 2
                            error_terms.append({"seed": seed, "noise": noise, "region": region,
                                "member_a": MEMBERS[a], "member_b": MEMBERS[b],
                                "error_product": float(products[a, b]),
                                "weighted_mse_contribution": float(factor*contributions[a, b])})
    OUT.mkdir(parents=True, exist_ok=True)
    save_csv("remaining_false_negatives.csv", missed)
    save_csv("loss_decomposition.csv", losses)
    save_csv("weighted_error_terms.csv", error_terms)
    save_csv("outer_false_negative_examples.csv", examples)
    save_csv("q100_last_false_negative.csv", endpoints)
    verification = {"status": "passed", "new_model_fits": 0, "new_model_predictions": 0,
        "weights_changed": False, "input_files": inputs,
        "max_reconstructed_five_prediction_difference_kW_m2": maximum_reconstruction_error,
        "verified": ["saved outer prediction reconstruction", "disjoint exhaustive FN partition", "q100 against source metrics",
            "weighted error-product MSE identity", "ExtraTrees-relative MSE identity", "weighted dispersion identity"],
        "row_counts": {"remaining_false_negatives": len(missed), "loss_decomposition": len(losses),
            "weighted_error_terms": len(error_terms), "outer_false_negative_examples": len(examples),
            "q100_last_false_negative": len(endpoints)},
        "scope": "post hoc descriptive analysis of reused development data; no selection; pointwise bounds are oracle diagnostics"}
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({k: verification[k] for k in ["status", "new_model_fits", "max_reconstructed_five_prediction_difference_kW_m2", "row_counts"]}))


if __name__ == "__main__":
    main()
