"""Independently check metrics, endpoint suffixes, splits and saved predictions."""
import json
from collections import defaultdict

import joblib
import numpy as np

from pilot import (KEYS, NOISE_DIRS, OUT, THRESHOLDS, candidates, identity,
                   load_features, read_csv, read_json, representation,
                   save_json, setup)


def close(actual, expected):
    np.testing.assert_allclose(float(actual), expected, rtol=1e-10, atol=1e-9, equal_nan=True)


def validate_metrics(metrics, predictions, y_lookup=None, days_lookup=None):
    count, endpoints = 0, 0
    for row in metrics:
        data = predictions[(row["method"], row["noise"], row["unit"])]
        y, p, days = data
        if row["day"] != "two_day":
            mask = days == row["day"]
            y, p, days = y[mask], p[mask], days[mask]
        t = np.asarray([THRESHOLDS[d] for d in days])
        actual, pred = y >= t, p >= t
        assert int(row["n"]) == len(y)
        assert int(row["fp"]) == int((~actual & pred).sum())
        assert int(row["fn"]) == int((actual & ~pred).sum())
        close(row["rmse"], np.linalg.norm(p-y)/np.sqrt(len(y)))
        close(row["mae"], np.abs(p-y).sum()/len(y))
        close(row["r2"], 1 - np.sum((p-y)**2)/np.sum((y-y.mean())**2))
        close(row["bias"], (p-y).mean())
        close(row["fpr"], int((~actual & pred).sum())/int((~actual).sum()))
        close(row["recall"], int((actual & pred).sum())/int(actual.sum()))
        near = np.abs(y-t) <= .1*np.abs(t)
        assert int(row["n_onb"]) == int(near.sum())
        close(row["rmse_onb"], np.linalg.norm(p[near]-y[near])/np.sqrt(near.sum()))
        if row["day"] != "two_day":
            stages = np.unique(y)
            failures = y[~pred]
            possible = stages[stages > failures.max()] if len(failures) else stages
            q = float(possible[0]) if len(possible) else float("nan")
            close(row["q100"], q)
            close(row["g100"], q-THRESHOLDS[row["day"]])
            if np.isfinite(q):
                assert pred[y >= q].all()
                earlier = stages[stages < q]
                if len(earlier):
                    assert not pred[y >= earlier[-1]].all()
            endpoints += 1
        count += 1
    return count, endpoints


def group_predictions(filename, unit):
    grouped = defaultdict(list)
    for row in read_csv(OUT / filename):
        grouped[(row["method"], row["noise"], unit)].append(row)
    result = {}
    for key, rows in grouped.items():
        ids = [identity(r) for r in rows]
        assert len(ids) == len(set(ids)) == 540
        result[key] = (np.asarray([float(r["y_kW_m2"]) for r in rows]),
                       np.asarray([float(r["prediction_kW_m2"]) for r in rows]),
                       np.asarray([r["source_wav_id"][:8] for r in rows]))
    return result, grouped


def main():
    rows, folds, data_root, outer, base, y, days = setup()
    held_ids = {identity(r) for r in outer}
    assert not {identity(r) for r in rows} & held_ids
    training = read_json(OUT / "training_predictions.json")
    assert [tuple(r) for r in training["ids"]] == [identity(r) for r in rows]
    training_summary = read_csv(OUT / "training_comparison.csv")
    train_pred = {(r["method"], "clean", r["unit"]): (y, np.asarray(training["methods"][r["method"]]), days) for r in training_summary}
    train_counts = validate_metrics(training_summary, train_pred)
    outer_pred, outer_rows = group_predictions("outer_predictions.csv", "outer_unused_chunk")
    outer_counts = validate_metrics(read_csv(OUT / "outer_metrics.csv"), outer_pred)
    frozen = read_json(OUT / "frozen_integration.json")
    nested_checks, final_checks, integration_checks = 0, 0, 0
    registry = {r["method"]: r for r in candidates()}
    # Reproduce full held-fold predictions after model reload, not just a probe.
    train_features = {n: load_features(rows, data_root, n, "train")[0] for n in ["clean", "-20"]}
    for rep in ["band10", "frequency34", "temporal46"]:
        for family in ["SVR", "HGB"]:
            audit = read_json(OUT / f"{rep}_{family}_audit.json")
            for item in audit["nested_fits"]:
                held = np.asarray(item["held_indices"], int)
                fit = np.asarray(item["fit_indices"], int)
                assert len(fit) == 1080 and len(held) == 540 and set(fit).isdisjoint(held)
                assert not {identity(rows[i]) for i in fit} & held_ids
                model = joblib.load(OUT / item["artifact"])
                name = f"{rep}_{family}_{item['policy']}"
                with np.load(OUT / f"{name}_oof.npz") as saved:
                    for noise, source in [("clean", "clean"), ("-20", "minus20")]:
                        expected = saved[source][held]
                        p = model.predict(representation(train_features[noise], rep)[held])
                        np.testing.assert_allclose(p, expected, atol=1e-10, rtol=1e-12)
                        nested_checks += 1
    for noise in NOISE_DIRS:
        X = load_features(outer, data_root, noise, "outer")[0]
        pool = {}
        for name, item in registry.items():
            model = joblib.load(OUT / item["final_artifact"])
            p = model.predict(representation(X, item["representation"]))
            expected_rows = outer_rows[(name, noise, "outer_unused_chunk")]
            assert [identity(r) for r in expected_rows] == [identity(r) for r in outer]
            expected = outer_pred[(name, noise, "outer_unused_chunk")][1]
            np.testing.assert_allclose(p, expected, rtol=1e-12, atol=1e-10)
            pool[name] = p
            final_checks += 1
        base_matrix = np.column_stack([outer_pred[(key, noise, "outer_unused_chunk")][1] for key in KEYS])
        for method in frozen["methods"]:
            matrix = base_matrix if method["candidate"] is None else np.column_stack([base_matrix, pool[method["candidate"]]])
            p = matrix @ method["weights"] if method["kind"] == "linear" else joblib.load(OUT / method["artifact"]).predict(matrix)
            np.testing.assert_allclose(p, outer_pred[(method["method"], noise, "outer_unused_chunk")][1], rtol=1e-12, atol=1e-10)
            integration_checks += 1
    ablation_pred, _ = group_predictions("fixed_input_ablation_predictions.csv", "outer_unused_chunk")
    with np.load(OUT / "fixed_input_ablation_oof.npz") as saved:
        for rep in saved:
            ablation_pred[(f"fixed15_{rep}", "clean", "training_fixed_oof")] = y, saved[rep], days
    ablation_counts = validate_metrics(read_csv(OUT / "fixed_input_ablation_metrics.csv"), ablation_pred)
    counts = {"metric_rows": train_counts[0]+outer_counts[0]+ablation_counts[0], "independent_endpoint_checks": train_counts[1]+outer_counts[1]+ablation_counts[1], "nested_reloaded_prediction_checks": nested_checks, "final_reloaded_prediction_checks": final_checks, "frozen_integration_prediction_checks": integration_checks, "outer_train_chunk_overlap": 0, "status": "passed"}
    save_json("verification.json", counts)
    print(json.dumps(counts))


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        main()
