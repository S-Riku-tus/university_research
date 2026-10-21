"""Read saved tables/predictions only; write this audit directory only.

No model imports, fitting, inference, weight selection or raw-array loading.
The per-case upper bound is an oracle diagnostic, never an evaluation score.
"""
from pathlib import Path
import csv
import hashlib
import json
import os
import subprocess
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone, timedelta

import numpy as np

ROOT_NATIVE = Path(__file__).resolve().parents[2]
ROOT = Path("\\\\?\\" + str(ROOT_NATIVE)) if os.name == "nt" else ROOT_NATIVE
OUT = Path(__file__).resolve().parent
OLD = ROOT / "experiments/2026-10-15_clean_only_result_analysis"
SINGLES = ["randomforest", "conformer", "alexnet", "hgb", "extra_trees"]
FIVE = "ensemble__original3_hgb_extra_trees"
CORE = "ensemble__original3_mse"
THRESHOLDS = {"20250611": 221505.1102, "20250618": 271677.6816}


def read_json(p):
    return json.loads(p.read_text(encoding="utf-8"))


def rows(p):
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(name, data):
    with (OUT / name).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def identity(r):
    return (r["source_wav_id"], int(r["chunk_index"]))


def main():
    now = datetime.now(timezone(timedelta(hours=9))).isoformat()
    cond = read_json(OLD / "conditions.json")
    run = ROOT / cond["source_run"]
    ver = read_json(OLD / "verification.json")
    recorded_hashes = {p: sha(ROOT / p) == h for p, h in ver["input_hashes"].items()}
    assert all(recorded_hashes.values()), "A main-run source changed since 10/15"
    pooled = {(r["noise"], r["model_key"]): r for r in rows(OLD / "pooled_metrics.csv")}
    saved_q100 = {(r["noise"], r["model_key"], r["source_day"][:10].replace(".", "")): r
                  for r in rows(OLD / "q100_by_source_day.csv")}
    weights = rows(OLD / "weights.csv")
    internal = read_json(run / "heatflux_no_noise/internal_validation_fold1.json")
    training = internal["samples"]
    train_ids = {identity(r) for r in training}
    assert len(train_ids) == len(training) == 1620
    baseline_ids = None
    all_metrics, cases, onset_rows, band_rows, q100_rows = [], [], [], [], []
    recon_error, metric_error = 0., 0.
    main_sources = {}
    for noise in ["clean", "0", "-4", "-8", "-12", "-16", "-20"]:
        folder = "heatflux_no_noise" if noise == "clean" else f"heatflux_reference_SNR={noise}"
        suffix = "no_noise" if noise == "clean" else noise
        pred_path = run / folder / "fold_pred" / f"pred_f1_{suffix}.csv"
        pred = rows(pred_path)
        main_sources[str(pred_path.relative_to(ROOT)).replace("\\", "/")] = sha(pred_path)
        ids = [identity(r) for r in pred]
        assert len(ids) == len(set(ids)) == 540 and not train_ids.intersection(ids)
        if baseline_ids is None:
            baseline_ids = ids
        assert ids == baseline_ids
        counts = Counter(r["source_wav_id"] for r in pred)
        assert len(counts) == 36 and set(counts.values()) == {15}
        assert set(Counter(r["source_wav_id"] for r in training).values()) == {45}
        y = np.array([float(r["y_true"]) for r in pred])
        th = np.array([THRESHOLDS[r["source_wav_id"][:8]] for r in pred])
        positive = y >= th
        onset = np.isclose(y, th, rtol=0, atol=1e-5)
        near = np.abs(y - th) <= .1 * th
        assert np.array_equal(onset, near) and int(onset.sum()) == 30
        models = [k for k in pred[0] if k in SINGLES or k.startswith("ensemble__")]
        arrays = {k: np.array([float(r[k]) for r in pred]) for k in models}
        for strategy in sorted({r["strategy_name"] for r in weights}):
            w = {r["model_key"]: float(r["weight"]) for r in weights if r["strategy_name"] == strategy}
            rebuilt = sum(arrays[k] * v for k, v in w.items())
            recon_error = max(recon_error, float(np.max(np.abs(rebuilt - arrays["ensemble__" + strategy]))))
        for model, p in arrays.items():
            e = (p-y)/1000
            ph = p >= th
            m = dict(noise=noise, model_key=model, n=540,
                     rmse_all_kW_m2=float(np.sqrt(np.mean(e**2))),
                     mae_all_kW_m2=float(np.mean(np.abs(e))),
                     rmse_pre_onb_kW_m2=float(np.sqrt(np.mean(e[~positive]**2))),
                     rmse_onb_kW_m2=float(np.sqrt(np.mean(e[near]**2))),
                     rmse_high_kW_m2=float(np.sqrt(np.mean(e[positive]**2))),
                     rmse_high_excluding_near_kW_m2=float(np.sqrt(np.mean(e[positive & ~near]**2))),
                     tp=int(np.sum(ph & positive)), fn=int(np.sum(~ph & positive)),
                     fp=int(np.sum(ph & ~positive)), tn=int(np.sum(~ph & ~positive)),
                     onset_detected=int(np.sum(ph & onset)), onset_n=int(onset.sum()))
            for key in ["n", "rmse_all_kW_m2", "mae_all_kW_m2", "rmse_pre_onb_kW_m2", "rmse_onb_kW_m2", "rmse_high_kW_m2", "tp", "fn", "fp", "tn"]:
                metric_error = max(metric_error, abs(float(pooled[noise, model][key])-m[key]))
            all_metrics.append(m)
            for day in THRESHOLDS:
                dm = np.array([r["source_wav_id"].startswith(day) for r in pred])
                stages = sorted(set(y[dm]))
                q100 = next((q/1000 for q in stages if np.all(ph[dm & (y >= q)])), None)
                saved = saved_q100[noise, model, day]["q100_kW_m2"]
                assert (q100 is None and not saved) or (q100 is not None and abs(q100-float(saved)) < 1e-8)
                q100_rows.append(dict(noise=noise, model_key=model, day=day, q100_kW_m2=q100))
        if noise == "clean":
            for fraction in [.05, .1, .15, .2, .25, .3]:
                for day in THRESHOLDS:
                    mask = np.array([r["source_wav_id"].startswith(day) for r in pred]) & (np.abs(y-th) <= fraction*th)
                    band_rows.append(dict(day=day, fraction=fraction, n=int(mask.sum()),
                                          n_pre=int(np.sum(mask & ~positive)), n_post=int(np.sum(mask & positive)),
                                          n_wavs=len({r["source_wav_id"] for r, b in zip(pred, mask) if b}),
                                          heat_flux_stages_kW_m2=";".join(f"{v/1000:.7f}" for v in sorted(set(y[mask])))))
            for partition, records, target_key in [("train", training, "heat_flux"), ("evaluation", pred, "y_true")]:
                for r in records:
                    threshold = THRESHOLDS[r["source_wav_id"][:8]]
                    if abs(float(r[target_key])-threshold) < 1e-5:
                        onset_rows.append(dict(partition=partition, source_wav_id=r["source_wav_id"],
                                               chunk_index=int(r["chunk_index"]), q_kW_m2=float(r[target_key])/1000,
                                               sample_index=r.get("sample_index", "")))
        if noise in ["clean", "-20"]:
            # w_core >= .25, w_HGB >= .05, w_ET >= .05; sum = 1.
            # For ONE sample only, the remaining .65 can go to its largest block.
            blocks = np.column_stack([arrays[CORE], arrays["hgb"], arrays["extra_trees"]])
            upper = blocks @ np.array([.25, .05, .05]) + .65*blocks.max(axis=1)
            any_single = np.column_stack([arrays[k] for k in SINGLES]).max(axis=1) >= th
            failed = positive & (arrays[FIVE] < th)
            for i in np.where(failed)[0]:
                r = pred[i]
                cases.append(dict(noise=noise, sample_index=r["sample_index"], source_wav_id=r["source_wav_id"],
                                  chunk_index=r["chunk_index"], at_recorded_onb=bool(onset[i]),
                                  q_kW_m2=y[i]/1000, threshold_kW_m2=th[i]/1000,
                                  any_single_positive=bool(any_single[i]),
                                  positive_singles=";".join(k for k in SINGLES if arrays[k][i] >= th[i]),
                                  fixed_core_per_case_upper_kW_m2=float(upper[i]/1000),
                                  fixed_core_per_case_possible=bool(upper[i] >= th[i]),
                                  five_prediction_kW_m2=float(arrays[FIVE][i]/1000)))
    assert metric_error < 1e-8 and recon_error < 1e-6
    write_csv("recomputed_primary_metrics.csv", all_metrics)
    write_csv("onb_failure_feasibility.csv", cases)
    write_csv("onb_stage_partition.csv", onset_rows)
    write_csv("near_band_sensitivity_counts.csv", band_rows)
    write_csv("recomputed_q100.csv", q100_rows)
    failure_summary = []
    for noise in ["clean", "-20"]:
        for scope in ["all_FN", "recorded_onb_FN"]:
            subset = [r for r in cases if r["noise"] == noise and (scope == "all_FN" or r["at_recorded_onb"])]
            failure_summary.append(dict(noise=noise, scope=scope, n=len(subset),
                                        all_singles_negative=sum(not r["any_single_positive"] for r in subset),
                                        any_single_positive=sum(r["any_single_positive"] for r in subset),
                                        fixed_core_per_case_possible=sum(r["fixed_core_per_case_possible"] for r in subset)))
    # Inspect the actual saved recovery outputs, rather than assuming preflight is current.
    recovery_dir = ROOT / "experiments/2026-10-21_cnn_three_tuning_recovery"
    stopped = read_json(recovery_dir / "interrupted_summary.json")
    source_audits_unchanged = all(sha(ROOT / s["path"]) == s["sha256"] for s in stopped["source_files"])
    tuning_month = (ROOT / stopped["source_run"]).parent.parent
    by_signature = {}
    run_inventory = []
    for directory in sorted(tuning_month.glob("*/*")):
        audits = list((directory / "candidate_audits").glob("*.json"))
        counter = Counter()
        for p in sorted(audits):
            d = read_json(p)
            ps = d["parameter_set"]
            key = ps["model_key"]
            counter[key] += 1
            sig = (key, json.dumps(ps["models"][key], sort_keys=True))
            by_signature[sig] = dict(model_key=key, params=ps["models"][key],
                                     rmse_kW_m2=d["metrics"]["rmse_all"]/1000,
                                     path=str(p.relative_to(ROOT)).replace("\\", "/"), sha256=sha(p))
        run_inventory.append(dict(path=str(directory.relative_to(ROOT)).replace("\\", "/"),
                                  completed_audits=dict(counter),
                                  selected_candidates_exists=(directory / "selected_candidates.json").exists()))
    tuning_counts = Counter(r["model_key"] for r in by_signature.values())
    tuning_best = {k: min([r for r in by_signature.values() if r["model_key"] == k], key=lambda r: r["rmse_kW_m2"])
                   for k in tuning_counts}
    snapshot = dict(captured_at_JST=now,
                    branch=subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT_NATIVE, text=True).strip(),
                    head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT_NATIVE, text=True).strip(),
                    main_run=cond["source_run"], run_hash=cond["run_hash"], fit_id=cond["fit_id"],
                    additional_source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in [
                        run / "heatflux_no_noise/internal_validation_fold1.json",
                        run / "heatflux_no_noise/split_manifest.json",
                        OLD / "conditions.json", OLD / "weights.csv",
                        ROOT / "code/utils/ensemble/fixed_core_stacking.py",
                        ROOT / "code/utils/dataloading/acoustic_summary_features.py"]},
                    matched_results_read=False, model_fit_or_inference=False, raw_arrays_loaded=False,
                    raw_prediction_hashes=main_sources, legacy_main_source_hashes_unchanged=recorded_hashes,
                    recomputed_model_noise_rows=len(all_metrics), metric_max_difference=metric_error,
                    q100_checks=len(q100_rows),
                    reconstruction_max_difference_W_m2=recon_error,
                    train_n=1620, evaluation_n=540, source_wavs=36,
                    train_evaluation_same_chunk_overlap=0, shared_source_wavs=36,
                    recorded_onb_wavs=sorted({r["source_wav_id"] for r in onset_rows}),
                    onset_train_n=90, onset_evaluation_n=30,
                    failure_summary=failure_summary,
                    feasibility_scope="One sample at a time; oracle upper bound only; no common weights fitted, no FP constraint tested",
                    tuning_original_59_hashes_unchanged=source_audits_unchanged,
                    tuning_actual_unique_complete_counts=dict(tuning_counts), tuning_best=tuning_best,
                    tuning_runs=run_inventory)
    assert source_audits_unchanged
    # Keep compact, reproducible extracts of the primary plans and selected slides.
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
          "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    office = ROOT / "研究進捗報告"
    extracts = ["# 計画原本・発表原本の選択抽出", "", "原本を変更せずXMLの文章を抽出。図中の画素は未抽出。", ""]
    office_hashes = {}
    for pattern in ["*基本となる*.docx", "2026_AE*.docx"]:
        for path in office.glob(pattern):
            rel = path.relative_to(ROOT).as_posix()
            office_hashes[rel] = sha(path)
            extracts += [f"## {rel}", ""]
            with zipfile.ZipFile(path) as z:
                xml = ET.fromstring(z.read("word/document.xml"))
                extracts += ["".join(t.text or "" for t in p.findall(".//w:t", ns))
                             for p in xml.findall(".//w:p", ns)]
    for marker, indices in [("522", [6, 7]), ("619", [4, 5, 13, 14, 15]), ("724", [4, 5, 11]), ("1016", list(range(1, 15)))]:
        folder = next(p for p in (office / "2026").iterdir() if p.name.startswith(marker))
        paths = list(folder.glob("*.pptx"))
        for path in paths:
            with zipfile.ZipFile(path) as z:
                slides = sorted([n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)],
                                key=lambda n: int(re.search(r"(\d+)\.xml$", n).group(1)))
                if marker == "1016" and len(slides) != 14:
                    continue
                rel = path.relative_to(ROOT).as_posix()
                office_hashes[rel] = sha(path)
                extracts += [f"## {rel} ({len(slides)} slides)", ""]
                for i in indices:
                    if i <= len(slides):
                        xml = ET.fromstring(z.read(slides[i-1]))
                        extracts += [f"### slide {i}", "", " | ".join(t.text or "" for t in xml.findall(".//a:t", ns)), ""]
    (OUT / "primary_document_extracts.md").write_text("\n".join(extracts)+"\n", encoding="utf-8")
    snapshot["primary_office_source_sha256"] = office_hashes
    (OUT / "verification.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: snapshot[k] for k in ["captured_at_JST", "recomputed_model_noise_rows", "metric_max_difference",
                      "reconstruction_max_difference_W_m2", "failure_summary", "tuning_actual_unique_complete_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
