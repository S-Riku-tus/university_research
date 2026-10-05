"""Audit an alternative known-WAV time-block validation; do not train or adopt it."""

import numpy as np

from analyze import OUTPUT, THRESHOLDS, read_csv, save_csv, training_rows


def block_split(rows, fold, gap_chunks=1):
    held = np.asarray([i for i, r in enumerate(rows) if int(r["chunk_index"])//20+1 == fold])
    fit_candidates = [i for i, r in enumerate(rows) if int(r["chunk_index"])//20+1 != fold]
    held_chunks = {}
    for i in held:
        held_chunks.setdefault(rows[i]["source_wav_id"], []).append(int(rows[i]["chunk_index"]))
    fit = [i for i in fit_candidates if all(abs(int(rows[i]["chunk_index"])-h) > gap_chunks
           for h in held_chunks[rows[i]["source_wav_id"]])]
    return np.asarray(fit), held, len(fit_candidates)-len(fit)


def main():
    rows = training_rows()
    assert all(0 <= int(r["chunk_index"]) < 60 for r in rows)
    audits, indices = [], []
    for gap in [0, 1, 2]:
        coverage = np.zeros(len(rows), int)
        for fold in [1, 2, 3]:
            fit, held, purged = block_split(rows, fold, gap)
            coverage[held] += 1
            assert set(fit).isdisjoint(held)
            fit_groups = {rows[i]["source_wav_id"] for i in fit}
            held_groups = {rows[i]["source_wav_id"] for i in held}
            assert len(fit_groups) == len(held_groups) == 36
            assert fit_groups == held_groups
            held_lookup = {g: [int(rows[i]["chunk_index"]) for i in held if rows[i]["source_wav_id"] == g]
                           for g in held_groups}
            assert all(all(abs(int(rows[i]["chunk_index"])-h) > gap
                           for h in held_lookup[rows[i]["source_wav_id"]]) for i in fit)
            for day in THRESHOLDS:
                own_fit = [i for i in fit if rows[i]["source_wav_id"].startswith(day)]
                own_held = [i for i in held if rows[i]["source_wav_id"].startswith(day)]
                audits.append({"gap_chunks": gap, "fold": fold, "day": day,
                    "fit_chunks": len(fit), "held_chunks": len(held), "purged_fit_chunks": purged,
                    "fit_chunks_this_day": len(own_fit), "held_chunks_this_day": len(own_held),
                    "fit_stage_wavs_this_day": len({rows[i]["source_wav_id"] for i in own_fit}),
                    "fit_near_onb_wavs_this_day": len({rows[i]["source_wav_id"] for i in own_fit
                        if .9 <= float(rows[i]["y_kW_m2"])/float(rows[i]["onb_kW_m2"]) <= 1.1}),
                    "shared_source_wavs": 36, "same_chunk_shared": False,
                    "evaluation_scope": "Known-source-WAV separated time blocks; not unknown-WAV validation"})
            for role, subset in [("fit", fit), ("held", held)]:
                indices.extend({"gap_chunks": gap, "fold": fold, "role": role,
                    "source_wav_id": rows[i]["source_wav_id"], "chunk_index": rows[i]["chunk_index"]} for i in subset)
        assert np.all(coverage == 1)
    save_csv("known_wav_time_block_audit.csv", audits)
    save_csv("known_wav_time_block_indices.csv", indices)
    print("3 contiguous time folds, gaps 0/1/2: every fit retains all 18 stages per day; no shared chunk")
    print("Prototype only; same source WAV intentionally shared; temporal independence not proven")


if __name__ == "__main__":
    main()
