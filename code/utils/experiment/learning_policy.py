"""学習日・学習ノイズの選択と、元WAVを共有しない評価分割を組み立てる。"""

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from utils.experiment.dataset_jobs import has_input_files


DEFAULT_POLICY = {"split_mode": "within_day", "training_noise": "matched"}
CLEAN_NOISE = "heatflux_no_noise"


def normalize_learning_policy(policy, experiment_names, color_channel=1):
    resolved = {**DEFAULT_POLICY, **(policy or {})}
    if set(resolved) != set(DEFAULT_POLICY):
        raise ValueError(f"未知のlearning_policy設定です: {set(resolved) - set(DEFAULT_POLICY)}")
    if resolved["split_mode"] not in {"within_day", "leave_one_day_out"}:
        raise ValueError("split_modeはwithin_dayまたはleave_one_day_outです。")
    if resolved["training_noise"] not in {"matched", "clean_only"}:
        raise ValueError("training_noiseはmatchedまたはclean_onlyです。")
    if len(set(experiment_names)) != len(experiment_names):
        raise ValueError("experiment_namesに同じ実験日を重複指定できません。")
    if resolved["split_mode"] == "leave_one_day_out" and len(experiment_names) < 2:
        raise ValueError("別日評価にはdata.experiment_namesで2日以上を有効にしてください。")
    if color_channel != 1:
        raise ValueError("元WAVを分離する評価にはchunk_manifest.csv付きのNPY入力が必要です。")
    return resolved


def policy_result_date_dir(result_date_dir, policy):
    """従来方式の保存先を保ち、一般化評価の結果を別の実行フォルダへ分ける。"""
    if policy == DEFAULT_POLICY:
        return result_date_dir
    day = "wd" if policy["split_mode"] == "within_day" else "lodo"
    noise = "clean" if policy["training_noise"] == "clean_only" else "matched"
    return f"{result_date_dir}__{day}_{noise}"


def build_learning_families(jobs, policy, experiment_names):
    """clean_onlyでは複数の評価ノイズを一つの学習処理にまとめる。"""
    lookup = {(j["experiment_name"], j["max_freq_hz"]): j for j in jobs}
    grouped = {}
    for job in jobs:
        noise = CLEAN_NOISE if policy["training_noise"] == "clean_only" else job["noise_dir_name"]
        key = (job["experiment_name"], job["max_freq_hz"], noise)
        grouped.setdefault(key, []).append(job)
    families = []
    for (test_day, frequency, train_noise), evaluation_jobs in grouped.items():
        train_days = ([test_day] if policy["split_mode"] == "within_day"
                      else [day for day in experiment_names if day != test_day])
        training_jobs = []
        for day in train_days:
            template = lookup.get((day, frequency))
            if template is None:
                raise FileNotFoundError(f"学習に必要な実験日・周波数のデータがありません: {day}, {frequency}")
            data_path = Path(template["source_dir"]) / frequency / train_noise
            if not has_input_files(data_path, 1):
                raise FileNotFoundError(f"学習に必要なノイズ条件がありません: {data_path}")
            training_jobs.append({**template, "data_path": data_path, "noise_dir_name": train_noise})
        context = {
            **policy,
            "training_experiments": train_days,
            "evaluation_experiments": [test_day],
            "training_noise_dir": train_noise,
            "generalization_scope": ("within_experiment_source_wav_oof"
                                     if policy["split_mode"] == "within_day"
                                     else "held_out_experiment_day"),
            "outer_folds_per_evaluation_day": None if policy["split_mode"] == "within_day" else 1,
        }
        families.append({"training_jobs": training_jobs, "evaluation_jobs": evaluation_jobs,
                         "context": context})
    return families


def checked_metadata(metadata, experiment_name):
    """ノイズ版の照合に使う実験日・元WAV・chunk番号を検査する。"""
    rows, seen = [], set()
    for row in metadata:
        row = dict(row)
        if row.get("experiment_name") not in (None, "", experiment_name):
            raise ValueError("manifestの実験日と読込対象の実験日が一致しません。")
        row["experiment_name"] = experiment_name
        if not row.get("source_wav_id") or row.get("chunk_index") in (None, ""):
            raise ValueError("chunk_manifest.csvにsource_wav_idとchunk_indexが必要です。")
        key = sample_key(row)
        if key in seen:
            raise ValueError(f"元WAV・chunk番号が重複しています: {key}")
        seen.add(key)
        rows.append(row)
    if not rows:
        raise ValueError("評価対象のchunkがありません。")
    return rows


def sample_key(row):
    return (row["experiment_name"], row["source_wav_id"], int(row["chunk_index"]))


def wav_groups(metadata):
    # 日付が異なる同名WAVを混同しない。JSON配列で区切り文字の衝突も避ける。
    return np.asarray([json.dumps([row["experiment_name"], row["source_wav_id"]],
                                  ensure_ascii=False) for row in metadata])


def targets_from_metadata(metadata):
    return np.asarray([float(row["sample_filename"].split("_")[0]) for row in metadata])


def aligned_indices(reference, other):
    """ファイル順に依存せず、同じ録音の同じ時間区間をノイズ間で対応付ける。"""
    lookup = {sample_key(row): index for index, row in enumerate(other)}
    if len(lookup) != len(other) or set(lookup) != {sample_key(row) for row in reference}:
        raise ValueError("ノイズ条件間で元WAV・chunkの集合が一致しません。")
    indices = np.asarray([lookup[sample_key(row)] for row in reference], dtype=int)
    if not np.allclose(targets_from_metadata(reference), targets_from_metadata(other)[indices],
                       rtol=0, atol=1e-6):
        raise ValueError("ノイズ条件間で熱流束ラベルが一致しません。")
    for ref, index in zip(reference, indices):
        for field in ("chunk_start_seconds", "chunk_duration_seconds"):
            if ref.get(field) not in (None, "") or other[index].get(field) not in (None, ""):
                if not np.isclose(float(ref[field]), float(other[index][field]), rtol=0, atol=1e-6):
                    raise ValueError(f"ノイズ条件間で{field}が一致しません。")
    return indices


def outer_splits(training_metadata, evaluation_metadata, split_mode, folds):
    """学習側と評価側の添字を返し、元WAV・実験日の重複を検出する。"""
    train_groups = wav_groups(training_metadata)
    test_groups = wav_groups(evaluation_metadata)
    if split_mode == "within_day":
        alignment = aligned_indices(training_metadata, evaluation_metadata)
        splitter = GroupKFold(n_splits=folds)
        splits = [(fit, alignment[test]) for fit, test in
                  splitter.split(np.zeros(len(train_groups)), groups=train_groups)]
    else:
        training_days = {row["experiment_name"] for row in training_metadata}
        test_days = {row["experiment_name"] for row in evaluation_metadata}
        if training_days & test_days:
            raise ValueError("別日評価の学習日と評価日が重複しています。")
        splits = [(np.arange(len(train_groups)), np.arange(len(test_groups)))]
    for fit, test in splits:
        if set(train_groups[fit]) & set(test_groups[test]):
            raise ValueError("学習側と評価側で元WAVが重複しています。")
    return splits
