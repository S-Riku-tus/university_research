"""学習日・学習ノイズの選択と、元WAVを共有しない評価分割を組み立てる。"""

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from utils.experiment.dataset_jobs import has_input_files


DEFAULT_POLICY = {"training_noise": "matched"}
CLEAN_NOISE = "heatflux_no_noise"


def _day_list(policy, key):
    values = policy.get(key)
    if not isinstance(values, (list, tuple)) or not values or any(
        not isinstance(day, str) or not day for day in values
    ):
        raise ValueError(f"learning_policy.{key}には重複のない実験日名のリストが必要です。")
    if len(set(values)) != len(values):
        raise ValueError(f"learning_policy.{key}に同じ実験日を重複指定できません。")
    return list(values)


def experiment_split_kind(policy):
    """日付リストだけから評価方式を一意に決める。"""
    train_days = _day_list(policy, "train_experiments")
    test_days = _day_list(policy, "test_experiments")
    train_set, test_set = set(train_days), set(test_days)
    if train_set.isdisjoint(test_set):
        return "cross_day"
    if train_set == test_set:
        return "within_day" if len(train_set) == 1 else "leave_one_day_out"
    raise ValueError(
        "train_experimentsとtest_experimentsは、完全分離（別日評価）または"
        "同一集合（日内評価／leave-one-day-out）にしてください。部分重複は使用できません。"
    )


def resolve_experiment_names(data_config, policy):
    """実行対象日を学習日とテスト日の和集合から一元化する。"""
    policy = policy or {}
    configured = data_config.get("experiment_names")
    if configured is not None and (
        not isinstance(configured, (list, tuple))
        or any(not isinstance(day, str) or not day for day in configured)
    ):
        raise ValueError("data.experiment_namesには実験日名のリストを指定してください。")
    train_days = _day_list(policy, "train_experiments")
    test_days = _day_list(policy, "test_experiments")
    derived = list(dict.fromkeys([*train_days, *test_days]))
    if configured is None:
        return sorted(derived)
    if set(configured) != set(derived) or len(configured) != len(derived):
        raise ValueError("data.experiment_namesは学習日とテスト日の和集合に一致させてください。")
    return list(configured)


def normalize_learning_policy(policy, experiment_names, color_channel=1):
    resolved = {**DEFAULT_POLICY, **(policy or {})}
    extra_keys = {"train_experiments", "test_experiments"}
    if set(resolved) - set(DEFAULT_POLICY) - extra_keys:
        raise ValueError(f"未知のlearning_policy設定です: {set(resolved) - set(DEFAULT_POLICY)}")
    if resolved["training_noise"] not in {"matched", "clean_only"}:
        raise ValueError("training_noiseはmatchedまたはclean_onlyです。")
    if len(set(experiment_names)) != len(experiment_names):
        raise ValueError("experiment_namesに同じ実験日を重複指定できません。")
    for key in ("train_experiments", "test_experiments"):
        days = _day_list(resolved, key)
        if set(days) - set(experiment_names):
            raise ValueError(f"{key}にdata.experiment_names外の実験日があります。")
    unused = set(experiment_names) - set(resolved["train_experiments"]) - set(resolved["test_experiments"])
    if unused:
        raise ValueError(f"experiment_namesに未使用の実験日があります: {sorted(unused)}")
    experiment_split_kind(resolved)
    if color_channel != 1:
        raise ValueError("元WAVを分離する評価にはchunk_manifest.csv付きのNPY入力が必要です。")
    return resolved


def policy_result_date_dir(result_date_dir, policy):
    """保存系列パスの末尾へ学習・評価方針を付け、結果を分離する。"""
    day = {"within_day": "wd", "leave_one_day_out": "lodo", "cross_day": "days"}[
        experiment_split_kind(policy)
    ]
    noise = "clean" if policy["training_noise"] == "clean_only" else "matched"
    return f"{result_date_dir}__{day}_{noise}"


def build_learning_families(jobs, policy, experiment_names):
    """Build the independent fit/weight scope for each evaluation family.

    matched creates one family per noise, so models and ensemble weights are
    fitted again from that noise's training data.  clean_only deliberately
    creates one clean family and shares that fitted state across evaluation
    noises.
    """
    policy = normalize_learning_policy(policy, experiment_names)
    split_kind = experiment_split_kind(policy)
    lookup = {(j["experiment_name"], j["max_freq_hz"]): j for j in jobs}
    grouped = {}
    for job in jobs:
        if job["experiment_name"] not in policy["test_experiments"]:
            continue
        noise = CLEAN_NOISE if policy["training_noise"] == "clean_only" else job["noise_dir_name"]
        key = (job["experiment_name"], job["max_freq_hz"], noise)
        grouped.setdefault(key, []).append(job)
    families = []
    for (test_day, frequency, train_noise), evaluation_jobs in grouped.items():
        if split_kind == "within_day":
            train_days = [test_day]
        elif split_kind == "leave_one_day_out":
            train_days = [day for day in policy["train_experiments"] if day != test_day]
        else:
            train_days = list(policy["train_experiments"])
        training_jobs = []
        for day in train_days:
            template = lookup.get((day, frequency))
            if template is None:
                raise FileNotFoundError(f"学習に必要な実験日・周波数のデータがありません: {day}, {frequency}")
            data_path = Path(template["source_dir"]) / frequency / train_noise
            if not has_input_files(data_path, 1):
                raise FileNotFoundError(f"学習に必要なノイズ条件がありません: {data_path}")
            training_jobs.append({**template, "data_path": data_path, "noise_dir_name": train_noise})
        evaluation_noises = {job["noise_dir_name"] for job in evaluation_jobs}
        if policy["training_noise"] == "matched":
            if evaluation_noises != {train_noise}:
                raise RuntimeError(
                    "matched must create one independent training/weight family per noise."
                )
            ensemble_weight_scope = "per_training_noise"
        else:
            if train_noise != CLEAN_NOISE:
                raise RuntimeError("clean_only must fit models and weights from clean data.")
            ensemble_weight_scope = "shared_clean_across_evaluation_noises"
        context = {
            **policy,
            "training_experiments": train_days,
            "evaluation_experiments": [test_day],
            "training_noise_dir": train_noise,
            "ensemble_weight_scope": ensemble_weight_scope,
            "evaluation_scheme": split_kind,
            "generalization_scope": ("within_experiment_source_wav_oof"
                                     if split_kind == "within_day"
                                     else "held_out_experiment_day"),
            "outer_folds_per_evaluation_day": None if split_kind == "within_day" else 1,
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


def outer_splits(training_metadata, evaluation_metadata, folds):
    """学習側と評価側の添字を返し、元WAV・実験日の重複を検出する。"""
    train_groups = wav_groups(training_metadata)
    test_groups = wav_groups(evaluation_metadata)
    training_days = {row["experiment_name"] for row in training_metadata}
    test_days = {row["experiment_name"] for row in evaluation_metadata}
    if training_days == test_days and len(training_days) == 1:
        alignment = aligned_indices(training_metadata, evaluation_metadata)
        splitter = GroupKFold(n_splits=folds)
        splits = [(fit, alignment[test]) for fit, test in
                  splitter.split(np.zeros(len(train_groups)), groups=train_groups)]
    elif training_days.isdisjoint(test_days):
        splits = [(np.arange(len(train_groups)), np.arange(len(test_groups)))]
    else:
        raise ValueError("学習日と評価日は、同一の1日または完全に分離した日集合である必要があります。")
    for fit, test in splits:
        if set(train_groups[fit]) & set(test_groups[test]):
            raise ValueError("学習側と評価側で元WAVが重複しています。")
    return splits
