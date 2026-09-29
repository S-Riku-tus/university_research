"""学習日・学習ノイズの選択と、明示した単位の評価分割を組み立てる。"""

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedShuffleSplit

from utils.experiment.dataset_jobs import has_input_files


DEFAULT_POLICY = {"training_noise": "matched"}
CLEAN_NOISE = "heatflux_no_noise"


def _resolved_day_policy(policy):
    """明示した評価方式の実効日付を解決する。省略時は従来の日付推論。"""
    policy = dict(policy or {})
    mode = policy.get("evaluation_mode", "auto")
    if mode not in {"auto", "cross_day", "within_day", "within_wav_chunk"}:
        raise ValueError(
            "evaluation_modeはcross_day、within_day、within_wav_chunk、autoです。"
        )
    settings_by_mode = policy.pop("evaluation_settings", None)
    if settings_by_mode is not None:
        if not isinstance(settings_by_mode, dict):
            raise ValueError("evaluation_settingsは評価方式ごとの辞書にしてください。")
        unknown_modes = set(settings_by_mode) - {"cross_day", "within_day", "within_wav_chunk"}
        if unknown_modes:
            raise ValueError(f"evaluation_settingsに未知の評価方式があります: {unknown_modes}")
        if mode == "auto":
            raise ValueError("evaluation_settingsを使う場合はevaluation_modeを明示してください。")
        selected = settings_by_mode.get(mode)
        if not isinstance(selected, dict):
            raise ValueError(f"evaluation_settings.{mode}を辞書で指定してください。")
        legacy_keys = {
            "train_experiments", "test_experiments", "within_day_experiment",
            "test_fraction", "test_split_seed", "test_stratify",
        }
        duplicated = set(policy) & legacy_keys
        if duplicated:
            raise ValueError(
                "evaluation_settings使用時は方式別の変数を外側へ重複指定しないでください: "
                f"{duplicated}"
            )
        allowed = (
            {"train_experiments", "test_experiments"}
            if mode == "cross_day"
            else {"experiment", "test_fraction", "test_split_seed", "test_stratify"}
            if mode == "within_day"
            else {"experiment", "test_fraction", "test_split_seed"}
        )
        unknown = set(selected) - allowed
        if unknown:
            raise ValueError(f"evaluation_settings.{mode}に未知の変数があります: {unknown}")
        if mode == "cross_day":
            policy["train_experiments"] = selected.get("train_experiments")
            policy["test_experiments"] = selected.get("test_experiments")
        else:
            policy["within_day_experiment"] = selected.get("experiment")
            for key in ("test_fraction", "test_split_seed", "test_stratify"):
                if key in selected:
                    policy[key] = selected[key]
            if mode == "within_wav_chunk":
                policy["test_stratify"] = "none"
    if mode in {"within_day", "within_wav_chunk"}:
        day = policy.get("within_day_experiment")
        if not isinstance(day, str) or not day.strip():
            raise ValueError(
                "within_day_experimentに同一実験内で評価する実験名を指定してください。"
            )
        # cross_day用リストを編集せず、1個の実験日指定だけで切替可能にする。
        policy["train_experiments"] = [day]
        policy["test_experiments"] = [day]
    return policy


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
    """明示方式を優先し、省略時は従来どおり日付リストから決める。"""
    policy = _resolved_day_policy(policy)
    train_days = _day_list(policy, "train_experiments")
    test_days = _day_list(policy, "test_experiments")
    train_set, test_set = set(train_days), set(test_days)
    if policy.get("evaluation_mode") == "within_day":
        return "within_day_holdout"
    if policy.get("evaluation_mode") == "within_wav_chunk":
        return "within_wav_chunk_holdout"
    if policy.get("evaluation_mode") == "cross_day" and not train_set.isdisjoint(test_set):
        raise ValueError("cross_dayでは学習日とテスト日を完全に分離してください。")
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
    policy = _resolved_day_policy(policy)
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
    resolved = {**DEFAULT_POLICY, **_resolved_day_policy(policy)}
    extra_keys = {"train_experiments", "test_experiments", "evaluation_mode",
                  "within_day_experiment", "test_fraction", "test_split_seed", "test_stratify"}
    if set(resolved) - set(DEFAULT_POLICY) - extra_keys:
        raise ValueError(f"未知のlearning_policy設定です: {set(resolved) - set(DEFAULT_POLICY)}")
    if resolved["training_noise"] not in {"matched", "clean_only"}:
        raise ValueError("training_noiseはmatchedまたはclean_onlyです。")
    if resolved.get("evaluation_mode") in {"within_day", "within_wav_chunk"}:
        fraction = resolved.setdefault("test_fraction", 0.25)
        seed = resolved.setdefault("test_split_seed", 42)
        if resolved.setdefault("test_stratify", "none") not in {"none", "onb"}:
            raise ValueError("test_stratifyはonbまたはnoneです。")
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction < 1:
            unit = "chunk数" if resolved.get("evaluation_mode") == "within_wav_chunk" else "元WAV数"
            raise ValueError(f"test_fractionは0より大きく1より小さい{unit}の割合です。")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("test_split_seedは0以上2**32未満の整数です。")
        if (resolved.get("evaluation_mode") == "within_wav_chunk"
                and resolved["test_stratify"] != "none"):
            raise ValueError(
                "within_wav_chunkでは全WAVから同じ割合を抽出するため、"
                "test_stratifyはnoneにしてください。"
            )
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
    day = {"within_day": "wd", "within_day_holdout": "wh",
           "within_wav_chunk_holdout": "wc",
           "leave_one_day_out": "lodo", "cross_day": "days"}[
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
        if split_kind in {"within_day", "within_day_holdout", "within_wav_chunk_holdout"}:
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
                                     else "within_experiment_source_wav_holdout"
                                     if split_kind == "within_day_holdout"
                                     else "within_known_source_wav_unseen_chunk_holdout"
                                     if split_kind == "within_wav_chunk_holdout"
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


def outer_splits(training_metadata, evaluation_metadata, folds, policy=None, threshold=None):
    """学習側と評価側の添字を返し、元WAV・実験日の重複を検出する。"""
    train_groups = wav_groups(training_metadata)
    test_groups = wav_groups(evaluation_metadata)
    training_days = {row["experiment_name"] for row in training_metadata}
    test_days = {row["experiment_name"] for row in evaluation_metadata}
    split_kind = experiment_split_kind(policy) if policy else None
    if training_days == test_days and len(training_days) == 1:
        alignment = aligned_indices(training_metadata, evaluation_metadata)
        if split_kind == "within_day_holdout":
            policy = normalize_learning_policy(policy, sorted(training_days))
            if policy["test_stratify"] == "onb":
                if threshold is None or not np.isfinite(threshold):
                    raise ValueError("ONB層化分割には実験日の有限なONB閾値が必要です。")
                unique = np.unique(train_groups)
                targets = targets_from_metadata(training_metadata)
                # 同じWAVの全chunkをまとめ、ONB前／以上の構成を保つ。
                labels = []
                for group in unique:
                    group_labels = np.unique(targets[train_groups == group] >= threshold)
                    if len(group_labels) != 1:
                        raise ValueError("同じ元WAV内でONB前後が混在しているため層化できません。")
                    labels.append(group_labels[0])
                _, counts = np.unique(labels, return_counts=True)
                if len(counts) != 2 or min(counts) < 2:
                    raise ValueError("ONB層化にはONB前・以上それぞれ2本以上の元WAVが必要です。")
                splitter = StratifiedShuffleSplit(n_splits=1, test_size=policy["test_fraction"],
                                                 random_state=policy["test_split_seed"])
                group_fit, group_test = next(splitter.split(unique, labels))
                fit = np.flatnonzero(np.isin(train_groups, unique[group_fit]))
                test = np.flatnonzero(np.isin(train_groups, unique[group_test]))
                splits = [(fit, alignment[test])]
            else:
                splitter = GroupShuffleSplit(n_splits=1, test_size=policy["test_fraction"],
                                             random_state=policy["test_split_seed"])
                splits = [(fit, alignment[test]) for fit, test in
                          splitter.split(np.zeros(len(train_groups)), groups=train_groups)]
        elif split_kind == "within_wav_chunk_holdout":
            policy = normalize_learning_policy(policy, sorted(training_days))
            unique_groups = sorted(set(train_groups))
            has_provenance = [
                bool(str(row.get("source_experiment_name", "")).strip())
                or bool(str(row.get("original_source_wav_id", "")).strip())
                for row in training_metadata
            ]
            if any(has_provenance):
                if not all(has_provenance):
                    raise ValueError(
                        "統合データの出典情報は全chunkに指定してください。"
                    )
                provenance_groups = {}
                seen_original_groups = set()
                for group in unique_groups:
                    rows = [
                        training_metadata[int(index)]
                        for index in np.flatnonzero(train_groups == group)
                    ]
                    source_days = {
                        str(row.get("source_experiment_name", "")).strip()
                        for row in rows
                    }
                    original_wavs = {
                        str(row.get("original_source_wav_id", "")).strip()
                        for row in rows
                    }
                    if (len(source_days) != 1 or "" in source_days
                            or len(original_wavs) != 1 or "" in original_wavs):
                        raise ValueError(
                            "同じ元WAV内で統合データの出典日または元WAV IDが一致しません。"
                        )
                    source_day = next(iter(source_days))
                    original_wav = next(iter(original_wavs))
                    original_key = (source_day, original_wav)
                    if original_key in seen_original_groups:
                        raise ValueError(
                            "統合データ内で出典日・元WAV IDが重複しています: "
                            f"{original_key}"
                        )
                    seen_original_groups.add(original_key)
                    provenance_groups.setdefault(source_day, []).append((original_wav, group))
                # 単日runと同じ乱数列を各出典日に独立して適用する。これにより、
                # 統合runでも各日の同じchunkを外側テストとして再利用できる。
                group_pools = [
                    [group for _, group in sorted(provenance_groups[source_day])]
                    for source_day in sorted(provenance_groups)
                ]
            else:
                group_pools = [unique_groups]
            selected = []
            for groups in group_pools:
                rng = np.random.RandomState(policy["test_split_seed"])
                for group in groups:
                    group_indices = np.flatnonzero(train_groups == group)
                    group_indices = np.asarray(sorted(
                        group_indices,
                        key=lambda index: sample_key(training_metadata[int(index)]),
                    ), dtype=int)
                    if len(group_indices) < 2:
                        raise ValueError(
                            "within_wav_chunkには各元WAVにつき2 chunk以上必要です。"
                        )
                    n_test = max(1, int(np.ceil(len(group_indices) * policy["test_fraction"])))
                    n_test = min(n_test, len(group_indices) - 1)
                    positions = np.sort(rng.choice(len(group_indices), size=n_test, replace=False))
                    selected.extend(group_indices[positions].tolist())
            test_reference = np.asarray(sorted(selected), dtype=int)
            fit = np.setdiff1d(
                np.arange(len(training_metadata), dtype=int), test_reference,
                assume_unique=True,
            )
            splits = [(fit, alignment[test_reference])]
        else:
            splitter = GroupKFold(n_splits=folds)
            splits = [(fit, alignment[test]) for fit, test in
                      splitter.split(np.zeros(len(train_groups)), groups=train_groups)]
    elif training_days.isdisjoint(test_days):
        splits = [(np.arange(len(train_groups)), np.arange(len(test_groups)))]
    else:
        raise ValueError("学習日と評価日は、同一の1日または完全に分離した日集合である必要があります。")
    for fit, test in splits:
        if split_kind == "within_wav_chunk_holdout":
            fit_keys = {sample_key(training_metadata[int(index)]) for index in fit}
            test_keys = {sample_key(evaluation_metadata[int(index)]) for index in test}
            if fit_keys & test_keys:
                raise ValueError("学習側と評価側で同じchunkが重複しています。")
            if set(train_groups[fit]) != set(test_groups[test]):
                raise ValueError("within_wav_chunkでは全元WAVを学習側と評価側へ含めます。")
        elif set(train_groups[fit]) & set(test_groups[test]):
            raise ValueError("学習側と評価側で元WAVが重複しています。")
    return splits
