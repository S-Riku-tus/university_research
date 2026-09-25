"""周波数／ノイズの保存階層と、旧階層の結果参照を一元管理する。"""

import re
from datetime import datetime
from pathlib import Path

from utils.experiment.run_helpers import path_exists, safe_tag


MAX_STUDY_DIR_LENGTH = 52


def normalize_result_date_dir(result_date_dir):
    """解析日を ``YYYYMM/DD`` に分け、既存の相対パス指定も保つ。

    ``YYYYMMDD/study`` と旧来の ``YYYYMMDD_study`` の両方を受け付ける。
    すでに ``YYYYMM/DD/study`` なら何も変えない。日付で始まらない旧系列名は
    推測で分類せず、そのまま返す。
    """
    value = str(result_date_dir).strip()
    if not value:
        raise ValueError("result_date_dir must not be empty")
    parts = list(Path(value).parts)
    if not parts:
        raise ValueError("result_date_dir must not be empty")
    match = re.fullmatch(r"(\d{8})(?:_(.+))?", parts[0])
    if match:
        date_value, suffix = match.groups()
        try:
            datetime.strptime(date_value, "%Y%m%d")
        except ValueError:
            return Path(*parts).as_posix()
        normalized = [date_value[:6], date_value[6:]]
        if suffix:
            normalized.append(suffix)
        normalized.extend(parts[1:])
        return Path(*normalized).as_posix()
    return Path(*parts).as_posix()


def _compact_day(day):
    """実験日名を人が比較しやすいMMDDへ縮め、日付でなければ安全に短縮する。"""
    match = re.match(r"^(\d{4})[.-](\d{2})[.-](\d{2})", str(day))
    if match:
        return f"{match.group(2)}{match.group(3)}"
    return safe_tag(day, max_len=8)


def _day_list(days):
    return "+".join(_compact_day(day) for day in days)


def _policy_segment(job, config, compact=False):
    from utils.experiment.learning_policy import experiment_split_kind
    policy = config.get("learning_policy", {})
    split_kind = experiment_split_kind(policy)
    test_day = job.get("experiment_name")
    if split_kind == "cross_day":
        train_days = list(policy.get("train_experiments", []))
        test_days = list(policy.get("test_experiments", []))
        code = "xd"
    elif split_kind == "leave_one_day_out":
        train_days = [day for day in policy["train_experiments"] if day != test_day]
        test_days = [test_day]
        code = "lo"
    else:
        train_days = [test_day]
        test_days = [test_day]
        code = "wd"
    if compact:
        train = f"{len(train_days)}d"
        test = _compact_day(test_days[0]) if len(test_days) == 1 else f"{len(test_days)}d"
    else:
        train = _day_list(train_days)
        test = _day_list(test_days)
    return f"{code}-t{train}-v{test}"


def _selection_segment(config):
    selection = config.get("acoustic_selection", {})
    threshold = selection.get("peak_height_threshold")
    if not selection.get("enabled", threshold is not None) or threshold is None:
        return "s0"
    value = format(float(threshold), ".3g").replace("+", "")
    value = re.sub(r"e(-?)0+(\d+)$", r"e\1\2", value)
    return "s" + value


def result_scope_dir_name(
    base_name,
    job,
    config,
    execution_id,
    run_hash,
    parameter_index=1,
    parameter_count=1,
):
    """短く読める主要条件と6桁の実行時刻から実行系列名を作る。"""
    if not execution_id or not run_hash:
        raise ValueError("execution_id and run_hash are required for scoped results")
    if not re.fullmatch(r"\d{6}", str(execution_id)):
        raise ValueError("execution_id must be HHMMSS (6 digits)")
    if not 1 <= parameter_index <= parameter_count:
        raise ValueError("parameter_index must be within parameter_count")
    policy = config.get("learning_policy", {})
    validation = "iw" + str(config.get("run", {}).get("folds", "x"))
    noise = "nc" if policy.get("training_noise") == "clean_only" else "nm"
    epochs = config.get("run", {}).get("epochs", "x")
    prefix = safe_tag(str(base_name).split("__", 1)[0], max_len=12)
    suffix = str(execution_id)

    def build(compact_days=False, include_epochs=True):
        parts = [prefix, _policy_segment(job, config, compact=compact_days),
                 f"{validation}-{noise}", _selection_segment(config)]
        if include_epochs:
            parts.append(f"e{epochs}")
        if parameter_count > 1:
            parts.append(f"p{parameter_index:02d}")
        parts.append(suffix)
        return "_".join(parts)

    name = build()
    if len(name) > MAX_STUDY_DIR_LENGTH:
        name = build(include_epochs=False)
    if len(name) > MAX_STUDY_DIR_LENGTH:
        name = build(compact_days=True, include_epochs=False)
    if len(name) > MAX_STUDY_DIR_LENGTH:
        # 最後の保険。実行時刻を必ず保持し、不完全なtokenを残さない。
        semantic_budget = MAX_STUDY_DIR_LENGTH - len(suffix) - 1
        semantic = safe_tag("_".join(name.split("_")[:-1]), max_len=semantic_budget)
        name = f"{semantic}_{suffix}"
    return name


def scoped_result_job(
    job,
    execution_id,
    run_hash,
    config,
    parameter_index=1,
    parameter_count=1,
):
    """主要条件が読める名前で、実行とパラメータ条件ごとに保存先を分ける。"""
    if not execution_id or not run_hash:
        raise ValueError("execution_id and run_hash are required for scoped results")
    base = Path(job["save_base_path"])
    scope_name = result_scope_dir_name(
        base.name,
        job,
        config,
        execution_id,
        run_hash,
        parameter_index=parameter_index,
        parameter_count=parameter_count,
    )
    return {**job, "save_base_path": base.with_name(scope_name)}


def result_run_path(job, run_dir):
    return Path(job["save_base_path"]) / job["max_freq_hz"] / job["noise_dir_name"] / run_dir


def existing_result_run_path(job, run_dir):
    current = result_run_path(job, run_dir)
    legacy = Path(job["save_base_path"]) / job["noise_dir_name"] / job["max_freq_hz"] / run_dir
    # 新階層に結果があれば優先し、未移行の既存結果だけ旧階層から読む。
    return current if path_exists(current) or not path_exists(legacy) else legacy


def noise_trend_path(job, run_dir):
    return Path(job["save_base_path"]) / "noise_trends" / run_dir
