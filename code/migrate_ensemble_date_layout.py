"""ensemble結果の日付階層を YYYYMMDD から YYYYMM/DD へ安全に移行する。"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re


LEGACY_DATE_DIR = re.compile(r"^(\d{6})(\d{2})(?:_(.+))?$")
TEXT_SUFFIXES = {".csv", ".json", ".md", ".txt"}


def _long_path(path):
    absolute = os.path.abspath(path)
    if os.name == "nt" and not absolute.startswith("\\\\?\\"):
        return Path("\\\\?\\" + absolute)
    return Path(absolute)


def _normal_path(path):
    value = str(path)
    return value[4:] if value.startswith("\\\\?\\") else value


def _is_link(path):
    return path.is_symlink() or (
        hasattr(path, "is_junction") and path.is_junction()
    )


def _assert_inside(path, root):
    path.resolve().relative_to(root.resolve())


def _destination_for_name(name):
    match = LEGACY_DATE_DIR.fullmatch(name)
    if not match:
        return None
    month, day, suffix = match.groups()
    try:
        datetime.strptime(month + day, "%Y%m%d")
    except ValueError:
        return None
    relative = Path(month) / day
    return relative / suffix if suffix else relative


def find_ensemble_roots(search_root):
    search_root = Path(search_root).resolve()
    roots = []
    # ensemble配下には長い生成物パスがあるため、そこを再帰探索しない。
    # データ本体も対象外として枝刈りし、regression_resultだけを探す。
    def onerror(error):
        raise error

    for current, directories, _ in os.walk(search_root, topdown=True, onerror=onerror):
        current = Path(current)
        if current.name == "regression_result":
            path = current / "npy" / "ensemble"
            if path.is_dir():
                _assert_inside(path, search_root)
                roots.append(path)
            directories[:] = []
            continue
        directories[:] = [
            name for name in directories
            if name not in {"data", ".git", ".agents", ".codex"}
        ]
    return sorted(roots)


def migration_plan(search_root):
    search_root = Path(search_root).resolve()
    if not search_root.is_dir():
        raise FileNotFoundError(search_root)
    moves = []
    for root in find_ensemble_roots(search_root):
        for source in sorted(root.iterdir()):
            if not source.is_dir():
                continue
            relative_target = _destination_for_name(source.name)
            if relative_target is None:
                continue
            if _is_link(source):
                raise ValueError(f"リンクまたはjunctionは移動できません: {source}")
            target = root / relative_target
            _assert_inside(source, root)
            _assert_inside(target, root)
            moves.append({
                "root": root,
                "source": source,
                "target": target,
                "has_suffix": len(relative_target.parts) > 2,
            })

    targets = {}
    for move in moves:
        target = move["target"]
        if target in targets:
            raise FileExistsError(
                f"複数の移動元が同じ移動先を指しています: {target}"
            )
        targets[target] = move["source"]
        if target.exists():
            raise FileExistsError(f"移動先が既に存在します。統合しません: {target}")

    # 同じ日の YYYYMMDD と YYYYMMDD_suffix が併存する場合、先に日付本体を
    # 移してからsuffixをその下へ置く。日付本体内の同名フォルダとも衝突させない。
    exact_by_target = {
        move["target"]: move for move in moves if not move["has_suffix"]
    }
    for move in moves:
        if not move["has_suffix"]:
            continue
        exact = exact_by_target.get(move["target"].parent)
        if exact is not None:
            nested = exact["source"] / move["target"].name
            if nested.exists():
                raise FileExistsError(
                    f"日付本体内の既存フォルダとsuffixの移動先が衝突します: {nested}"
                )

    return sorted(
        moves,
        key=lambda item: (
            str(item["root"]),
            item["target"].parts,
            item["has_suffix"],
        ),
    )


def _tree_stats(path):
    files = 0
    total_bytes = 0
    for item in _long_path(path).rglob("*"):
        if _is_link(item):
            raise ValueError(f"移動対象内にリンクまたはjunctionがあります: {item}")
        if item.is_file():
            files += 1
            total_bytes += item.stat().st_size
    return {"files": files, "bytes": total_bytes}


def _replacement_pairs(moves):
    replacements = []
    for move in moves:
        source = move["source"]
        target = move["target"]
        values = [
            (str(source), str(target)),
            (source.as_posix(), target.as_posix()),
            (
                json.dumps(str(source), ensure_ascii=False)[1:-1],
                json.dumps(str(target), ensure_ascii=False)[1:-1],
            ),
        ]
        old_component = source.name
        new_component = (target.relative_to(move["root"])).as_posix()
        if move["has_suffix"]:
            values.append((old_component, new_component))
        else:
            values.extend([
                (old_component + "/", new_component + "/"),
                (old_component + "\\", new_component.replace("/", "\\") + "\\"),
                (
                    old_component + "\\\\",
                    new_component.replace("/", "\\\\") + "\\\\",
                ),
            ])
        for before, after in values:
            pair = (before.encode("utf-8"), after.encode("utf-8"))
            if pair not in replacements:
                replacements.append(pair)
    return sorted(replacements, key=lambda pair: len(pair[0]), reverse=True)


def _minimal_target_roots(moves):
    targets = sorted({move["target"] for move in moves}, key=lambda path: len(path.parts))
    return [
        target for index, target in enumerate(targets)
        if not any(parent in target.parents for parent in targets[:index])
    ]


def _rewrite_moved_references(moves):
    replacements = _replacement_pairs(moves)
    rewritten = 0
    for root in _minimal_target_roots(moves):
        for path in _long_path(root).rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            original = path.read_bytes()
            updated = original
            for before, after in replacements:
                updated = updated.replace(before, after)
            if updated != original:
                path.write_bytes(updated)
                rewritten += 1
    return rewritten


def _write_report(path, report):
    if path is None:
        return
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def migrate_ensemble_dates(search_root, apply=False, report_path=None):
    search_root = Path(search_root).resolve()
    moves = migration_plan(search_root)
    report = {
        "search_root": _normal_path(search_root),
        "applied": apply,
        "status": "planned",
        "move_count": len(moves),
        "moves": [
            {
                "from": _normal_path(move["source"]),
                "to": _normal_path(move["target"]),
            }
            for move in moves
        ],
    }
    _write_report(report_path, report)
    if not apply or not moves:
        return report

    try:
        for move, record in zip(moves, report["moves"]):
            record["before"] = _tree_stats(move["source"])
        report["status"] = "in_progress"
        _write_report(report_path, report)

        for move, record in zip(moves, report["moves"]):
            target = move["target"]
            target.parent.mkdir(parents=True, exist_ok=True)
            _long_path(move["source"]).rename(_long_path(target))
            record["after"] = _tree_stats(move["target"])
            if record["before"] != record["after"]:
                raise RuntimeError(
                    f"移動前後でファイル数または総byte数が一致しません: {move['target']}"
                )
        report["reference_files_updated"] = _rewrite_moved_references(moves)
        report["status"] = "complete"
        _write_report(report_path, report)
        return report
    except Exception as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        _write_report(report_path, report)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = migrate_ensemble_dates(
        args.search_root,
        apply=args.apply,
        report_path=args.report,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
