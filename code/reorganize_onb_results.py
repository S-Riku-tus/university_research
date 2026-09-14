"""指定した実行日の生成結果を、周波数／ノイズの順へ移す。無指定では確認だけ行う。"""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path


def long_path(path):
    absolute = os.path.abspath(path)
    return Path("\\\\?\\" + absolute) if os.name == "nt" and not absolute.startswith("\\\\?\\") else Path(absolute)


def migration_plan(result_root):
    resolved_root = Path(result_root).resolve()
    root = long_path(resolved_root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    moves = []
    for noise in sorted(root.glob("heatflux_*")):
        if not noise.is_dir():
            continue
        for frequency in sorted(noise.glob("maxfreq=*")):
            if frequency.is_dir():
                moves.append((frequency, root / frequency.name / noise.name))
    old_graphs = root / "noise_trends"
    if old_graphs.is_dir():
        for frequency in sorted(old_graphs.glob("maxfreq=*")):
            if frequency.is_dir():
                moves.append((frequency, root / frequency.name / "noise_trends"))
    for source, target in moves:
        # 移動前に全対象を検査する。明示した実行日の外や既存の移動先へは書き込まない。
        source.resolve().relative_to(resolved_root)
        target.resolve().relative_to(resolved_root)
        if source.is_symlink() or (hasattr(source, "is_junction") and source.is_junction()):
            raise ValueError(f"リンク先の結果フォルダは移動できません: {source}")
        if target.exists():
            raise FileExistsError(f"移動先が既に存在します。上書きしません: {target}")
    return root, moves


def _normal_path(path):
    value = str(path)
    return value[4:] if value.startswith("\\\\?\\") else value


def migrate_results(result_root, apply=False):
    root, moves = migration_plan(result_root)
    report = {"result_root": _normal_path(root), "applied": apply,
              "moves": [{"from": _normal_path(source), "to": _normal_path(target)} for source, target in moves]}
    if not apply or not moves:
        return report
    # 残った指標の値が変わらないことを、移動前後のファイルhashで確認する。
    metric_hashes = {}
    for source, target in moves:
        for path in source.rglob("*.csv"):
            if path.name.startswith(("metrics_summary_", "wav_metrics_", "pred_f")):
                metric_hashes[target / path.relative_to(source)] = hashlib.sha256(path.read_bytes()).hexdigest()
    log_path = root / f"layout_migration_{datetime.now():%Y%m%d_%H%M%S}.json"
    report["status"] = "in_progress"
    log_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for source, target in moves:
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
    replacements = []
    for source, target in moves:
        old, new = _normal_path(source), _normal_path(target)
        for before, after in ((old, new), (old.replace("\\", "/"), new.replace("\\", "/")),
                              (json.dumps(old, ensure_ascii=False)[1:-1], json.dumps(new, ensure_ascii=False)[1:-1])):
            replacements.append((before.encode("utf-8"), after.encode("utf-8")))
    # CSV内のsave_path・グラフの参照元など、移動した生成結果への参照だけ更新する。
    rewritten = 0
    for path in root.rglob("*"):
        if path == log_path or not path.is_file() or path.suffix.lower() not in {".json", ".csv", ".txt"}:
            continue
        original = path.read_bytes()
        updated = original
        for before, after in replacements:
            updated = updated.replace(before, after)
        if updated != original:
            path.write_bytes(updated)
            rewritten += 1
    for path, expected in metric_hashes.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"移動前後で評価値のCSVが変化しました: {path}")
    # 空になった旧階層だけを取り除く。中に別の結果がある場合は残す。
    for parent in {source.parent for source, _ in moves}:
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
    report.update(status="complete", metric_files_verified=len(metric_hashes), reference_files_updated=rewritten)
    log_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="移動と参照パスの更新を実施する")
    args = parser.parse_args()
    print(json.dumps(migrate_results(args.result_root, args.apply), ensure_ascii=False, indent=2))
