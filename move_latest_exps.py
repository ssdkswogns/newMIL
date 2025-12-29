#!/usr/bin/env python3
"""
Move or copy the latest exp_* directory per dataset into a new root folder.

Selection logic matches collect_aopcr.py:
- Default picks the newest Train_log.log by mtime per dataset.
- --max-exp-id picks the exp_* with the largest numeric suffix instead.
Optionally filter by an option.txt reference with --match-options/--ignore-keys.
"""
import argparse
import shutil
from pathlib import Path
from typing import List, Optional

from collect_aopcr import find_latest_logs, parse_options_file


def move_or_copy_latest(
    root: Path,
    dest: Path,
    match_options: Optional[Path],
    ignore_keys: List[str],
    use_max_exp_id: bool,
    mode: str,
    dry_run: bool,
) -> None:
    ref_opts = parse_options_file(match_options) if match_options else None
    latest = find_latest_logs(root, ref_opts, ignore_keys, prefer_max_exp_id=use_max_exp_id)
    if not latest:
        msg = "No Train_log.log files found under datasets."
        if match_options:
            msg += " (no exp matched the given option file)"
        raise SystemExit(msg)

    dest.mkdir(parents=True, exist_ok=True)

    for dataset, log_path, _ in latest:
        exp_dir = log_path.parent
        target_dir = dest / dataset / exp_dir.name

        print(f"{mode.upper()}: {exp_dir} -> {target_dir}")
        if dry_run:
            continue

        target_dir.parent.mkdir(parents=True, exist_ok=True)
        if target_dir.exists():
            raise SystemExit(f"Target already exists: {target_dir}")

        if mode == "move":
            shutil.move(str(exp_dir), str(target_dir))
        else:
            shutil.copytree(exp_dir, target_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move or copy the latest exp_* per dataset into a new root folder."
    )
    parser.add_argument(
        "--root",
        default="savemodel/InceptBackbone",
        type=str,
        help="Root directory containing dataset subfolders (default: savemodel/InceptBackbone)",
    )
    parser.add_argument(
        "--dest",
        required=True,
        type=str,
        help="Destination root; dataset/exp_* structure will be created under this path.",
    )
    parser.add_argument(
        "--match-options",
        default=None,
        type=str,
        help="If set, only use exp_* whose option.txt matches this file (ignoring --ignore-keys).",
    )
    parser.add_argument(
        "--ignore-keys",
        nargs="*",
        default=["save_dir", "seed"],
        help="Keys to ignore when matching option.txt (default: save_dir seed)",
    )
    parser.add_argument(
        "--max-exp-id",
        action="store_true",
        help="Pick the exp_* with the largest numeric suffix instead of newest mtime.",
    )
    parser.add_argument(
        "--mode",
        choices=["move", "copy"],
        default="move",
        help="Move (default) or copy the selected exp_* directories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be moved/copied without performing the operation.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    dest = Path(args.dest)
    match_path = Path(args.match_options) if args.match_options else None

    move_or_copy_latest(
        root=root,
        dest=dest,
        match_options=match_path,
        ignore_keys=args.ignore_keys,
        use_max_exp_id=args.max_exp_id,
        mode=args.mode,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
