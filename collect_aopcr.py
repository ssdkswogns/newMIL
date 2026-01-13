#!/usr/bin/env python3
"""
Collect the most recent AOPCR summaries from Train_log.log files.

Looks under savemodel/InceptBackbone/<dataset>/exp_*/Train_log.log,
finds the latest log (by mtime) per dataset, and prints the final
Weighted/Mean/Sum AOPCR along with the per-class entries that
precede that summary block.
"""
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SUMMARY_RE = re.compile(
    r"Weighted AOPCR:\s*([0-9eE+\-\.]+),\s*Mean AOPCR:\s*([0-9eE+\-\.]+)"
    r"(?:,\s*Sum AOPCR:\s*([0-9eE+\-\.]+))?"
)
CLASS_RE = re.compile(
    r"\[AOPCR\]\s*class\s+(\S+)\s+count=(\d+)\s+AOPCR=([0-9eE+\-\.]+|NaN)",
    re.IGNORECASE,
)
BEST_RE = re.compile(r"Best Results \|", re.IGNORECASE)
METRIC_KV_RE = re.compile(r"([A-Za-z0-9_()/]+)=([0-9eE+\-\.]+)")
COVERAGE_RE = re.compile(r"Coverage@(\d+)%:")
COVERAGE_DETAIL_RE = re.compile(
    r"(Explanation|Random|Gain):\s*([0-9eE+\-\.]+)\s*\(.*?([0-9eE+\-\.]+)\)"
)


def parse_log(log_path: Path) -> Dict[str, Optional[Dict]]:
    """Parse a Train_log.log and return the last Best Results and AOPCR blocks."""
    aopcr_blocks: List[Dict] = []
    current_classes: List[Dict] = []
    best_block: Optional[Dict] = None
    fallback_metrics: Optional[Dict] = None  # e.g., ED1NN lines without "Best Results"
    coverage_blocks: List[Dict] = []
    current_coverage: Optional[Dict] = None
    lines = log_path.read_text(errors="ignore").splitlines()

    for line in lines:
        # Best Results
        if BEST_RE.search(line):
            metrics = dict(METRIC_KV_RE.findall(line))
            # Convert numeric values
            metrics = {k: float(v) for k, v in metrics.items()}
            best_block = metrics
            continue

        # AOPCR class lines
        m_class = CLASS_RE.search(line)
        if m_class:
            current_classes.append(
                {
                    "class": m_class.group(1),
                    "count": int(m_class.group(2)),
                    "aopcr": m_class.group(3),
                }
            )
            continue

        # AOPCR summary line
        m_sum = SUMMARY_RE.search(line)
        if m_sum:
            aopcr_blocks.append(
                {
                    "weighted": float(m_sum.group(1)),
                    "mean": float(m_sum.group(2)),
                    "sum": float(m_sum.group(3)) if m_sum.group(3) is not None else None,
                    "classes": current_classes,
                }
            )
            current_classes = []
            continue

        # Coverage section start
        m_cov = COVERAGE_RE.search(line)
        if m_cov:
            threshold = int(m_cov.group(1)) / 100.0
            current_coverage = {"threshold": threshold}
            continue

        # Coverage detail lines
        if current_coverage is not None:
            m_cov_detail = COVERAGE_DETAIL_RE.search(line)
            if m_cov_detail:
                metric_type = m_cov_detail.group(1).lower()
                mean_val = float(m_cov_detail.group(2))
                weighted_val = float(m_cov_detail.group(3))
                current_coverage[f"{metric_type}_mean"] = mean_val
                current_coverage[f"{metric_type}_weighted"] = weighted_val

                # If we have all three metrics, save the block
                if "gain_mean" in current_coverage:
                    coverage_blocks.append(current_coverage)
                    current_coverage = None
                continue

        # Fallback metric line (e.g., ED1NN logs)
        metric_pairs = METRIC_KV_RE.findall(line)
        if metric_pairs and best_block is None:
            keys_lower = [k.lower() for k, _ in metric_pairs]
            # Only consider lines that look like global metrics
            if any(k.startswith(("f1", "acc", "bal", "roc", "map")) for k in keys_lower):
                fallback_metrics = {k: float(v) for k, v in metric_pairs}

    if best_block is None:
        best_block = fallback_metrics

    return {
        "best": best_block,
        "aopcr": aopcr_blocks[-1] if aopcr_blocks else None,
        "coverage": coverage_blocks if coverage_blocks else None,
    }


def parse_options_file(path: Path) -> Dict[str, str]:
    """Parse option.txt into a dict of key -> value (as raw strings)."""
    opts: Dict[str, str] = {}
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("-"):
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            opts[k.strip()] = v.strip()
    return opts


def options_match(opts: Dict[str, str], ref: Dict[str, str], ignore: List[str]) -> bool:
    for k, v in ref.items():
        if k in ignore:
            continue
        if k not in opts:
            return False
        if str(opts[k]) != str(v):
            return False
    return True


def find_latest_logs(
    root: Path,
    ref_opts: Optional[Dict[str, str]],
    ignore_keys: List[str],
    prefer_max_exp_id: bool = False,
) -> List[Tuple[str, Path, Optional[Path]]]:
    """
    Return (dataset, log_path, option_path) for the newest exp_* per dataset.
    If ref_opts is provided, only consider exp_* whose option.txt matches it (ignoring ignore_keys).
    If prefer_max_exp_id is True, choose the exp_* with the largest numeric suffix instead of newest mtime.
    """
    result: List[Tuple[str, Path, Optional[Path]]] = []
    for ds_dir in root.glob("*"):
        if not ds_dir.is_dir():
            continue

        candidates: List[Tuple[float, Optional[int], Path, Optional[Path]]] = []
        for exp_dir in ds_dir.glob("exp_*"):
            opt_path = exp_dir / "option.txt"
            log_path = exp_dir / "Train_log.log"
            if not log_path.exists():
                continue

            if ref_opts is not None:
                if not opt_path.exists():
                    continue
                opts = parse_options_file(opt_path)
                if not options_match(opts, ref_opts, ignore_keys):
                    continue

            exp_num_match = re.search(r"exp_(\d+)", exp_dir.name)
            exp_num = int(exp_num_match.group(1)) if exp_num_match else None
            candidates.append(
                (log_path.stat().st_mtime, exp_num, log_path, opt_path if opt_path.exists() else None)
            )

        if candidates:
            def _sort_key(candidate: Tuple[float, Optional[int], Path, Optional[Path]]) -> Tuple:
                mtime, exp_num, *_ = candidate
                if prefer_max_exp_id:
                    return (exp_num if exp_num is not None else -1, mtime)
                return (mtime,)

            _, _, log_path, opt_path = max(candidates, key=_sort_key)
            result.append((ds_dir.name, log_path, opt_path))

    return sorted(result, key=lambda x: x[0].lower())


def main():
    parser = argparse.ArgumentParser(description="Collect AOPCR summaries (and Best Results) from Train_log.log files.")
    parser.add_argument(
        "--root",
        default="savemodel/InceptBackbone",
        type=str,
        help="Root directory containing dataset subfolders (default: savemodel/InceptBackbone)",
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
        help="If set, pick the exp_* with the largest numeric suffix instead of the newest mtime.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    ref_opts = parse_options_file(Path(args.match_options)) if args.match_options else None

    latest_logs = find_latest_logs(root, ref_opts, args.ignore_keys, prefer_max_exp_id=args.max_exp_id)
    if not latest_logs:
        msg = "No Train_log.log files found."
        if args.match_options:
            msg += " (no exp matched the given option file)"
        raise SystemExit(msg)

    for dataset, log_path, opt_path in latest_logs:
        parsed = parse_log(log_path)
        best = parsed.get("best")
        aopcr = parsed.get("aopcr")
        coverage = parsed.get("coverage")

        print(f"\n[{dataset}] {log_path.parent.name} | {log_path}")
        if opt_path and args.match_options:
            print(f"  option matched: {opt_path}")

        # Best Results
        if best:
            best_parts = []
            for key in ["F1(mi)", "F1(Ma)", "P(mi)", "P(Ma)", "R(mi)", "R(Ma)", "ROC_AUC(Ma)", "mAP(Ma)", "Acc", "Inst_Acc"]:
                if key in best:
                    best_parts.append(f"{key}={best[key]:.4f}")
            if best_parts:
                print("  Best Results: " + "  ".join(best_parts))
        else:
            print("  Best Results: (not found)")

        # AOPCR
        if aopcr:
            if aopcr["sum"] is not None:
                print(f"  AOPCR: Weighted={aopcr['weighted']:.6f}, Mean={aopcr['mean']:.6f}, Sum={aopcr['sum']:.6f}")
            else:
                print(f"  AOPCR: Weighted={aopcr['weighted']:.6f}, Mean={aopcr['mean']:.6f}")

            if aopcr["classes"]:
                print("  AOPCR per class:")
                for c in aopcr["classes"]:
                    print(
                        f"    class {c['class']:>4} | count={c['count']:>5} | AOPCR={c['aopcr']}"
                    )
        else:
            print("  AOPCR: (not found)")

        # Coverage metrics
        if coverage:
            print("  Coverage Metrics:")
            for cov in sorted(coverage, key=lambda x: x["threshold"], reverse=True):
                thr = cov["threshold"]
                print(f"    Coverage@{thr:.0%}:")
                print(f"      Expl: {cov.get('explanation_mean', 0):.4f} (w: {cov.get('explanation_weighted', 0):.4f})")
                print(f"      Rand: {cov.get('random_mean', 0):.4f} (w: {cov.get('random_weighted', 0):.4f})")
                print(f"      Gain: {cov.get('gain_mean', 0):.4f}")


if __name__ == "__main__":
    main()
