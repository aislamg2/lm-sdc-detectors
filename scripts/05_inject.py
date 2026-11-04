"""Run the resumable fault-injection campaign, writing Parquet shards.

Action item: Part I.2 Fault-injection pipeline (stop/resume between Pi sessions)

Stop at any time with Ctrl-C, then run the same command again to continue: only the
shard in progress is lost. Shards go to <paths.fault_runs>/fit and /eval.
"""

import argparse
import sys
from pathlib import Path

from sdc.config import load_config
from sdc.injection.campaign import run_campaign
from sdc.records.io import load_runs


def summarize(split: str, out_dir: Path) -> None:
    runs = load_runs(out_dir)
    counts = runs["label"].value_counts().to_dict()
    reasons = runs["crash_reason"].dropna().value_counts().to_dict()
    print(f"{split}: {len(runs)} runs saved; masked={counts.get('masked', 0)} "
          f"sdc={counts.get('sdc', 0)} crash={counts.get('crash', 0)} {reasons or ''}")


def main(config_path: str | None, split: str, max_shards: int | None) -> None:
    config = load_config(config_path)
    root = Path(config["paths"]["fault_runs"])
    for name in (["fit", "eval"] if split == "all" else [split]):
        status = run_campaign(config, name, root / name, max_shards)
        print(f"{name}: wrote {status['written']}, already had {status['skipped']}, "
              f"{status['remaining']} of {status['total_shards']} shards remaining")
        summarize(name, root / name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="YAML overlay on configs/default.yaml")
    parser.add_argument("--split", choices=["fit", "eval", "all"], default="all")
    parser.add_argument("--max-shards", type=int, default=None,
                        help="stop after this many new shards per split")
    args = parser.parse_args()
    try:
        main(args.config, args.split, args.max_shards)
    except KeyboardInterrupt:
        sys.exit("interrupted: run the same command again to resume")
