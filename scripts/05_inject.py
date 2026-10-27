"""Run the resumable PyTorchFI campaign for one split, writing Parquet shards.

Action item: Part I.2 Fault-injection pipeline (stop/resume between Pi sessions)
"""

import argparse


def main(config_path: str) -> None:
    # TODO: implement.
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/default.yaml")
    main(parser.parse_args().config)
