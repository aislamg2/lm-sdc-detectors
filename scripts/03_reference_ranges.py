"""Compute per-layer normal ranges from clean fitting-set runs.

Action item: Compute normal ranges from the fitting set
"""

import argparse


def main(config_path: str) -> None:
    # TODO: implement.
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/default.yaml")
    main(parser.parse_args().config)
