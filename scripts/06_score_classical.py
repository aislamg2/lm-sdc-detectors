"""Fit classical detectors on the fitting set, freeze thresholds, score both sets.

Action item: Part I.3b Implement the classical detectors
"""

import argparse


def main(config_path: str) -> None:
    # TODO: implement.
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/default.yaml")
    main(parser.parse_args().config)
