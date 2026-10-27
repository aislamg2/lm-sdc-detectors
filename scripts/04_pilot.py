"""Small pilot: per-run cost and SDC rate, to choose injections per image, LM size, and subsampling.

Action item: Run a small pilot of the injection loop and LM detectors
"""

import argparse


def main(config_path: str) -> None:
    # TODO: implement.
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/default.yaml")
    main(parser.parse_args().config)
