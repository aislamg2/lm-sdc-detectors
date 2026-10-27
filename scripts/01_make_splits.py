"""Load the held-out test split, keep correctly classified images, balance classes, split fit/eval.

Action item: Part I.1 Clean runs; split images into fitting and evaluation sets
"""

import argparse


def main(config_path: str) -> None:
    # TODO: implement.
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/default.yaml")
    main(parser.parse_args().config)
