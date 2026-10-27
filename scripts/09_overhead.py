"""Measure each detector's latency, memory, and energy on the Pi (separate session).

Action item: Part I.3c / Part II.2c
"""

import argparse


def main(config_path: str) -> None:
    # TODO: implement.
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/default.yaml")
    main(parser.parse_args().config)
