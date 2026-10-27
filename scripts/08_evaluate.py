"""AUROC, recall at 5% FPR, bootstrap CIs, stratified breakdowns, ablations, figures.

Action item: Part I.3d / Part III ablations
"""

import argparse


def main(config_path: str) -> None:
    # TODO: implement.
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/default.yaml")
    main(parser.parse_args().config)
