"""Export the trained ResNet-18 checkpoint to FP32 (and INT8) TFLite.

Action item: model prep (ported conversion in sdc.model.tflite)
"""

import argparse


def main(config_path: str) -> None:
    # TODO: implement.
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/default.yaml")
    main(parser.parse_args().config)
