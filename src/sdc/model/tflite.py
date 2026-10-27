"""TFLite (LiteRT) conversion and inference.

`convert_fp32` and `convert_int8` are ported from Reem's notebook. Conversion
runs on a workstation (needs litert-torch); inference runs on the Pi
(needs ai-edge-litert). Install with `pip install -e .[tflite]`.
"""

import copy

import numpy as np
import torch

SAMPLE_INPUTS = (torch.randn(1, 3, 224, 224),)


def convert_fp32(model: torch.nn.Module, out_path: str) -> None:
    import litert_torch

    model = model.eval().cpu()
    edge_model = litert_torch.convert(copy.deepcopy(model), SAMPLE_INPUTS)
    edge_model.export(out_path)


def convert_int8(model: torch.nn.Module, out_path: str) -> None:
    """PT2E dynamic per-channel INT8 quantization (stretch goal in the proposal)."""
    import litert_torch
    from litert_torch.quantize.pt2e_quantizer import (
        PT2EQuantizer,
        get_symmetric_quantization_config,
    )
    from litert_torch.quantize.quant_config import QuantConfig
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    model_int8 = copy.deepcopy(model).eval().cpu()
    exported_program = torch.export.export(model_int8, SAMPLE_INPUTS).module()
    pt2e_quantizer = PT2EQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True)
    )
    prepared_model = prepare_pt2e(exported_program, pt2e_quantizer)
    with torch.no_grad():
        prepared_model(*SAMPLE_INPUTS)
    quantized_torch_model = convert_pt2e(prepared_model, fold_quantize=False)
    edge_model = litert_torch.convert(
        quantized_torch_model,
        SAMPLE_INPUTS,
        quant_config=QuantConfig(pt2e_quantizer=pt2e_quantizer),
    )
    edge_model.export(out_path)


def load_interpreter(path: str):
    """Return an allocated LiteRT interpreter for `path`.

    TODO: implement with ai_edge_litert.interpreter.Interpreter.
    """
    raise NotImplementedError


def run_tflite(interpreter, x: np.ndarray) -> np.ndarray:
    """Run one NCHW float input through the interpreter and return logits.

    TODO: port `run_tflite` from the notebook (handles int8/uint8 (de)quantization).
    """
    raise NotImplementedError


def check_layer_alignment(torch_model, tflite_path: str) -> dict:
    """Verify weight faults stay attributable to their original layer after conversion.

    TFLite may fuse or restructure layers (proposal §6.1). TODO: map each PyTorch
    weight tensor to its TFLite tensor and report anything fused or missing.
    """
    raise NotImplementedError
