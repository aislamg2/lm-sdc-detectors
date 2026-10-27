# lm-sdc-detectors
Comparing small language models against classical detectors for catching silent data corruption (bit-flip faults) in a ResNet-18 EuroSAT classifier on a Raspberry Pi 5.

Reem Shehayib, Micah Brody, Arvin Islam-Gomes (Notre Dame)

> **Status: skeleton.** Most functions are stubs that raise `NotImplementedError` with a TODO describing what to build. The ported pieces from Reem's notebook (transforms, test split, model definition, TFLite conversion) are real code.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # core
pip install -e ".[tflite,lm]"      # TFLite conversion/inference, llama.cpp
pytest
```

Data, checkpoints, models, and run outputs are gitignored. Put them at the paths in `configs/default.yaml`:
- `data/EuroSAT_RGB/`: unzipped EuroSAT RGB
- `checkpoints/resnet18_eurosat.pt`: trained weights (**still to export from Reem's Colab run**)
- `models/*.tflite`, `models/*.gguf`

## Layout
| Path | What | Planning item |
|---|---|---|
| `src/sdc/data/eurosat.py` | transforms, held-out test split (seed 42), correct-image filter, fit/eval split | Part I.1 |
| `src/sdc/model/` | ResNet-18, checkpoint loading, TFLite conversion, hooks on the 6 monitored points | |
| `src/sdc/records/` | run-record schema (**draft; freeze it**), Parquet shard I/O | Freeze run-record format |
| `src/sdc/injection/` | FP32 bit flips, PyTorchFI weight/activation faults, labeling, resumable campaign | Part I.2 |
| `src/sdc/observations/` | black-box / white-box stats, reference ranges, text serialization for prompts | Part I.3a |
| `src/sdc/detectors/classical/` | sanity, confidence (entropy), range, quantile, learned (logreg) | Part I.3b |
| `src/sdc/detectors/lm/` | llama.cpp client, prompts, generative, single-pass, probe, optional API reference | Part II |
| `src/sdc/eval/` | AUROC, recall@5%FPR, bootstrap CIs over images, stratification, ablations, figures | Part I.3d, Part III |
| `src/sdc/overhead/` | latency, memory, thermal, energy, analytical LM cost model | Part I.3c, II.2c |
| `scripts/00–09` | pipeline entry points in run order | |
| `pi/` | Raspberry Pi 5 setup checklist | Part II.1 |
| `notebooks/reem_TFLite_EuroSAT_ResNet18.ipynb` | Reem's original training/conversion notebook (reference) | |

## Pipeline
```
00_export_models   -> 01_make_splits -> 02_clean_runs -> 03_reference_ranges
-> 04_pilot -> 05_inject (fit, eval) -> 06_score_classical -> 07_score_lm
-> 08_evaluate -> 09_overhead (separate Pi session)
```

## Owners (from the proposal)
- Reem & Arvin: fault injection, classical detectors, overhead measurement
- Micah: language-model detectors

## Credits
Model training and TFLite conversion code adapted from
[reemshehayib/TFLite-Deployment-of-ResNet18-on-EuroSAT-Dataset](https://github.com/reemshehayib/TFLite-Deployment-of-ResNet18-on-EuroSAT-Dataset).
