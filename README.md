# lm-sdc-detectors
Comparing small language models against classical detectors for catching silent data corruption (bit-flip faults) in a ResNet-18 EuroSAT classifier on a Raspberry Pi 5.

Reem Shehayib, Micah Brody, Arvin Islam-Gomes (Notre Dame)

> **Status: in progress, no real results yet.** The fault-injection pipeline through the campaign loop is built and tested on fake data, but hasn't run on real data or on the Pi. Detectors, evaluation, and overhead measurement are still stubs that raise `NotImplementedError`.
>
> **Start with [docs/README.md](docs/README.md)** for full project context: what is built, how it fits together, decisions, known issues, and what is next.

## Setup
```bash
python3.12 -m venv .venv && source .venv/bin/activate   # Python 3.11 or newer
pip install -e ".[dev]"
pytest                                                  # no real data needed
```

The optional extras add TFLite and llama.cpp (`pip install -e ".[tflite,lm]"`). They aren't needed yet and haven't been tested here.

Data, checkpoints, models, and run outputs are gitignored. Put them at the paths in `configs/default.yaml`:
- `data/EuroSAT_RGB/`: unzipped EuroSAT RGB, the same zip Reem trained on (**still needed**)
- `checkpoints/resnet18_eurosat.pt`: trained weights as a plain state_dict (**still needed from Reem**)
- `checkpoints/test_indices.json`: optional list of test-split indices (**still needed from Reem**); without it, the seed-42 split is reproduced
- `models/*.tflite`, `models/*.gguf`

## Layout
| Path | What | Planning item | Status |
|---|---|---|---|
| `src/sdc/data/eurosat.py` | transforms, held-out test split (seed 42), correct-image filter, fit/eval split | Part I.1 | Done |
| `src/sdc/model/` | ResNet-18, checkpoint loading, TFLite conversion, hooks on the 6 monitored points | | Done, except TFLite: conversion ported but never run, inference is a stub |
| `src/sdc/records/` | run-record schema (frozen at version 1, awaiting team review), Parquet shard I/O | Freeze run-record format | Done |
| `src/sdc/injection/` | FP32 bit flips, weight faults (in place), activation faults (PyTorchFI), labeling, resumable campaign | Part I.2 | Done |
| `src/sdc/observations/` | black-box / white-box stats, reference ranges, text serialization for prompts | Part I.3a | Done, except serialization and `feature_vector` |
| `src/sdc/detectors/classical/` | sanity, confidence (entropy), range, quantile, learned (logreg) | Part I.3b | Stub |
| `src/sdc/detectors/lm/` | llama.cpp client, prompts, generative, single-pass, probe, optional API reference | Part II | Stub |
| `src/sdc/eval/` | AUROC, recall@5%FPR, bootstrap CIs over images, stratification, ablations, figures | Part I.3d, Part III | Stub |
| `src/sdc/overhead/` | latency, memory, thermal, energy, analytical LM cost model | Part I.3c, II.2c | Stub |
| `scripts/00–09` | pipeline entry points in run order | | `01`, `02`, `03`, `05` done; the rest are stubs |
| `pi/` | Raspberry Pi 5 setup checklist | Part II.1 | Nothing done yet |
| `notebooks/reem_TFLite_EuroSAT_ResNet18.ipynb` | Reem's original training/conversion notebook (reference) | | Reference only |

## Pipeline
```
00_export_models   -> 01_make_splits -> 02_clean_runs -> 03_reference_ranges
-> 04_pilot -> 05_inject (fit, eval) -> 06_score_classical -> 07_score_lm
-> 08_evaluate -> 09_overhead (separate Pi session)
```

`05_inject` can be stopped at any time. Run the same command again to resume.

## Owners (from the proposal)
- Reem & Arvin: fault injection, classical detectors, overhead measurement
- Micah: language-model detectors

## Credits
Model training and TFLite conversion code adapted from
[reemshehayib/TFLite-Deployment-of-ResNet18-on-EuroSAT-Dataset](https://github.com/reemshehayib/TFLite-Deployment-of-ResNet18-on-EuroSAT-Dataset).
