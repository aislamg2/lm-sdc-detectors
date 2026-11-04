# Project status and context

**Last updated: 2026-11-04, at commit `2091728` on `main`, which includes the campaign loop.**

This is the single place to get up to speed on `lm-sdc-detectors`: what the project is, what has been built, how it fits together, what was decided and why, and what is left. It is written for teammates and for AI agents picking up the work. If something here disagrees with the code, trust the code and fix this file.

## Contents
1. [Current status in brief](#1-current-status-in-brief)
2. [The project](#2-the-project)
3. [Status against the plan](#3-status-against-the-plan)
4. [How the pipeline fits together](#4-how-the-pipeline-fits-together)
5. [What is implemented](#5-what-is-implemented)
6. [Data formats](#6-data-formats)
7. [Decisions and why](#7-decisions-and-why)
8. [Differences from the proposal](#8-differences-from-the-proposal)
9. [Known issues and things to watch](#9-known-issues-and-things-to-watch)
10. [Open questions for the team](#10-open-questions-for-the-team)
11. [Setup, running, and testing](#11-setup-running-and-testing)
12. [What is next](#12-what-is-next)
13. [Notes for AI agents](#13-notes-for-ai-agents)

---

## 1. Current status in brief

- **No real results exist yet.** Everything has been tested only on fake images, made-up records, and randomly initialized ResNet-18 weights, on a Mac. Nothing has run on a Raspberry Pi.
- **The main blocker is three files from Reem:** the trained checkpoint, `test_indices.json`, and the EuroSAT zip used for training. See [section 9](#9-known-issues-and-things-to-watch).
- **Built and tested:** bit flips, run labeling, observation stats, activation monitoring, checkpoint and config loading, the frozen run-record schema, Parquet shard I/O, image splits, clean runs, reference ranges, weight and activation fault injection, and the resumable campaign loop.
- **The real campaign has not been run.** It needs Reem's files, real splits and clean runs, and the pilot's choice of `injections_per_image`.
- **Not built yet:** the pilot, all detectors, evaluation metrics, and overhead measurement.
- **Tests:** 461 passing.

---

## 2. The project

**Paper:** "Can Small Language Models Detect Silent Data Corruption on the Edge? An Evaluation on an Embedded CNN."

**Team and ownership (from the proposal):**
- Reem Shehayib and Arvin Islam-Gomes: fault injection, classical detectors, overhead measurement.
- Micah Brody: language-model detectors.

**The problem.** Hardware faults such as radiation-induced bit flips can corrupt a neural network during inference. Some faults change nothing (masked), some cause crashes or NaN/inf outputs (easy to notice), and some silently change the prediction. The last kind is a **silent data corruption (SDC)**, and it's the dangerous one.

**The question.** Given identical observations, can a small language model detect SDCs as well as classical detectors, at an acceptable cost on edge hardware?

**The setup.**
- **Monitored model:** ResNet-18 with about 11 million parameters, trained in PyTorch on EuroSAT (10 land-cover classes, 224x224 inputs, ImageNet normalization). It comes from Reem's prior work.
- **Application:** on-board land-cover classification for an Earth-observation satellite.
- **Hardware:** Raspberry Pi 5 CPU (4x Cortex-A76 at 2.4 GHz, 8 GB). The Hailo NPU is out of scope because it can't be fault-injected or observed layer by layer.
- **Faults:** one flipped bit per run, in a weight (persistent) or an activation (transient), stratified by sign, exponent, and mantissa bits.
- **Labels:** each faulty run is compared with the fault-free run on the same image.
  - `masked`: finite outputs, prediction unchanged.
  - `sdc`: finite outputs, prediction changed.
  - `crash`: error, timeout, or non-finite output.

**Research questions.**
- **RQ1:** Given the same observations, how does language-model detection quality compare to classical detectors?
- **RQ2:** What latency, memory, and energy overhead does each detector add on a Pi 5?
- **RQ3:** Which faults does each detector miss, by fault site, bit position, and layer?
- **RQ4:** How do model internals, reference information, generated reasoning, and a trained decision head affect the language-model detectors?

**Detectors.** Each gives a run a score, where higher means more likely an SDC.
- **Classical:** sanity check, confidence (entropy), range check, quantile monitor, and a learned logistic regression.
- **Language model** (Qwen3 at 0.6B or 1.7B parameters, 4-bit, in llama.cpp):
  - **Generative:** thinking mode, then reads yes/no probabilities.
  - **Single-pass:** direct answer, reads the next-token yes/no probabilities.
  - **Probe:** logistic regression on the final hidden state.
- **Stretch goals:** final-layer checksums, redundant execution, and INT8.

**Access levels.**
- **Black-box:** top-1 softmax probability and entropy.
- **White-box:** black-box plus min, max, mean, std, and non-finite count at 6 monitored points: `stem` (maxpool output), `layer1`, `layer2`, `layer3`, `layer4`, and `fc` (logits).

**Prompt conditions.** `zero_shot` (observation only) and `reference` (observation plus normal ranges from fault-free runs).

**Metrics.** The main task is SDC versus masked, with crashes reported separately. The metrics are AUROC and SDC recall at 5% false-positive rate (threshold frozen on the fitting set), with 95% bootstrap confidence intervals resampled over images. Results are broken down by fault site, bit group, and layer. Overhead (latency, peak memory, energy) is always measured on the Pi.

**Schedule from the proposal:** main injection campaign Nov 2–13, language-model detectors Nov 9–20, overhead Nov 16–25, analysis and poster Nov 30–Dec 4, poster Dec 7, final paper Dec 11.

---

## 3. Status against the plan

### Items due Oct 30

| Item | Status | What is left |
|---|---|---|
| Set up the shared GitHub repo | Done | Nothing |
| Freeze the run-record format | Frozen at version 1 | Reem and Micah still need to review it |
| Split images into fitting and evaluation sets | Code done | Run on real data |
| Run clean inference, keep correct images | Code done | Run on real data |
| Compute normal ranges from the fitting set | Code done | Run on real data |
| Build the resumable injection loop | Code done | Tested with `kill -9` on fake data; not yet run on real data or on the Pi |

### Later items

None of these have started: Qwen3 on the Pi, literature, pilot, full campaign, classical and language-model detectors, optional hosted-API reference detector, threshold freezing, evaluation scoring, breakdowns, ablations, Pi overhead measurement, figures, poster, and final paper.

Stub files with docstrings already exist for most of the code.

---

## 4. How the pipeline fits together

Scripts run in number order. `[done]` means implemented and tested on fake data. `[stub]` means it still raises `NotImplementedError`.

```text
Reem's files: checkpoint, test_indices.json, EuroSAT
        |
00_export_models      [stub]  TFLite conversion for overhead measurement
        |
01_make_splits        [done]  -> runs/splits.json
        |
02_clean_runs         [done]  -> runs/clean/fit.parquet, eval.parquet
        |
03_reference_ranges   [done]  -> runs/reference_ranges.json   (fit data only)
        |
04_pilot              [stub]  sets injections per image, LM size, subsampling
        |
05_inject             [done]  -> runs/faults/fit, eval: Parquet shards, campaign.json
        |
06_score_classical    [stub]  -> runs/scores/
07_score_lm           [stub]  -> runs/scores/
        |
08_evaluate           [stub]  -> results/
        |
09_overhead           [stub]  separate session on the Pi
```

**One faulty run** (`run_one` in `campaign.py`):
1. `sample_fault` picks the layer, element, and bit.
2. `weight_fault` or `ActivationInjector.inject` applies the fault.
3. The model runs on one image with `ActivationMonitor` attached.
4. `blackbox_stats` and `whitebox_stats` build the observation.
5. `label_run` assigns `masked`, `sdc`, or `crash`.
6. A `RunRecord` is built, and `write_shard` saves batches of them.

**The campaign** (`run_campaign`) verifies its inputs, builds the deterministic plan, checks `campaign.json`, then runs each unfinished shard. Before saving a shard, it checks that the weights are bit-for-bit unchanged.

---

## 5. What is implemented

Tests are in `tests/`, in a file named after the module.

| Module | What it provides | Tests |
|---|---|---|
| `src/sdc/config.py` | `load_config(path=None)`: reads `configs/default.yaml`, deep-merges an overlay, rejects unknown keys, and resolves `paths` from the repo root | 20 |
| `src/sdc/injection/bitflip.py` | `bit_group_of`, `sample_bit`, `flip_bit_fp32`. Bit 31 is sign, 30–23 exponent, 22–0 mantissa | 237 |
| `src/sdc/injection/labeling.py` | `label_run(clean_pred, faulty_logits, crashed)` | 12 |
| `src/sdc/injection/faults.py` | `injectable_layers`, `weight_shapes`, `activation_shapes`, `sample_fault`, `weight_fault`, `ActivationInjector` | 34 |
| `src/sdc/injection/campaign.py` | `plan_faults`, `plan_hash`, `run_one`, `run_campaign`, `time_limit`, `weights_fingerprint`, `current_git_commit` | 35 |
| `src/sdc/model/resnet18.py` | `build_resnet18`, `load_checkpoint`, `checkpoint_sha256` | 10 |
| `src/sdc/model/monitor.py` | `MONITORED_POINTS`, `ActivationMonitor` (forward hooks, removed even after a crash) | 14 |
| `src/sdc/observations/stats.py` | `blackbox_stats`, `layer_stats`, `whitebox_stats`, `reference_ranges` | 17 + 14 |
| `src/sdc/observations/clean.py` | `clean_run_row`, `run_clean`, `load_clean_runs`, `CLEAN_ARROW_SCHEMA` | 16 |
| `src/sdc/records/schema.py` | Frozen `RunRecord`, `LayerStats`, and the `Label`, `CrashReason`, `FaultSite`, `BitGroup` enums | 5 |
| `src/sdc/records/io.py` | `record_to_row`, `write_shard`, `write_table_atomic`, `completed_shards`, `load_runs`, `ARROW_SCHEMA` | 21 |
| `src/sdc/data/eurosat.py` | `load_test_split`, `select_correct_images`, `balanced_sample`, `fit_eval_split`, `build_splits`, `load_splits`, `verify_splits` | 25 |
| `scripts/01_make_splits.py` | Writes `runs/splits.json` | In `test_eurosat.py` |
| `scripts/02_clean_runs.py` | Writes the clean Parquet files | In `test_clean_runs.py` |
| `scripts/03_reference_ranges.py` | Writes `runs/reference_ranges.json` | In `test_reference_ranges.py` |
| `scripts/05_inject.py` | Runs the campaign and prints label counts. Options: `--split fit\|eval\|all`, `--max-shards N` | In `test_campaign.py` |
| `tests/test_imports.py` | Smoke test: every module imports | 1 |

**Ported from Reem's notebook.** The transforms, the seed-42 split, and the model definition are in use and tested. `convert_fp32` and `convert_int8` in `src/sdc/model/tflite.py` were ported but **have never been run** in this repo. The original notebook is at `notebooks/reem_TFLite_EuroSAT_ResNet18.ipynb`, from [reemshehayib/TFLite-Deployment-of-ResNet18-on-EuroSAT-Dataset](https://github.com/reemshehayib/TFLite-Deployment-of-ResNet18-on-EuroSAT-Dataset).

**Still stubs:**
- `src/sdc/observations/serialize.py`, plus `feature_vector` in `stats.py`
- `src/sdc/detectors/` (all of it, including `freeze_threshold` in `base.py`)
- `src/sdc/eval/` and `src/sdc/overhead/`
- `load_interpreter`, `run_tflite`, and `check_layer_alignment` in `src/sdc/model/tflite.py`
- `scripts/00`, `04`, `06`, `07`, `08`, and `09`

---

## 6. Data formats

Everything under `data/`, `checkpoints/`, `models/`, `runs/`, and `results/` is gitignored. These locations come from `configs/default.yaml`.

### Input files
| Path | Contents |
|---|---|
| `data/EuroSAT_RGB/` | Unzipped EuroSAT, one folder per class. The folder names must match `CLASS_NAMES` |
| `checkpoints/resnet18_eurosat.pt` | A plain state_dict, saved with `torch.save(model.state_dict(), path)` |
| `checkpoints/test_indices.json` | Optional. Assumed to be a JSON list of full-dataset indices. If absent, the seed-42 split is reproduced |

### `runs/splits.json`
```json
{
  "seed": 42,
  "num_test_images": 2700,
  "num_correct": 2500,
  "clean_accuracy": 0.926,
  "per_class": 200,
  "fit": [{"image_id": 0, "image_path": "Forest/Forest_1.jpg", "true_class": 1}],
  "eval": [{"image_id": 1, "image_path": "River/River_2.jpg", "true_class": 8}],
  "split_source": "test_indices",
  "checkpoint_sha256": "..."
}
```
The numbers above only show the shape; they aren't results. **`image_id` is an image's position within the test split, not within the full dataset.** `split_source` is `test_indices` or `reproduced_seed_split`.

### `runs/clean/fit.parquet` and `eval.parquet`
One row per image, with these columns:
- `image_id`, `image_path`, `true_class`, `clean_pred`
- `clean_logits` (10 values), `top1_prob`, `entropy`
- For each monitored point: `{point}_min`, `_max`, `_mean`, `_std`, `_nonfinite`

### `runs/reference_ranges.json`
```json
{
  "checkpoint_sha256": "...",
  "num_fit_images": 1000,
  "quantile_low": 0.001,
  "quantile_high": 0.999,
  "points": {
    "stem": {"min": 0.0, "max": 1.5, "quantile_low": 0.0, "quantile_high": 1.4}
  }
}
```
- `min` and `max` are `a_l` and `b_l` in the proposal's Eq. 1: the lowest per-run min and highest per-run max across clean fitting runs.
- `quantile_low` is the 0.1% quantile of per-run mins, and `quantile_high` is the 99.9% quantile of per-run maxes.

### Fault runs: `runs/faults/fit/` and `runs/faults/eval/`
Each folder holds `campaign.json` and `shard_000123.parquet` files. Shard `k` always covers the same runs.

`campaign.json` records `schema_version`, `split`, `seed`, `injections_per_image`, `sites`, `bit_groups`, `shard_size`, `checkpoint_sha256`, `plan_hash`, `num_runs`, and `num_shards`. If any of these differ on resume, the campaign stops.

Shards are written with `write_shard`. These are the frozen `RunRecord` fields (`SCHEMA_VERSION = 1`) with `layer_stats` flattened into columns:

| Group | Fields |
|---|---|
| Identity | `run_id`, `plan_idx`, `seed`, `split`, `image_id`, `image_path`, `true_class` |
| Fault | `fault_site`, `layer`, `element`, `bit_position`, `bit_group`, `original_value`, `flipped_value` |
| Outcome | `clean_pred`, `faulty_pred`, `label`, `crash_reason`, `faulty_logits` |
| Black-box | `top1_prob`, `entropy` |
| Reproducibility | `checkpoint_sha256`, `git_commit`, `device`, `timestamp`, `inference_time_s` |
| White-box | `{point}_min`, `_max`, `_mean`, `_std`, `_nonfinite` for each monitored point |
| Bookkeeping | `model_format`, `schema_version` |

**Reading the values:**
- `crash_reason` is `exception`, `timeout`, `nonfinite`, or empty.
- **NaN and empty mean different things.** NaN means a value was computed and was bad. Empty means no value exists, for example at a point a crashed run never reached. `load_runs` and `load_clean_runs` return Arrow-backed columns that keep them apart, so `isna()` is true only for empty values.
- For scikit-learn and NumPy, convert with `.to_numpy(dtype=float)`.
- `git_commit` ends in `-dirty` if the code had uncommitted changes. Commit before a real campaign so results can be traced to exact code.

**Kept out of the run record on purpose:** clean runs (their own table, joined on `image_id`), detector scores (their own table, joined on `run_id`), and raw activations and LM hidden states (too large).

---

## 7. Decisions and why

| Decision | Reason |
|---|---|
| Faults are injected on PyTorch, not LiteRT | Activation faults can't be applied inside `invoke()`, and conversion can merge layers, so a LiteRT weight flip may not be the same fault. LiteRT is for overhead measurement |
| Sample a layer uniformly, then an element within it | The proposal specifies it, and every layer gets enough faults for per-layer results |
| Target the 21 Conv2d and Linear layers | They hold 11,172,032 of 11,181,642 parameters. Batch norm parameters and the Linear bias are skipped |
| Weight faults flip in place and restore in `finally` | Avoids copying the model per fault. The restore also runs after a crash, and tests compare weights bit by bit |
| Activation faults use PyTorchFI with a custom hook | The proposal names PyTorchFI. The hook checks `module is target` because PyTorchFI attaches it to all 21 layers |
| Bit flips use the raw 32 bits in NumPy | Converting through a 64-bit Python float can alter NaN bits |
| Crash check runs before `argmax` in `label_run` | `argmax` still returns a class when logits contain NaN, which would mislabel crashes as SDC or masked |
| Stats are computed in float64 on the CPU | When several values near `3e38` add up, the float32 mean and std overflow to `inf` and finite activations look broken. A single huge value doesn't overflow |
| Entropy uses `log_softmax`, in nats | Stays accurate for very confident outputs |
| Standard deviation is the population version | The activations are the whole layer, not a sample |
| Save 10 raw logits, not probabilities | Probabilities can be computed from logits, but very confident outputs can't be converted back |
| Parquet shards with one fixed layout | A shard with no crashes still has typed columns, so all shards combine cleanly |
| Shards are written to a temp file, synced, then renamed | A shard either exists completely or not at all, which is what makes resuming safe |
| Each fault has its own random generator, seeded by its position in the plan | The plan is repeatable, and one image's faults don't depend on the others |
| Exceptions during inference become crash records, but errors while setting up a fault stop the campaign | A fault can cause the first, while the second is a bug. A shard where every run raises an exception also stops |
| The time limit sits inside `weight_fault` | The alarm can't interrupt the weight restore |
| Results files are never overwritten | Later results depend on them. Delete a file by hand to rebuild it |
| Reference ranges read only `fit.parquet` | Using eval data would leak it into the detectors. A test enforces this |
| Each image is listed with its path, class, and checkpoint hash | Catches a different dataset or different weights instead of silently using them |
| Quantile monitor bounds come from existing per-run stats | Needs no new fields, and every detector keeps seeing the same observation |
| `.gitignore` folder rules are anchored, such as `/data/` | An unanchored `data/` also matched `src/sdc/data/`, which kept that package out of the first commit |
| No `torch_tpu` or `tpu-agent-suite` | They target Google Cloud TPUs, while this project measures cost on a Pi CPU. A later TPU side project would mainly replace `faults.py`, but activation faults on a TPU are untested |

---

## 8. Differences from the proposal

These need to be reflected in the paper, or changed if the team disagrees.

1. **Weight faults don't go through PyTorchFI.** They're flipped directly in place. Activation faults do use PyTorchFI.
2. **The quantile monitor** is in the planning notes but not the proposal. Its definition here is proposed and unconfirmed.
3. **Overall SDC rates aren't hardware-realistic.** Sampling evenly across layers means a small layer gets as many faults as a large one, while a real hit is more likely to land in a large layer. Per-layer results are unaffected, but the paper should say this.
4. **Target layers** exclude batch norm parameters and the Linear bias.

---

## 9. Known issues and things to watch

**Blocking:**
- **The trained weights may no longer exist.** The notebook's only `torch.save` writes a temporary `temp.pt` and deletes it. Reem may need to retrain by re-running the notebook. The split stays the same because it's seeded, but **the TFLite files must then be regenerated from the new weights**, or injection and overhead would use different models.
- **Get the same EuroSAT zip Reem used.** The split depends on the dataset's file list.

**Not yet verified:**
- **PyTorchFI on the Pi.** Version 0.6.0 works on torch 2.15 on a Mac, for activation injection through our custom hook. Its own weight injection was never tried, because we don't use it.
- **Real power loss.** `kill -9` on fake data lost nothing and resumed to identical results, except for time fields. That stops the process but isn't a real power cut, and that test's kill landed just after a shard finished.
- **Timeouts on real hangs.** `signal.setitimer` is tested against a slow Python hook. It can't interrupt a hang inside one compiled operation, and it only works in the main thread on macOS and Linux.
- **Per-fault cost on the Pi.** PyTorchFI copies the model per activation fault, which took about 9 ms on a Mac. The pilot will measure the Pi.
- **The format of `test_indices.json`.** A plain list of full-dataset indices is assumed. A list of the wrong kind of numbers wouldn't be caught.
- **The TFLite conversion code**, which has never been run here.

**Good to know:**
- **Dramatic faults can still be masked.** A weight flipped to about `1.1e37` changed nothing in a test, because the input channel it multiplies was entirely zero. Negative activations can similarly be erased by ReLU. A wild `flipped_value` with a `masked` label isn't necessarily a bug.
- **Batch versus single-image rounding.** Step 3 picks correct images in batches of 32, and step 4 runs one at a time. An image on a decision boundary could flip and stop `02_clean_runs.py` with a `clean prediction` error. If that isn't caused by wrong weights, build the splits with batch size 1.
- **Sanity-check `clean_accuracy`.** It should be high on real data. Near 10% means the wrong weights or a class mismatch.
- **Forward hooks must return `None`.** A hook that returns a value replaces the layer's output.
- **`.numpy()` shares memory with the tensor.** Copy it before modifying the tensor. This caused a real bug in `faults.py`, now fixed.
- **Odd class sizes round down.** With 5 images per class, fit gets 2 and eval gets 3. An even `images_per_class` gives an exact 50/50.

**Housekeeping:**
- **Reem's training had no augmentation.** All three subsets share one dataset object in the notebook, so the eval transform also applied during training. This doesn't affect this repo's split, but Reem may want to know.
- **Reem's repo has no license file.** Get an OK from Reem for reusing the code.
- **Package versions aren't pinned** in `pyproject.toml`. Pin them before the main campaign so all runs use the same software.
- **Scripts `00`, `04`, and `06`–`09` still use a relative `--config` default.** Scripts `01`–`03` and `05` use `None`, and the rest should match when they're implemented.
- **The config setting was renamed:** `lm.gguf_path` is now `paths.qwen_gguf`. Micah should know.

---

## 10. Open questions for the team

1. Does everyone agree with the frozen schema (`src/sdc/records/schema.py`)? It's cheap to change only before the campaign starts.
2. Is direct weight flipping acceptable, as long as the paper describes it?
3. Is the quantile monitor definition right? The bounds would be the 0.1% and 99.9% quantiles of each layer's per-run min and max across clean fitting runs.
4. Do the original weights still exist, or does the model need retraining?
5. Where will the main campaign run: the Pi or a faster machine? The proposal allows detection results from another machine, but overhead must be measured on the Pi.
6. Is the Pi set up, and does someone have the inline power tester?
7. Which Python version does the Pi run? The project requires 3.11 or newer.

---

## 11. Setup, running, and testing

```bash
python3.12 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -e ".[dev]"
```

The `tflite` and `lm` extras in `pyproject.toml` add LiteRT and llama.cpp. They aren't needed for current work and haven't been installed or tested here.

**Run the tests** (no real data needed; takes about 15 seconds on a Mac):

```bash
pytest
```

**Environment the tests passed on:** macOS, Python 3.12.12, torch 2.15.0, torchvision 0.29.2, pytorchfi 0.6.0, numpy 2.5.4, pandas 3.1.0, pyarrow 26.0.1, scikit-learn 1.9.1.

**Run the pipeline** once Reem's files are in place:

```bash
python scripts/01_make_splits.py
```

```bash
python scripts/02_clean_runs.py
```

```bash
python scripts/03_reference_ranges.py
```

The campaign comes after the pilot, once `injection.injections_per_image` is set:

```bash
python scripts/05_inject.py
```

Stop it at any time and run the same command to resume.

Each script takes `--config some_overlay.yaml` to override settings in `configs/default.yaml`. Unknown keys in an overlay raise an error, so typos can't pass silently.

---

## 12. What is next

**Now, no real data needed:**
1. Build the classical detectors and evaluation metrics, tested on fake records.

**When Reem's files arrive:**
2. Run scripts `01`–`03` and check `clean_accuracy`. Runs made with random weights must be thrown away.
3. Write and run the pilot (`scripts/04_pilot.py` is still a stub), set `injections_per_image`, and choose the language-model size.
4. Get team sign-off on [section 10](#10-open-questions-for-the-team), then run the full campaign.

**In parallel:**
- Micah: Qwen3 in llama.cpp on the Pi, confirming that yes/no probabilities and hidden states can be read.
- Reem and Arvin: Pi setup and the power-measurement procedure (`pi/README.md` has a checklist).

---

## 13. Notes for AI agents

**Read first:** this file, then `configs/default.yaml`, `src/sdc/records/schema.py`, and the module you're changing.

**Do:**
- Run `pytest` before and after changes, using `.venv/bin/python -m pytest` if the environment isn't active.
- Match the existing style: small functions, short docstrings, and errors that say what to fix.
- Treat `tests/` as the spec of current behavior.
- Update this file when status or decisions change, including the commit at the top.

**Don't:**
- Push, commit, or start long runs unless the person you're working with asks. Arvin wants to review and push changes personally.
- Change `RunRecord` without bumping `SCHEMA_VERSION` and telling the team. `tests/test_schema.py` fails if the fields change.
- Use eval data to fit, tune, or set thresholds for any detector.
- Report or keep any output from random weights or fake images as a result.
- Commit data, checkpoints, models, or anything in `runs/` or `results/`.
- Assume a stub works. Check for `NotImplementedError`.

**Terms used in the code:**
- **Run:** one inference on one image with one flipped bit.
- **Campaign:** the full experiment across all planned faults.
- **Shard:** one Parquet file holding a batch of runs.
- **Fit / eval:** disjoint image sets. Fit is for tuning, and eval is the only source of reported results.
- **Observation:** what a detector sees for a run, at black-box or white-box level.
- **Monitored point:** one of the 6 places where activations are summarized.
- **Reference ranges:** normal per-layer values from clean fitting runs.
