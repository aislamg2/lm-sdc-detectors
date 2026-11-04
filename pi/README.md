# Raspberry Pi 5 setup

Target: Pi 5, 8 GB, 4x Cortex-A76 @ 2.4 GHz, CPU only (no Hailo NPU: it cannot be
fault-injected or observed layer by layer). The classifier and the LM detectors
share the same 8 GB.

**Status: nothing here has been done yet.** All code so far has only run on a Mac. See
[docs/README.md](../docs/README.md) for full project context.

## TODO
- [ ] OS and Python version (record here so everyone matches; the project needs 3.11 or newer)
- [ ] `pip install -e .[tflite,lm]` on the Pi; note any wheels that must be built from source
- [ ] Build llama.cpp / llama-cpp-python for ARM (OpenBLAS? NEON flags?)
- [ ] Download Qwen3 0.6B and 1.7B GGUF (4-bit) into `models/`
- [ ] Confirm yes/no next-token probabilities and final hidden states are readable
      (`sdc.detectors.lm.llama_client.QwenClient`)
- [ ] Confirm PyTorchFI runs on the Pi, or decide to inject on a workstation and
      only measure overhead on the Pi. It works on a Mac with torch 2.15
- [ ] Pin package versions so the Pi and other machines match
- [ ] If the campaign runs here, try stopping and resuming `scripts/05_inject.py`
- [ ] Power measurement: inline tester model, idle-power procedure (Shehayib & Kogge, 2026)
- [ ] Thermal: cooler/fan setup, log `/sys/class/thermal/thermal_zone0/temp`
