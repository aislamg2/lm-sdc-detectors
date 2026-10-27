# Raspberry Pi 5 setup

Target: Pi 5, 8 GB, 4x Cortex-A76 @ 2.4 GHz, CPU only (no Hailo NPU: it cannot be
fault-injected or observed layer by layer). The classifier and the LM detectors
share the same 8 GB.

## TODO
- [ ] OS and Python version (record here so everyone matches)
- [ ] `pip install -e .[tflite,lm]` on the Pi; note any wheels that must be built from source
- [ ] Build llama.cpp / llama-cpp-python for ARM (OpenBLAS? NEON flags?)
- [ ] Download Qwen3 0.6B and 1.7B GGUF (4-bit) into `models/`
- [ ] Confirm yes/no next-token probabilities and final hidden states are readable
      (`sdc.detectors.lm.llama_client.QwenClient`)
- [ ] Confirm PyTorchFI runs on the Pi, or decide to inject on a workstation and
      only measure overhead on the Pi
- [ ] Power measurement: inline tester model, idle-power procedure (Shehayib & Kogge, 2026)
- [ ] Thermal: cooler/fan setup, log `/sys/class/thermal/thermal_zone0/temp`
