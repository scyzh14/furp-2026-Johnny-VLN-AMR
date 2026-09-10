# Matterport3DSimulator R2R Baseline Modern Migration

Modern reproduction of the Room-to-Room (R2R) navigation baseline on NVIDIA Blackwell GPU.

---

## Why This Migration Is Needed

The original Matterport3DSimulator R2R baseline ships with prebuilt binaries
and a Docker image built for **CUDA 9.2 + PyTorch 1.1.0 + Ubuntu 18.04**.
These binaries contain GPU kernels compiled for architectures up to sm_72
(Volta). The target hardware is an **RTX 5070 Ti Laptop GPU (Blackwell,
compute capability sm_120)**, which is not supported by any CUDA version
prior to 12.8. Concretely:

- The original CUDA 9.2 container can start and even report
  `torch.cuda.is_available() == True` (the driver JIT-compiles a fallback
  SASS), but no actual kernel runs — every GPU operation raises
  `RuntimeError: no kernel image is available for execution on the device`.
- CUDA 9.2's nvcc cannot emit sm_120 code; cuDNN 7 has no Blackwell support.
- This is not a driver-version mismatch or a configuration issue — it is a
  hard architectural incompatibility. The only path forward is a modern
  CUDA runtime (≥ 12.8) and a PyTorch wheel built with cu128.

### Exploration Before Migration

Before committing to a local migration, several alternatives were explored:

1. **Cloud GPU platforms.** Most managed cloud environments do not offer
   legacy CUDA 9.2 images, and those that do run on GPU architectures
   (Turing/Ampere) that still cannot run sm_120-native code. Using a cloud
   instance with a newer CUDA stack (e.g., CUDA 11.8 + PyTorch 2.0.0)
   seemed promising, but immediately surfaced a deeper problem: the
   original R2R training logic is **structurally dependent on the MatterSim
   simulator object**. The training loop calls
   `sim.newEpisode()`, `sim.makeAction()`, `sim.getState()` to drive
   real state transitions, and the `teacher action` supervision is derived
   from the simulator's navigable-location graph at each step — not from a
   pre-computed action sequence.

2. **Pre-computing teacher actions offline.** A common shortcut in modern
   VLN repos is to pre-calculate shortest-path action labels and feed them
   as a static table, bypassing the simulator entirely. This was rejected:
   it changes the algorithm's data flow (R2RBatch → MatterSim → observation
   → feature → instruction → agent) and severs the link between the
   simulator's state machine and the supervision signal. The original
   baseline's teacher actions are **tied to live simulator state**, and
   replacing them with a static table is a Category-D (algorithm) change.

3. **Conclusion.** The only way to faithfully reproduce the original R2R
   Seq2Seq student-forcing baseline on this hardware is to perform a
   **minimal necessary migration of the software stack** — keeping the
   algorithm, data flow, simulator integration, and evaluation metrics
   completely unchanged — while rebuilding MatterSim and upgrading PyTorch
   to versions that support sm_120.

For the full record of what problems were discovered, how each was solved,
and what pitfalls were hit along the way, see
[**MIGRATION_LOG.md**](MIGRATION_LOG.md).

---

## Goal

Migrate the original Matterport3DSimulator R2R Seq2Seq student-forcing
baseline from:

- CUDA 9.2
- torch 1.1.0
- Ubuntu 18.04

to:

- RTX 5070 Ti (Blackwell sm_120)
- CUDA 12.8 container
- PyTorch 2.7.1 + cu128
- Ubuntu 22.04 Docker environment

**Principle:** Only A (environment), B (PyTorch API), and C (MatterSim
compilation) changes are permitted. Category-D (algorithm) changes are
forbidden. The original Dockerfile is preserved; no data files are
modified.

---

## Project Timeline

| Phase | Scope | Status |
|---|---|---|
| 1. Repository audit | Full codebase inspection; modification/inventory report (structure, dependencies, PyTorch & MatterSim compatibility risks); no file changes | Done |
| 2. Modern Docker environment | `Dockerfile.modern` (CUDA 12.8 + Python 3.10); PyTorch 2.7.1+cu128; GPU recognized by PyTorch on RTX 5070 Ti (sm_120) | Done |
| 3. MatterSim build | Build Python binding in container (EGL path); fix build-only issues (pybind11, OpenCV4 macro); navigation state-machine test (rendering off) | Done |
| 4. R2R environment | Verify data flow: R2R dataset → R2RBatch → MatterSim → observations → cached image features; simulator stays in the loop | Not started |
| 5. PyTorch 2.x migration | Minimal API changes only (`G.node`→`G.nodes`, `mask.byte()`→`mask.bool()`); architecture/loss/metrics unchanged | Not started |
| 6. Minimal validation | 12-step test chain: torch import → MatterSim → scan → action → R2RBatch → model forward → backward → optimizer step | Not started |
| 7. Training | Debug mode (tiny batch/iterations) → small sanity run → full R2R Seq2Seq student-forcing baseline | Not started |
| 8. Evaluation | Run `eval.py`; output baseline metrics (NE / OSR / SR / SPL) | Not started |

## Environment

| Component | Value |
|---|---|
| GPU | RTX 5070 Ti Laptop GPU (Blackwell, sm_120, 12 GB) |
| NVIDIA Driver | 595.84 |
| Container base | `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04` |
| PyTorch | 2.7.1+cu128 |
| Python | 3.10 |
| MatterSim build | EGL rendering, pybind11 v2.9.2 |

## Reproduction

```bash
# 1. Build the modern image
docker build -f Dockerfile.modern -t mattersim:modern-cu128-py310 .

# 2. Start a container with source + data mounted
docker run -d --name mattersim_build --gpus all \
  -v "$(pwd):/root/Matterport3DSimulator" \
  -v "$HOME/mp3d_data:/mp3d_data" \
  -w /root/Matterport3DSimulator \
  mattersim:modern-cu128-py310 sleep infinity

# 3. Populate pybind11 submodule (v2.9.2, supports Python 3.10)
docker exec mattersim_build git clone --depth 1 --branch v2.9.2 \
  https://gitee.com/mirrors/pybind11.git pybind11

# 4. Build MatterSim (EGL path, Python binding only)
docker exec mattersim_build bash -c \
  "cd /root/Matterport3DSimulator && mkdir -p build && cd build && \
   cmake -DEGL_RENDERING=ON .. && make -j\$(nproc) MatterSimPython"

# 5. Verify state machine (rendering off)
docker exec mattersim_build python3 tests/test_mattersim.py

# 6. R2R training (coming after phase 5-7)
# docker exec mattersim_build python3 tasks/R2R/train.py
```

Coming soon — full R2R training and evaluation instructions will be added
after PyTorch 2.x API migration (phases 5-7).
