# Quickstart: Run the R2R Baseline

This guide walks through the entire pipeline: from building the Docker
image to training and evaluating the R2R Seq2Seq student-forcing baseline.

Full migration details are in [MIGRATION_LOG.md](experiments/MIGRATION_LOG.md).

## Prerequisites

- NVIDIA GPU with driver >= 570 (tested on RTX 5070 Ti, Blackwell sm_120)
- Docker + NVIDIA Container Toolkit (`nvidia-docker2` / `nvidia-container-toolkit`)
- ~50 GB disk for Matterport scan data (skybox images, not included in repo)
- Matterport3D dataset placed at `~/mp3d_data/v1/scans/` (or adjust the bind mount path)

## 1. Build the Docker Image

```bash
docker build -f Dockerfile.modern -t mattersim:modern-cu128-py310 .
```

If Docker Hub is unreachable, the Dockerfile uses `docker.m.daocloud.io`
as a mirror. Adjust if your network requires a different mirror.

## 2. Start a Container

```bash
docker run -d \
  --name mattersim_build \
  --gpus all \
  -v "$(pwd):/root/Matterport3DSimulator" \
  -v ~/mp3d_data:/mp3d_data \
  -w /root/Matterport3DSimulator \
  mattersim:modern-cu128-py310 \
  sleep infinity
```

All subsequent commands run inside the container:

```bash
docker exec -it mattersim_build bash
```

## 3. Build MatterSim

```bash
# Initialize pybind11 submodule (v2.9.2, supports Python 3.10)
git submodule update --init --recursive

# CMake with EGL off-screen rendering
mkdir -p build && cd build
cmake -DEGL_RENDERING=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
make -j$(nproc) MatterSimPython
```

Verify the binding loads:

```bash
python3 -c "import MatterSim; print('MatterSim OK')"
```

## 4. Run Tests (in order)

```bash
# 1. MatterSim state machine (rendering off)
python3 tests/test_mattersim.py

# 2. R2R data pipeline (simulator in the loop)
python3 tests/test_r2r_env.py

# 3. Seq2Seq model forward / loss / backward / optimizer step
python3 tests/test_agent_forward.py

# 4. Real train.py entry (smoke: 3 iters, batch 4)
python3 tests/test_train_debug.py

# 5. Standalone eval.py scoring + simple agent baselines
python3 tests/test_eval.py
```

## 5. Train

### Debug / smoke run (3 iterations, no checkpoints)

```bash
python3 tasks/R2R/train.py --debug
```

### Sanity run (100 iterations, batch 32, no checkpoints)

```bash
python3 tasks/R2R/train.py --debug --batch-size 32 --n-iters 100
```

### Short pipeline run (1000 iterations, batch 100, with validation + checkpoints)

```bash
python3 tasks/R2R/train.py --batch-size 100 --n-iters 1000
```

### Full baseline (20000 iterations, batch 100, ~53 min on RTX 5070 Ti)

```bash
python3 tasks/R2R/train.py
```

Output:
- Checkpoints: `tasks/R2R/snapshots/` (200 enc/dec pairs)
- Validation results: `tasks/R2R/results/` (200 JSONs per split)
- Training log: `tasks/R2R/plots/seq2seq_sample_imagenet_log.csv`

## 6. Evaluate

```bash
# Score the trained model + run Stop/Shortest/Random baselines
python3 tests/test_eval.py

# Or run eval.py directly (simple agents only):
python3 tasks/R2R/eval.py
```

Metrics produced: Navigation Error (NE), Oracle Success Rate (OSR),
Success Rate (SR), Success weighted by Path Length (SPL).

## 7. Plot Training Curves

```bash
python3 tasks/R2R/plot.py
```

Output:
- `tasks/R2R/plots/training.png` — loss / NE / SR curves
- `tasks/R2R/plots/val_seen_error.png` — navigation error histogram

## Expected Results (student-forcing, iter 20000)

| Metric | val_seen | val_unseen |
|---|---|---|
| NE (m) | 6.23 | 7.87 |
| OSR | 0.499 | 0.286 |
| SR | 0.370 | 0.218 |
| SPL | 0.311 | 0.187 |

These match the original paper's reported baseline within expected variance.

## Data Layout

```
Matterport3DSimulator/
  connectivity/           # nav graphs (in repo)
  img_features/           # ResNet-152 TSV (4.1 GB, not in repo)
  tasks/R2R/data/         # R2R JSON splits + vocab (not in repo)
  tasks/R2R/plots/        # training CSV + PNG outputs
  tasks/R2R/snapshots/    # checkpoints (git-ignored)
  tasks/R2R/results/     # eval JSONs (git-ignored)
  build/                  # MatterSim .so (git-ignored)
~/mp3d_data/v1/scans/     # Matterport skybox images (bind-mounted)
```
