# MIGRATION_LOG.md

Technical migration log for the Matterport3DSimulator R2R baseline.
For a project overview, see [PROJECT.md](PROJECT.md).

This document focuses on **how each problem was discovered, how it was
diagnosed, what pitfalls were hit, and how it was fixed** — organized as
"discovery → diagnosis → fix" rather than as a chronological diary.

Change categories:
- **A — Environment compatibility** (Docker, apt, system libraries): allowed
- **B — PyTorch API** (1.1 → 2.7 syntax differences): allowed, minimal edits
- **C — MatterSim build** (C++ toolchain, OpenCV macros): allowed, core logic untouched
- **D — Algorithm** (encoder/decoder/attention/teacher forcing/loss/eval): forbidden

---

## Pitfall 1: The legacy container "looks usable", but every GPU computation fails

### Discovery

During the audit, a running legacy container `mattersim:9.2-devel-ubuntu18.04`
(built from the original Dockerfile) was found. Running
`torch.cuda.is_available()` inside it returns `True`, and `nvidia-smi` works
fine because driver 595.84 is backward-compatible with the CUDA 9.2 runtime.
**On the surface the GPU appears usable.**

### Diagnosis

Actually performing a matrix multiply:

```python
x = torch.randn(100, 100, device='cuda')
y = x @ x  # RuntimeError: no kernel image is available for execution on the device
```

Root cause: the RTX 5070 Ti is a Blackwell GPU with compute capability
**sm_120**. The CUDA 9.2 wheel only ships SASS kernels for sm_30 through
sm_72. The driver can JIT-compile PTX for backward compatibility (which is
why `is_available()` returns `True`), but there is no matching kernel to
execute.

### Conclusion

This is not a driver problem or a configuration problem — it is a **hard
architectural incompatibility**. Any CUDA wheel ≤ 12.7 cannot run compute
on sm_120. The only path forward is CUDA ≥ 12.8 plus PyTorch ≥ 2.7 (the
first official release with Blackwell support).

---

## Pitfall 2: The default PyPI torch 2.7.1 is still cu126 and does not support sm_120

### Discovery

When the first version of the image was built in phase 2 with a plain
`pip install torch==2.7.1`, the GPU check looked fine:

```
torch version: 2.7.1
CUDA available: True
Device name(0): NVIDIA GeForce RTX 5070 Ti Laptop GPU
```

But executing `torch.randn(1000, 1000, device='cuda')` raised:

```
RuntimeError: no kernel image is available for execution on the device
```

### Diagnosis

`torch.version.cuda` prints `12.6`, not 12.8. The `pip install torch==2.7.1`
from PyPI installs the **+cu126** build by default (supporting up to
sm_90 / Hopper), which contains no sm_120 SASS kernels.

### Fix

Install from the official PyTorch cu128 index:

```bash
pip3 install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.7.1 torchvision==0.22.1
```

After the fix `torch.version.cuda` is `12.8`, the device capability is
`(12, 0)`, and the matrix multiply succeeds. **This is the most subtle
pitfall**: both builds are named torch 2.7.1 with an identical version
number; the only difference is the `+cu128` suffix and the
`torch.version.cuda` value.

---

## Pitfall 3: Docker Hub direct connection times out; large image downloads are interrupted

### Discovery

`docker pull nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04` against
`registry-1.docker.io` times out (`i/o timeout`). Even when a connection is
occasionally established, the 5GB+ devel image has its connection reset
mid-download.

### Diagnosis

```bash
curl -sI https://registry-1.docker.io/v2/   # timeout
curl -sI https://docker.m.daocloud.io/v2/   # 200 OK
curl -sI https://nvcr.io/v2/                 # 200 OK
```

### Fix

Pull through the daocloud registry mirror:

```bash
docker pull docker.m.daocloud.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
```

The Dockerfile.modern `FROM` line uses the full mirror address directly.
apt uses the Tsinghua mirror and pip uses the Tsinghua PyPI mirror; however
the **PyTorch cu128 wheels** must be fetched directly from
`download.pytorch.org` (verified reachable; the Tsinghua mirror does not
host a cu128 index).

---

## Pitfall 4: The pybind11 submodule directory is empty

### Discovery

The very first `cmake` run in phase 3 failed with:

```
CMake Error at CMakeLists.txt:57 (add_subdirectory):
  The source directory .../pybind11 does not contain CMakeLists.txt
```

### Diagnosis

`git submodule status` shows the pybind11 submodule is uninitialized
(prefix `-`). `.gitmodules` points at `pybind/pybind11`, pinned to commit
`86e2ad4` from 2018 — which **does not support Python 3.10**.

### Fix

Without touching `.gitmodules` or the binding source, clone pybind11
**v2.9.2** into the empty directory inside the container (supports Python
3.10 and is C++11 compatible):

```bash
git clone --depth 1 --branch v2.9.2 https://gitee.com/mirrors/pybind11.git pybind11
```

A direct GitHub clone times out, so the gitee mirror is used.

**Gotcha:** because pybind11 is a git submodule, cloning into it makes the
parent repository show a submodule-pointer diff. This is silenced locally
with `git config submodule.pybind11.ignore all` in `.git/config` (not
committed, produces no diff). pybind11 is a build dependency, not a build
artifact; `ignore=all` is sufficient.

---

## Pitfall 5: cmake cannot find OpenGL (missing GLVND libOpenGL.so)

### Discovery

```
CMake Error: Could NOT find OpenGL (missing: OPENGL_opengl_LIBRARY)
```

### Diagnosis

Inside the container, `find_package(OpenGL COMPONENTS OpenGL EGL)` needs
the GLVND `libOpenGL.so` (distinct from the traditional `libGL.so`).
`libgl-dev` provides `libGL.so`, but `libOpenGL.so` is provided by
`libopengl-dev`, which **the original Dockerfile.modern was missing**.

### Fix

`apt install libopengl-dev` in the container, and add the package back to
Dockerfile.modern (one line). cmake configures successfully afterwards.

---

## Pitfall 6: OpenCV 4 removed CV_LOAD_IMAGE_ANYDEPTH

### Discovery

```
src/lib/NavGraph.cpp:59: error: 'CV_LOAD_IMAGE_ANYDEPTH' was not declared
```

### Diagnosis

Ubuntu 18.04 shipped OpenCV 3.2 where `CV_LOAD_IMAGE_ANYDEPTH` is a C-style
macro. Ubuntu 22.04 ships OpenCV 4.5 where these legacy macros were removed
in favor of the `cv::IMREAD_*` enums. **The values are identical** (both
are `2`); only the namespace and naming style changed.

### Fix

One line in [NavGraph.cpp:59](src/lib/NavGraph.cpp#L59):

```cpp
// before
cv::imread(..., CV_LOAD_IMAGE_ANYDEPTH);
// after
cv::imread(..., cv::IMREAD_ANYDEPTH);
```

Zero logic change — a constant-name migration only. This is the single
legacy OpenCV macro in the whole repository.

---

## Pitfall 7: The C++ unit test fails to build (but is irrelevant to R2R)

### Discovery

`make` fails on `src/test/main.cpp` with two errors:

1. `Catch.hpp:6631` — `SIGSTKSZ` is no longer a compile-time constant in
   glibc ≥ 2.34
2. `main.cpp:379` — `CV_L2` was removed in OpenCV 4

### Diagnosis

Both errors come from the **vendored Catch.hpp** and the upstream C++ test
executable (the `tests` target) — not from the MatterSim library itself and
not from the Python binding. The R2R baseline only needs the
`MatterSimPython` target (the `.so` binding).

### Fix

**No source code is modified.** Build only the required target:

```bash
make MatterSimPython   # compiles only the Python binding + libMatterSim.so
```

Artifacts: `build/MatterSim.cpython-310-x86_64-linux-gnu.so` +
`build/libMatterSim.so`

---

## Pitfall 8: networkx 3.x removed `G.node[...]` (phase 4)

### Discovery

As soon as `R2RBatch` exercises the teacher-action path
(`_shortest_path_action` when the next viewpoint is not currently visible):

```
AttributeError: 'Graph' object has no attribute 'node'
```

### Diagnosis

`env.py` calls `self.graphs[scan].node[vp]['position']`. The `Graph.node`
attribute was deprecated in networkx 2.4 and **removed in networkx 3.x**
(the modern image pins networkx 3.4.2). The replacement is `Graph.nodes`,
which returns the same node-attribute view. This is a pure API rename;
the returned `'position'` attribute (a numpy array set via
`nx.set_node_attributes`) is unchanged.

### Fix

One line in [env.py:184](tasks/R2R/env.py#L184):

```python
# before
self.graphs[state.scanId].node[nextViewpointId]['position'] - pos
# after
self.graphs[state.scanId].nodes[nextViewpointId]['position'] - pos
```

After this single change the full R2R environment data flow ran with no
further errors (`csv.field_size_limit(sys.maxsize)`,
`nx.set_node_attributes(..., values=, name=)` keyword call, and
`all_pairs_dijkstra_path` all work unchanged on Python 3.10 / numpy 1.26 /
networkx 3.4).

---

## clangd support: compile_commands.json

Re-configured the build with `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` so clangd
gets exact include paths/defines (EGL flag, OpenCV4 include path, C++11):

```bash
cd build
cmake -DEGL_RENDERING=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
ln -sf build/compile_commands.json ../compile_commands.json   # clangd auto-discovery
```

The root symlink is git-ignored (machine-specific build artifact). The
compile database covers all seven translation units, including
`src/lib_python/MatterSimPython.cpp`.

Note: the database contains container-absolute paths
(`/root/Matterport3DSimulator/...`, `/usr/include/opencv4`), so clangd
must run **inside the container** (e.g. VS Code Dev Containers / remote
clangd); a host-side clangd would not resolve those paths without path
mapping.

---

## Phase 5: PyTorch 1.1 → 2.7 API audit and minimal migration

### Audit of tasks/R2R/{agent.py, model.py, train.py}

Verified empirically in the container (probe script against torch 2.7.1+cu128),
not guessed:

| Item | Location | Verdict in torch 2.7 |
|---|---|---|
| `mask.byte().cuda()` passed to `attn.data.masked_fill_` | [agent.py:178](tasks/R2R/agent.py#L178) → [model.py:96](tasks/R2R/model.py#L96) | **Hard error**: `RuntimeError: masked_fill_ only supports boolean masks, but got mask with dtype unsigned char` → must change |
| `from torch.autograd import Variable` / `Variable(t, requires_grad=False)` | model.py, agent.py, train.py | OK (deprecated but functional, zero semantic change) — kept to minimize diff |
| `pack_padded_sequence(embeds, lengths)` with `lengths` = list of 0-dim tensors | model.py:50 ← agent.py `_sort_batch` | OK (probe passed; batch already sorted descending upstream, so `enforce_sorted=True` default is satisfied) |
| `h0.cuda(), c0.cuda()` / `.long().cuda()` / `torch.LongTensor(n)` | model.py:42, agent.py:177/191/229 | OK (GPU-only container; device handling left as-is per minimal-change principle) |
| `attn.data.masked_fill_` via `.data` | model.py:96 | OK (discouraged but functional; avoiding rewrite of attention internals) |
| `logit[i, idx] = -float('inf')` in-place masking | agent.py:243 | OK (autograd handles index_put) |
| `logit.max(1)`, `.detach()`, `F.softmax(dim=1)`, `D.Categorical`, `.item()` | agent.py:253-281 | OK (identical semantics in 2.x) |
| `torch.cuda.manual_seed`, `.cuda()` on models | train.py | OK |

Known original-code quirk (NOT a version issue, not fixed): `action_embeds.squeeze()`
in model.py:133 collapses the batch dim when `batch_size == 1`; the test uses
batch 4 (upstream default is 100).

### Probe evidence (torch 2.7.1+cu128)

```
PROBE1 byte-mask masked_fill_: FAIL -> RuntimeError masked_fill_ only supports
       boolean masks, but got mask with dtype unsigned char
PROBE1b bool-mask masked_fill_: OK
PROBE2 list-of-tensor lengths: OK
PROBE3 Variable: OK
```

### Fix (single line, Category B)

[agent.py:178](tasks/R2R/agent.py#L178):

```python
# before
mask.byte().cuda(),
# after
mask.bool().cuda(),
```

`mask` is a padding-position comparison (`sorted_tensor == padding_idx`), so
`.byte()` and `.bool()` carry identical mask semantics; `masked_fill_(-inf)`
behavior in SoftDotAttention is unchanged.

### Verification: tests/test_agent_forward.py

One full learning step of the ORIGINAL pipeline, asserted stage by stage:

```
torch 2.7.1+cu128 (cuda 12.8, available=True)
R2RBatch loaded with 14039 instructions, using splits: train
[1] R2RBatch ready (14039 instructions, batch=4)
[2] EncoderLSTM (2093312 params) + AttnDecoderLSTM (6102278 params) on cuda:0
[3] forward + teacher-forced loss: 33.4266
[4] backward OK: encoder grad-norm 3.3882, decoder grad-norm 24.4694
[5] optimizer.step: both optimizers updated parameters
[6] second iteration with sample feedback: loss 15.4673
ALL SEQ2SEQ MODEL TESTS PASSED
```

Coverage: R2RBatch → Seq2SeqAgent.rollout → EncoderLSTM → AttnDecoderLSTM →
SoftDotAttention (BoolTensor mask) → CrossEntropyLoss(ignore_index='<ignore>')
→ loss.backward() (non-zero gradient norms asserted) → Adam optimizer.step()
(parameter tensors asserted changed) → a second iteration with 'sample'
student-forcing feedback. Architecture, teacher forcing, and loss math are
untouched. (The LSTM dropout warning is upstream behavior: dropout=0.5 with
num_layers=1 — left as-is.)

### Net PyTorch migration surface

Exactly **one line** (`mask.byte()` → `mask.bool()`) was required to move the
whole Seq2Seq training step from torch 1.1 to torch 2.7. Everything else
(Variable, `.data`, `.cuda()`, list-of-tensor lengths) is verified-compatible
and intentionally preserved.

---

## Phase 6: real train.py entry on PyTorch 2.7 (debug run)

### Discovery (audit of train.py)

- **No argparse at all**: every knob is a module-level constant
  (`batch_size=100`, `feedback_method='sample'`,
  `n_iters=20000`, `model_prefix=...`); `__main__` calls `train_val()`
  unconditionally. There is no way to run a small config from the CLI.
- `train_val()` unconditionally creates **full validation environments**
  for `val_seen` and `val_unseen` (`R2RBatch` + `Evaluation` per split);
  `train()` runs two full-split `agent.test()` passes per env every
  `log_every=100` iterations — a debug run must skip these.
- `train()` unconditionally calls `agent.save()` (≈32 MB per pair) at the
  end of every interval — must be gated for debug runs.
- Paths audit: `TRAIN_VOCAB` / `TRAINVAL_VOCAB` exist (so `setup()` writes
  nothing); `IMAGENET_FEATURES` exists; `RESULT_DIR` / `SNAPSHOT_DIR` /
  `PLOT_DIR` exist and are empty — a debug run cannot overwrite anything.
- CUDA/optimizer APIs (`torch.cuda.manual_seed`, `optim.Adam`, `.cuda()`)
  verified compatible in phase 5.

### Diagnosis

The entry point needed a minimal debug configuration without touching any
training logic. Constraints: default behavior of `python3 tasks/R2R/train.py`
(no flag) must remain byte-for-byte equivalent to upstream; debug must use a
small batch, few iterations, the train split only, no validation, and no
snapshot writes.

### Fix (Category A/B, ~10 lines in train.py, no algorithm change)

[train.py](tasks/R2R/train.py):

1. `import argparse` + `__main__` gains `--debug` flag. With `--debug`:
   `batch_size=4`, `n_iters=3`. Without it: constants untouched, so the
   default full-baseline path is unchanged.
2. `train_val(debug=False)` gates validation-environment creation
   (`val_envs = {}` in debug).
3. `train(..., debug=False)` gates `agent.save(...)` only (CSV loss log
   still written — it is the iteration evidence).

No change to encoder/decoder/attention, feedback logic, loss, or eval.

### Verification: tests/test_train_debug.py

Launches the **real CLI entry** as a subprocess and asserts the chain
stage by stage:

```
[run] /usr/bin/python3 tasks/R2R/train.py --debug (cwd=/root/Matterport3DSimulator)
Loading image features from img_features/ResNet-152-imagenet.tsv
Loading navigation graphs for 61 scans
R2RBatch loaded with 14039 instructions, using splits: train
Training with sample feedback
0m 0s (- 0m 0s) (3 100%) train loss: 1.1941

[1] R2RBatch initialized on train split: OK
[2] Seq2SeqAgent created with original default feedback (sample): OK
[3] iterations completed: 3/3 (100%), train loss 1.1941: OK
[4] valid loss produced (forward+backward+optimizer.step all ran): OK
[5] training log written: tasks/R2R/plots/seq2seq_sample_imagenet_log.csv
[6] no snapshot checkpoints written (snapshots dir unchanged): OK

TRAIN.PY DEBUG ENTRY TEST PASSED
```

Loss-log CSV content (one row per log interval, iteration counter reaches
`n_iters`):

```
,iteration,train loss
0,3,1.1940708478291828
```

`train.py --help` confirms the parser; the no-flag default path is unchanged.
No new PyTorch 2.x incompatibilities surfaced in the entry point itself —
the only hard blocker remains the phase-5 `mask.bool()` fix.

---

## Phase 7: sanity training (100 iters, batch 32)

### Debug entry extension (Category A, ~8 lines)

[train.py](tasks/R2R/train.py) `--debug` gained two optional overrides so the
same entry serves both the fast smoke test and sanity runs; defaults are
unchanged:

- `--batch-size` (debug default 4)
- `--n-iters` (debug default 3)

Sanity command: `python3 tasks/R2R/train.py --debug --batch-size 32 --n-iters 100`
(train split only, no validation, no snapshot writes — debug semantics).

### Pitfall 9: host NVIDIA driver auto-upgrade breaks all GPU access (blocker)

#### Discovery

Immediately after launching the sanity run, the container was down and
`docker start` failed:

```
failed to fulfil mount request: open /run/nvidia-persistenced/socket:
no such file or directory
```

#### Diagnosis

On the HOST (not in the repo, not in the container):

```
$ nvidia-smi
Failed to initialize NVML: Driver/library version mismatch
NVML library version: 595.91

$ cat /proc/driver/nvidia/version
NVRM version: NVIDIA UNIX Open Kernel Module ... 595.84

$ modinfo nvidia | grep version
version: 595.91.07
$ dpkg -l | grep nvidia-driver-595
ii  nvidia-driver-595-open  595.91.07-0ubuntu0.26.04.1
```

The host driver package was auto-upgraded (unattended-upgrades) from
595.84 to **595.91.07** since the last GPU run. The userspace libraries
(NVML etc.) are now 595.91 while the **loaded kernel module is still
595.84** — the driver stack is split across the reboot boundary. All GPU
access fails on host and in containers until the kernel module is
reloaded (practically: reboot, since the display server holds the old
module).

#### Fix

Host action required (user, with root): **reboot**, or unload/reload the
nvidia kernel modules (`sudo systemctl stop display-manager; sudo rmmod
nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia`).
No repo file can fix this; no repo file was affected by it.

#### Resolution

After host reboot the driver stack is unified at 595.91.07; the stale
container was removed and recreated from the image with the same bind
mounts (the compiled `.so` files live on the repo bind mount and survive).
No repo change was needed — and no commit was made for the driver event.
One operational note: `docker run ... python3 - <<EOF` needs `-i` or stdin
is not attached and the script is empty.

### Sanity run (batch 32, 100 iters, debug)

```
Training with sample feedback
0m 5s (- 0m 0s) (100 100%) train loss: 1.2051
```

GPU peak **742 MiB** of 12227 MiB, 45 °C; no snapshots/results written
(debug); CSV loss log written. Stable, no anomalies.

### Checkpoint/validation run (batch 100, 1000 iters, full pipeline)

`--batch-size` / `--n-iters` overrides were extended to also apply to a
normal run (without `--debug`), so validation and snapshot saving stay on
while the iteration count is shortened:

```
python3 tasks/R2R/train.py --batch-size 100 --n-iters 1000
```

Selected log (loss + success rate every 100 iters, val on both splits via
the original eval.py metrics):

```
 (100 10%) train loss 1.1620  val_seen loss 0.9633 SR 0.048  val_unseen loss 0.9461 SR 0.038
(1000 100%) train loss 0.9043  val_seen loss 0.8997 SR 0.122  val_unseen loss 0.9221 SR 0.084
```

Full 1000-iter run with ten validation rounds: **2m51s**. Final metrics
(iter 1000): val_seen NE 8.24 m / OSR 0.164 / SR 0.122 / SPL 0.109;
val_unseen NE 8.60 m / OSR 0.115 / SR 0.084 / SPL 0.077 — a healthy
early-training trajectory (the original paper's fully-trained
student-forcing sample baseline is about SR 0.22 / 0.17 at 20k iters;
these 1k-iter numbers are consistent with it).

- GPU peak **1534 MiB** (13% of 12 GB), 56 °C — batch 100 has large headroom.
- Checkpoints: 10 encoder + 10 decoder pairs (24.4 MB each, 313 MB total)
  in `tasks/R2R/snapshots/`; 20 validation-result JSONs; both verified
  loadable with `torch.load(..., weights_only=True)` (enc 7 / dec 9 state
  keys). These are runtime artifacts (git-ignored), not committed.
- This run also exercises `eval.py` scoring (NE/OSR/SR/SPL) end to end —
  Phase 8 evaluation path is already verified on torch 2.7.

---

## Phase 8: standalone evaluation on PyTorch 2.7

### Audit

[eval.py](tasks/R2R/eval.py) is **pure Python** (json / networkx / numpy /
collections / pprint) with **zero PyTorch imports**. No `Variable`, no
`.cuda()`, no autograd. The `Evaluation` class reads JSON trajectory files
and computes NE/OSR/SR/SPL via networkx graph distances. No compatibility
issue with torch 2.7 is possible — eval.py doesn't touch torch.

Checkpoint loading happens in `agent.py`'s `test()` method (already
verified in Phase 7 training validation). `eval_seq2seq()` (line 127,
currently commented out in `__main__`) references a teacher-forcing
result file that we didn't produce, but this is irrelevant: the
`Evaluation.score()` method is the actual scorer and works on any
result JSON.

### Verification: tests/test_eval.py

**(1) Trained model scoring (iter 20000, standalone Evaluation)**

Scored the result JSONs from the final training checkpoint using
eval.py's `Evaluation` class as a standalone entry point:

```
val_seen (eval.py standalone):
  length                 11.1050
  nav_error              6.2335
  oracle success_rate    0.4990
  success_rate           0.3696
  spl                    0.3106
  [OK] all metrics match training CSV within 1e-4

val_unseen (eval.py standalone):
  length                  8.1546
  nav_error               7.8669
  oracle success_rate     0.2857
  success_rate            0.2175
  spl                     0.1874
  [OK] all metrics match training CSV within 1e-4
```

Cross-check: every metric (NE/OSR/SR/SPL/length) matches the training CSV
final row to within 1e-4 — confirming the standalone eval path and the
in-training validation path produce identical results.

**(2) Simple agent baselines (eval.py __main__ entry)**

Ran `eval_simple_agents()` (the original `python3 tasks/R2R/eval.py`
entry) on all three splits:

| Agent | Split | NE (m) | OSR | SR | SPL |
|---|---|---|---|---|---|
| Stop | val_seen | 10.19 | 0.000 | 0.000 | 0.000 |
| Shortest | val_seen | 0.00 | 1.000 | 1.000 | 1.000 |
| Random | val_seen | 9.49 | 0.211 | 0.163 | 0.149 |
| Stop | val_unseen | 9.48 | 0.000 | 0.000 | 0.000 |
| Shortest | val_unseen | 0.00 | 1.000 | 1.000 | 1.000 |
| Random | val_unseen | 9.22 | 0.215 | 0.160 | 0.140 |

These are the expected reference baselines: Stop (SR=0, never moves),
Shortest (SR=1, follows optimal path), Random (SR~0.14–0.16). All three
splits and three agents ran without error on the modern stack.

### Result

**eval.py required zero modifications.** No Category A/B/C/D fix was
needed. The evaluation pipeline (result JSON → Evaluation.score →
NE/OSR/SR/SPL) is fully functional on PyTorch 2.7 / CUDA 12.8 /
RTX 5070 Ti.



### GPU verification (phase 2)

```
torch version: 2.7.1+cu128
CUDA version: 12.8
CUDA available: True
Device capability(0): (12, 0)
Compute test (matmul 1000x1000): OK
```

### MatterSim navigation state machine (phase 3, rendering off)

[tests/test_mattersim.py](tests/test_mattersim.py), run inside the container:

```
[TEST 2] import MatterSim: OK
[TEST 3] Simulator constructed and initialized (rendering off): OK
[TEST 4] newEpisode/getState: OK
[TEST 5a] forward action -> expected viewpoint, step=1: OK
[TEST 5b-5e] turn/look state transitions (heading, elevation, viewIndex): OK
ALL MATTERSIM STATE-MACHINE TESTS PASSED
```

Coverage:
- `import MatterSim` — pybind11 binding loads.
- `newEpisode(scan, vp, heading, elev)` + `getState()` — scanId /
  viewpointId / step=0 / discretized viewIndex all correct.
- `makeAction` — forward moves to `navigableLocations[1]` with a viewpointId
  matching the connectivity graph; turn right → heading π/6 / viewIndex
  12→13; turn left → 11×π/6; look up/down → elevation ±π/6 / pitch bands
  24/0.
- The whole run uses `setRenderingEnabled(False)`: no GL context is created
  and no skybox images are read (consistent with the R2R baseline training
  path).

### R2R environment data flow (phase 4, simulator in the loop)

[tests/test_r2r_env.py](tests/test_r2r_env.py), run from the repo root in
the container:

```
[1] vocab size=991, tokenizer ready
R2RBatch loaded with 14039 instructions, using splits: train
[2] R2RBatch ready: 14039 instructions across 61 scans
[3] EnvBatch wraps a live MatterSim.Simulator; cached feature store loaded: 10567 viewpoints
[4] reset() produced 4 observations
[5] teacher-forced rollout through MatterSim: all 4/4 trajectories reached goal
ALL R2R ENVIRONMENT DATA-FLOW TESTS PASSED
```

Coverage (original data flow preserved, no pre-computed teacher table):
- `R2R_train.json` → 14,039 separate instruction entries across 61 scans;
  vocab 991; TSV feature store decoded to 10,567 viewpoint entries
  (36 × 2048 each).
- `R2RBatch.reset()` drives the **live** `MatterSim.newEpisode` (asserted
  `env.env.sim` is a `MatterSim.Simulator` instance).
- Every observation's `feature` is checked with `np.array_equal` against
  `features[scan_viewpoint][viewIndex]` — i.e. the cached feature is keyed
  by the simulator's live viewpoint/viewIndex, not assumed in advance.
- Teacher action indices always reference a real navigable location.
- A teacher-forced rollout calls `env.step` (→ live `makeAction`); the
  simulator step counter increments correctly and **all 4 trajectories end
  exactly at their goal viewpoint (Dijkstra graph distance 0)**, proving
  the shortest-path teacher supervision stays coupled to live simulator
  state transitions.

---

## Change inventory (through phase 6)

| File | Category | Change | Lines |
|---|---|---|---|
| [Dockerfile.modern](Dockerfile.modern) | A | New: CUDA 12.8 + PyTorch 2.7.1+cu128 + MatterSim build deps; includes the `libopengl-dev` backfill | +54 |
| [src/lib/NavGraph.cpp](src/lib/NavGraph.cpp#L59) | C | `CV_LOAD_IMAGE_ANYDEPTH` → `cv::IMREAD_ANYDEPTH` (same-value constant) | 1 |
| [tasks/R2R/env.py](tasks/R2R/env.py#L184) | B | `G.node[...]` → `G.nodes[...]` (networkx 3.x API rename) | 1 |
| [tasks/R2R/agent.py](tasks/R2R/agent.py#L179) | B | `mask.byte()` → `mask.bool()` (torch 2.x requires BoolTensor for masked_fill_) | 1 |
| [tasks/R2R/train.py](tasks/R2R/train.py) | A/B | Minimal `--debug` flag (batch 4 / 3 iters / no validation / no snapshots) + `--batch-size`/`--n-iters` overrides for sanity runs; default no-flag path unchanged | ~18 |
| [tests/test_mattersim.py](tests/test_mattersim.py) | New | Minimal state-machine test (does not modify existing code) | +106 |
| [tests/test_r2r_env.py](tests/test_r2r_env.py) | New | R2R data-flow test with simulator in the loop | +121 |
| [tests/test_agent_forward.py](tests/test_agent_forward.py) | New | Minimal Seq2Seq forward/loss/backward/step test | +117 |
| [tests/test_train_debug.py](tests/test_train_debug.py) | New | Real train.py entry, debug run, stage-by-stage assertions | +90 |
| [tests/test_eval.py](tests/test_eval.py) | New | Standalone eval.py scoring (iter 20000) + simple agent baselines | +79 |
| [.gitignore](.gitignore) | A | Ignore root `compile_commands.json` symlink (clangd build artifact) | 1 |

Untouched (algorithm): R2R agent/model/train/eval logic and utils data
processing; MatterSim core logic (MatterSim.cpp / MatterSimPython.cpp);
the original Dockerfile; data files. The only R2R source edits are the two
one-line API renames in env.py (networkx) and agent.py (torch), both
Category B with no behavior change.

---

## Git record (branch migration/modern-pytorch)

```
64980b6 Phase 3: build MatterSim in modern container, verify nav state machine
13697db Add MIGRATION_LOG.md: record migration plan and phase 1-2 results
1e441d1 Add modern CUDA 12.8 + PyTorch 2.7.1 cu128 Dockerfile
589d091 (upstream master) Merge pull request #100 ...
```

Not pushed. The local repo has `submodule.pybind11.ignore=all` set (not
committed).

---

## TODO

- [x] Phase 4: R2R environment — `env.py` `G.node`→`G.nodes` (networkx 3.x); data flow verified
- [x] Phase 5: minimal PyTorch API migration — `agent.py` `mask.byte()`→`mask.bool()`; forward/loss/backward/step verified
- [x] Phase 6: real train.py entry — `--debug` config; 3 iterations completed end-to-end
- [x] Phase 7: sanity training (batch 32 / 100 iters loss 1.2051, 742 MiB) + full-pipeline run (batch 100 / 1000 iters, val + checkpoints, loss 0.9043, 1534 MiB); full 20k-iter baseline launched
- [x] Phase 8: standalone eval.py — trained model (iter 20000) scored with Evaluation class, all metrics match training CSV within 1e-4; Stop/Shortest/Random baselines verified; eval.py required zero modifications
