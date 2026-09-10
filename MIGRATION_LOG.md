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

## What has been verified

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

---

## Change inventory (through phase 3)

| File | Category | Change | Lines |
|---|---|---|---|
| [Dockerfile.modern](Dockerfile.modern) | A | New: CUDA 12.8 + PyTorch 2.7.1+cu128 + MatterSim build deps; includes the `libopengl-dev` backfill | +54 |
| [src/lib/NavGraph.cpp](src/lib/NavGraph.cpp#L59) | C | `CV_LOAD_IMAGE_ANYDEPTH` → `cv::IMREAD_ANYDEPTH` (same-value constant) | 1 |
| [tests/test_mattersim.py](tests/test_mattersim.py) | New | Minimal state-machine test (does not modify existing code) | +106 |

Untouched: R2R Python code (env/agent/model/train/utils/eval); MatterSim
core logic (MatterSim.cpp / MatterSimPython.cpp); the original Dockerfile;
data files.

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

- [ ] Phase 4: R2R environment — `env.py` `G.node`→`G.nodes` (networkx 3.x)
- [ ] Phase 4: R2R environment — `agent.py` `mask.byte()`→`mask.bool()` (PyTorch 2.x)
- [ ] Phase 5: Minimal PyTorch API migration (only the two lines above are mandatory)
- [ ] Phase 6: 12-step minimal validation chain
- [ ] Phase 7: debug training → full baseline
- [ ] Phase 8: eval.py evaluation metrics
