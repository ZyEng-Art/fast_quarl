# Quarl on the torch_xmlir XPU backend

This branch targets the Baidu XPU environment in `xsgl_zengyong`, using
PyTorch 2.9.0 and `torch_xmlir`'s CUDA API facade. It does not use Intel's
`torch.xpu` API. Quartz rule matching and DGL graph construction stay on CPU.
QGNN feature embeddings, message aggregation, policy/value networks and their
gradients execute on the selected accelerator.

## Changes

`QUARL_GRAPH_BACKEND=auto` selects tensor message passing when XMLIR is loaded.
`torch` explicitly selects it; `dgl` preserves the original DGL path for
CPU/CUDA comparisons. Graph topology remains on CPU in tensor mode, while
features and edge indices move to the network's parameter device. The QGNN
weights, message features and sum aggregation are unchanged. QGIN is rejected
in tensor mode rather than silently substituting a different architecture.

XMLIR's shutdown can fail after multithreaded autograd uses the accelerator.
The runtime therefore disables autograd worker threads in XMLIR processes.
The XMLIR exit hook remains enabled. Device selection also supports CPU, and
single-rank DDP returns the original model. `QUARL_DIST_BACKEND` optionally
selects a process-group backend; otherwise the existing NCCL name is retained
for accelerator runs (XMLIR maps it to its collective backend).

Unused RPC initialization is skipped when there are no observers. RPC uses a
300-second timeout instead of zero, which newer PyTorch interprets as an
immediate timeout. WandB receives a plain configuration dictionary for
compatibility with newer releases. Runs can disable WandB with
`c.wandb.en=false`.

## Build

Use an isolated Python environment with the installed XMLIR PyTorch, DGL
1.1.3, Cython 3, pybind11, Qiskit, z3-solver, Hydra and the dependencies in
`env_ppo.yml`. Do not install a CUDA build of DGL for the XMLIR tensor path.

From the repository root, with the environment activated:

```bash
export QUARTZ_INSTALL_PREFIX="$PWD/.xpu-native"
export LD_LIBRARY_PATH="$QUARTZ_INSTALL_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/python:${PYTHONPATH:-}"
export DGLBACKEND=pytorch
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cmake -S . -B build-xpu -DQUARTZ_BUILD_TOOLS=OFF \
  -DPython_EXECUTABLE="$(command -v python)" \
  -DCMAKE_INSTALL_PREFIX="$QUARTZ_INSTALL_PREFIX"
cmake --build build-xpu --target quartz_runtime -j4
cmake --install build-xpu
(cd python && python setup.py build_ext --inplace)
```

## Validate before optimization experiments

```bash
python experiment/ppo-new/tools/check_xpu.py
QUARL_PYTHON="$(command -v python)" bash experiment/ppo-new/tools/run_xpu_smoke.sh
python experiment/ppo-new/tools/check_xpu_search.py ../runs/xpu_smoke
```

The first check compares six-layer QGNN outputs and all parameter gradients
against CPU/DGL, including parallel edges, an isolated node and a graph with
no edges. It verifies an Adam update and finite parameters. The training check
uses `barenco_tof_3`, six GNN layers and two small PPO iterations. It is an
integration check, not a reproduction of the paper's training budget.

The default gate-count objective, ECC rules, rewards and PPO estimator are
preserved. CPU graph bookkeeping and tensor transfers add overhead; compare
both elapsed time and attempted/applied transformations in later experiments.
Validation of one accelerator does not establish multi-accelerator correctness.

On 2026-10-02, the CPU and XPU forward checks had maximum absolute errors of
`1.19e-7` and `1.04e-6` against DGL respectively. All gradient comparisons and
the Adam update passed, and the process exited with status zero. Two PPO
iterations collected 160 transitions and saved two checkpoints. Loading the
second checkpoint and searching for ten seconds completed about 700 expansion
attempts and passed a Qiskit unitary-equivalence check. The best gate count
remained 58. These short checks establish execution correctness only; they
provide no evidence of an optimization gain from reinforcement learning.

For the RL attribution experiment, compare updating PPO, a frozen initial
policy, and a uniform-valid-action policy under identical ECC rules, buffers,
increase limits and search masks. Repeat seeds and report best gate count
against both wall time and transformations. If using pretrained weights,
record that the paper includes `barenco_tof_3` in its pretraining set; freezing
such a model isolates additional fine-tuning, not learning altogether.
