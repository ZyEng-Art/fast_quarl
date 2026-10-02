#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
quarl_root=$(cd "$(dirname "$0")/../../.." && pwd)
quarl_python=${QUARL_PYTHON:-python}
export PYTHONPATH="$quarl_root/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$quarl_root/.xpu-native/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export DGLBACKEND=pytorch QUARL_GRAPH_BACKEND=torch WANDB_MODE=disabled
cd "$quarl_root/experiment/ppo-new"
exec "$quarl_python" ppo.py c=nam2_ft c.resume=false 'c.gpus=[0]' \
  c.num_eps_per_iter=4 c.agent_batch_size=4 c.max_eps_len=20 c.min_eps_len=20 \
  c.dyn_eps_len=false c.max_iterations=2 c.k_epochs=1 c.mini_batch_size=32 \
  c.gnn_num_layers=6 c.wandb.en=false c.omp_num_threads=4 c.ddp_port=24536 \
  "hydra.run.dir=$quarl_root/../runs/xpu_smoke" "$@"
