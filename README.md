# fast_quarl

Quarl 的昆仑芯 P800 / torch_xmlir 适配与 GNN 加速实现，保留 NVIDIA CUDA 与 CPU 运行路径。

基于 [quantum-compiler/Quarl](https://github.com/quantum-compiler/Quarl) 的 `385cf5e04a41590a48a2fe82b99f5e99e6cd2279`，XPU 适配提交为 `c719a205b75af90f01ab67affb764cdfff15a738`。原项目遵循 Apache-2.0，许可证见 [LICENSE](LICENSE)，原安装说明见 [INSTALL.md](INSTALL.md)。论文：[Quarl: A Learning-Based Quantum Circuit Optimizer](https://arxiv.org/abs/2307.10120)。

## 改动

- 使用 PyTorch tensor 图聚合适配 torch_xmlir 的 CUDA 接口；在该环境中 DGL 拓扑保留在 CPU。
- 每次 GNN 前向只准备一次边索引和边特征，供所有卷积层共享；不缓存随参数更新而变化的 embedding。
- 提供并发 PPO 训练与搜索、双向最好电路交换、等价性验证、35 门目标停止和均匀随机无学习对照。
- 提供 Barenco、GF 优化轨迹的离线 DAG 网页：动作依赖、单步子图大小、整条轨迹累计覆盖范围。

## 已测结果

| 实验 | 首次达到 Barenco 35 门 |
| --- | ---: |
| XPU 适配原版，旋转合并预处理 | 16,562.92 秒（276.05 分钟） |
| 共享拓扑加速版，相同预处理 | 9,004.64 秒（150.08 分钟） |

均从头训练，种子 98766，无预训练权重，最终电路经过等价性验证。单次达到目标耗时约减少 45.6%；这包含搜索随机性与运行时负载差异，不能视为稳定的硬件或算法加速比。

GNN 微基准批量 64：前向 4.49 ms → 2.64 ms（1.70×），前向＋反向 8.04 ms → 6.08 ms（1.32×）。基准覆盖批量 8、64、256；测试输出和梯度最大差异均为 0。微基准不包含 PPO actor/critic、优化器、候选匹配或完整搜索。实际 PPO minibatch 为 4800，不能直接套用批量 64 的反向加速比。

原始记录、硬件说明和最终 QASM 见 [results/barenco_tof_3](results/barenco_tof_3)。

## 环境与构建

已验证环境：Linux、Python 3.10、PyTorch 2.9.0＋torch_xmlir、DGL 1.1.3、NumPy 1.26.4。torch_xmlir 由昆仑芯环境提供，它的 `cuda` 设备名称不表示实际使用 NVIDIA GPU。普通 NVIDIA 环境使用匹配自身 CUDA 的 PyTorch/DGL 安装。

保留上游 Python 环境描述：[env_ppo.yml](experiment/ppo-new/env_ppo.yml)。XPU 环境不要直接照该文件替换厂商提供的 PyTorch。额外需要 CMake、C++17 编译器、Cython、OmegaConf、Hydra、Qiskit、W&B（实验中禁用联网记录）等原项目依赖。

```bash
export QUARL_PYTHON=python
bash tools/fast_quarl/build_xpu.sh
export PYTHONPATH="$PWD/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$PWD/.xpu-native/lib:${LD_LIBRARY_PATH:-}"
export DGLBACKEND=pytorch QUARL_GRAPH_BACKEND=torch
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 WANDB_MODE=disabled
```

## 验证与运行

```bash
python experiment/ppo-new/tools/check_xpu.py --cpu-only
python tools/fast_quarl/benchmark.py --gpu 0 --output runs/benchmark.json

# 两张卡：前 5 分钟仅训练，随后启动并发搜索；达到已验证的 35 门停止。
python tools/fast_quarl/run_experiment.py controller \
  --train-gpu 0 --search-gpu 1 --seed 98766 --run-dir runs/learning_rm

# 无学习对照：均匀策略、冻结完整模型状态，无 backward/optimizer.step。
python tools/fast_quarl/run_experiment.py controller \
  --mode uniform --train-gpu 2 --search-gpu 3 --run-dir runs/uniform_rm
```

使用 `--no-preprocess` 从原始 58 门电路开始；默认旋转合并为 46 门。每次实验必须使用新的 `--run-dir`。默认没有时间上限；随机对照可能长时间停留在 38 门。比较策略时同时报告时间、采样转换数和种子，不能把均匀对照更高的采样吞吐误认为学习收益。

## 轨迹网页

下载仓库后用 Chrome/Safari 打开：

- [动作 DAG、子图大小与累计覆盖](analysis/trajectory_action_detail_20261002/action_viewer.html)
- [门数与完整轨迹总览](analysis/trajectory_analysis_20261002/trajectory_viewer.html)

网页数据内嵌，可离线使用；GitHub 文件预览不会直接执行 HTML。轨迹 replay 使用 `nam_ecc.json`，训练使用 `nam_325_ecc.json`，两者 action ID 不可直接互换。累计覆盖按门 GUID 追踪：原始节点覆盖与历次新增节点分别统计；历次节点数不是同时存在的子图大小。保存轨迹只有一种可重现后继的匹配恢复，不保证唯一。

分析数据仅覆盖已保存的成功轨迹，不能单独证明强化学习有效。当前随机对照仍在运行，其最终结果未包含在本仓库的已完成记录中。


## Global circuit context experiment

Branch `feat/global-context` adds `QGNNGlobal`: each node retains a six-layer QGNN local representation, then fuses it with the mean representation of every node in its circuit and log(1 + circuit gate count). Pooling respects DGL batch boundaries. A Linear + ReLU fusion keeps the existing actor/critic input width. This is a pooled-context baseline, not a Graphormer or a claim of improved optimization quality.

Use complete circuit observations for both rollout and PPO: `QGNNGlobal` requires `--no-subgraph-opt`. The localized `next_nodes` value target and all rewrite legality checks remain unchanged. The existing `QGNN` default and subgraph default are preserved.

```bash
python tools/fast_quarl/check_global_context.py --gpu 4 --output results/global_context/checks.json
python tools/fast_quarl/run_experiment.py controller --gnn-type QGNNGlobal --no-subgraph-opt --train-gpu 4 --search-gpu 5 --ddp-port 25200 --run-dir runs/global_context_rm
```

`QUARL_NATIVE_PYTHON` optionally points to a compatible existing Quartz Python build; normally build this checkout following the instructions above. On node36 the experiment reuses the existing compiled Quartz extension while loading the model and PPO code from this branch. Results must be compared with the full-graph QGNN arm using transitions and wall time to first verified 35 gates. Extra model parameters and changed initial actor/critic RNG state mean this is an architecture comparison rather than identical-model timing.

Checks cover per-circuit pooling isolation, node permutation equivariance, influence beyond six hops, finite gradients, Adam updates, and actor/critic output shapes on CPU and P800. Node36 evidence is stored in `results/global_context/node36_checks.json`.
