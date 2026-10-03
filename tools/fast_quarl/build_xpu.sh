#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
quarl_root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$quarl_root"
export QUARTZ_INSTALL_PREFIX="$PWD/.xpu-native"
export LD_LIBRARY_PATH="$QUARTZ_INSTALL_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export DGLBACKEND=pytorch
export QUARL_GRAPH_BACKEND=torch
quarl_python=${QUARL_PYTHON:-python}
cmake -S . -B build-xpu -DQUARTZ_BUILD_TOOLS=OFF -DPython_EXECUTABLE="$quarl_python" -DCMAKE_INSTALL_PREFIX="$QUARTZ_INSTALL_PREFIX"
cmake --build build-xpu --target quartz_runtime -j4
cmake --install build-xpu
cd python
"$quarl_python" setup.py build_ext --inplace
cd ..
PYTHONPATH="$PWD/python" "$quarl_python" -c 'import quartz; c=quartz.QuartzContext(gate_set=["h","cx","x","rz","add","neg"],filename="experiment/ecc_set/nam_325_ecc.json",no_increase=False,include_nop=True); g=quartz.PyGraph.from_qasm(context=c,filename="experiment/circs/nam_circs/barenco_tof_3.qasm"); print("NATIVE_READY",c.num_xfers,g.gate_count)'
echo BUILD_PASSED
