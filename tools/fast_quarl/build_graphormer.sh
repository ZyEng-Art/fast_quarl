#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$ROOT/.graphormer-native"
"${CXX:-c++}" -O3 -std=c++17 -shared -fPIC "$ROOT/tools/fast_quarl/graphormer_preprocess.cpp" -o "$ROOT/.graphormer-native/libgraphormer.so.tmp"
mv "$ROOT/.graphormer-native/libgraphormer.so.tmp" "$ROOT/.graphormer-native/libgraphormer.so"
