#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
python3 "$ROOT/configure_native.py"
SRC="$ROOT/third_party/ORB_SLAM3"
cmake -S "$SRC/Thirdparty/DBoW2" -B "$ROOT/build/dbow2" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS='-include boost/serialization/library_version_type.hpp'
cmake --build "$ROOT/build/dbow2" -j2
cmake -S "$SRC" -B "$ROOT/build/orb" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS='-include boost/serialization/library_version_type.hpp'
cmake --build "$ROOT/build/orb" --target g2o mast3r_backend -j2
echo BUILD_SUCCESS
