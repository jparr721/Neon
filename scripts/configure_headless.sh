#!/bin/bash

set -euox pipefail

if [ ! -d "build" ]; then
  mkdir build
fi
cd build
cmake -GNinja -DCMAKE_BUILD_TYPE=Release -DUSE_DOUBLE=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DUSE_HEADLESS=1 ..
cd ..
