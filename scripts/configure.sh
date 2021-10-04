#!/bin/bash

set -euox pipefail

if [ -d "build" ]; then
  rm -r build
fi

mkdir build
cd build
cmake -GNinja -DCMAKE_BUILD_TYPE=Release -DUSE_DOUBLE=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=1 ..
cd ..
