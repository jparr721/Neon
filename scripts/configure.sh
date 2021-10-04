#!/bin/bash

set -euox pipefail

mkdir build
cd build
cmake -GNinja -DCMAKE_BUILD_TYPE=Release -DUSE_DOUBLE=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=1 ..
cd ..
