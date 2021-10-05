#!/bin/bash

set -euox pipefail

if [ ! -d "build" ]; then
  mkdir build
fi

cd third_party
if [ ! -d "libtorch" ]; then
  wget https://download.pytorch.org/libtorch/cu111/libtorch-shared-with-deps-1.9.1%2Bcu111.zip
  mkdir libtorch
  unzip libtorch-shared-with-deps-1.9.1+cu111.zip -d libtorch
fi

cd build
cmake -GNinja -DCMAKE_BUILD_TYPE=Release -DUSE_DOUBLE=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DUSE_HEADLESS=1 -DUSE_NEURAL_NETWORK=1 ..
cd ..
ln -s $(pwd)/build/compile_commands.json $(pwd)
