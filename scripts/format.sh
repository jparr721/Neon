#!/bin/bash

set -euox pipefail

clang-format $(find datasets meshing solvers utilities visualizer -maxdepth 5 -name '*.cpp' -o name '*.h') -i
