#!/bin/bash

set -euox pipefail

cd build
cmake --build . --target $1
