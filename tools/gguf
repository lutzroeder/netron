#!/bin/bash

set -e
pushd $(cd $(dirname ${0})/..; pwd) > /dev/null

clean() {
    echo "mslite clean"
    rm -rf "./third_party/source/llama.cpp"
}

sync() {
    echo "ggml sync"
    [ -d "./third_party/source/llama.cpp" ] || git clone --quiet https://github.com/ggerganov/llama.cpp.git "./third_party/source/llama.cpp"
}

while [ "$#" != 0 ]; do
    command="$1" && shift
    case "${command}" in
        "clean") clean;;
        "sync") sync;;
    esac
done
