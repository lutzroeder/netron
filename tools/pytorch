#!/bin/bash

set -e
pushd $(cd $(dirname ${0})/..; pwd) > /dev/null

case "${OSTYPE}" in
    msys*) python="winpty python";;
    *) python="python";;
esac

venv() {
    env_dir=./third_party/env/pytorch
    [ -d "${env_dir}" ] || ${python} -m venv ${env_dir}
    case "${OSTYPE}" in
        msys*) source ${env_dir}/Scripts/activate;;
        *) source ${env_dir}/bin/activate;;
    esac
    ${python} -m pip install --quiet --upgrade pip requests
}

clean() {
    echo "pytorch clean"
    rm -rf "./third_party/source/pytorch"
    rm -rf "./third_party/source/executorch/schema"
}

sync() {
    echo "pytorch sync"
    [ -d "./third_party/source/pytorch" ] || git clone --quiet https://github.com/pytorch/pytorch.git "./third_party/source/pytorch"
    git -C "./third_party/source/pytorch" pull --quiet --prune
    mkdir -p "./third_party/source/executorch/schema"
    curl --silent --location --output "./third_party/source/executorch/schema/scalar_type.fbs" "https://github.com/pytorch/executorch/raw/main/schema/scalar_type.fbs"
    curl --silent --location --output "./third_party/source/executorch/schema/program.fbs" "https://github.com/pytorch/executorch/raw/main/schema/program.fbs"
}

install() {
    echo "pytorch install"
    venv
    ${python} -m pip install --quiet --upgrade wheel
    ${python} -m pip install --quiet --upgrade  --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
    # ${python} -m pip install --quiet --upgrade torch-neuron --index-url https://pip.repos.neuron.amazonaws.com
    deactivate
}


schema() {
    [[ $(grep -U $'\x0D' ./source/pytorch-schema.js) ]] && crlf=1
    echo "pytorch schema"
    node ./tools/flatc.js --root torch --out ./source/pytorch-schema.js ./third_party/source/pytorch/torch/csrc/jit/serialization/mobile_bytecode.fbs ./third_party/source/executorch/schema/program.fbs
    if [[ -n ${crlf} ]]; then
        unix2dos --quiet --newfile ./source/pytorch-schema.js ./source/pytorch-schema.js
    fi
}

metadata() {
    echo "pytorch metadata"
    [[ $(grep -U $'\x0D' ./source/pytorch-metadata.json) ]] && crlf=1
    ${python} ./tools/pytorch_script.py
    if [[ -n ${crlf} ]]; then
        unix2dos --quiet --newfile ./source/pytorch-metadata.json ./source/pytorch-metadata.json
    fi
}

while [ "$#" != 0 ]; do
    command="$1" && shift
    case "${command}" in
        "clean") clean;;
        "install") install;;
        "sync") sync;;
        "schema") schema;;
        "metadata") metadata;;
    esac
done
