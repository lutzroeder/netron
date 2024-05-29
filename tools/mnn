#!/bin/bash

set -e
pushd $(cd $(dirname ${0})/..; pwd) > /dev/null

clean() {
    echo "mnn clean"
    rm -rf "./third_party/source/mnn"
}

sync() {
    echo "mnn sync"
    mkdir -p "./third_party/source/mnn/schema/default"
    curl --silent --location --output "./third_party/source/mnn/schema/default/CaffeOp.fbs" "https://github.com/alibaba/MNN/raw/master/schema/default/CaffeOp.fbs"
    curl --silent --location --output "./third_party/source/mnn/schema/default/ExtraInfo.fbs" "https://github.com/alibaba/MNN/raw/master/schema/default/ExtraInfo.fbs"
    curl --silent --location --output "./third_party/source/mnn/schema/default/MNN.fbs" "https://github.com/alibaba/MNN/raw/master/schema/default/MNN.fbs"
    curl --silent --location --output "./third_party/source/mnn/schema/default/Tensor.fbs" "https://github.com/alibaba/MNN/raw/master/schema/default/Tensor.fbs"
    curl --silent --location --output "./third_party/source/mnn/schema/default/TensorflowOp.fbs" "https://github.com/alibaba/MNN/raw/master/schema/default/TensorflowOp.fbs"
    curl --silent --location --output "./third_party/source/mnn/schema/default/TFQuantizeOp.fbs" "https://github.com/alibaba/MNN/raw/master/schema/default/TFQuantizeOp.fbs"
    curl --silent --location --output "./third_party/source/mnn/schema/default/Type.fbs" "https://github.com/alibaba/MNN/raw/master/schema/default/Type.fbs"
    curl --silent --location --output "./third_party/source/mnn/schema/default/UserDefine.fbs" "https://github.com/alibaba/MNN/raw/master/schema/default/UserDefine.fbs"
}

schema() {
    echo "mnn schema"
    [[ $(grep -U $'\x0D' ./source/mnn-schema.js) ]] && crlf=1
    node ./tools/flatc.js --text --root mnn --out ./source/mnn-schema.js ./third_party/source/mnn/schema/default/MNN.fbs
    if [[ -n ${crlf} ]]; then
        unix2dos --quiet --newfile ./source/mnn-schema.js ./source/mnn-schema.js
    fi
}

while [ "$#" != 0 ]; do
    command="$1" && shift
    case "${command}" in
        "clean") clean;;
        "sync") sync;;
        "schema") schema;;
    esac
done
