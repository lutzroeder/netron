#!/bin/bash

set -e
pushd $(cd $(dirname ${0})/..; pwd) > /dev/null

case "${OSTYPE}" in
    msys*) python="winpty python";;
    *) python="python";;
esac

clean() {
    echo "caffe2 clean"
    rm -rf "./third_party/source/caffe2"
}

sync() {
    echo "caffe2 sync"
    mkdir -p "./third_party/source/caffe2/proto"
    curl --silent --location --output "./third_party/source/caffe2/proto/caffe2.proto" "https://github.com/pytorch/pytorch/raw/ff8042bcfb518127c86ad5b4af4fa9171a499904/caffe2/proto/caffe2.proto"
}

schema() {
    [[ $(grep -U $'\x0D' ./source/caffe2-proto.js) ]] && crlf=1
    echo "caffe2 schema"
    node ./tools/protoc.js --text --root caffe2 --out ./source/caffe2-proto.js ./third_party/source/caffe2/proto/caffe2.proto
    if [[ -n ${crlf} ]]; then
        unix2dos --quiet --newfile ./source/caffe2-proto.js ./source/caffe2-proto.js
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
