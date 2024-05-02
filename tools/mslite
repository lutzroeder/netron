#!/bin/bash

set -e
pushd $(cd $(dirname ${0})/..; pwd) > /dev/null

clean() {
    echo "mslite clean"
    rm -rf "./third_party/source/mindspore"
}

sync() {
    echo "mslite sync"
    mkdir -p "./third_party/source/mindspore/mindspore/lite/schema/"
    curl --silent --location --output "./third_party/source/mindspore/mindspore/lite/schema/model.fbs" "https://github.com/mindspore-ai/mindspore/raw/master/mindspore/lite/schema/model.fbs"
    curl --silent --location --output "./third_party/source/mindspore/mindspore/lite/schema/ops.fbs" "https://github.com/mindspore-ai/mindspore/raw/master/mindspore/lite/schema/ops.fbs"
    curl --silent --location --output "./third_party/source/mindspore/mindspore/lite/schema/ops_types.fbs" "https://github.com/mindspore-ai/mindspore/raw/master/mindspore/lite/schema/ops_types.fbs"
}

schema() {
    echo "mslite schema"
    [[ $(grep -U $'\x0D' ./source/mslite-schema.js) ]] && crlf=1
    node ./tools/flatc.js --text --root mslite --out ./source/mslite-schema.js ./third_party/source/mindspore/mindspore/lite/schema/model.fbs
    if [[ -n ${crlf} ]]; then
        unix2dos --quiet --newfile ./source/mslite-schema.js ./source/mslite-schema.js
    fi
}

metadata() {
    echo "mslite metadata"
    [[ $(grep -U $'\x0D' ./source/mslite-metadata.json) ]] && crlf=1
    node ./tools/mslite-script.js
    if [[ -n ${crlf} ]]; then
        unix2dos --quiet --newfile ./source/mslite-metadata.json ./source/mslite-metadata.json
    fi
}

while [ "$#" != 0 ]; do
    command="$1" && shift
    case "${command}" in
        "clean") clean;;
        "sync") sync;;
        "schema") schema;;
        "metadata") metadata;;
    esac
done
