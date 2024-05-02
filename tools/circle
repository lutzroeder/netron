#!/bin/bash

set -e
pushd $(cd $(dirname ${0})/..; pwd) > /dev/null

clean() {
    echo "circle clean"
    rm -rf "./third_party/source/circle"
}

sync() {
    echo "circle sync"
    mkdir -p "./third_party/source/circle/nnpackage/schema"
    curl --silent --location --output "./third_party/source/circle/nnpackage/schema/circle_schema.fbs" "https://github.com/Samsung/ONE/raw/master/nnpackage/schema/circle_schema.fbs"
}

schema() {
    echo "circle schema"
    [[ $(grep -U $'\x0D' ./source/circle-schema.js) ]] && crlf=1
    node ./tools/flatc.js --text --root circle --out ./source/circle-schema.js ./third_party/source/circle/nnpackage/schema/circle_schema.fbs
    if [[ -n ${crlf} ]]; then
        unix2dos --quiet --newfile ./source/circle-schema.js ./source/circle-schema.js
    fi
}

metadata() {
    echo "circle metadata"
    if [[ $(grep -U $'\x0D' ./source/circle-metadata.json) ]]; then crlf=1; else crlf=; fi
    node ./tools/circle-script.js
    if [[ -n ${crlf} ]]; then
        unix2dos --quiet --newfile ./source/circle-metadata.json ./source/circle-metadata.json
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
