#!/bin/bash

set -e
pushd $(cd $(dirname ${0})/..; pwd) > /dev/null

schema() {
    echo "rknn schema"
    [[ $(grep -U $'\x0D' ./source/rknn-schema.js) ]] && crlf=1
    node ./tools/flatc.js --root rknn --out ./source/rknn-schema.js ./tools/rknn.fbs
    if [[ -n ${crlf} ]]; then
        unix2dos --quiet --newfile ./source/rknn-schema.js ./source/rknn-schema.js
    fi
}

while [ "$#" != 0 ]; do
    command="$1" && shift
    case "${command}" in
        "schema") schema;;
    esac
done
