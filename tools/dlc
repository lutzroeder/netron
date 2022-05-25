#!/bin/bash

set -e
pushd $(cd $(dirname ${0})/..; pwd) > /dev/null

schema() {
    echo "dlc schema"
    [[ $(grep -U $'\x0D' ./source/dlc-schema.js) ]] && crlf=1
    node ./tools/flatc.js --root dlc --out ./source/dlc-schema.js ./tools/dlc.fbs
    if [[ -n ${crlf} ]]; then
        unix2dos --quiet --newfile ./source/dlc-schema.js ./source/dlc-schema.js
    fi
}

while [ "$#" != 0 ]; do
    command="$1" && shift
    case "${command}" in
        "schema") schema;;
    esac
done
