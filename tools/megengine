#!/bin/bash

set -e
pushd $(cd $(dirname ${0})/..; pwd) > /dev/null

clean() {
    echo "megengine clean"
    rm -rf ./third_party/source/megengine
}

sync() {
    echo "megengine sync"
    mkdir -p "./third_party/source/megengine/src/serialization/fbs"
    curl --silent --location --output "./third_party/source/megengine/src/serialization/fbs/dtype.fbs" "https://github.com/MegEngine/MegEngine/raw/master/ci/compatibility/fbs/V2-backup/dtype.fbs"
    curl --silent --location --output "./third_party/source/megengine/src/serialization/fbs/mgb_opr_param_defs.fbs" "https://github.com/MegEngine/MegEngine/raw/master/ci/compatibility/fbs/V2-backup/mgb_opr_param_defs.fbs"
    curl --silent --location --output "./third_party/source/megengine/src/serialization/fbs/mgb_cpp_opr.fbs" "https://github.com/MegEngine/MegEngine/raw/master/ci/compatibility/fbs/V2-backup/mgb_cpp_opr.fbs"
    curl --silent --location --output "./third_party/source/megengine/src/serialization/fbs/opr_param_defs.fbs" "https://github.com/MegEngine/MegEngine/raw/master/ci/compatibility/fbs/V2-backup/opr_param_defs.fbs"
    curl --silent --location --output "./third_party/source/megengine/src/serialization/fbs/schema_v2.fbs" "https://github.com/MegEngine/MegEngine/raw/master/ci/compatibility/fbs/V2-backup/schema_v2.fbs"
}

schema() {
    echo "megengine schema"
    [[ $(grep -U $'\x0D' ./source/megengine-schema.js) ]] && crlf=1
    node ./tools/flatc.js --root megengine --out ./source/megengine-schema.js ./third_party/source/megengine/src/serialization/fbs/schema_v2.fbs
    if [[ -n ${crlf} ]]; then
        unix2dos --quiet --newfile ./source/megengine-schema.js ./source/megengine-schema.js
    fi
}

metadata() {
    echo "megengine metadata"
    if [[ $(grep -U $'\x0D' ./source/megengine-metadata.json) ]]; then crlf=1; else crlf=; fi
    node ./tools/megengine-script.js
    if [[ -n ${crlf} ]]; then
        unix2dos --quiet --newfile ./source/megengine-metadata.json ./source/megengine-metadata.json
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
