#!/bin/bash

pushd "$(dirname $0)" > /dev/null

if [[ "$1" == "-lazy" ]] && [[ -d current ]]; then
  popd > /dev/null
  echo "done ..."
  exit
fi

# check is flatbuffer installed or not
FLATC=../3rd/flatbuffers/tmp/flatc
if [ ! -e $FLATC ]; then
  echo "building flatc ..."

  # make tmp dir
  pushd ../3rd/flatbuffers > /dev/null
  [ ! -d tmp ] && mkdir tmp
  cd tmp && rm -rf *

  # build
  cmake .. && cmake --build . --target flatc -- -j4

  # dir recover
  popd > /dev/null
fi

# determine directory to use
DIR="fbs"
if [ -d "private" ]; then
  DIR="private"
fi

# clean up
echo "cleaning up ..."
rm -f gen/*.h
[ ! -d gen ] && mkdir gen

# flatc all fbs
pushd gen > /dev/null
echo "generating fbs under $DIR ..."
find ../$DIR/*.fbs | xargs ../$FLATC -c -b --gen-object-api --reflect-names
popd > /dev/null

# finish
popd > /dev/null
echo "done ..."
