#!/bin/bash

if [ $# -eq 0 ]; then
    dst="mnist"
elif [ $# -eq 1 ]; then
    dst="$1"
else
    echo "Too many arguments." >&2
    exit 1
fi

mkdir -p "$dst"
cd "$dst"
wget http://yann.lecun.com/exdb/mnist/{train,t10k}-{images-idx3,labels-idx1}-ubyte.gz
