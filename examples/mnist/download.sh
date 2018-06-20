#!/bin/bash

if [ $# -gt 1 ]; then
    echo "Too many arguments." >&2
    exit 1
fi

dir=${1:-mnist}

mkdir -p "$dir"
cd "$dir"
wget http://yann.lecun.com/exdb/mnist/{train,t10k}-{images-idx3,labels-idx1}-ubyte.gz
