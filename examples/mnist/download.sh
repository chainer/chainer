#!/bin/bash

if [ $# != 1 ]; then
    echo "Too many or too few arguments." >&2
    exit 1
fi

mkdir -p $1
cd $1
wget http://yann.lecun.com/exdb/mnist/{train,t10k}-{images-idx3,labels-idx1}-ubyte.gz
