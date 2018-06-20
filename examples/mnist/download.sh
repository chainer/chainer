#!/bin/bash
if [ ! -d mnist ]; then
    mkdir -p mnist
    cd mnist
    wget http://yann.lecun.com/exdb/mnist/{train,t10k}-{images-idx3,labels-idx1}-ubyte.gz
else
    echo "Found already existing MNIST dataset directory. Skipping download."
fi
