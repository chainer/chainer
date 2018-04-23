#!/bin/bash
mkdir -p mnist
cd mnist
wget http://yann.lecun.com/exdb/mnist/{train,t10k}-{images-idx3,labels-idx1}-ubyte.gz
