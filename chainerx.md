# ChainerX User Guide

*ChainerX*, or `chainerx` is a standalone Python package that's integrated into Chainer.
It is implemented almost purely in C++ where the C++ implementations are exposed via Python bindings, allowing Chainer to make use of it.
It does in other words **not** replace Chainer. It aims to instead improve the performance of Chainer in terms of speed by reducing the Python overhead.

This guide is aimed toward users familiar with the Chainer interface but want to improve their training/inference speed, using ChainerX.
It explains how to install it, the motivation behind it and how to modify your existing Chainer code to use ChainerX.

- [Installation](#installation)
- [About ChainerX](#about-chainerx)
- [Known issues](#known-issues)
- [FAQ](#faq)

## Installation

See [ChainerX installation](docs/source/chainerx/install/index.rst).

## About ChainerX

See [ChainerX tutorial](docs/source/chainerx/tutorial/index.rst).

## Known Issues

See: [ChainerX limitations](docs/source/chainerx/limitations.rst)

## FAQ

### Can I use ChainerX without Chainer?

Yes, it is possible. See the code samples below.

- [Train an MLP with MNIST](chainerx_cc/examples/mnist)
- [Train a CNN with ImageNet](chainerx_cc/examples/imagenet)
- [`chainerx` basic usage examples](tests/chainerx_tests/acceptance_tests)

### What does the C++ interface look like?

It is almost identical to the Python interface with a 1-to-1 mapping.
The binding layer is thin and many of the types defined in C++ have Python equivalent classes.
The bindings are defined [here](https://github.com/pfnet/chainerx/tree/master/chainerx_cc/chainerx/python).
