<div align="center"><img src="https://raw.githubusercontent.com/chainer/chainer/master/docs/image/chainer_red_h.png" width="400"/></div>

# Chainer: A deep learning framework

[![pypi](https://img.shields.io/pypi/v/chainer.svg)](https://pypi.python.org/pypi/chainer)
[![GitHub license](https://img.shields.io/github/license/chainer/chainer.svg)](https://github.com/chainer/chainer)
[![travis](https://img.shields.io/travis/chainer/chainer/master.svg)](https://travis-ci.org/chainer/chainer)
[![coveralls](https://img.shields.io/coveralls/chainer/chainer.svg)](https://coveralls.io/github/chainer/chainer)
[![Read the Docs](https://readthedocs.org/projects/chainer/badge/?version=stable)](https://docs.chainer.org/en/stable/?badge=stable)

[**Website**](https://chainer.org/)
| [**Docs**](https://docs.chainer.org/en/stable/)
| [**Install Guide**](https://docs.chainer.org/en/stable/install.html)
| **Tutorials** ([ja](https://tutorials.chainer.org/ja/))
| **Examples** ([Official](https://github.com/chainer/chainer/tree/master/examples), [External](https://github.com/chainer-community/awesome-chainer))
| [**Concepts**](https://docs.chainer.org/en/stable/guides/)
| [**ChainerX**](#chainerx)

**Forum** ([en](https://groups.google.com/forum/#!forum/chainer), [ja](https://groups.google.com/forum/#!forum/chainer-jp))
| **Slack invitation** ([en](https://bit.ly/join-chainer-slack), [ja](https://bit.ly/join-chainer-jp-slack))
| **Twitter** ([en](https://twitter.com/ChainerOfficial), [ja](https://twitter.com/ChainerJP))

*Chainer* is a Python-based deep learning framework aiming at flexibility.
It provides automatic differentiation APIs based on the **define-by-run** approach (a.k.a. dynamic computational graphs) as well as object-oriented high-level APIs to build and train neural networks.
It also supports CUDA/cuDNN using [CuPy](https://github.com/cupy/cupy) for high performance training and inference.
For more details about Chainer, see the documents and resources listed above and join the community in Forum, Slack, and Twitter.

## My Contributions (@dido1998)
### Before GSoC selection
- [Implementation of sigmoid for ChainerX](https://github.com/chainer/chainer/pull/6472)[Merged]

  Implemented the sigmoid routine and wrote corresponding tests.
- [Dot Product for higher dimensions for ChainerX](https://github.com/chainer/chainer/pull/6476)[Merged]

  ChainerX only supported dot for <=2-dimensional arrays, after this it was able to support higher dimensions also.
- [Elementwise power operator for ChainerX](https://github.com/chainer/chainer/pull/6496)[Merged]

  This supports array^array, scalar^array and array^scalar.
- [Implementation of absolute for ChainerX](https://github.com/chainer/chainer/pull/6715) [Merged]
- [Implementation of Relu for ChainerX] (https://github.com/chainer/chainer/pull/6731)[Merged]

### After GSoC selection
- [LSTM implementation for ChainerX](https://github.com/chainer/chainer/pull/7282)[open]

  This includes the CPU and GPU(CUDNN) implementation of multilayer uni-directional and bi-directional LSTMs.
- [GRU implementation for ChainerX](https://github.com/chainer/chainer/pull/7678)[open]

  This includes the CPU and GPU(CUDNN) implementation of multilayer uni-directional and bi-directional GRUs.
- [Vanilla RNN implementation for ChainerX](https://github.com/chainer/chainer/pull/7764)[open]

  This includes the CPU and GPU(CUDNN) implementation of multilayer uni-directional and bi-directional RNNs.
- [TreeLSTM implementation for ChainerX](https://github.com/chainer/chainer/pull/7720)[open]

  This includes the implementation of tree-lstm.
- [SLSTM implementation for ChainerX](https://github.com/chainer/chainer/pull/7783)[open]

  This includes the implementation of slstm.
- [Word embeddings for ChainerX](https://github.com/chainer/chainer/pull/7784)[open]
  Word embeddings support for ChainerX
  

## Stable version

The stable version of current Chainer is separated in here: [v5](https://github.com/chainer/chainer/tree/v5).

## Installation

To install Chainer, use `pip`.

```sh
$ pip install chainer
```

To enable CUDA support, [set up CUDA](https://docs.nvidia.com/cuda/index.html#installation-guides) and install [CuPy](https://github.com/cupy/cupy).

```sh
$ pip install cupy
```

[See the installation guide for more details](https://docs.chainer.org/en/stable/install.html).


## Docker image

We are providing the official Docker image.
This image supports [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).
Login to the environment with the following command, and run the Python interpreter to use Chainer with CUDA and cuDNN support.

```
$ nvidia-docker run -it chainer/chainer /bin/bash
```


## Contribution

Any contributions to Chainer are welcome!
If you want to file an issue or send a pull request, [please follow the contribution guide](https://docs.chainer.org/en/stable/contribution.html).


## ChainerX

See the [ChainerX documentation](https://docs.chainer.org/en/stable/chainerx/index.html).


## License

MIT License (see `LICENSE` file).


## More information

- [Release notes](https://github.com/chainer/chainer/releases)


## Reference

Tokui, S., Oono, K., Hido, S. and Clayton, J.,
Chainer: a Next-Generation Open Source Framework for Deep Learning,
*Proceedings of Workshop on Machine Learning Systems(LearningSys) in
The Twenty-ninth Annual Conference on Neural Information Processing Systems (NIPS)*, (2015)
[URL](http://learningsys.org/papers/LearningSys_2015_paper_33.pdf), [BibTex](chainer_bibtex.txt)


Akiba, T., Fukuda, K. and Suzuki, S.,
ChainerMN: Scalable Distributed Deep Learning Framework,
*Proceedings of Workshop on ML Systems in
The Thirty-first Annual Conference on Neural Information Processing Systems (NIPS)*, (2017)
[URL](http://learningsys.org/nips17/assets/papers/paper_25.pdf), [BibTex](chainermn_bibtex.txt)
