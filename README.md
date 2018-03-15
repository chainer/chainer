<div align="center"><img src="docs/image/chainer_red_h.png" width="400"/></div>

# Chainer: A deep learning framework

[![pypi](https://img.shields.io/pypi/v/chainer.svg)](https://pypi.python.org/pypi/chainer)
[![GitHub license](https://img.shields.io/github/license/chainer/chainer.svg)](https://github.com/chainer/chainer)
[![travis](https://img.shields.io/travis/chainer/chainer/master.svg)](https://travis-ci.org/chainer/chainer)
[![coveralls](https://img.shields.io/coveralls/chainer/chainer.svg)](https://coveralls.io/github/chainer/chainer)
[![Read the Docs](https://readthedocs.org/projects/chainer/badge/?version=stable)](https://docs.chainer.org/en/stable/?badge=stable)

[**Website**](https://chainer.org/)
| [**Docs**](https://docs.chainer.org/en/stable/)
| [**Install Guide**](https://docs.chainer.org/en/stable/install.html)
| [**Tutorial**](https://docs.chainer.org/en/stable/tutorial/)
| **Examples** ([Official](https://github.com/chainer/chainer/tree/master/examples), [External](https://github.com/chainer/chainer/wiki/External-examples))

**Forum** ([en](https://groups.google.com/forum/#!forum/chainer), [ja](https://groups.google.com/forum/#!forum/chainer-jp))
| **Slack invitation** ([en](https://bit.ly/join-chainer-slack), [ja](https://bit.ly/join-chainer-jp-slack))
| **Slack archive** ([en](https://chainer.slackarchive.io), [ja](https://chainer-jp.slackarchive.io))
| **Twitter** ([en](https://twitter.com/ChainerOfficial), [ja](https://twitter.com/ChainerJP))

*Chainer* is a Python-based deep learning framework aiming at flexibility.
It provides automatic differentiation APIs based on the **define-by-run** approach (a.k.a. dynamic computational graphs) as well as object-oriented high-level APIs to build and train neural networks.
It also supports CUDA/cuDNN using [CuPy](https://github.com/cupy/cupy) for high performance training and inference.
For more details of Chainer, see the documents and resources listed above and join the community in Forum, Slack, and Twitter.

## Stable version

The stable version of current Chainer is separated in here: [v3](https://github.com/chainer/chainer/tree/v3).

## Installation

To install Chainer, use `pip`.

```sh
$ pip install chainer
```

To enable CUDA support, [set up CUDA](https://docs.nvidia.com/cuda/index.html#installation-guides) and install [CuPy](https://github.com/cupy/cupy).

```sh
$ pip install cupy
```

[See the installation guide for more details.](https://docs.chainer.org/en/stable/install.html).


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


## License

MIT License (see `LICENSE` file).


## More information

- [Release notes](https://github.com/chainer/chainer/releases)
- [Research projects using Chainer](https://github.com/chainer/chainer/wiki/Research-projects-using-Chainer)


## Reference

Tokui, S., Oono, K., Hido, S. and Clayton, J.,
Chainer: a Next-Generation Open Source Framework for Deep Learning,
*Proceedings of Workshop on Machine Learning Systems(LearningSys) in
The Twenty-ninth Annual Conference on Neural Information Processing Systems (NIPS)*, (2015)
[URL](http://learningsys.org/papers/LearningSys_2015_paper_33.pdf), [BibTex](chainer_bibtex.txt)
