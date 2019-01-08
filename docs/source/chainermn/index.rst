.. ChainerMN documentation master file, created by
   sphinx-quickstart on Fri Apr  7 14:20:41 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Distributed Deep Learning with ChainerMN
========================================

ChainerMN enables multi-node distributed deep learning with the following features:

* **Scalable** --- it makes full use of the latest technologies such as NVIDIA NCCL and CUDA-Aware MPI,
* **Flexible** --- even dynamic neural networks can be trained in parallel thanks to Chainer's flexibility, and
* **Easy** --- minimal changes to existing user code are required.

`This blog post <http://chainer.org/general/2017/02/08/Performance-of-Distributed-Deep-Learning-Using-ChainerMN.html>`__ provides our benchmark results using up to 128 GPUs.

ChainerMN can be used for both inner-node (i.e., multiple GPUs inside a node) and inter-node settings.
For inter-node settings, we highly recommend to use high-speed interconnects such as InfiniBand.

ChainerMN examples are available on `GitHub <https://github.com/chainer/chainer/tree/master/examples/chainermn/>`__.
These examples are based on the `examples of Chainer <https://github.com/chainer/chainer/tree/master/examples/>`__ and the differences are highlighted.

.. toctree::
   :maxdepth: 2

   installation/index
   tutorial/index
   model_parallel/index
   reference/index

