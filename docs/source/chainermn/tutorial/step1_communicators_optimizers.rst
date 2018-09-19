Step 1: Communicators and Optimizers
====================================

In the following, we explain how to modify your code using Chainer
to enable distributed training with ChainerMN.
We take `Chainer's MNIST example <https://github.com/pfnet/chainer/blob/master/examples/mnist/train_mnist.py>`_
and modify it in a step-by-step manner
to see the standard way of using ChainerMN.


Creating a Communicator
~~~~~~~~~~~~~~~~~~~~~~~

We first need to create a *communicator*.
A communicator is in charge of communication between workers.
A communicator can be created as follows::

  comm = chainermn.create_communicator()


Workers in a node have to use different GPUs.
For this purpose, ``intra_rank`` property of communicators is useful.
Each worker in a node is assigned a unique ``intra_rank`` starting from zero.
Therefore, it is often convenient to use the ``intra_rank``-th GPU.

The following line of code is found in the original MNIST example::

  chainer.cuda.get_device_from_id(args.gpu).use()

which we modify as follows::

  device = comm.intra_rank
  chainer.cuda.get_device_from_id(device).use()


Creating a Multi-Node Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the most important step.
We need to insert the communication right after backprop
and right before optimization.
In ChainerMN, it is done by creating a *multi-node optimizer*.

Method ``create_multi_node_optimizer`` receives a standard Chainer optimizer,
and it returns a new optimizer. The returned optimizer is called multi-node optimizer.
It behaves exactly same as the supplied original standard optimizer
(e.g., you can add hooks such as ``WeightDecay``),
except that it communicates model parameters and gradients properly in a multi-node setting.

The following is the code line found in the original MNIST example::

  optimizer = chainer.optimizers.Adam()


To obtain a multi-node optimizer, we modify that part as follows::

  optimizer = chainermn.create_multi_node_optimizer(
      chainer.optimizers.Adam(), comm)


Run
~~~

With the above two changes, your script is ready for distributed
training.  Invoke your script with ``mpiexec`` or ``mpirun`` (see your
MPI's manual for details).  The following is an example of executing the
training with four processes at localhost::

  $ mpiexec -n 4 python train_mnist.py

In the non-GPU mode, you may see a warning like shown below, 
but this message is harmless, and you can ignore it for now ::

  Warning: using naive communicator because only naive supports CPU-only execution


If you have multiple GPUs on the localhost, 4 for example, you
may also want to try::

  $ mpiexec -n 4 python train_mnist.py --gpu


Multi-node execution
~~~~~~~~~~~~~~~~~~~~

If you can successfully run the multi-process version of the MNIST
example, you are almost ready for multi-node execution. The simplest
way is to specify the ``--host`` argument to the :command:`mpiexec`
command. Let's suppose you have two GPU-equipped computing nodes:
``host00`` and ``host01``, each of which has 4 GPUs, and so you have 8 GPUs
in total::

  $ mpiexec -n 8 -host host00,host01 python train_mnist.py

The script should print similar results to the previous intra-node execution.

Copying datasets
~~~~~~~~~~~~~~~~

In the MNIST example, the rank 0 process reads the entire portion of
the dataset and scatters it to other processes. In some applications,
such as the ImageNet ChainerMN example, however, only the pathes to
each data file are scattered and each process reads the actual data
files. In such cases, all datasets must be readable on all computing
nodes in the same location. You don't need to worry about this if you
use NFS (Network File System) or any other similar data synchronizing
system. Otherwise, you need to manually copy data files between nodes
using :command:`scp` or :command:`rsync`.


If you have trouble
~~~~~~~~~~~~~~~~~~~

If you have any trouble running the sample programs in your
environment, go to the :ref:`troubleshooting` page and follow the
steps to check your environment and configuration.

Next Steps
~~~~~~~~~~

With only the above two changes
distributed training is already performed.
Thus,
the model parameters are updated
by using gradients that are aggregated over all the workers.
However,
this MNIST example still has a few areas in need of improvment.
In the next page, we will see how to address the following problems:

* Training period is wrong; 'one epoch' is not one epoch.
* Evaluation is not parallelized.
* Status outputs to stdout are repeated and annoying.

