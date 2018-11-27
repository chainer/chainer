Model Parallel on ChainerMN
===========================


Step 1: Communicators
~~~~~~~~~~~~~~~~~~~~~

To perform multi-node communications, a *communicator* is needed.
Basic usages are the same with the case of the data parallel, see :doc:`../tutorial/step1_communicators_optimizers`::

    comm = chainermn.create_communicator()

If you want to define collective communications among limited number of processes later, it is useful to split the communicator::

    subcomm = comm.split(comm.rank % 2, comm.rank)

.. figure:: ../../../image/model_parallel/comm_split.png
    :align: center
    :scale: 50%

For further detail about the communicator split, please refer to `MPI tutorial <http://mpitutorial.com/tutorials/introduction-to-groups-and-communicators/>`__.



Step 2: Datasets and Iterators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In model parallel, there are three types of processes regarding to datasets.

1. model inputs come from datasets, and each process takes different mini-batches
2. model inputs come from datasets, and several processes share the same mini-batches
3. model inputs come from other processes


1. scatter_dataset
------------------

For the first case, you may use ``scatter_dataset`` as is introduced in :doc:`../tutorial/step2_datasets_evaluators`.

.. figure:: ../../../image/model_parallel/scatter_dataset.png
    :align: center

2. multi node iterator
----------------------

For the second case, iterator need to be modified, where ``create_multi_node_iterator`` is useful::

    train, test = chainer.datasets.get_mnist()
    train_iter = chainermn.iterators.create_multi_node_iterator(
        chainer.iterators.SerialIterator(train, batchsize), comm)
    test_iter = chainermn.iterators.create_multi_node_iterator(
        chainer.iterators.SerialIterator(test, batchsize), comm)

The resulting iterators return the same mini-batches among processes specified by the communicator.

.. figure:: ../../../image/model_parallel/multi_node_iterator.png
    :align: center

3. empty dataset
----------------

For the last case, you may use ``create_empty_dataset``, which returns a dataset with the same number of empty tuples as the original dataset::

    train, test = chainer.datasets.get_mnist()
    train = chainermn.datasets.create_empty_dataset(train)
    test = chainermn.datasets.create_empty_dataset(test)

Note that datasets are required in Chainer's API. The empty dataset can be used as a dummy dataset.

.. figure:: ../../../image/model_parallel/empty_dataset.png
    :align: center
    :scale: 40%


Step 3: Define Communications 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ChainerMN supports most of the MPI communications *as Chainer functions*, including point-to-point and collective communications.
To know usages of each communication, please refer to :doc:`../reference/index`.

Example 1: Point-to-point Communication
---------------------------------------

This is an example to use point-to-point communications::

    def __call__(self, x):
        h = f(x)
        h = chainermn.functions.send(x)
        return h

Note that the return value of ``send`` is often not negligible.
Please refer to :ref:`pseudo-connect`.


Example 2: Collective Communication
-----------------------------------

Here is another example to use collective communications::

    def __call__(self, x):
        h = f(x)
        h = chainermn.functions.allgather(h)
        h = F.stack(h, axis=0)
        h = F.average(h, axis=0)
        return h

This pattern often appears in the averaging ensemble training.


.. _pseudo-connect:

Note: Define-by-Run and Model Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
