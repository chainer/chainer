.. module:: chainermn

API Reference
=============


Communicators
~~~~~~~~~~~~~

.. autofunction:: create_communicator
.. autoclass:: CommunicatorBase
    :members: rank, intra_rank, inter_rank, inter_size, size,
              alltoall, split, send, recv, bcast, gather, allreduce,
              send_obj, recv_obj, bcast_obj, gather_obj,
              allreduce_obj, bcast_data, allreduce_grad


Optimizers and Evaluators
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: create_multi_node_optimizer
.. autofunction:: create_multi_node_evaluator


Dataset Utilities
~~~~~~~~~~~~~~~~~

.. autofunction:: scatter_dataset
.. autofunction:: chainermn.datasets.create_empty_dataset


Links
~~~~~

.. autoclass:: MultiNodeChainList
    :members: add_link
.. autoclass:: chainermn.links.MultiNodeBatchNormalization
.. autofunction:: chainermn.links.create_mnbn_model


Functions
~~~~~~~~~

.. autofunction:: chainermn.functions.send
.. autofunction:: chainermn.functions.recv
.. autofunction:: chainermn.functions.pseudo_connect
.. autofunction:: chainermn.functions.bcast
.. autofunction:: chainermn.functions.gather
.. autofunction:: chainermn.functions.scatter
.. autofunction:: chainermn.functions.alltoall
.. autofunction:: chainermn.functions.allgather


Iterators
~~~~~~~~~

.. autofunction:: chainermn.iterators.create_multi_node_iterator
.. autofunction:: chainermn.iterators.create_synchronized_iterator


Trainer extensions
~~~~~~~~~~~~~~~~~~

.. autoclass:: chainermn.extensions.AllreducePersistent
.. autofunction:: chainermn.create_multi_node_checkpointer

Configurations
~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   configurations
