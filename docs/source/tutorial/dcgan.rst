DCGAN: Generate the images from Deep Convolutinal GAN
******************************************************

.. currentmodule:: chainer

0. Introduction
================


1. Basic Idea of Generative Model
==================================

2. the Difference among GAN and Other Generative Models
========================================================


=====================   ===============================================================================================================================
Symbol                  Definition                                               
=====================   ===============================================================================================================================
:math:`|\mathcal{V}|`   The size of vocabulary                                   
:math:`D`               The size of embedding vector                             
:math:`{\bf v}_t`       A one-hot center word vector                             
:math:`V_{t \pm C}`     A set of :math:`2C` context vectors around :math:`{\bf v}_t`, namely, :math:`\{{\bf v}_{t+c}\}_{c=-C}^C \backslash {\bf v}_t`
:math:`{\bf l}_H`       An embedding vector of an input word vector              
:math:`{\bf l}_O`       An output vector of the network                          
:math:`{\bf W}_H`       The embedding matrix for inputs                          
:math:`{\bf W}_O`       The embedding matrix for outputs                         
=====================   ===============================================================================================================================

.. note::

    Using **negative sampling** or **hierarchical softmax** for the loss
    function is very common, however, in this tutorial, we will use the
    **softmax over all words** and skip the other variants for the sake
    of simplicity.

2.1 Skip-gram
--------------

3. Details of GAN
==================

3.1 How GAN works?
-------------------

3.1 What is DCGAN?
-------------------

4. Implementation of DCGAN in Chainer
==========================================

There is an example of Word2vec in the official repository of Chainer,
so we will explain how to implement Skip-gram based on this:
`chainer/examples/dcgan <https://github.com/chainer/chainer/tree/master/examples/dcgan>`_

4.1 Preparation
--------------------------

First, let's import necessary packages:

.. literalinclude:: ../../../examples/dcgan/train_dcgan.py
   :language: python
   :lines: 6-20
   :caption: train_dcgan.py

4.2 Define a Skip-gram model
-----------------------------

Next, let's define a network for Skip-gram.

.. literalinclude:: ../../../examples/dcgan/train_dcgan.py
   :language: python
   :pyobject: SkipGram
   :caption: train_dcgan.py

.. literalinclude:: ../../../examples/dcgan/train_dcgan.py
   :language: python
   :pyobject: SoftmaxCrossEntropyLoss
   :caption: train_dcgan.py

.. note::

    * The weight matrix ``self.embed.W`` is the embedding matrix for input vector
      ``x``.
    * The function call ``__call__`` takes the word ID of a center word ``x`` and
      word IDs of context words contexts as inputs, and outputs the error calculated
      by the loss function ``loss_func`` s.t. ``SoftmaxCrossEntropyLoss``.
    * Note that the initial shape of ``x`` and contexts are ``(batch_size,)``
      and ``(batch_size, n_context)``, respectively.
    * The ``batch_size`` means the size of mini-batch, and ``n_context`` means the
      number of context words.
      
First, we obtain the embedding vectors of contexts by ``e = self.embed(contexts)``.
Then ``F.broadcast_to(x[:, None], (shape[0], shape[1]))`` performs broadcasting of
``x`` (its shape is ``(batch_size,)``) to ``(batch_size, n_context)`` by copying the
same value ``n_context`` time to fill the second axis, and then the broadcasted ``x``
is reshaped into 1-D vector ``(batchsize * n_context,)`` while ``e`` is reshaped to
``(batch_size * n_context, n_units)``.
In Skip-gram model, predicting a context word from the center word is the same as
predicting the center word from a context word because the center word is always
a context word when considering the context word as a center word. So, we create
``batch_size * n_context`` center word predictions by applying ``self.out`` linear
layer to the embedding vectors of context words. Then, calculate softmax cross
entropy between the broadcasted center word ID x and the predictions.

4.3 Prepare dataset and iterator
---------------------------------

Let's retrieve the Penn Tree Bank (PTB) dataset by using Chainer's dataset utility
``get_ptb_words()`` method.

.. literalinclude:: ../../../examples/dcgan/train_dcgan.py
   :language: python
   :start-after: Load the dataset
   :end-before: counts.update
   :dedent: 4

Then define an iterator to make mini-batches that contain a set of center words with their context words.
``train`` and ``val`` means training data and validation data. Each data contains
the list of Document IDs:

    .. code-block:: console
    
        >>> train
        array([ 0,  1,  2, ..., 39, 26, 24], dtype=int32)
        >>> val
        array([2211,  396, 1129, ...,  108,   27,   24], dtype=int32)

.. literalinclude:: ../../../examples/dcgan/train_dcgan.py
   :language: python
   :pyobject: WindowIterator
   :caption: train_dcgan.py

* In the constructor, we create an array ``self.order`` which denotes shuffled
  indices of ``[window, window + 1, ..., len(dataset) - window - 1]`` in order to
  choose a center word randomly from dataset in a mini-batch.
* The iterator definition ``__next__`` returns ``batch_size`` sets of center word
  and context words.
* The code ``self.order[i:i_end]`` returns the indices for a set of center words
  from the random-ordered array ``self.order``. The center word IDs center at the
  random indices are retrieved by ``self.dataset.take``.
* ``np.concatenate([np.arange(-w, 0), np.arange(1, w + 1)])`` creates a set of
  offsets to retrieve context words from the dataset.
* The code ``position[:, None] + offset[None, :]`` generates the indices of context
  words for each center word index in position. The context word IDs context are
  retrieved by ``self.dataset.take``.

4.4 Prepare model, optimizer, and updater
------------------------------------------

.. literalinclude:: ../../../examples/dcgan/train_dcgan.py
   :language: python
   :start-after: if args.model
   :end-before: elif args.model
   :caption: train_dcgan.py
   :dedent: 4

.. literalinclude:: ../../../examples/dcgan/train_dcgan.py
   :language: python
   :start-after: Set up an optimizer
   :end-before: Set up an iterator
   :caption: train_dcgan.py
   :dedent: 4

.. literalinclude:: ../../../examples/dcgan/train_dcgan.py
   :language: python
   :start-after: Set up an iterator
   :end-before: Set up a trainer
   :caption: train_dcgan.py
   :dedent: 4

.. literalinclude:: ../../../examples/dcgan/train_dcgan.py
   :language: python
   :start-after: Set up a trainer
   :end-before: Save the word2vec model
   :caption: train_dcgan.py
   :dedent: 4

4.5 Start training
-------------------

.. code-block:: console

    $ pwd
    /root2chainer/chainer/examples/word2vec

5. Reference
=============
* [1] `Mikolov, Tomas; et al. "Efficient Estimation of Word Representations in Vector Space". arXiv:1301.3781 <https://arxiv.org/abs/1301.3781>`_
* [2] `Distributional Hypothesis <https://aclweb.org/aclwiki/Distributional_Hypothesis>`_
