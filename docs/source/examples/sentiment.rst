Sentiment Analysis with Recursive Neural Network
*************************************************

.. currentmodule:: chainer

0. Introduction
================

In this tutorial, we will use the Recursive Neural Network to analyze sentiment of sentences.

Sentiment analysis is one of the major tasks of Natural Language Processing (NLP),
which identifies writers’ sentiments for sentences. When expressing sentiment,
we basically uses labels whether it is positive or negative. For example,
in the case of the dataset used this time, emotions are expressed in 5 stages
like 1(really negative), 2(negative), 3(neutral), 4(positive), 5(really positive).

.. figure:: ../../image/sentiment/sentiment-treebank.png
    :scale: 100%

    cited from [1]

Sentiment analysis is implemented with Recursive Neural Network. Recursive Neural
Network is a recursive neural net with a tree structure. NLP often expresses
sentences in a tree structure, Recursive Neural Network is often used in NLP.
First, we explain the training method of Recursive Neural Network without
mini-batch processing. After that, as a advanced story, I explain the training
method of mini-batched Recursive Neural Network.

In this tutorial, you will understand the following:

#. What is Recursive Neural Network?
#. Implementation of sentiment analysis by Recursive Neural Network using Chainer

    * Training method of Recursive Neural Network without mini-batch
    * Training method of mini-batched Recursive Neural Network

1. What is Recursive Neural Network? [2]
========================================

Recursive Neural Network is one of Recurrent Neural Networks that extended to
a tree structure. As both networks is often written as RNN, so you need to be
careful which one you are expressing. In many cases, it refers to the Recurrent
Neural Network in many cases, but in natural language processing it sometimes
refers to the Recursive Neural Network.

Recursive Neural Network uses a tree structure with a fixed number of branches.
In the case of a binary tree, the hidden state vector of the current node is
computed from the hidden state vectors of the left and right child nodes,
as follows:

.. math::
    {\bf h}_P = a \left( {\bf W} 
    \left[ \begin{array}{l}
        {\bf h}_L \\
        {\bf h}_R \\
    \end{array} \right]
    + {\bf b} \right)

.. image:: ../../image/sentiment/rnn.png

This operation is sequentially calculated from the leaf nodes toward the root node.
Recursive Neural Network is expected to express relationships between long-distance
elements compared to Recurrent Neural Network, because the depth is enough with
:math:`log_2(T)` if the element count is :math:`T`.

2. Implementation of sentiment analysis by Recursive Neural Network
====================================================================


4. Implementation of Skip-gram in Chainer
==========================================

There is an example of Word2vec in the official repository of Chainer,
so we will explain how to implement Skip-gram based on this:
:tree:`examples/word2vec`

4.1 Preparation
--------------------------

First, let's import necessary packages:

.. literalinclude:: ../../../examples/word2vec/train_word2vec.py
   :language: python
   :lines: 6-20
   :caption: train_word2vec.py

4.2 Define a Skip-gram model
-----------------------------

Next, let's define a network for Skip-gram.

.. literalinclude:: ../../../examples/word2vec/train_word2vec.py
   :language: python
   :pyobject: SkipGram
   :caption: train_word2vec.py

.. literalinclude:: ../../../examples/word2vec/train_word2vec.py
   :language: python
   :pyobject: SoftmaxCrossEntropyLoss
   :caption: train_word2vec.py

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
Then ``F.broadcast_to(x[:, None], (batch_size, n_context))`` performs broadcasting of
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

.. literalinclude:: ../../../examples/word2vec/train_word2vec.py
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

.. literalinclude:: ../../../examples/word2vec/train_word2vec.py
   :language: python
   :pyobject: WindowIterator
   :caption: train_word2vec.py

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

.. literalinclude:: ../../../examples/word2vec/train_word2vec.py
   :language: python
   :start-after: if args.model
   :end-before: elif args.model
   :caption: train_word2vec.py
   :dedent: 4

.. literalinclude:: ../../../examples/word2vec/train_word2vec.py
   :language: python
   :start-after: Set up an optimizer
   :end-before: Set up an iterator
   :caption: train_word2vec.py
   :dedent: 4

.. literalinclude:: ../../../examples/word2vec/train_word2vec.py
   :language: python
   :start-after: Set up an iterator
   :end-before: Set up a trainer
   :caption: train_word2vec.py
   :dedent: 4

.. literalinclude:: ../../../examples/word2vec/train_word2vec.py
   :language: python
   :start-after: Set up a trainer
   :end-before: Save the word2vec model
   :caption: train_word2vec.py
   :dedent: 4

4.5 Start training
-------------------

.. code-block:: console

    $ pwd
    /root2chainer/chainer/examples/word2vec
    $ python train_word2vec.py --test  # run by test mode. If you want to use all data, remove "--test".
    GPU: -1
    # unit: 100
    Window: 5
    Minibatch-size: 1000
    # epoch: 20
    Training model: skipgram
    Output type: hsm
    
    n_vocab: 10000
    data length: 100
    epoch       main/loss   validation/main/loss
    1           4233.75     2495.33               
    2           1411.14     4990.66               
    3           4233.11     1247.66               
    4           2821.66     4990.65               
    5           4231.94     1247.66               
    6           5642.04     2495.3                
    7           5640.82     4990.64               
    8           5639.31     2495.28               
    9           2817.89     4990.62               
    10          1408.03     3742.94               
    11          5633.11     1247.62               
    12          4221.71     2495.21               
    13          4219.3      4990.56               
    14          4216.57     2495.16               
    15          4213.52     2495.12               
    16          5616.03     1247.55               
    17          5611.34     3742.78               
    18          2800.31     3742.74               
    19          1397.79     2494.95               
    20          2794.1      3742.66

4.5 Search the similar words
-----------------------------

.. code-block:: console

    $ pwd
    /root2chainer/chainer/examples/word2vec
    $ python search.py 
    >> apple
    query: apple
    compaq: 0.6169619560241699
    chip: 0.49579331278800964
    retailer: 0.4904134273529053
    maker: 0.4684058427810669
    computer: 0.4652436673641205
    >> animal      
    query: animal
    beauty: 0.5680124759674072
    human: 0.5404794216156006
    insulin: 0.5365156531333923
    cell: 0.5186758041381836
    photographs: 0.5077002048492432

5. Reference
=============
* [1] `Socher, Richard; et al. "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank". <https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf>`_
* [2] 深層学習による自然言語処理 (機械学習プロフェッショナルシリーズ)
