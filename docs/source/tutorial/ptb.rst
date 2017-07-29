The Implementation of Recurrent Neural Net Language Model with Chainer
**********************************************************************

.. currentmodule:: chainer

0. Introduction
================

The **language model** is modeling the probability of generating natural languages,
such that sentences or documents. You can use the language model to estimate how
natural a sentence or a document is. Also, with the language model, you can
generate new sentences or documents.

Let's start with modeling the probability of generating sentences. We represent
a sentence be :math:`X = (x_0, x_1, ..., x_T)`, in which :math:`x_t` is a
one-hot vector.
Generally, :math:`x_0` is the one-hot vector of **BOS** (beginning of sentence),
and :math:`x_T` is that  of **EOS** (end of sentence).

In the case of language model, we usually model the probability of generating
the next word under the condition of the previous words. Let :math:`X[i, j]` be
:math:`(x_i, x_{i+1}, ..., x_j)`, the occurrence probabilitx of sentence :math:`X`
can be

.. math:: P(X) = P(x_0) \prod_{t=1}^T P(x_t|X[0, t-1])

As you see above, the occurrence probability :math:`P(X)` can be decomposed into
the conditional probability by the previous words :math:`P(x_t|X[0, t-1])`.
So, in the language model, we model :math:`P(x_t|X[0, t-1])`.


1. Basic Idea of Recurrent Neural Net Language Model
=====================================================

**Recurrent Neurral Net Language Model** (RNN language model) is the neural net
language model which contains the RNN in the network. Since the RNN can deal with
the variable length inputs, it is suitable for modeling the sequential data such
that natural languages. 

The probablity of generating the sentence :math:`X` is denoted as
:math:`P_{\rm model}(X)`,

.. math:: P_{\rm model}(X) = P(x_0) \prod_{t=1}^T P(x_t|X[0, t-1])

We show the one layer of the RNN language model with these parameters.

* :math:`x_t` : the one-hot vector of :math:`t`-th word.
* :math:`y_t` : the :math:`t`-th output.
* :math:`h_t^i` : the :math:`t`-th hidden layer of `i`-th layer.
* :math:`p_t` : the next word's probability of :math:`t`-th word.
* :math:`E` : Embedding matrix
* :math:`W_h` : Hidden layer matrix
* :math:`W_o` : Output layer matrix

.. image:: ../../image/ptb/rnnlm.png

#. Get the embedding vector

    .. math:: h_t^0 = Ex_t

#. Calculate the hidden layer

    .. math:: h_t^1 = {\rm tanh}(W_h [h_t^0; h_{t-1}^1])

#. Calculate the output layer

    .. math:: y_t = W_o h_t^1

#. Transform to probability

    .. math:: p_t = {\rm softmax}(y_t)

2. Implementation of Recurrent Neural Net Language Model
=========================================================

* There is an example related to the RNN language model on the GitHub repository,
  so we will explain based on that.

    * `chainer/examples/ptb <https://github.com/chainer/chainer/tree/master/examples/ptb>`_

2.1 Overview of the Example
----------------------------

.. image:: ../../image/ptb/rnnlm_example.png

* In this example, we use the recurrent neural model above.

    * :math:`x_t` : the one-hot vector of :math:`t`-th input.
    * :math:`y_t` : the :math:`t`-th output.
    * :math:`h_t^i` : the :math:`t`-th hidden layer of `i`-th layer.
    * :math:`E` : Embedding matrix
    * :math:`W_o` : Output layer matrix

* We use the **LSTM** (long short-term memory) for the connection of hidden layers.
  The LSTM is the one of major recurrent neural net unit. It is desined for
  remembering the long-term memory, which means the the relation of the separated
  words, such that the word at beggining of sentence and that at end. 
* We also use the **dropout** before the LSTM and linear transformation. Dropout is
  the one of the refularization techniques for reducing overfitting on training
  data.

2.2 Implementation Method
--------------------------

Import Package
^^^^^^^^^^^^^^^

.. literalinclude:: ../../../examples/ptb/train_ptb.py
   :language: python
   :lines: 14-18
   :caption: train_ptb.py
   :lineno-match:

* Basically, if you use chainer, you import in this way.
* Importing functions as ``F`` and links as ``L`` makes it easy to use.

Define Network Structure
^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../examples/ptb/train_ptb.py
   :language: python
   :pyobject: RNNForLM
   :caption: train_ptb.py
   :lineno-match:

* Next, we define the network structure of the RNN language model.
* When we call the constructor ``__init__``, we pass the vocavulary size
  ``n_vocab`` and the size of the hidden vectors ``n_units``.

    * As the connection of layers, :class:`~chainer.links.LSTM`,
      :class:`~chainer.links.Linear`, and :class:`~chainer.functions.dropout` are
      used.
    * The :class:`~chainer.Parameter` s are initialized in ``self.init_scope()``.

        * It is recommended to initialize :class:`~chainer.Parameter` here.
        * Since we set :class:`~chainer.Parameter` as the attribute of Link,
          there are effects such as making IDE easier to follow code.
        * For details, see :ref:`upgrade-new-param-register`.

    * You can access all the parameters by ``self.params()`` and initialze by
      ``np.random.uniform(-0.1, 0.1, param.data.shape)``.

* The function call ``__call__`` takes the input word's ID ``x``. In this function,
  the nerwork structure of RNN language model is defined.

    * The input word' ID ``x`` is converted to the embedding vector  ``h0``
      by ``self.embed(x)``.
    * After tha embedding vector ``h0`` passes through the network, the output ``y``
      is returned.

Define Iterator for Data
^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../examples/ptb/train_ptb.py
   :language: python
   :pyobject: ParallelSequentialIterator
   :caption: train_ptb.py

Define Updater
^^^^^^^^^^^^^^^

.. literalinclude:: ../../../examples/ptb/train_ptb.py
   :language: python
   :pyobject: BPTTUpdater
   :caption: train_ptb.py

Define Evaluation Function (Perplexity)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../examples/ptb/train_ptb.py
   :language: python
   :pyobject: compute_perplexity
   :caption: train_ptb.py

Main Function
^^^^^^^^^^^^^^

.. literalinclude:: ../../../examples/ptb/train_ptb.py
   :language: python
   :start-after: parser.parse_args
   :end-before: n_vocab
   :caption: train_ptb.py

* We download the Penn Treebank dataset by :class:`~chainer.datasets.get_ptb_words`.
  Each data contains the list of Document IDs
  
    * ``train`` is the training data.
    * ``val`` is the validation data. 
    * ``test`` is the test data. 

.. literalinclude:: ../../../examples/ptb/train_ptb.py
   :language: python
   :start-after: test[:100]
   :end-before: Prepare an RNNLM model
   :caption: train_ptb.py

* From the datasets ``train``, ``val`` and ``test``, we create the iterators for
  the datasets.
  
.. literalinclude:: ../../../examples/ptb/train_ptb.py
   :language: python
   :start-after: Prepare an RNNLM model
   :end-before: model.compute_accuracy
   :caption: train_ptb.py

* We create the recurrent neural net ``rnn`` and the classification model ``model``
  by :class:`~chainer.links.Classifier`.
  
.. literalinclude:: ../../../examples/ptb/train_ptb.py
   :language: python
   :start-after: Set up an optimizer
   :end-before: Set up a trainer
   :caption: train_ptb.py

* We setup the optimizer by :class:`~chainer.optimizers.SGD`.

.. literalinclude:: ../../../examples/ptb/train_ptb.py
   :language: python
   :start-after: Set up a trainer
   :end-before: Evaluate the final model
   :caption: train_ptb.py

* We setup and run the trainer.

.. literalinclude:: ../../../examples/ptb/train_ptb.py
   :language: python
   :start-after: Evaluate the final model
   :end-before: Serialize the final model
   :caption: train_ptb.py

* We evaluate the final model.

2.3 Run Example
----------------

Training the model
^^^^^^^^^^^^^^^^^^^

.. code-block:: console

    $ pwd
    /root2chainer/chainer/examples/ptb
    $ python train_ptb.py --test  # run by test mode. If you want to use all data, remove "--test".
    Downloading from https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt...
    Downloading from https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt...
    Downloading from https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt...
    #vocab = 10000
    test
    test perplexity: 29889.9857364

Generating sentences
^^^^^^^^^^^^^^^^^^^^^

* You can generate the sentence which starts with the word in the vocavulary.

    * In this exmaple, we generate the sentence which starts with the word **animal**. 

.. code-block:: console

    $ pwd
    /root2chainer/chainer/examples/ptb
    $ python gentxt.py -m model.npz -p "animal"
    animal automotive moscow <unk> percentage <unk> kia sim advance <unk> rubens <unk> <unk> <unk> <unk> <unk> tom <unk> <unk> jewelry .
