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

As shown below, we explain the implementation with Colaboratory. If you have a
browser, you can immediately run the tutorial in the GPU environment.
So, please try it!

`chainer-community/chainer-colab-notebook/OfficialExample(ja)/sentiment_ja.ipynb <https://colab.research.google.com/drive/1NQOxHw-JINkDbY4JDT5Tgy8fOuEcbJbv>`_

.. toctree::

    notebooks/sentiment_en


3. Reference
=============
* [1] `Socher, Richard; et al. "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank". <https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf>`_
* [2] 深層学習による自然言語処理 (機械学習プロフェッショナルシリーズ)
