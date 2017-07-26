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
a sentence be :math:`Y = (y_0, y_1, ..., y_T)`, in which :math:`y_t` is a
one-hot vector.
Generally, :math:`y_0` is the one-hot vector of **BOS** (beginning of sentence),
and :math:`y_T` is that  of **EOS** (end of sentence).

In the case of language model, we usually model the probability of generating
the next word under the condition of the previous words. Let :math:`Y[i, j]` be
:math:`(y_i, y_{i+1}, ..., y_j)`, the occurrence probability of sentence :math:`Y`
can be

.. math:: P(Y) = P(y_0) \prod_{t=1}^T P(y_t|Y[0, t-1])

As you see above, the occurrence probability :math:`P(Y)` can be decomposed into
the conditional probability by the previous words :math:`P(y_t|Y[0, t-1])`.
So, in the language model, we model :math:`P(y_t|Y[0, t-1])`.


1. Basic Idea of Recurrent Neural Net Language Model
=====================================================

**Recurrent Neurral Net Language Model** (RNN language model) is the neural net
language model which contains the RNN in the network. Since the RNN can deal with
the variable length inputs, it is suitable for modeling the sequential data such
that natural languages. 

The probablity of generating the sentence :math:`Y` is denoted as
:math:`P_{model}(Y)`,

.. math:: P_{model}(Y) = P(y_0) \prod_{t=1}^T P(y_t|Y[0, t-1])

We show the one layer of the RNN language model with these parameters.

- :math:`E` : Embedding matrix
- :math:`W_h` : Hidden layer matrix
- :math:`W_o` : Output layer matrix

.. image:: ../../image/ptb/rnnlm.png

#. Get the embedding vector

    .. math:: y_t^* = Ey_{t-1}

#. Calculate the hidden layer

    .. math:: h_t = tanh(W_h [y_t^*; h_{t-1}])

#. Calculate the output layer

    .. math:: o_t = W_o h_t

#. Transform to probability

    .. math:: p_t = softmax(o_t)

