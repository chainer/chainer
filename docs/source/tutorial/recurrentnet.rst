Recurrent Nets and their Computational Graph
--------------------------------------------

.. currentmodule:: chainer

In this section, you will learn how to write

* recurrent nets with full backprop,
* recurrent nets with truncated backprop,
* evaluation of networks with few memory.

After reading this section, you will be able to:

* Handle input sequences of variable length
* Truncate upper stream of the network during forward computation
* Use volatile variables to prevent network construction


Recurrent Nets
~~~~~~~~~~~~~~

Recurrent nets are neural networks with loops.
They are often used to learn from sequential input/output.
Given an input stream :math:`x_1, x_2, \dots, x_t, \dots` and the initial state :math:`h_0`, a recurrent net iteratively updates its state by :math:`h_t = f(x_t, h_{t-1})`, and at some or every point in time :math:`t`, it outputs :math:`y_t = g(h_t)`.
If we expand the procedure along the time axis, it looks like a regular feed-forward network except that same parameters are periodically used within the network.

Here we learn how to write a simple one-layer recurrent net.
The task is language modeling: given a finite sequence of words, we want to predict the next word at each position without peeking the successive words.
Suppose that there are 1,000 different word types, and that we use 100 dimensional real vectors to represent each word (a.k.a. word embedding).

Let's start from defining the recurrent neural net language model (RNNLM) as a link.
We can use the :class:`chainer.links.LSTM` link that implements a fully-connected stateful LSTM layer.
This link looks like a linear layer.
On construction, you pass the input and output size to the constructor:

.. doctest::

   >>> l = L.LSTM(100, 50)

Then, call on this instance ``l(x)`` runs *one step of LSTM layer*:

.. doctest::

   >>> l.reset_states(10)  # 10 is the batch size
   >>> x = Variable(np.random.randn((10, 100)).astype(np.float32))
   >>> y = l(x)

Do not forget to reset the internal state of the LSTM layer before the forward computation!
Every recurrent layer holds its internal state (i.e. the output of the previous call).
At the first application of the recurrent layer, you must reset the internal state.
Then, the next input can be direclty fed to the LSTM instance:

.. doctest::

   >>> x2 = Variable(np.random.randn((10, 100)).astype(np.float32))
   >>> y2 = l(x2)

Based on this LSTM link, let's write our RNNLM network as a new link:

.. testcode::

   class RNNLM(DictLink):
       def __init__(self):
           super(RNNLM, self).__init__(
               embed=L.EmbedID(1000, 100),  # word embedding
               mid=L.LSTM(100, 50),  # the first LSTM layer
               out=L.Linear(50, 1000),  # the feed-forward output layer
           )

       def reset_states(self, batchsize):
           self['mid'].reset_states(batchsize)

       def forward_one_step(self, cur_word, next_word):
           # Given the current and the next word IDs, compute the loss of the
           # next-word prediction.
           x = self['embed'](cur_word)
           h = self['mid'](x)
           y = self['out'](h)
           return F.softmax_cross_entropy(y, next_word)

       def forward(self, x_list):
           # Given a list of input variables x_list, compute the whole loss on
           # this word sequence.
           self.reset_states(len(x_list[0].data))  # pass the minibatch size
           loss = 0
           for cur_word, next_word in zip(x_list, x_list[1:]):
               loss += self.forward_one_step(cur_word, next_word)
           return loss

   model = RNNLM()
   optimizer = optimizers.SGD()
   optimizer.setup(model)

Here :class:`~chainer.links.EmbedID` is a link for word embedding.
It converts input integers into corresponding fixed-dimensional embedding vectors.
The last linear link ``out`` represents the feed-forward output layer.

We implemented the one-step-forward computation as a separate method, which is a best practice of writing recurrent nets for higher extensibility.
The ``forward`` method is very simple and no special care needs to be taken with respect to the length of the input sequence.
This code actually handles variable length input sequences without any tricks.

Of course, the accumulated loss is a Variable object with the full history of computation.
So we can just call its :meth:`~Variable.backward` method to compute gradients of the total loss according to the model parameters:

.. testcode::
   :hide:

   x_list = [Variable(np.random.randint(255, size=(1,)).astype(np.int32))
             for _ in range(100)]

.. testcode::

   optimizer.zero_grads()
   loss = forward(x_list)
   loss.backward()
   optimizer.update()

Do not forget to call :meth:`Optimizer.zero_grads` before the backward computation!

Truncate the Graph by Unchaining
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Learning from very long sequences is also a typical use case of recurrent nets.
Suppose that the input and state sequence is too long to fit into memory.
In such cases, we often truncate the backpropagation into a short time range.
This technique is called *truncated backprop*.
It is heuristic, and it makes the gradients biased.
However, this technique works well in practice if the time range is long enough.

How to implement truncated backprop in Chainer?
Chainer has a smart mechanism to achieve truncation, called **backward unchaining**.
It is implemented in the :meth:`Variable.unchain_backward` method.
Backward unchaining starts from the Variable object, and it chops the computation history backwards from the variable.
The chopped variables are disposed automatically (if they are not referenced explicitly from any other user object).
As a result, they are no longer a part of computation history, and are not involved in backprop anymore.

Let's write an example of truncated backprop.
Here we use the same network as the one used in the previous subsection.
Suppose that we are given a very long sequence, and we want to run backprop truncated at every 30 time steps.
We can write truncated backprop using the ``forward_one_step`` method that we wrote above:

.. testcode::

   loss = 0
   count = 0
   seqlen = len(x_list[1:])

   model.reset_states(1)  # minibatch size == 1
   for cur_word, next_word in zip(x_list, x_list[1:]):
       loss += model.forward_one_step(cur_word, next_word)
       count += 1
       if count % 30 == 0 or count == seqlen:
           optimizer.zero_grads()
           loss.backward()
           loss.unchain_backward()
           optimizer.update()

State is updated at ``foward_one_step``, and the losses are accumulated to ``loss`` variable.
At each 30 steps, backprop takes place at the accumulated loss.
Then, the :meth:`~Variable.unchain_backward` method is called, which deletes the computation history backward from the accumulated loss.
Note that the last state of ``model`` is not lost, since the RNNLM instance holds a reference to it.

The implementation of truncated backprop is simple, and since there is no complicated trick on it, we can generalize this method to different situations.
For example, we can easily extend the above code to use different schedules between backprop timing and truncation length.

Network Evaluation without Storing the Computation History
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On evaluation of recurrent nets, there is typically no need to store the computation history.
While unchaining enables us to walk through unlimited length of sequences with limited memory, it is a bit of a work-around.

As an alternative, Chainer provides an evaluation mode of forward computation which does not store the computation history.
This is enabled by just passing ``volatile`` flag to all input variables.
Such variables are called *volatile variables*.

.. warning::

   It is not allowed to mix volatile and non-volatile variables as arguments to same function.

Note that the parameters of links are all variables, so each of them also has a volatile flag.
Volatile flags of all the parameter variables can be switched by substituting to the :attr:`Link.volatile` attribute:

.. testcode::

   model.volatile = True

Then, pass volatile variables to the forward method::

   x_list = [Variable(..., volatile=True) for _ in range(100)]  # list of 100 words
   loss = model.forward(x_list)

Volatile variables are also useful to evaluate feed-forward networks.

Variable's volatility can be changed directly by setting the :attr:`Variable.volatile` attribute.
This enables us to combine a fixed feature extractor network and a trainable predictor network.
For example, suppose that we want to train a feed-forward network ``predictor_func``, which is located on top of another fixed pretrained network ``fixed_func``.
We want to train ``predictor_func`` without storing the computation history for ``fixed_func``.
This is simply done by following code snippets (suppose ``x_data`` and ``y_data`` indicate input data and label, respectively)::

   x = Variable(x_data, volatile=True)
   feat = fixed_func(x)
   feat.volatile = False
   y = predictor_func(feat)
   y.backward()

At first, the input variable ``x`` is volatile, so ``fixed_func`` is executed in volatile mode, i.e. without memorizing the computation history.
Then the intermediate variable ``feat`` is manually set to non-volatile, so ``predictor_func`` is executed in non-volatile mode, i.e., with memorizing the history of computation.
Since the history of computation is only memorized between variables ``feat`` and ``y``, the backward computation stops at the ``feat`` variable.

---------

In this section we have demonstrated how to write recurrent nets in Chainer and some fundamental techniques to manage the history of computation (a.k.a. computational graph).
The example in the ``examples/ptb`` directory implements truncated backprop learning of a LSTM language model from the Penn Treebank corpus.
In the next section, we will review how to use GPU(s) in Chainer.
