Recurrent Nets and their Computational Graph
````````````````````````````````````````````

.. currentmodule:: chainer

In this section, you will learn how to write

* recurrent nets with full backprop,
* recurrent nets with truncated backprop,
* evaluation of networks with few memory.

After reading this section, you will be able to:

* Handle input sequences of variable length
* Truncate upper stream of the network during forward computation
* Use no-backprop mode to prevent network construction


Recurrent Nets
''''''''''''''

Recurrent nets are neural networks with loops.
They are often used to learn from sequential input/output.
Given an input stream :math:`x_1, x_2, \dots, x_t, \dots` and the initial state :math:`h_0`, a recurrent net iteratively updates its state by :math:`h_t = f(x_t, h_{t-1})`, and at some or every point in time :math:`t`, it outputs :math:`y_t = g(h_t)`.
If we expand the procedure along the time axis, it looks like a regular feed-forward network except that same parameters are repeatedly used within the network.

Here we learn how to write a simple one-layer recurrent net.
The task is language modeling: given a finite sequence of words, we want to predict the next word at each position without peeking the successive words.
Suppose there are 1,000 different word types, and that we use 100 dimensional real vectors to represent each word (a.k.a. word embedding).

Let's start from defining the recurrent neural net language model (RNNLM) as a chain.
We can use the :class:`chainer.links.LSTM` link that implements a fully-connected stateful LSTM layer.
This link looks like an ordinary fully-connected layer.
On construction, you pass the input and output size to the constructor:

.. doctest::

   >>> l = L.LSTM(100, 50)

Then, call on this instance ``l(x)`` executes *one step of LSTM layer*:

.. doctest::

   >>> l.reset_state()
   >>> x = Variable(np.random.randn(10, 100).astype(np.float32))
   >>> y = l(x)

Do not forget to reset the internal state of the LSTM layer before the forward computation!
Every recurrent layer holds its internal state (i.e. the output of the previous call).
At the first application of the recurrent layer, you must reset the internal state.
Then, the next input can be directly fed to the LSTM instance:

.. doctest::

   >>> x2 = Variable(np.random.randn(10, 100).astype(np.float32))
   >>> y2 = l(x2)

Based on this LSTM link, let's write our recurrent network as a new chain:

.. testcode::

   class RNN(Chain):
       def __init__(self):
           super(RNN, self).__init__()
           with self.init_scope():
               self.embed = L.EmbedID(1000, 100)  # word embedding
               self.mid = L.LSTM(100, 50)  # the first LSTM layer
               self.out = L.Linear(50, 1000)  # the feed-forward output layer

       def reset_state(self):
           self.mid.reset_state()

       def __call__(self, cur_word):
           # Given the current word ID, predict the next word.
           x = self.embed(cur_word)
           h = self.mid(x)
           y = self.out(h)
           return y

   rnn = RNN()
   model = L.Classifier(rnn)
   optimizer = optimizers.SGD()
   optimizer.setup(model)

Here :class:`~chainer.links.EmbedID` is a link for word embedding.
It converts input integers into corresponding fixed-dimensional embedding vectors.
The last linear link ``out`` represents the feed-forward output layer.

The ``RNN`` chain implements a *one-step-forward computation*.
It does not handle sequences by itself, but we can use it to process sequences by just feeding items in a sequence straight to the chain.

Suppose we have a list of word variables ``x_list``.
Then, we can compute loss values for the word sequence by simple ``for`` loop.

.. testcode::

   def compute_loss(x_list):
       loss = 0
       for cur_word, next_word in zip(x_list, x_list[1:]):
           loss += model(cur_word, next_word)
       return loss

Of course, the accumulated loss is a Variable object with the full history of computation.
So we can just call its :meth:`~Variable.backward` method to compute gradients of the total loss according to the model parameters:

.. testcode::
   :hide:

   x_list = [Variable(np.random.randint(255, size=(1,)).astype(np.int32))
             for _ in range(100)]

.. testcode::

   # Suppose we have a list of word variables x_list.
   rnn.reset_state()
   model.cleargrads()
   loss = compute_loss(x_list)
   loss.backward()
   optimizer.update()

Or equivalently we can use the ``compute_loss`` as a loss function:

.. testcode::

   rnn.reset_state()
   optimizer.update(compute_loss, x_list)


Truncate the Graph by Unchaining
''''''''''''''''''''''''''''''''

Learning from very long sequences is also a typical use case of recurrent nets.
Suppose the input and state sequence is too long to fit into memory.
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
Suppose we are given a very long sequence, and we want to run backprop truncated at every 30 time steps.
We can write truncated backprop using the model defined above:

.. testcode::

   loss = 0
   count = 0
   seqlen = len(x_list[1:])

   rnn.reset_state()
   for cur_word, next_word in zip(x_list, x_list[1:]):
       loss += model(cur_word, next_word)
       count += 1
       if count % 30 == 0 or count == seqlen:
           model.cleargrads()
           loss.backward()
           loss.unchain_backward()
           optimizer.update()

State is updated at ``model()``, and the losses are accumulated to ``loss`` variable.
At each 30 steps, backprop takes place at the accumulated loss.
Then, the :meth:`~Variable.unchain_backward` method is called, which deletes the computation history backward from the accumulated loss.
Note that the last state of ``model`` is not lost, since the RNN instance holds a reference to it.

The implementation of truncated backprop is simple, and since there is no complicated trick on it, we can generalize this method to different situations.
For example, we can easily extend the above code to use different schedules between backprop timing and truncation length.


Network Evaluation without Storing the Computation History
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

On evaluation of recurrent nets, there is typically no need to store the computation history.
While unchaining enables us to walk through unlimited length of sequences with limited memory, it is a bit of a work-around.

As an alternative, Chainer provides an evaluation mode of forward computation which does not store the computation history.
This is enabled by just calling :func:`~chainer.no_backprop_mode` context::

   with chainer.no_backprop_mode():
       x_list = [Variable(...) for _ in range(100)]  # list of 100 words
       loss = compute_loss(x_list)

Note that we cannot call ``loss.backward()`` to compute the gradient here, since the variable created in the no-backprop context does not remember the computation history.

No-backprop context is also useful to evaluate feed-forward networks to reduce the memory footprint.

We can combine a fixed feature extractor network and a trainable predictor network using :func:`~chainer.no_backprop_mode`.
For example, suppose we want to train a feed-forward network ``predictor_func``, which is located on top of another fixed pre-trained network ``fixed_func``.
We want to train ``predictor_func`` without storing the computation history for ``fixed_func``.
This is simply done by following code snippets (suppose ``x_data`` and ``y_data`` indicate input data and label, respectively)::

   with chainer.no_backprop_mode():
       x = Variable(x_data)
       feat = fixed_func(x)
   y = predictor_func(feat)
   y.backward()

At first, the input variable ``x`` is in no-backprop mode, so ``fixed_func`` does not memorize the computation history.
Then ``predictor_func`` is executed in backprop mode, i.e., with memorizing the history of computation.
Since the history of computation is only memorized between variables ``feat`` and ``y``, the backward computation stops at the ``feat`` variable.


Making it with Trainer
''''''''''''''''''''''

The above codes are written with plain Function/Variable APIs.
When we write a training loop, it is better to use Trainer, since we can then easily add functionalities by extensions.

Before implementing it on Trainer, let's clarify the training settings.
We here use Penn Tree Bank dataset as a set of sentences.
Each sentence is represented as a word sequence.
We concatenate all sentences into one long word sequence, in which each sentence is separated by a special word ``<eos>``, which stands for "End of Sequence".
This dataset is easily obtained by :func:`chainer.datasets.get_ptb_words`.
This function returns train, validation, and test dataset, each of which is represented as a long array of integers.
Each integer represents a word ID.

Our task is to learn a recurrent neural net language model from the long word sequence.
We use words in different locations to form mini-batches.
It means we maintain :math:`B` indices pointing to different locations in the sequence, read from these indices at each iteration, and increment all indices after the read.
Of course, when one index reaches the end of the whole sequence, we turn the index back to 0.

In order to implement this training procedure, we have to customize the following components of Trainer:

- Iterator.
  Built-in iterators do not support reading from different locations and aggregating them into a mini-batch.
- Update function.
  The default update function does not support truncated BPTT.

When we write a dataset iterator dedicated to the dataset, the dataset implementation can be arbitrary; even the interface is not fixed.
On the other hand, the iterator must support the :class:`~chainer.dataset.Iterator` interface.
The important methods and attributes to implement are ``batch_size``, ``epoch``, ``epoch_detail``, ``is_new_epoch``, ``iteration``, ``__next__``, and ``serialize``.
Following is a code from the official example in the :tree:`examples/ptb` directory.

.. code-block:: python

   from __future__ import division

   class ParallelSequentialIterator(chainer.dataset.Iterator):
       def __init__(self, dataset, batch_size, repeat=True):
           self.dataset = dataset
           self.batch_size = batch_size
           self.epoch = 0
           self.is_new_epoch = False
           self.repeat = repeat
           self.offsets = [i * len(dataset) // batch_size for i in range(batch_size)]
           self.iteration = 0

       def __next__(self):
           length = len(self.dataset)
           if not self.repeat and self.iteration * self.batch_size >= length:
               raise StopIteration
           cur_words = self.get_words()
           self.iteration += 1
           next_words = self.get_words()

           epoch = self.iteration * self.batch_size // length
           self.is_new_epoch = self.epoch < epoch
           if self.is_new_epoch:
               self.epoch = epoch

           return list(zip(cur_words, next_words))

       @property
       def epoch_detail(self):
           return self.iteration * self.batch_size / len(self.dataset)

       def get_words(self):
           return [self.dataset[(offset + self.iteration) % len(self.dataset)]
                   for offset in self.offsets]

       def serialize(self, serializer):
           self.iteration = serializer('iteration', self.iteration)
           self.epoch = serializer('epoch', self.epoch)

   train_iter = ParallelSequentialIterator(train, 20)
   val_iter = ParallelSequentialIterator(val, 1, repeat=False)

Although the code is slightly long, the idea is simple.
First, this iterator creates ``offsets`` pointing to positions equally spaced within the whole sequence.
The i-th examples of mini-batches refer the sequence with the i-th offset.
The iterator returns a list of tuples of the current words and the next words.
Each mini-batch is converted to a tuple of integer arrays by the ``concat_examples`` function in the standard updater (see the previous tutorial).

Backprop Through Time is implemented as follows.

.. code-block:: python

   class BPTTUpdater(training.updaters.StandardUpdater):

       def __init__(self, train_iter, optimizer, bprop_len):
           super(BPTTUpdater, self).__init__(train_iter, optimizer)
           self.bprop_len = bprop_len

       # The core part of the update routine can be customized by overriding.
       def update_core(self):
           loss = 0
           # When we pass one iterator and optimizer to StandardUpdater.__init__,
           # they are automatically named 'main'.
           train_iter = self.get_iterator('main')
           optimizer = self.get_optimizer('main')

           # Progress the dataset iterator for bprop_len words at each iteration.
           for i in range(self.bprop_len):
               # Get the next batch (a list of tuples of two word IDs)
               batch = train_iter.__next__()

               # Concatenate the word IDs to matrices and send them to the device
               # self.converter does this job
               # (it is chainer.dataset.concat_examples by default)
               x, t = self.converter(batch)

               # Compute the loss at this time step and accumulate it
               loss += optimizer.target(chainer.Variable(x), chainer.Variable(t))

           optimizer.target.cleargrads()  # Clear the parameter gradients
           loss.backward()  # Backprop
           loss.unchain_backward()  # Truncate the graph
           optimizer.update()  # Update the parameters

   updater = BPTTUpdater(train_iter, optimizer, bprop_len)  # instantiation
   
In this case, we update the parameters on every ``bprop_len`` consecutive words.
The call of ``unchain_backward`` cuts the history of computation accumulated to the LSTM links.
The rest of the code for setting up Trainer is almost same as one given in the previous tutorial.


---------

In this section we have demonstrated how to write recurrent nets in Chainer and some fundamental techniques to manage the history of computation (a.k.a. computational graph).
The example in the :tree:`examples/ptb` directory implements truncated backprop learning of a LSTM language model from the Penn Treebank corpus.
In the next section, we will review how to use GPU(s) in Chainer.
