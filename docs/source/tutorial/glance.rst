Chainer at a Glance
-----------------------

.. currentmodule:: chainer

Welcome to Chainer!

Chainer is a rapidly growing neural network platform. The strengths of Chainer are:
* 100% pure Python -- Chainer is developed from the beginning in Python, source code and errors follow the Pythonic way
* Define by Run -- neural networks definitions are defined on-the-fly at run time, allowing for easier customisation
* Broad and deep support -- Chainer is actively used for most of the current approaches for neural nets (CNN, RNN, RL, etc.), aggressively adds new approaches as they're developed, and provides support for many kinds of hardware as well as parallelization for multiple GPUs


Mushrooms -- tastey or deathly?
~~~~~~~~~~~~

Let's take a look at a basic program of Chainer to see how it works. For a dataset, we'll work with the edible vs. poisonous mushroom dataset, which has over 8,000 examples of mushrooms, labelled by 22 categories including odor, cap color, habitat, etc.

How will Chainer learn which mushrooms are edible and which mushrooms will kill you? Let's see!

Let's start our python program. Matplotlib is used for the graphs to show training progress.
 
.. doctest::

   #!/usr/bin/env python
   
   from __future__ import print_function
   
   try:
       import matplotlib
       matplotlib.use('Agg')
   except ImportError:
       pass

Typical imports for a Chainer program. Links contain trainable parameters and functions do not.

.. doctest::

   import chainer
   import chainer.functions as F
   import chainer.links as L
   from chainer import training, datasets
   from chainer.training import extensions
   
   import numpy as np
   import sklearn.preprocessing as sp
   
From the raw mushroom.csv, we format the data into a Chainer dataset. Chainer requires a numpy array for the features in the X matrix and a flattened array if the label is one-dimensional.

.. doctest::

   data_array = np.genfromtxt('mushrooms.csv', delimiter=',',dtype=str, skip_header=1)
   labelEncoder = sp.LabelEncoder()
   for col in range(data_array.shape[1]):
       data_array[:, col] = labelEncoder.fit_transform(data_array[:, col])
   
   X = data_array[:, 1:].astype(np.float32)
   Y = np.ndarray.flatten(data_array[:, 0].astype(np.int32))
   train, test = datasets.split_dataset_random(datasets.TupleDataset(X, Y), 623)
   
Define the neural network. For our mushrooms, we'll use two fully-connected, hidden layers between the input and output layers.

.. doctest::

   # Network definition
   class MLP(chainer.Chain):
   
       def __init__(self, n_units, n_out):
   	super(MLP, self).__init__()
   	with self.init_scope():
   	    # the size of the inputs to each layer will be inferred
   	    self.l1 = L.Linear(n_units)  # n_in -> n_units
   	    self.l2 = L.Linear(n_units)  # n_units -> n_units
   	    self.l3 = L.Linear(n_out)  # n_units -> n_out
   
As an activation function, we'll use standard Rectified Linear Units (relu).

.. doctest::

       def __call__(self, x):
   	h1 = F.relu(self.l1(x))
   	h2 = F.relu(self.l2(h1))
   	return self.l3(h2)
   
Since mushrooms are either edible or poisonous (no information on psychedelic effects!) in the dataset, we'll use a classifier Link for the output, with 100 units in the hidden layers and 2 possible categories to be classified into.
   
.. doctest::

   model = L.Classifier(MLP(100, 2))

If using a CPU instead of the GPU, set gpu_id to -1. Otherwise, use the ID of the GPU, usually 0.

.. doctest::

   gpu_id = -1
   
   if gpu_id >= 0:
       # Make a specified GPU current
       chainer.cuda.get_device_from_id(gpu_id).use()
       model.to_gpu()  # Copy the model to the GPU
   
Pick and optimizer, and set up the model to use it.

.. doctest::

   # Setup an optimizer
   optimizer = chainer.optimizers.SGD()
   optimizer.setup(model)
   
Configure iterators to step through batches of the data for training and for testing validation. In this case, we'll use a batch size of 100, no repeating, and shuffling not required since we already shuffled the dataset on reading it in.

.. doctest::

   train_iter = chainer.iterators.SerialIterator(train, 100)
   test_iter = chainer.iterators.SerialIterator(test, 100,
   					     repeat=False, shuffle=False)

Set up the updater to be called after the training batches and set the number of batches per epoch to 100. The learning rate per epoch will be output to the directory `result`.
   
.. doctest::

   # Set up a trainer
   updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)
   trainer = training.Trainer(updater, (100, 'epoch'), out='result')
   
Set the model to be evaluated after each epoch.

.. doctest::

   trainer.extend(extensions.Evaluator(test_iter, model, device=gpu_id))
   
Dump a computational graph from 'loss' variable at the first iteration. The "main" refers to the target link of the "main" optimizer.

.. doctest::

   trainer.extend(extensions.dump_graph('main/loss'))
   
Take a snapshot of the training every 20 epochs.

.. doctest::

   trainer.extend(extensions.snapshot(), trigger=(20, 'epoch'))
   
Write a log of evaluation statistics for each epoch.

.. doctest::

   trainer.extend(extensions.LogReport())
   
Save two plot images to the result directory.

.. doctest::

   if extensions.PlotReport.available():
       trainer.extend(
   	extensions.PlotReport(['main/loss', 'validation/main/loss'],
   			      'epoch', file_name='loss.png'))
       trainer.extend(
   	extensions.PlotReport(
   	    ['main/accuracy', 'validation/main/accuracy'],
   	    'epoch', file_name='accuracy.png'))
   
Print selected entries of the log to standard output.

.. doctest::

   trainer.extend(extensions.PrintReport(
       ['epoch', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
   
Print a progress bar to standard output.

.. doctest::

   trainer.extend(extensions.ProgressBar())
   
Run the training.

.. doctest::

   trainer.run()
   
