"""Recurrent neural network language model with static graph optimizations.

This is a modified version of the standard Chainer Penn Tree Bank (ptb)
example that
includes static subgraph optimizations. It is mostly unchanged
from the original model except that that the RNN is unrolled for `bproplen`
slices inside of a static chain.

This was required because the `LSTM` link used by the ptb example
is not fully compatible with the static subgraph
optimizations feature. Specifically, it does not support
multiple calls in the same iteration unless it is called from
inside a single static chain.

This code is ported from the following implementation written in Torch.
https://github.com/tomsercu/lstm

This code is a custom loop version of train_ptb.py. That is, we train
models without using the Trainer class in chainer and instead write a
training loop that manually computes the loss of minibatches and
applies an optimizer to update the model.
"""
from __future__ import print_function
import argparse
import copy
import numpy as np
import random

import chainer
from chainer import configuration
from chainer.dataset import convert
import chainer.functions as F
from chainer.functions.loss import softmax_cross_entropy
import chainer.links as L
from chainer import serializers
from chainer import static_graph


# Definition of a recurrent net for language modeling
class RNNForLMSlice(chainer.Chain):

    def __init__(self, n_vocab, n_units):
        super(RNNForLMSlice, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units)
            self.l1 = L.LSTM(n_units, n_units)
            self.l2 = L.LSTM(n_units, n_units)
            self.l3 = L.Linear(n_units, n_vocab)

        for param in self.params():
            param.array[...] = np.random.uniform(-0.1, 0.1, param.array.shape)

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()

    def __call__(self, x):
        h0 = self.embed(x)
        h1 = self.l1(F.dropout(h0))
        h2 = self.l2(F.dropout(h1))
        y = self.l3(F.dropout(h2))
        return y


class RNNForLMUnrolled(chainer.Chain):
    def __init__(self, n_vocab, n_units):
        super(RNNForLMUnrolled, self).__init__()
        with self.init_scope():
            self.rnn = RNNForLMSlice(n_vocab, n_units)

    @static_graph(verbosity_level=1)
    def __call__(self, words):
        """Perform a forward pass on the supplied list of words.

        The RNN is unrolled for a number of time slices equal to the
        length of the supplied word sequence.

        Args:
            words_labels (list of Variable): The list of input words to the
                unrolled neural network.

        Returns the corresponding lest of output variables of the same
        length as the input sequence.
        """
        outputs = []
        for ind in range(len(words)):
            word = words[ind]
            y = self.rnn(word)
            outputs.append(y)

        return outputs


# Dataset iterator to create a batch of sequences at different positions.
# This iterator returns a pair of current words and the next words. Each
# example is a part of sequences starting from the different offsets
# equally spaced within the whole sequence.
class ParallelSequentialIterator(chainer.dataset.Iterator):

    def __init__(self, dataset, batch_size, repeat=True):
        self.dataset = dataset
        self.batch_size = batch_size  # batch size
        # Number of completed sweeps over the dataset. In this case, it is
        # incremented if every word is visited at least once after the last
        # increment.
        self.epoch = 0
        # True if the epoch is incremented at the last iteration.
        self.is_new_epoch = False
        self.repeat = repeat
        length = len(dataset)
        # Offsets maintain the position of each sequence in the mini-batch.
        self.offsets = [i * length // batch_size for i in range(batch_size)]
        # NOTE: this is not a count of parameter updates. It is just a count of
        # calls of ``__next__``.
        self.iteration = 0
        # use -1 instead of None internally
        self._previous_epoch_detail = -1.

    def __next__(self):
        # This iterator returns a list representing a mini-batch. Each item
        # indicates a different position in the original sequence. Each item is
        # represented by a pair of two word IDs. The first word is at the
        # "current" position, while the second word at the next position.
        # At each iteration, the iteration count is incremented, which pushes
        # forward the "current" position.
        length = len(self.dataset)
        if not self.repeat and self.iteration * self.batch_size >= length:
            # If not self.repeat, this iterator stops at the end of the first
            # epoch (i.e., when all words are visited once).
            raise StopIteration
        cur_words = self.get_words()
        self._previous_epoch_detail = self.epoch_detail
        self.iteration += 1
        next_words = self.get_words()

        epoch = self.iteration * self.batch_size // length
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch:
            self.epoch = epoch

        return list(zip(cur_words, next_words))

    @property
    def epoch_detail(self):
        # Floating point version of epoch.
        return self.iteration * self.batch_size / len(self.dataset)

    @property
    def previous_epoch_detail(self):
        if self._previous_epoch_detail < 0:
            return None
        return self._previous_epoch_detail

    def get_words(self):
        # It returns a list of current words.
        return [self.dataset[(offset + self.iteration) % len(self.dataset)]
                for offset in self.offsets]

    def serialize(self, serializer):
        # It is important to serialize the state to be recovered on resume.
        self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)
        try:
            self._previous_epoch_detail = serializer(
                'previous_epoch_detail', self._previous_epoch_detail)
        except KeyError:
            # guess previous_epoch_detail for older version
            self._previous_epoch_detail = self.epoch + \
                (self.current_position - self.batch_size) / len(self.dataset)
            if self.epoch_detail > 0:
                self._previous_epoch_detail = max(
                    self._previous_epoch_detail, 0.)
            else:
                self._previous_epoch_detail = -1.


def main():
    np.random.seed(0)
    random.seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=20,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--bproplen', '-l', type=int, default=25,
                        help='Number of words in each mini-batch '
                             '(= length of truncated BPTT)')
    parser.add_argument('--epoch', '-e', type=int, default=39,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gradclip', '-c', type=float, default=5,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--test', action='store_true',
                        help='Use tiny datasets for quick tests')
    parser.set_defaults(test=False)
    parser.add_argument('--unit', '-u', type=int, default=650,
                        help='Number of LSTM units in each layer')
    args = parser.parse_args()

    def evaluate(model, iter):
        # Evaluation routine to be used for validation and test.
        evaluator = model.copy()  # to use different state
        evaluator.rnn.reset_state()  # initialize state
        sum_perp = 0
        data_count = 0
        words = []
        labels = []
        lossfun = softmax_cross_entropy.softmax_cross_entropy
        with configuration.using_config('train', False):
            for batch in copy.copy(iter):
                word, label = convert.concat_examples(batch, args.gpu)
                words.append(word)
                labels.append(label)
                data_count += 1
            outputs = evaluator(words)

            for ind in range(len(outputs)):
                y = outputs[ind]
                label = labels[ind]
                loss = lossfun(y, label)
                sum_perp += loss.array
        return np.exp(float(sum_perp) / data_count)

    # Load the Penn Tree Bank long word sequence dataset
    train, val, test = chainer.datasets.get_ptb_words()
    n_vocab = max(train) + 1  # train is just an array of integers
    print('#vocab =', n_vocab)

    if args.test:
        train = train[:100]
        val = val[:100]
        test = test[:100]

    # Create the dataset iterators
    train_iter = ParallelSequentialIterator(train, args.batchsize)
    val_iter = ParallelSequentialIterator(val, 1, repeat=False)
    test_iter = ParallelSequentialIterator(test, 1, repeat=False)

    # Prepare an RNNLM model
    model = RNNForLMUnrolled(n_vocab, args.unit)
    lossfun = softmax_cross_entropy.softmax_cross_entropy
    if args.gpu >= 0:
        # Make the specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # Set up an optimizer
    optimizer = chainer.optimizers.SGD(lr=1.0)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))

    sum_perp = 0
    count = 0
    iteration = 0

    while train_iter.epoch < args.epoch:
        iteration += 1
        words = []
        labels = []
        # Progress the dataset iterator for bprop_len words at each iteration.
        for i in range(args.bproplen):
            # Get the next batch (a list of tuples of two word IDs)
            batch = train_iter.__next__()
            # Concatenate the word IDs to matrices and send them to the device
            # self.converter does this job
            # (it is chainer.dataset.concat_examples by default)
            word, label = convert.concat_examples(batch, args.gpu)
            words.append(word)
            labels.append(label)
            count += 1

        outputs = model(words)

        loss = 0
        for ind in range(len(outputs)):
            y = outputs[ind]
            label = labels[ind]
            loss += lossfun(y, label)

        sum_perp += loss.array
        optimizer.target.cleargrads()  # Clear the parameter gradients
        loss.backward()  # Backprop
        loss.unchain_backward()  # Truncate the graph
        optimizer.update()  # Update the parameters

        if iteration % 20 == 0:
            print('iteration: ', iteration)
            print('training perplexity: ', np.exp(float(sum_perp) / count))
            sum_perp = 0
            count = 0

        if train_iter.is_new_epoch:
            print('Evaluating model on validation set...')
            print('epoch: ', train_iter.epoch)
            print('validation perplexity: ', evaluate(model, val_iter))

    # Evaluate on test dataset
    print('test')
    test_perp = evaluate(model, test_iter)
    print('test perplexity:', test_perp)

    # Save the model and the optimizer
    print('save the model')
    serializers.save_npz('rnnlm.model', model)
    print('save the optimizer')
    serializers.save_npz('rnnlm.state', optimizer)


if __name__ == '__main__':

    main()
