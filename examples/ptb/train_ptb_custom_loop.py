#!/usr/bin/env python
"""Sample script of recurrent neural network language model.

This code is ported from the following implementation written in Torch.
https://github.com/tomsercu/lstm

This code is a custom loop version of train_ptb.py. That is, we train
models without using the Trainer class in chainer and instead write a
training loop that manually computes the loss of minibatches and
applies an optimizer to update the model.
"""
import argparse
import os
import sys

import numpy as np

import chainer
from chainer import configuration
from chainer.dataset import convert
import chainer.links as L
from chainer import serializers
import chainerx

import train_ptb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=20,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--bproplen', '-l', type=int, default=35,
                        help='Number of words in each mini-batch '
                             '(= length of truncated BPTT)')
    parser.add_argument('--epoch', '-e', type=int, default=39,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--device', '-d', type=str, default='-1',
                        help='Device specifier. Either ChainerX device '
                        'specifier or an integer. If non-negative integer, '
                        'CuPy arrays with specified device id are used. If '
                        'negative integer, NumPy arrays are used')
    parser.add_argument('--gradclip', '-c', type=float, default=5,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', type=str,
                        help='Directory that has `rnnln.model`'
                        ' and `rnnlm.state`')
    parser.add_argument('--test', action='store_true',
                        help='Use tiny datasets for quick tests')
    parser.set_defaults(test=False)
    parser.add_argument('--unit', '-u', type=int, default=650,
                        help='Number of LSTM units in each layer')
    group = parser.add_argument_group('deprecated arguments')
    group.add_argument('--gpu', '-g', dest='device',
                       type=int, nargs='?', const=0,
                       help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    device = chainer.get_device(args.device)
    if device.xp is chainerx:
        sys.stderr.write('This example does not support ChainerX devices.\n')
        sys.exit(1)

    device.use()

    def evaluate(model, iter):
        # Evaluation routine to be used for validation and test.
        evaluator = model.copy()  # to use different state
        evaluator.predictor.reset_state()  # initialize state
        sum_perp = 0
        data_count = 0
        # Enable evaluation mode.
        with configuration.using_config('train', False):
            # This is optional but can reduce computational overhead.
            with chainer.using_config('enable_backprop', False):
                iter.reset()
                for batch in iter:
                    x, t = convert.concat_examples(batch, device)
                    loss = evaluator(x, t)
                    sum_perp += loss.array
                    data_count += 1
        return np.exp(float(sum_perp) / data_count)

    # Load the Penn Tree Bank long word sequence dataset
    train, val, test = chainer.datasets.get_ptb_words()
    n_vocab = max(train) + 1  # train is just an array of integers
    print('#vocab = {}'.format(n_vocab))

    if args.test:
        train = train[:100]
        val = val[:100]
        test = test[:100]

    # Create the dataset iterators
    train_iter = train_ptb.ParallelSequentialIterator(train, args.batchsize)
    val_iter = train_ptb.ParallelSequentialIterator(val, 1, repeat=False)
    test_iter = train_ptb.ParallelSequentialIterator(test, 1, repeat=False)

    # Prepare an RNNLM model
    rnn = train_ptb.RNNForLM(n_vocab, args.unit)
    model = L.Classifier(rnn)
    model.compute_accuracy = False  # we only want the perplexity
    model.to_device(device)

    # Set up an optimizer
    optimizer = chainer.optimizers.SGD(lr=1.0)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))

    # Load model and optimizer
    if args.resume is not None:
        resume = args.resume
        if os.path.exists(resume):
            serializers.load_npz(os.path.join(resume, 'rnnlm.model'), model)
            serializers.load_npz(
                os.path.join(resume, 'rnnlm.state'), optimizer)
        else:
            raise ValueError(
                '`args.resume` ("{}") is specified,'
                ' but it does not exist'.format(resume)
            )

    sum_perp = 0
    count = 0
    iteration = 0
    while train_iter.epoch < args.epoch:
        loss = 0
        iteration += 1
        # Progress the dataset iterator for bprop_len words at each iteration.
        for i in range(args.bproplen):
            # Get the next batch (a list of tuples of two word IDs)
            batch = train_iter.__next__()
            # Concatenate the word IDs to matrices and send them to the device
            # self.converter does this job
            # (it is chainer.dataset.concat_examples by default)
            x, t = convert.concat_examples(batch, device)
            # Compute the loss at this time step and accumulate it
            loss += optimizer.target(x, t)
            count += 1

        sum_perp += loss.array
        optimizer.target.cleargrads()  # Clear the parameter gradients
        loss.backward()  # Backprop
        loss.unchain_backward()  # Truncate the graph
        optimizer.update()  # Update the parameters

        if iteration % 20 == 0:
            print('iteration: {}'.format(iteration))
            print('training perplexity: {}'.format(
                np.exp(float(sum_perp) / count)))
            sum_perp = 0
            count = 0

        if train_iter.is_new_epoch:
            print('epoch: {}'.format(train_iter.epoch))
            print('validation perplexity: {}'.format(
                evaluate(model, val_iter)))

    # Evaluate on test dataset
    print('test')
    test_perp = evaluate(model, test_iter)
    print('test perplexity: {}'.format(test_perp))

    # Save the model and the optimizer
    out = args.out
    if not os.path.exists(out):
        os.makedirs(out)
    print('save the model')
    serializers.save_npz(os.path.join(out, 'rnnlm.model'), model)
    print('save the optimizer')
    serializers.save_npz(os.path.join(out, 'rnnlm.state'), optimizer)


if __name__ == '__main__':
    main()
