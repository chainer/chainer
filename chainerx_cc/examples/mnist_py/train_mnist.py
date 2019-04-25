#!/usr/bin/env python3

import argparse
import gzip
import pathlib
import time

import numpy as np

import chainerx as chx


class MLP(object):

    def __init__(self):
        self.W1, self.b1 = new_linear_params(784, 1000)
        self.W2, self.b2 = new_linear_params(1000, 1000)
        self.W3, self.b3 = new_linear_params(1000, 10)

    @property
    def params(self):
        return self.W1, self.b1, self.W2, self.b2, self.W3, self.b3

    def forward(self, x):
        h = chx.relu(chx.linear(x, self.W1, self.b1))
        h = chx.relu(chx.linear(h, self.W2, self.b2))
        return chx.linear(h, self.W3, self.b3)

    def update(self, lr):
        for param in self.params:
            # TODO(beam2d): make it easier
            p = param.as_grad_stopped()
            # TODO(beam2d): make grad not have graph by default
            p -= lr * param.grad.as_grad_stopped()
            param.cleargrad()

    def require_grad(self):
        for param in self.params:
            param.require_grad()


def new_linear_params(n_in, n_out):
    W = np.random.randn(n_out, n_in).astype(
        np.float32)  # TODO(beam2d): not supported in chx
    W /= np.sqrt(n_in)  # TODO(beam2d): not supported in chx
    W = chx.array(W)
    b = chx.zeros(n_out, dtype=chx.float32)
    return W, b


def compute_loss(y, t):
    # softmax cross entropy
    score = chx.log_softmax(y, axis=1)
    mask = (t[:, chx.newaxis] == chx.arange(
        10, dtype=t.dtype)).astype(score.dtype)
    # TODO(beam2d): implement mean
    return -(score * mask).sum() * (1 / y.shape[0])


def evaluate(model, X_test, Y_test, eval_size, batch_size):
    N_test = X_test.shape[0] if eval_size is None else eval_size

    if N_test > X_test.shape[0]:
        raise ValueError(
            'Test size can be no larger than {}'.format(X_test.shape[0]))

    with chx.no_backprop_mode():
        total_loss = chx.array(0, dtype=chx.float32)
        num_correct = chx.array(0, dtype=chx.int64)
        for i in range(0, N_test, batch_size):
            x = X_test[i:min(i + batch_size, N_test)]
            t = Y_test[i:min(i + batch_size, N_test)]

            y = model.forward(x)
            total_loss += compute_loss(y, t) * batch_size
            num_correct += (y.argmax(axis=1).astype(t.dtype)
                            == t).astype(chx.int32).sum()

    mean_loss = float(total_loss) / N_test
    accuracy = int(num_correct) / N_test
    return mean_loss, accuracy


def main():
    parser = argparse.ArgumentParser('Train a neural network on MNIST dataset')
    parser.add_argument(
        '--batchsize', '-B', type=int, default=100, help='Batch size')
    parser.add_argument(
        '--epoch', '-E', type=int, default=20,
        help='Number of epochs to train')
    parser.add_argument(
        '--iteration', '-I', type=int, default=None,
        help='Number of iterations to train. Epoch is ignored if specified.')
    parser.add_argument(
        '--data', '-p', default='mnist',
        help='Path to the directory that contains MNIST dataset')
    parser.add_argument(
        '--device', '-d', default='native', help='Device to use')
    parser.add_argument(
        '--eval-size', default=None, type=int,
        help='Number of samples to use from the test set for evaluation. '
        'None to use all.')
    args = parser.parse_args()

    chx.set_default_device(args.device)

    # Prepare dataset
    X, Y = get_mnist(args.data, 'train')
    X_test, Y_test = get_mnist(args.data, 't10k')

    # Prepare model
    model = MLP()

    # Training
    N = X.shape[0]   # TODO(beam2d): implement len
    # TODO(beam2d): support int32 indexing
    all_indices_np = np.arange(N, dtype=np.int64)
    batch_size = args.batchsize
    eval_size = args.eval_size

    # Train
    model.require_grad()

    it = 0
    epoch = 0
    is_finished = False
    start = time.time()

    while not is_finished:
        # TODO(beam2d): not suupported in chx
        np.random.shuffle(all_indices_np)
        all_indices = chx.array(all_indices_np)

        for i in range(0, N, batch_size):
            indices = all_indices[i:i + batch_size]
            x = X.take(indices, axis=0)
            t = Y.take(indices, axis=0)

            y = model.forward(x)
            loss = compute_loss(y, t)

            loss.backward()
            model.update(lr=0.01)

            it += 1
            if args.iteration is not None:
                mean_loss, accuracy = evaluate(
                    model, X_test, Y_test, eval_size, batch_size)
                elapsed_time = time.time() - start
                print(
                    'iteration {}... loss={},\taccuracy={},\telapsed_time={}'
                    .format(it, mean_loss, accuracy, elapsed_time))
                if it >= args.iteration:
                    is_finished = True
                    break

        epoch += 1
        if args.iteration is None:  # stop based on epoch, instead of iteration
            mean_loss, accuracy = evaluate(
                model, X_test, Y_test, eval_size, batch_size)
            elapsed_time = time.time() - start
            print(
                'epoch {}... loss={},\taccuracy={},\telapsed_time={}'
                .format(epoch, mean_loss, accuracy, elapsed_time))
            if epoch >= args.epoch:
                is_finished = True


def get_mnist(path, name):
    path = pathlib.Path(path)
    x_path = str(path / '{}-images-idx3-ubyte.gz'.format(name))
    y_path = str(path / '{}-labels-idx1-ubyte.gz'.format(name))

    with gzip.open(x_path, 'rb') as fx:
        fx.read(16)  # skip header
        # read/frombuffer is used instead of fromfile because fromfile does not
        # handle gzip file correctly
        x = np.frombuffer(fx.read(), dtype=np.uint8).reshape(-1, 784)

    with gzip.open(y_path, 'rb') as fy:
        fy.read(8)  # skip header
        y = np.frombuffer(fy.read(), dtype=np.uint8)

    assert x.shape[0] == y.shape[0]

    x = x.astype(np.float32)
    x /= 255
    y = y.astype(np.int32)
    return chx.array(x), chx.array(y)


if __name__ == '__main__':
    main()
