#!/usr/bin/env python
import argparse

import chainer
from train_mnist import MLP
from train_mnist_model_parallel import ParallelMLP


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--snapshot', '-s',
                        default='result/snapshot_iter_12000',
                        help='The path to a saved snapshot (NPZ)')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('')

    # Create a same model object as what you used for training
    if 'result_model_parallel' in args.snapshot:
        model = ParallelMLP(args.unit, 10, args.gpu, args.gpu)
    else:
        model = MLP(args.unit, 10)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    # Load saved parameters from a NPZ file of the Trainer object
    try:
        chainer.serializers.load_npz(
            args.snapshot, model, path='updater/model:main/predictor/')
    except Exception:
        chainer.serializers.load_npz(
            args.snapshot, model, path='predictor/')

    # Prepare data
    train, test = chainer.datasets.get_mnist()
    x, answer = test[0]
    if args.gpu >= 0:
        x = chainer.cuda.cupy.asarray(x)
    with chainer.using_config('train', False):
        prediction = model(x[None, ...])[0].array.argmax()

    print('Prediction:', prediction)
    print('Answer:', answer)


if __name__ == '__main__':
    main()
