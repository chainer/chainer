#!/usr/bin/env python
import argparse

import chainer
from train_mnist import MLP
from train_mnist_model_parallel import ParallelMLP


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--device', '-d', type=str, default='-1',
                        help='Device specifier. Either ChainerX device '
                        'specifier or an integer. If non-negative integer, '
                        'CuPy arrays with specified device id are used. If '
                        'negative integer, NumPy arrays are used')
    parser.add_argument('--snapshot', '-s',
                        default='result/snapshot_iter_12000',
                        help='The path to a saved snapshot (NPZ)')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    group = parser.add_argument_group('deprecated arguments')
    group.add_argument('--gpu', '-g', dest='device',
                       type=int, nargs='?', const=0,
                       help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    device = chainer.get_device(args.device)

    print('Device: {}'.format(device))
    print('# unit: {}'.format(args.unit))
    print('')

    device.use()

    # Create a same model object as what you used for training
    if 'result_model_parallel' in args.snapshot:
        model = ParallelMLP(args.unit, 10, args.gpu, args.gpu)
    else:
        model = MLP(args.unit, 10)

    # Load saved parameters from a NPZ file of the Trainer object
    try:
        chainer.serializers.load_npz(
            args.snapshot, model, path='updater/model:main/predictor/')
    except Exception:
        chainer.serializers.load_npz(
            args.snapshot, model, path='predictor/')

    model.to_device(device)

    # Prepare data
    train, test = chainer.datasets.get_mnist()
    x, answer = test[0]
    x = device.send(x)
    with chainer.using_config('train', False):
        prediction = model(x[None, ...])[0].array.argmax()

    print('Prediction:', prediction)
    print('Answer:', answer)


if __name__ == '__main__':
    main()
