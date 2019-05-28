import argparse

import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions

from chainer.datasets import get_cifar10
from chainer.datasets import get_cifar100

import chainermn

import models.VGG


def main():
    parser = argparse.ArgumentParser(description='ChainerMN example: CIFAR')
    parser.add_argument('--dataset', '-d', default='cifar10',
                        help='The dataset to use: cifar10 or cifar100')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images per GPU in a mini-batch')
    parser.add_argument('--learnrate', '-l', type=float, default=0.05,
                        help='Learning rate for SGD')
    parser.add_argument('--epoch', '-e', type=int, default=300,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', action='store_true',
                        help='Use GPU (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--communicator', type=str, default='flat',
                        help='Type of communicator')
    args = parser.parse_args()

    # Prepare ChainerMN communicator.
    if args.gpu:
        comm = chainermn.create_communicator(args.communicator)
        device = comm.intra_rank
    else:
        comm = chainermn.create_communicator('naive')
        device = -1

    if comm.rank == 0:
        print('==========================================')
        print('Num process (COMM_WORLD): {}'.format(comm.size))
        if args.gpu:
            print('Using GPUs')
        print('Using {} communicator'.format(args.communicator))
        print('Num Minibatch-size: {}'.format(args.batchsize))
        print('Num epoch: {}'.format(args.epoch))
        print('==========================================')

    # Set up a neural network to train.
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.

    dataset_info = {
        'cifar10': {'num_class_labels': 10, 'load_func': get_cifar10},
        'cifar100': {'num_class_labels': 100, 'load_func': get_cifar100},
    }

    if args.dataset not in dataset_info:
        raise RuntimeError('Invalid dataset choice.')

    num_class_labels = dataset_info[args.dataset]['num_class_labels']

    if comm.rank == 0:
        train, test = dataset_info[args.dataset]['load_func']()
    else:
        train, test = None, None

    train = chainermn.scatter_dataset(train, comm, shuffle=True)
    test = chainermn.scatter_dataset(test, comm, shuffle=True)

    model = L.Classifier(models.VGG.VGG(num_class_labels))
    if device >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(device).use()
        model.to_gpu()  # Copy the model to the GPU

    optimizer = chainermn.create_multi_node_optimizer(
        chainer.optimizers.MomentumSGD(args.learnrate), comm)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(5e-4))

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize,
                                                  shuffle=False)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=device)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    evaluator = extensions.Evaluator(test_iter, model, device=device)
    evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
    trainer.extend(evaluator)

    trainer.extend(extensions.ExponentialShift('lr', 0.5),
                   trigger=(25, 'epoch'))

    if comm.rank == 0:
        trainer.extend(extensions.dump_graph('main/loss'))
        trainer.extend(extensions.snapshot(
            filename='snaphot_epoch_{.updater.epoch}'))
        trainer.extend(extensions.LogReport())
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
        trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
