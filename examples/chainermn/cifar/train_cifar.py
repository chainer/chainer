import argparse

import chainermn
import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.training import triggers

from chainer.datasets import get_cifar10
from chainer.datasets import get_cifar100

import models.VGG


def main():
    parser = argparse.ArgumentParser(description='Chainer CIFAR example:')
    parser.add_argument('--dataset', '-d', default='cifar10',
                        help='The dataset to use: cifar10 or cifar100')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--learnrate', '-l', type=float, default=0.05,
                        help='Learning rate for SGD')
    parser.add_argument('--epoch', '-e', type=int, default=300,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', action='store_true',
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--early-stopping', type=str,
                        help='Metric to watch for early stopping')
    args = parser.parse_args()

    # Prepare ChainerMN communicator.
    if args.gpu:
        comm = chainermn.create_communicator('hierarchical')
        data_axis, model_axis = comm.rank % 2, comm.rank // 2
        data_comm = comm.split(data_axis, comm.rank)
        model_comm = comm.split(model_axis, comm.rank)
        device = comm.intra_rank
    else:
        comm = chainermn.create_communicator('naive')
        data_axis, model_axis = comm.rank % 2, comm.rank // 2
        data_comm = comm.split(data_axis, comm.rank)
        model_comm = comm.split(model_axis, comm.rank)
        device = -1

    if model_comm.size != 2:
        raise ValueError(
            'This example can only be executed on the even number'
            'of processes.')

    if comm.rank == 0:
        print('==========================================')
        print('Num process (COMM_WORLD): {}'.format(comm.size))
        if args.gpu:
            print('Using GPUs')
        print('Num Minibatch-size: {}'.format(args.batchsize))
        print('Num epoch: {}'.format(args.epoch))
        print('==========================================')

    # Set up a neural network to train.
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    if model_axis == 0:
        if args.dataset == 'cifar10':
            print('Using CIFAR10 dataset.')
            class_labels = 10
            train, test = get_cifar10()
        elif args.dataset == 'cifar100':
            print('Using CIFAR100 dataset.')
            class_labels = 100
            train, test = get_cifar100()
        else:
            raise RuntimeError('Invalid dataset choice.')
    model = L.Classifier(models.VGG.VGG(class_labels))

    if device >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(device).use()
        model.to_gpu()  # Copy the model to the GPU

    optimizer = chainermn.create_multi_node_optimizer(
        chainer.optimizers.MomentumSGD(args.learnrate), data_comm)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(5e-4))

    train = chainermn.scatter_dataset(train, data_comm, shuffle = True)
    test = chainermn.scatter_dataset(test, data_comm, shuffle = True)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize,
                                                 shuffle = False)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    stop_trigger = (args.epoch, 'epoch')
    # Early stopping option
    if args.early_stopping:
        stop_trigger = triggers.EarlyStoppingTrigger(
            monitor=args.early_stopping, verbose=True,
            max_trigger=(args.epoch, 'epoch'))

    # Set up a trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=device)
    trainer = training.Trainer(updater, stop_trigger, out=args.out)

    # Evaluate the model with the test dataset for each epoch
    evaluator = extensions.Evaluator(test_iter, model, device=device)
    evaluator = chainermn.create_multi_node_evaluator(evaluator, data_comm)
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
