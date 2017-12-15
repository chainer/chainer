import argparse

import chainer
from chainer.datasets import TransformDataset
from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer.training import extensions

import datasets
from model import ImageCaptionModel


class TestModeEvaluator(extensions.Evaluator):

    """Evaluates the model on the validation dataset."""

    def evaluate(self):
        with chainer.using_config('train', False), \
                chainer.no_backprop_mode():
            ret = super(TestModeEvaluator, self).evaluate()
        return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='result')
    parser.add_argument('--mscoco-root', type=str, default='data')
    parser.add_argument('--max-iters', type=int, default=50000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--dropout-ratio', type=float, default=0.5)
    parser.add_argument('--val-n-imgs', type=int, default=1000)
    parser.add_argument('--val-iter', type=int, default=100)
    parser.add_argument('--log-iter', type=int, default=1)
    parser.add_argument('--snapshot-iter', type=int, default=1000)
    parser.add_argument('--rnn', type=str, default='nsteplstm',
                        choices=['nsteplstm', 'lstm'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max-caption-length', type=int, default=30)
    args = parser.parse_args()

    # Load the MSCOCO dataset. Assumes that the dataset has been downloaded
    # already using e.g. the `download.py` script
    train, val = datasets.get_mscoco(args.mscoco_root)

    # Use any number of samples from the validation set for validation, using
    # all of them is usually quite slow since the number of validation images
    # are many
    val = val[:args.val_n_imgs]

    # Number of unique words that are found in the dataset
    vocab_size = len(train.vocab)

    # Instantiate the model to be trained either with LSTM layers or with
    # NStepLSTM layers
    model = ImageCaptionModel(
        vocab_size, dropout_ratio=args.dropout_ratio, rnn=args.rnn)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    def transform(in_data):
        # Called for each sample and applies necessary preprocessing to the
        # image such as resizing and normalizing
        img, caption = in_data
        img = model.prepare(img)
        return img, caption

    # Usually not required but we need to preprocess the images since their
    # sizes may vary (and the model requires that they have the exact same
    # fixed size)
    train = TransformDataset(train, transform)
    val = TransformDataset(val, transform)

    train_iter = iterators.SerialIterator(train, args.batch_size)
    val_iter = chainer.iterators.SerialIterator(
        val, args.batch_size, repeat=False, shuffle=False)

    optimizer = optimizers.Adam()
    optimizer.setup(model)

    def converter(batch, device):
        # The converted receives a batch of input samples any may modify it if
        # necessary. In our case, we need to align the captions depending on if
        # we are using LSTM layers of NStepLSTM layers in the model.
        if args.rnn == 'lstm':
            max_caption_length = args.max_caption_length
        elif args.rnn == 'nsteplstm':
            max_caption_length = None
        else:
            raise ValueError('Invalid RNN type.')
        return datasets.converter(
            batch, device, max_caption_length=max_caption_length)

    updater = training.updater.StandardUpdater(
        train_iter, optimizer=optimizer, device=args.gpu, converter=converter)

    trainer = training.Trainer(
        updater, out=args.out, stop_trigger=(args.max_iters, 'iteration'))
    trainer.extend(
        TestModeEvaluator(
            val_iter, model,
            eval_func=model,
            converter=converter,
            device=args.gpu
        ),
        trigger=(args.val_iter, 'iteration')
    )
    trainer.extend(
        extensions.LogReport(
            ['main/loss', 'validation/main/loss'],
            trigger=(args.log_iter, 'iteration')
        )
    )
    trainer.extend(
        extensions.PlotReport(
            ['main/loss', 'validation/main/loss'],
            trigger=(args.log_iter, 'iteration')
        )
    )
    trainer.extend(
        extensions.PrintReport(
            ['elapsed_time', 'epoch', 'iteration', 'main/loss',
             'validation/main/loss']
        ),
        trigger=(args.log_iter, 'iteration')
    )

    # Save model snapshots so that later on, we can load them and generate new
    # captions for any image. This can be done in the `predict.py` script
    trainer.extend(
        extensions.snapshot_object(model, 'model_{.updater.iteration}'),
        trigger=(args.snapshot_iter, 'iteration')
    )
    trainer.extend(extensions.ProgressBar())
    trainer.run()


if __name__ == '__main__':
    main()
