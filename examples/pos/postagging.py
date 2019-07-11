import argparse
import collections
import warnings

import nltk
import numpy
import six

import chainer
from chainer import datasets
import chainer.links as L
from chainer import reporter
from chainer import training
from chainer.training import extensions


class CRF(chainer.Chain):

    def __init__(self, n_vocab, n_pos):
        super(CRF, self).__init__()
        with self.init_scope():
            self.feature = L.EmbedID(n_vocab, n_pos)
            self.crf = L.CRF1d(n_pos)

    def forward(self, xs, ys):
        # h[i] is feature vector for each batch of words.
        hs = [self.feature(x) for x in xs]
        loss = self.crf(hs, ys, transpose=True)
        reporter.report({'loss': loss}, self)

        # To predict labels, call argmax method.
        _, predict = self.crf.argmax(hs, transpose=True)
        correct = 0
        total = 0
        for y, p in six.moves.zip(ys, predict):
            # NOTE y is ndarray because
            # it does not pass to transpose_sequence
            correct += self.xp.sum(y == p)
            total += len(y)
        reporter.report({'correct': correct}, self)
        reporter.report({'total': total}, self)

        return loss

    def argmax(self, xs):
        hs = [self.feature(x) for x in xs]
        return self.crf.argmax(hs, transpose=True)


@chainer.dataset.converter()
def convert(batch, device):
    sentences = [
        chainer.dataset.to_device(device, sentence) for sentence, _ in batch]
    poses = [chainer.dataset.to_device(device, pos) for _, pos in batch]
    return {'xs': sentences, 'ys': poses}


def main():
    parser = argparse.ArgumentParser(
        description='Chainer example: POS-tagging')
    parser.add_argument('--batchsize', '-b', type=int, default=30,
                        help='Number of images in each mini batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--device', '-d', type=str, default='-1',
                        help='Device specifier. Either ChainerX device '
                        'specifier or an integer. If non-negative integer, '
                        'CuPy arrays with specified device id are used. If '
                        'negative integer, NumPy arrays are used')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    group = parser.add_argument_group('deprecated arguments')
    group.add_argument('--gpu', '-g', dest='device',
                       type=int, nargs='?', const=0,
                       help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    if chainer.get_dtype() == numpy.float16:
        warnings.warn(
            'This example may cause NaN in FP16 mode.', RuntimeWarning)

    vocab = collections.defaultdict(lambda: len(vocab))
    pos_vocab = collections.defaultdict(lambda: len(pos_vocab))

    # Convert word sequences and pos sequences to integer sequences.
    nltk.download('brown')
    data = []
    for sentence in nltk.corpus.brown.tagged_sents():
        xs = numpy.array([vocab[lex] for lex, _ in sentence], numpy.int32)
        ys = numpy.array([pos_vocab[pos] for _, pos in sentence], numpy.int32)
        data.append((xs, ys))

    print('# of sentences: {}'.format(len(data)))
    print('# of words: {}'.format(len(vocab)))
    print('# of pos: {}'.format(len(pos_vocab)))

    device = chainer.get_device(args.device)
    device.use()

    model = CRF(len(vocab), len(pos_vocab))
    model.to_device(device)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

    test_data, train_data = datasets.split_dataset_random(
        data, len(data) // 10, seed=0)

    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test_data, args.batchsize,
                                                 repeat=False, shuffle=False)
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, converter=convert, device=device)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    evaluator = extensions.Evaluator(
        test_iter, model, device=device, converter=convert)
    # Only validate in each 1000 iteration
    trainer.extend(evaluator, trigger=(1000, 'iteration'))
    trainer.extend(extensions.LogReport(trigger=(100, 'iteration')),
                   trigger=(100, 'iteration'))

    trainer.extend(
        extensions.MicroAverage(
            'main/correct', 'main/total', 'main/accuracy'))
    trainer.extend(
        extensions.MicroAverage(
            'validation/main/correct', 'validation/main/total',
            'validation/main/accuracy'))

    trainer.extend(
        extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy', 'elapsed_time']),
        trigger=(100, 'iteration'))

    trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
