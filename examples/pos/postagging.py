import argparse
import collections
import itertools

from nltk.corpus import brown
import numpy

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
import chainer.optimizers as O
from chainer import reporter
from chainer import training
from chainer.training import extensions


class CRF(chainer.Chain):

    def __init__(self, n_vocab, n_pos):
        super(CRF, self).__init__(
            embed=L.EmbedID(n_vocab, n_pos),
            crf=L.CRF1d(n_pos),
        )

    def __call__(self, *args):
        xs = args[:len(args) / 2]
        ys = args[len(args) / 2:]

        inds = numpy.argsort([-len(x.data) for x in xs]).astype('i')
        xs = [xs[i] for i in inds]
        ys = [ys[i] for i in inds]

        xs = F.transpose_sequence(xs)
        ys = F.transpose_sequence(ys)

        hs = [self.embed(x) for x in xs]
        loss = self.crf(hs, ys)
        reporter.report({'loss': loss}, self)

        _, predict = self.crf.argmax(hs)
        correct = 0
        total = 0
        for y, p in zip(ys, predict):
            correct += xp.sum(y.data == p)
            total += len(y.data)
        reporter.report({'correct': float(correct)}, self)
        reporter.report({'total': float(total)}, self)

        return loss

    def argmax(self, xs):
        hs = [self.embed(x) for x in xs]
        return self.crf.argmax(hs)


def convert(batch, device):
    sentences = [cuda.to_gpu(sentence) for sentence, _ in batch]
    poses = [cuda.to_gpu(pos) for _, pos in batch]
    return tuple(sentences + poses)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=30,
                        help='Number of images in each mini batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()

    vocab = collections.defaultdict(lambda: len(vocab))
    pos_vocab = collections.defaultdict(lambda: len(pos_vocab))

    data = []
    for sentence in brown.tagged_sents():
        xs = numpy.array([vocab[lex] for lex, _ in sentence], 'i')
        ys = numpy.array([pos_vocab[pos] for _, pos in sentence] , 'i')
        data.append((xs, ys))

    print('# of sentences: {}'.format(len(data)))
    print('# of words: {}'.format(len(vocab)))
    print('# of pos: {}'.format(len(pos_vocab)))

    model = CRF(len(vocab), len(pos_vocab))
    model.to_gpu()
    xp = cuda.cupy
    optimizer = O.Adam()
    optimizer.setup(model)
    #opt.add_hook(chainer.optimizer.WeightDecay(0.1))

    train_data = data[len(data) / 4:]
    test_data = data[:len(data) / 4]

    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test_data, args.batchsize,
                                                 repeat=False, shuffle=False)
    updater = training.StandardUpdater(
        train_iter, optimizer, converter=convert)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'))

    trainer.extend(extensions.Evaluator(test_iter, model, device=0, converter=convert))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss', 'main/correct', 'main/total',
         'validation/main/correct', 'validation/main/total']))

    trainer.run()
