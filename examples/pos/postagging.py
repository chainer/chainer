import argparse
import collections

from nltk.corpus import brown
import numpy

import chainer
from chainer import cuda
from chainer import datasets
import chainer.functions as F
import chainer.links as L
import chainer.optimizers as O
from chainer import reporter
from chainer import training
from chainer.training import extensions


class CRF(chainer.Chain):

    def __init__(self, n_vocab, n_pos):
        super(CRF, self).__init__(
            feature=L.EmbedID(n_vocab, n_pos),
            crf=L.CRF1d(n_pos),
        )

    def __call__(self, *args):
        # NOTE: Chain.__call__ currently cannot use a list as an argument.
        # We simply concate two list and make a list.
        # Each xs[i] is a word-id sequences, and ys[i] is a pos-id sequence.
        # They of course have difference lengths.
        xs = args[:len(args) // 2]
        ys = args[len(args) // 2:]

        # Before making a transpose, you need to sort two lists in descending
        # order of length.
        inds = numpy.argsort([-len(x.data) for x in xs]).astype('i')
        xs = [xs[i] for i in inds]
        ys = [ys[i] for i in inds]

        # Make a transpose of sequences.
        # Now xs[t] is a batch of words at time t.
        xs = F.transpose_sequence(xs)
        ys = F.transpose_sequence(ys)

        # h[i] is feature vector for each batch of words.
        hs = [self.feature(x) for x in xs]
        loss = self.crf(hs, ys)
        reporter.report({'loss': loss}, self)

        # To predict labels, call argmax method.
        _, predict = self.crf.argmax(hs)
        correct = 0
        total = 0
        for y, p in zip(ys, predict):
            correct += self.xp.sum(y.data == p)
            total += len(y.data)
        reporter.report({'correct': correct}, self)
        reporter.report({'total': total}, self)

        return loss

    def argmax(self, xs):
        hs = [self.feature(x) for x in xs]
        return self.crf.argmax(hs)


def convert(batch, device):
    if device is None:
        def to_device(x):
            return x
    elif device < 0:
        to_device = cuda.to_cpu
    else:
        def to_device(x):
            return cuda.to_gpu(x, device, cuda.Stream.null)

    sentences = [to_device(sentence) for sentence, _ in batch]
    poses = [to_device(pos) for _, pos in batch]
    return tuple(sentences + poses)


def main():
    parser = argparse.ArgumentParser(
        description='Chainer example: POS-tagging')
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
    args = parser.parse_args()

    vocab = collections.defaultdict(lambda: len(vocab))
    pos_vocab = collections.defaultdict(lambda: len(pos_vocab))

    # Convert word sequences and pos sequences to integer sequences.
    data = []
    for sentence in brown.tagged_sents():
        xs = numpy.array([vocab[lex] for lex, _ in sentence], 'i')
        ys = numpy.array([pos_vocab[pos] for _, pos in sentence], 'i')
        data.append((xs, ys))

    print('# of sentences: {}'.format(len(data)))
    print('# of words: {}'.format(len(vocab)))
    print('# of pos: {}'.format(len(pos_vocab)))

    model = CRF(len(vocab), len(pos_vocab))
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)
    optimizer = O.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

    test_data, train_data = datasets.split_dataset_random(
        data, len(data) / 10, seed=0)

    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test_data, args.batchsize,
                                                 repeat=False, shuffle=False)
    updater = training.StandardUpdater(
        train_iter, optimizer, converter=convert, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(
        test_iter, model, device=args.gpu, converter=convert))
    trainer.extend(extensions.LogReport())

    trainer.extend(extensions.MicroAverage(
        'main/correct', 'main/total', 'main/accuracy'))
    trainer.extend(extensions.MicroAverage(
        'validation/main/correct', 'validation/main/total',
        'validation/main/accuracy'))

    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
