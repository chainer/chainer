import argparse
import collections

import nltk
import numpy
import six

import chainer
from chainer import datasets
import chainer.functions as F
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
        # Before making a transpose, you need to sort two lists in descending
        # order of length.
        inds = numpy.argsort([-len(x) for x in xs]).astype(numpy.int32)
        xs = [xs[i] for i in inds]
        ys = [ys[i] for i in inds]

        # Make transposed sequences.
        # Now xs[t] is a batch of words at time t.
        xs = F.transpose_sequence(xs)
        ys = F.transpose_sequence(ys)

        # h[i] is feature vector for each batch of words.
        hs = [self.feature(x) for x in xs]
        loss = self.crf(hs, ys)
        reporter.report({'loss': loss.data}, self)

        # To predict labels, call argmax method.
        _, predict = self.crf.argmax(hs)
        correct = 0
        total = 0
        for y, p in six.moves.zip(ys, predict):
            correct += self.xp.sum(y.data == p)
            total += len(y.data)
        reporter.report({'correct': correct}, self)
        reporter.report({'total': total}, self)

        return loss

    def argmax(self, xs):
        hs = [self.feature(x) for x in xs]
        return self.crf.argmax(hs)


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
    nltk.download('brown')
    data = []
    for sentence in nltk.corpus.brown.tagged_sents():
        xs = numpy.array([vocab[lex] for lex, _ in sentence], numpy.int32)
        ys = numpy.array([pos_vocab[pos] for _, pos in sentence], numpy.int32)
        data.append((xs, ys))

    print('# of sentences: {}'.format(len(data)))
    print('# of words: {}'.format(len(vocab)))
    print('# of pos: {}'.format(len(pos_vocab)))

    model = CRF(len(vocab), len(pos_vocab))
    if args.gpu >= 0:
        chainer.backends.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

    test_data, train_data = datasets.split_dataset_random(
        data, len(data) // 10, seed=0)

    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test_data, args.batchsize,
                                                 repeat=False, shuffle=False)
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, converter=convert, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    evaluator = extensions.Evaluator(
        test_iter, model, device=args.gpu, converter=convert)
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
