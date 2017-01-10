import argparse
import collections

from nltk.corpus import comtrans
from nltk.translate import bleu_score
import numpy

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import reporter
from chainer import training
from chainer.training import extensions


def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_section = numpy.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0, force_tuple=True)
    return exs


class Seq2seq(chainer.Chain):

    def __init__(self, n_layers, n_source_vocab, n_target_vocab, n_units):
        super(Seq2seq, self).__init__(
            embed_x=L.EmbedID(n_source_vocab, n_units),
            embed_y=L.EmbedID(n_target_vocab, n_units),
            encoder=L.NStepLSTM(n_layers, n_units, n_units, 0.1),
            decoder=L.NStepLSTM(n_layers, n_units, n_units, 0.1),
            W=L.Linear(n_units, n_target_vocab),
        )
        self.n_layers = n_layers
        self.n_units = n_units

    def __call__(self, *inputs):
        xs = inputs[:len(inputs) // 2]
        ys = inputs[len(inputs) // 2:]

        eos = self.xp.array([0], 'i')
        ys_in = [F.concat([eos, y], axis=0) for y in ys]
        ys_out = [F.concat([y, eos], axis=0) for y in ys]

        exs = sequence_embed(self.embed_x, xs)
        eys = sequence_embed(self.embed_y, ys_in)

        batch = len(xs)
        # Initial hidden variable and cell variable
        zero = self.xp.zeros((self.n_layers, batch, self.n_units), 'f')
        hx, cx, _ = self.encoder(zero, zero, exs)
        _, _, os = self.decoder(hx, cx, eys)
        loss = F.softmax_cross_entropy(
            self.W(F.concat(os, axis=0)), F.concat(ys_out, axis=0))

        reporter.report({'loss': loss}, self)
        return loss

    def translate(self, x):
        exs = sequence_embed(self.embed_x, [x])
        # Initial hidden variable and cell variable
        zero = self.xp.zeros((self.n_layers, 1, self.n_units), 'f')
        h, c, _ = self.encoder(zero, zero, exs)
        y = self.xp.zeros(1, 'i')
        result = []
        for i in range(10):
            ey = self.embed_y(y)
            h, c, ys = self.decoder(h, c, [ey])
            wy = self.W(ys[0])
            # TODO(unno): CuPy does not have argmax method
            yi = numpy.argmax(cuda.to_cpu(wy.data[0]))
            if yi == 0:
                # Found eos
                break
            result.append(yi)
            y = self.xp.array([yi], 'i')

        return result


def convert(batch, device):
    if device is None:
        def to_device(x):
            return x
    elif device < 0:
        to_device = cuda.to_cpu
    else:
        def to_device(x):
            return cuda.to_gpu(x, device, cuda.Stream.null)

    return tuple(
        [to_device(x) for x, _ in batch] + [to_device(y) for _, y in batch])


def main():
    parser = argparse.ArgumentParser(description='Chainer example: seq2seq')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=256,
                        help='Number of units')
    args = parser.parse_args()

    sentences = comtrans.aligned_sents('alignment-en-fr.txt')
    source_ids = collections.defaultdict(lambda: len(source_ids))
    target_ids = collections.defaultdict(lambda: len(target_ids))
    target_ids['eos']
    data = []
    for sentence in sentences:
        source = numpy.array([source_ids[w] for w in sentence.words], 'i')
        target = numpy.array([target_ids[w] for w in sentence.mots], 'i')
        data.append((source, target))
    print('Source vocabulary: %d' % len(source_ids))
    print('Target vocabulary: %d' % len(target_ids))

    test_data = data[:len(data) / 10]
    train_data = data[len(data) / 10:]
    target_words = {i: w for w, i in target_ids.items()}

    model = Seq2seq(3, len(source_ids), len(target_ids), args.unit)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(train_data, 50)
    updater = training.StandardUpdater(
        train_iter, optimizer, converter=convert, device=args.gpu)
    trainer = training.Trainer(updater, (10, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    @chainer.training.make_extension(trigger=(1, 'epoch'))
    def translate(trainer):
        words = ['Resumption', 'of', 'the', 'session']
        x = model.xp.array([source_ids[w] for w in words], 'i')
        ys = model.translate(x)
        words = [target_words[y] for y in ys]
        print(' '.join(words))

    @chainer.training.make_extension(trigger=(1, 'epoch'))
    def calc_bleu(trainer):
        references = []
        hypotheses = []
        for source, target in test_data:
            references.append([target])

            ys = model.translate(source)
            hypotheses.append(ys)

        bleu = bleu_score.corpus_bleu(
            references, hypotheses,
            smoothing_function=bleu_score.SmoothingFunction().method1)
        print(bleu)

    trainer.extend(translate)
    trainer.extend(calc_bleu)
    trainer.run()


if __name__ == '__main__':
    main()
