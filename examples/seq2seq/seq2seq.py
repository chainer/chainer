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

import europal


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

        reporter.report({'loss': loss.data}, self)
        return loss

    def translate(self, xs, max_length=50):
        batch = len(xs)
        with chainer.no_backprop_mode():
            exs = sequence_embed(self.embed_x, xs)
            # Initial hidden variable and cell variable
            zero = self.xp.zeros((self.n_layers, batch, self.n_units), 'f')
            h, c, _ = self.encoder(zero, zero, exs, train=False)
            ys = self.xp.zeros(batch, 'i')
            result = []
            for i in range(max_length):
                eys = self.embed_y(ys)
                eys = chainer.functions.split_axis(
                    eys, batch, 0, force_tuple=True)
                h, c, ys = self.decoder(h, c, eys, train=False)
                cys = chainer.functions.concat(ys, axis=0)
                wy = self.W(cys)
                ys = self.xp.argmax(wy.data, axis=1).astype('i')
                result.append(ys)

        result = cuda.to_cpu(self.xp.stack(result).T)

        # Remove EOS taggs
        outs = []
        for y in result:
            inds = numpy.argwhere(y == 0)
            if len(inds) > 0:
                y = y[:inds[0, 0]]
            outs.append(y)
        return outs


def convert(batch, device):
    if device is None:
        def to_device(x):
            return x
    elif device < 0:
        to_device = cuda.to_cpu
    else:
        def to_device(x):
            return cuda.to_gpu(x, device, cuda.Stream.null)

    def to_device_batch(batch):
        if device is None:
            return batch
        elif device < 0:
            return [to_device(x) for x in batch]
        else:
            xp = cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = numpy.cumsum([len(x) for x in batch[:-1]], dtype='i')
            concat_dev = to_device(concat)
            batch_dev = cuda.cupy.split(concat_dev, sections)
            return batch_dev

    return tuple(
        to_device_batch([x for x, _ in batch]) +
        to_device_batch([y for _, y in batch]))


class CalculateBleu(chainer.training.Extension):

    trigger = 1, 'epoch'

    def __init__(self, model, test_data, batch=100):
        self.model = model
        self.test_data = test_data
        self.batch = batch

    def __call__(self, trainer):
        with chainer.no_backprop_mode():
            references = []
            hypotheses = []
            for i in range(0, len(self.test_data), self.batch):
                sources, targets = zip(*self.test_data[i:i + self.batch])
                references.extend([[t.tolist()] for t in targets])

                ys = [y.tolist() for y in self.model.translate(sources)]
                hypotheses.extend(ys)

        bleu = bleu_score.corpus_bleu(
            references, hypotheses,
            smoothing_function=bleu_score.SmoothingFunction().method1)
        print(bleu)


def main():
    parser = argparse.ArgumentParser(description='Chainer example: seq2seq')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1024,
                        help='Number of units')
    args = parser.parse_args()

    if False:
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
    else:
        en_path = 'wmt/giga-fren.release2.fixed.en'
        source_vocab = ['<eos>', '<unk>'] + europal.count_words(en_path)
        source_data = europal.make_dataset(en_path, source_vocab)
        fr_path = 'wmt/giga-fren.release2.fixed.fr'
        target_vocab = ['<eos>', '<unk>'] + europal.count_words(fr_path)
        target_data = europal.make_dataset(fr_path, target_vocab)
        print('Original training data size: %d' % len(source_data))
        train_data = [(s, t) for s, t in zip(source_data, target_data)
                      if len(s) < 50 and len(t) < 50]
        print('Filtered training data size: %d' % len(train_data))

        en_path = 'wmt/dev/newstest2013.en'
        source_data = europal.make_dataset(en_path, source_vocab)
        fr_path = 'wmt/dev/newstest2013.fr'
        target_data = europal.make_dataset(fr_path, target_vocab)
        test_data = zip(source_data, target_data)

        source_ids = {word: index for index, word in enumerate(source_vocab)}
        target_ids = {word: index for index, word in enumerate(target_vocab)}

    target_words = {i: w for w, i in target_ids.items()}

    model = Seq2seq(3, len(source_ids), len(target_ids), args.unit)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    optimizer = chainer.optimizers.AdaGrad(0.5)
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(train_data, 50)
    updater = training.StandardUpdater(
        train_iter, optimizer, converter=convert, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport(trigger=(200, 'iteration')),
                   trigger=(200, 'iteration'))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']),
        trigger=(200, 'iteration'))

    @chainer.training.make_extension(trigger=(200, 'iteration'))
    def translate(trainer):
        words = ['Resumption', 'of', 'the', 'session']
        x = model.xp.array([source_ids[w] for w in words], 'i')
        ys = model.translate([x])[0]
        words = [target_words[y] for y in ys]
        print(' '.join(words))

    # trainer.extend(translate)
    trainer.extend(CalculateBleu(model, test_data), trigger=(200, 'iteration'))
    #trainer.extend(CalculateBleu(model, train_data))

    trainer.run()


if __name__ == '__main__':
    main()
