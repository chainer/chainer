import collections

from nltk.corpus import comtrans
import numpy

import chainer
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

        eos = numpy.array([0], 'i')
        ys_in = [F.concat([eos, y], axis=0) for y in ys]
        ys_out = [F.concat([y, eos], axis=0) for y in ys]

        exs = sequence_embed(self.embed_x, xs)
        eys = sequence_embed(self.embed_y, ys_in)

        batch = len(xs)
        # Initial hidden variable and cell variable
        zero = numpy.zeros((self.n_layers, batch, self.n_units), 'f')
        hx, cx, _ = self.encoder(zero, zero, exs)
        _, _, os = self.decoder(hx, cx, eys)
        loss = F.softmax_cross_entropy(
            self.W(F.concat(os, axis=0)), F.concat(ys_out, axis=0))

        reporter.report({'loss': loss}, self)
        return loss


def convert(batch, device):
    return tuple([x for x, _ in batch] + [y for _, y in batch])


def main():
    data = [
        (numpy.array([1,2,3], 'i'), numpy.array([1,2,3], 'i')),
        (numpy.array([1,2,3], 'i'), numpy.array([1,2,3], 'i')),
        (numpy.array([1,2,3], 'i'), numpy.array([1,2,3], 'i')),
    ]

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
    
    model = Seq2seq(3, len(source_ids), len(target_ids), 10)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(data, 50)
    updater = training.StandardUpdater(train_iter, optimizer, converter=convert)
    trainer = training.Trainer(updater, (10, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.run()


if __name__ == '__main__':
    main()
