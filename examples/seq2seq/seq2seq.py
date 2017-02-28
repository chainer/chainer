# encoding: utf-8

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


def get_topk(x, k=5, axis=1):
    ids_list = []
    scores_list = []
    xp = cuda.get_array_module(x)
    for i in range(k):
        ids = xp.argmax(x, axis=axis).astype('i')
        if axis == 0:
            scores = x[ids]
            x[ids] = - float('inf')
        else:
            scores = x[xp.arange(ids.shape[0]), ids]
            x[xp.arange(ids.shape[0]), ids] = - float('inf')
        ids_list.append(ids)
        scores_list.append(scores)
    return ids_list, scores_list


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

        xs = [x[::-1] for x in xs]

        eos = self.xp.zeros(1, 'i')
        ys_in = [F.concat([eos, y], axis=0) for y in ys]
        ys_out = [F.concat([y, eos], axis=0) for y in ys]

        exs = sequence_embed(self.embed_x, xs)
        eys = sequence_embed(self.embed_y, ys_in)

        batch = len(xs)
        # Initial hidden variable and cell variable
        zero = self.xp.zeros((self.n_layers, batch, self.n_units), 'f')
        hx, cx, _ = self.encoder(zero, zero, exs)
        _, _, os = self.decoder(hx, cx, eys)
        concat_os = F.concat(os, axis=0)
        concat_ys_out = F.concat(ys_out, axis=0)
        loss = F.softmax_cross_entropy(
            self.W(concat_os), concat_ys_out, normalize=False) \
            * concat_ys_out.shape[0] / batch

        reporter.report({'loss': loss.data}, self)
        perp = self.xp.exp(loss.data / concat_ys_out.shape[0] * batch)
        reporter.report({'perp': perp}, self)
        return loss

    def translate(self, xs, max_length=50):
        batch = len(xs)
        with chainer.no_backprop_mode():
            xs = [x[::-1] for x in xs]
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

    def translate_beam(self, xs, max_length=50, beam=16):
        with chainer.no_backprop_mode():
            xs = [x[::-1] for x in xs]
            exs = sequence_embed(self.embed_x, xs)
            # Initial hidden variable and cell variable
            zero = self.xp.zeros((self.n_layers, 1, self.n_units), 'f')
            h, c, _ = self.encoder(zero, zero, exs, train=False)
            h = F.broadcast_to(h, (self.n_layers, beam, self.n_units))
            c = F.broadcast_to(c, (self.n_layers, beam, self.n_units))
            ys = self.xp.zeros(beam, 'i')
            result = [[]] * beam

            sum_ws = self.xp.zeros(beam, 'f')
            for i in range(max_length):
                if i != 0 and (ys == 0).sum() == beam:
                    break

                eys = self.embed_y(ys)
                eys = chainer.functions.split_axis(
                    eys, beam, 0, force_tuple=True)
                h, c, hs = self.decoder(h, c, eys, train=False)
                hs_concat = chainer.functions.concat(hs, axis=0)
                ws_concat = F.log_softmax(self.W(hs_concat)).data

                if i != 0:
                    eos_sent_ids = self.xp.flatnonzero(ys == 0)
                    ws_concat[eos_sent_ids, :] = - float('inf')
                    ws_concat[eos_sent_ids, 0] = 0.
                    # eos-seq continue to choose eos with 0-score

                ys_list, ws_list = get_topk(ws_concat, beam, axis=1)

                if i == 0:
                    ys_list = [s[:1] for s in ys_list]
                    ws_list = [s[:1] for s in ws_list]
                    sum_ws_list = ws_list
                else:
                    sum_ws_list = [ws + sum_ws for ws in ws_list]

                sum_ws_concat = self.xp.concatenate(sum_ws_list, axis=0)
                ys_concat = self.xp.concatenate(ys_list, axis=0)

                idx_list, sum_ws_list = get_topk(sum_ws_concat, beam, axis=0)
                idx_concat = self.xp.stack(idx_list, axis=0)
                ys = ys_concat[idx_concat]
                sum_ws = self.xp.stack(sum_ws_list, axis=0)

                y_list = ys.tolist()
                old_idx_list = (idx_concat % beam).tolist()

                h = F.stack(
                    [h[:, idx] for idx in old_idx_list], axis=1)
                c = F.stack(
                    [c[:, idx] for idx in old_idx_list], axis=1)

                result = [result[idx] + [y]
                          for idx, y in zip(old_idx_list, y_list)]

        # Remove EOS taggs
        result = [[y for y in sent if y != 0]
                  for sent in result]

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
        print('BELU {}'.format(bleu))


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
        test_data = list(zip(source_data, target_data))

        source_ids = {word: index for index, word in enumerate(source_vocab)}
        target_ids = {word: index for index, word in enumerate(target_vocab)}

    target_words = {i: w for w, i in target_ids.items()}
    source_words = {i: w for w, i in source_ids.items()}

    model = Seq2seq(3, len(source_ids), len(target_ids), args.unit)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    updater = training.StandardUpdater(
        train_iter, optimizer, converter=convert, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport(trigger=(200, 'iteration')),
                   trigger=(200, 'iteration'))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'validation/main/loss',
         'main/perp', 'validation/main/perp', 'elapsed_time']),
        trigger=(200, 'iteration'))

    def translate_one(source, target):
        words = europal.split_sentence(source)
        print('# source : ' + ' '.join(words))
        x = model.xp.array(
            [source_ids.get(w, 1) for w in words], 'i')
        ys = model.translate([x])[0]
        words = [target_words[y] for y in ys]
        print('#  result : ' + ' '.join(words))

        words_list = model.translate_beam([x])[:3]
        for i, ys in enumerate(words_list, start=1):
            words = [target_words[y] for y in ys]
            print('#   beam{} : '.format(i) + ' '.join(words))
        print('#  expect : ' + target)

    @chainer.training.make_extension(trigger=(200, 'iteration'))
    def translate(trainer):
        translate_one(
            'Who are we ?',
            'Qui sommes-nous?')
        translate_one(
            'And it often costs over a hundred dollars ' +
            'to obtain the required identity card .',
            'Or, il en coûte souvent plus de cent dollars ' +
            'pour obtenir la carte d\'identité requise.')

        source, target = test_data[numpy.random.choice(len(test_data))]
        source = ' '.join([source_words[i] for i in source])
        target = ' '.join([target_words[i] for i in target])
        translate_one(source, target)

    trainer.extend(translate, trigger=(200, 'iteration'))
    trainer.extend(CalculateBleu(model, test_data),
                   trigger=(10000, 'iteration'))
    print('start training')
    trainer.run()


if __name__ == '__main__':
    main()
