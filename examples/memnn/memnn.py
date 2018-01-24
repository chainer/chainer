#!/usr/bin/env python

import argparse
import collections
import copy
import glob
import sys

import numpy
import six

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import initializers
import chainer.links as L
from chainer import optimizers
from chainer import training
from chainer.training import extensions

import babi


class BoWEncoder(object):

    """BoW sentence encoder.

    It is defined as:

    .. math::

       m = \sum_j A x_j,

    where :math:`A` is an embed matrix, and :math:`x_j` is :math:`j`-th word ID.

    """

    def __call__(self, embed, sentences):
        xp = cuda.get_array_module(sentences)
        e = embed(sentences)
        s = F.sum(e, axis=-2)
        return s


class PositionEncoder(object):

    """Position encoding.

    It is defined as:

    .. math::

       m = \sum_j l_j A x_j,

    where :math:`A` is an embed matrix, :math:`x_j` is :math:`j`-th word ID and

    .. math::

       l_{kj} = (1 - j / J) - (k / d)(1 - 2j / J).

    :math:`J` is length of a sentence and :math:`d` is the dimension of the
    embedding.

    """

    def __call__(self, embed, sentences):
        xp = cuda.get_array_module(sentences)
        e = embed(sentences)
        ndim = e.ndim
        n_words, n_units = e.shape[-2:]

        # To avoid 0/0, we use max(length, 1) here.
        # Note that when the length is zero, its embedding is always zero and
        # is igrenod.
        length = xp.maximum(
            xp.sum((sentences != 0).astype('f'), axis=-1), 1)
        length = length.reshape((length.shape + (1, 1)))
        k = xp.arange(1, n_units + 1, dtype=numpy.float32) / n_units
        i = xp.arange(1, n_words + 1, dtype=numpy.float32)[:, None]
        coeff = (1 - i / length) - k * (1 - 2.0 * i / length)
        e = coeff * e
        s = F.sum(e, axis=-2)
        return s


class Memory(object):

    """Memory component in a memory network.

    Args:
        A (chainer.links.EmbedID): Embed matrix for input. Its shape is
            ``(n_vocab, n_units)``.
        C (chainer.links.EmbedID): Embed matrix for output. Its shape is
            ``(n_vocab, n_units)``.
        TA (chainer.links.EmbedID): Embed matrix for temporal encoding for
            input. Its shape is ``(max_memory, n_units)``.
        TC (chainer.links.EmbedID): Embed matrix for temporal encoding for
            output. Its shape is ``(max_memory, n_units)``.
        encoder (callable): It encodes given stentences to embed vectors.

    """

    def __init__(self, A, C, TA, TC, encoder):
        self.A = A
        self.C = C
        self.TA = TA
        self.TC = TC
        self.encoder = encoder

    def register_all(self, sentences):
        self.m = self.encoder(self.A, sentences)
        self.c = self.encoder(self.C, sentences)

    def query(self, u):
        xp = cuda.get_array_module(u)
        size = self.m.shape[1]
        inds = xp.arange(size - 1, -1, -1, dtype=numpy.int32)
        tm = self.TA(inds)
        tc = self.TC(inds)
        tm = F.broadcast_to(tm, self.m.shape)
        tc = F.broadcast_to(tc, self.c.shape)
        p = F.softmax(F.batch_matmul(self.m + tm, u))
        o = F.batch_matmul(F.swapaxes(self.c + tc, 2, 1), p)
        o = F.squeeze(o, -1)
        u = o + u
        return u


class MemNN(chainer.Chain):

    def __init__(self, n_units, n_vocab, encoder, max_memory, hops):
        super(MemNN, self).__init__()

        with self.init_scope():
            self.embeds = chainer.ChainList()
            self.temporals = chainer.ChainList()

        normal = initializers.Normal()
        # Shares both embeded matrixes in adjacent layres
        for _ in six.moves.range(hops + 1):
            self.embeds.append(L.EmbedID(n_vocab, n_units, initialW=normal))
            self.temporals.append(
                L.EmbedID(max_memory, n_units, initialW=normal))

        self.memories = [
            Memory(self.embeds[i], self.embeds[i + 1],
                   self.temporals[i], self.temporals[i + 1], encoder)
            for i in six.moves.range(hops)
        ]
        # The question embedding is same as the input embedding of the
        # first layer
        self.B = self.embeds[0]
        # The answer prediction matrix W is same as the final output layer
        self.W = lambda u: F.linear(u, self.embeds[-1].W)

        self.encoder = encoder

    def fix_ignore_label(self):
        for embed in self.embeds:
            embed.W.data[0, :] = 0

    def register_all(self, sentences):
        for memory in self.memories:
            memory.register_all(sentences)

    def query(self, question):
        u = self.encoder(self.B, question)
        for memory in self.memories:
            u = memory.query(u)
        a = self.W(u)
        return a

    def __call__(self, sentences, question):
        self.register_all(sentences)
        a = self.query(question)
        return a


def convert_data(train_data, max_memory):
    all_data = []
    sentence_len = max(max(len(s.sentence) for s in story)
                       for story in train_data)
    for story in train_data:
        mem = numpy.zeros((max_memory, sentence_len), dtype=numpy.int32)
        i = 0
        for sent in story:
            if isinstance(sent, babi.Sentence):
                if i == max_memory:
                    mem[0:i - 1, :] = mem[1:i, :]
                    i -= 1
                mem[i, 0:len(sent.sentence)] = sent.sentence
                i += 1
            elif isinstance(sent, babi.Query):
                query = numpy.zeros(sentence_len, dtype=numpy.int32)
                query[0:len(sent.sentence)] = sent.sentence
                all_data.append({
                    'sentences': mem.copy(),
                    'question': query,
                    'answer': numpy.array(sent.answer, 'i'),
                })

    return all_data


def main():
    parser = argparse.ArgumentParser(
        description='Chainer example: End-to-end memory networks')
    parser.add_argument('data', help='Path to bAbI dataset')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--unit', '-u', type=int, default=20,
                        help='Number of units')
    parser.add_argument('--hop', '-H', type=int, default=3,
                        help='Number of hops')
    parser.add_argument('--max-memory', type=int, default=50,
                        help='Maximum number of memory')
    parser.add_argument('--sentence-repr',
                        choices=['bow', 'pe'], default='bow',
                        help='Sentence representation. '
                        'Select from BoW ("bow") or position encoding ("pe")')
    args = parser.parse_args()

    vocab = collections.defaultdict(lambda: len(vocab))
    vocab['<unk>'] = 0

    for data_id in six.moves.range(1, 21):

        train_data = babi.read_data(
            vocab,
            glob.glob('%s/qa%d_*train.txt' % (args.data, data_id))[0])
        test_data = babi.read_data(
            vocab,
            glob.glob('%s/qa%d_*test.txt' % (args.data, data_id))[0])
        print('Training data: %d' % len(train_data))

        train_data = convert_data(train_data, args.max_memory)
        test_data = convert_data(test_data, args.max_memory)

        if args.sentence_repr == 'bow':
            encoder = BoWEncoder()
        elif args.sentence_repr == 'pe':
            encoder = PositionEncoder()
        else:
            print('Unknonw --sentence-repr option: "%s"' % args.sentence_repr)
            sys.exit(1)

        memnn = MemNN(args.unit, len(vocab), encoder, args.max_memory, args.hop)
        model = L.Classifier(memnn, label_key='answer')
        opt = optimizers.Adam()

        if args.gpu >= 0:
            chainer.cuda.get_device(args.gpu).use()
            model.to_gpu()

        opt.setup(model)

        train_iter = chainer.iterators.SerialIterator(
            train_data, args.batchsize)
        test_iter = chainer.iterators.SerialIterator(
            test_data, args.batchsize, repeat=False, shuffle=False)
        updater = training.StandardUpdater(train_iter, opt, device=args.gpu)
        trainer = training.Trainer(updater, (args.epoch, 'epoch'))

        @training.make_extension()
        def fix_ignore_label(trainer):
            memnn.fix_ignore_label()

        trainer.extend(fix_ignore_label)
        trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
        trainer.extend(extensions.LogReport())
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy']))
        trainer.extend(extensions.ProgressBar(update_interval=10))
        trainer.run()


if __name__ == '__main__':
    main()
