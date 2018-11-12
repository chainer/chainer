import collections
import json
import os

import numpy
import six

import chainer
from chainer import backend
import chainer.functions as F
from chainer import initializers
import chainer.links as L

import babi


def bow_encode(embed, sentences):
    """BoW sentence encoder.

    It is defined as:

    .. math::

       m = \\sum_j A x_j,

    where :math:`A` is an embed matrix, and :math:`x_j` is :math:`j`-th word
    ID.

    """

    e = embed(sentences)
    s = F.sum(e, axis=-2)
    return s


def position_encode(embed, sentences):
    """Position encoding.

    It is defined as:

    .. math::

       m = \\sum_j l_j A x_j,

    where :math:`A` is an embed matrix, :math:`x_j` is :math:`j`-th word ID and

    .. math::

       l_{kj} = (1 - j / J) - (k / d)(1 - 2j / J).

    :math:`J` is length of a sentence and :math:`d` is the dimension of the
    embedding.

    """

    xp = backend.get_array_module(sentences)
    e = embed(sentences)
    n_words, n_units = e.shape[-2:]

    # To avoid 0/0, we use max(length, 1) here.
    # Note that when the length is zero, its embedding is always zero and
    # is ignored.
    length = xp.maximum(
        xp.sum((sentences != 0).astype('f'), axis=-1), 1)
    length = length.reshape((length.shape + (1, 1)))
    k = xp.arange(1, n_units + 1, dtype=numpy.float32) / n_units
    i = xp.arange(1, n_words + 1, dtype=numpy.float32)[:, None]
    coeff = (1 - i / length) - k * (1 - 2.0 * i / length)
    e = coeff * e
    s = F.sum(e, axis=-2)
    return s


def make_encoder(name):
    if name == 'bow':
        return bow_encode
    elif name == 'pe':
        return position_encode
    else:
        raise ValueError('Unknonw encoder type: "%s"' % name)


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
        xp = backend.get_array_module(u)
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

        self.n_units = n_units
        self.max_memory = max_memory
        self.hops = hops

    def fix_ignore_label(self):
        for embed in self.embeds:
            embed.W.array[0, :] = 0

    def register_all(self, sentences):
        for memory in self.memories:
            memory.register_all(sentences)

    def query(self, question):
        u = self.encoder(self.B, question)
        for memory in self.memories:
            u = memory.query(u)
        a = self.W(u)
        return a

    def forward(self, sentences, question):
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


def save_model(directory, model, vocab):
    """Saves a model to a given directory.

    Args:
        directory (str): Path to a directory where you store a model.
        model (chainer.Chain): Model to store.
        vocab (dict): Vocaburaly dictionary.

    """
    encoder = model.predictor.encoder
    if encoder == bow_encode:
        sentence_repr = 'bow'
    elif encoder == position_encode:
        sentence_repr = 'pe'
    else:
        raise ValueError('Cannot serialize encoder: %s' % str(encoder))

    os.makedirs(directory, exist_ok=True)
    parameters = {
        'unit': model.predictor.n_units,
        'hop': model.predictor.hops,
        'max_memory': model.predictor.max_memory,
        'sentence_repr': sentence_repr,
        'vocabulary': vocab,
    }
    with open(os.path.join(directory, 'parameter.json'), 'w') as f:
        json.dump(parameters, f)
    chainer.serializers.save_npz(
        os.path.join(directory, 'model.npz'), model)


def load_model(directory):
    """Loads a model saved.

    Args:
        directory (str): Path to a directory where you load a model.

    Returns:
        tuple: ``(model, vocab)`` where ``model`` is a loaded model and
        ``vocab`` is a ``dict`` storing its vocabulary.

    """
    with open(os.path.join(directory, 'parameter.json')) as f:
        parameters = json.load(f)

    max_memory = parameters['max_memory']
    vocab = collections.defaultdict(lambda: 0)
    vocab.update(parameters['vocabulary'])

    encoder = make_encoder(parameters['sentence_repr'])
    network = MemNN(
        parameters['unit'], len(vocab), encoder, max_memory, parameters['hop'])
    model = chainer.links.Classifier(network, label_key='answer')
    chainer.serializers.load_npz(
        os.path.join(directory, 'model.npz'), model)

    return model, vocab
