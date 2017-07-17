# encoding: utf-8
import numpy

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import reporter


def sequence_embed(embed, xs, dropout=0.):
    x_len = [len(x) for x in xs]
    x_section = numpy.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    ex = F.dropout(ex, ratio=dropout)
    exs = F.split_axis(ex, x_section, 0)
    return exs


def block_embed(embed, x, dropout=0.):
    batch, length = x.shape
    e = embed(x.reshape((batch * length, )))
    # (batch * length, units)
    e = F.transpose(F.stack(F.split_axis(e, batch, axis=0), axis=0), (0, 2, 1))
    # (batch, units, length)
    e = e[:, :, :, None]
    # (batch, units, length, 1)
    e = F.dropout(e, ratio=dropout)
    return e


class Classifier(chainer.Chain):

    def __init__(self, encoder, n_class, dropout=0.1):
        super(Classifier, self).__init__()
        with self.init_scope():
            self.encoder = encoder
            self.output = L.Linear(encoder.out_units, n_class)
        self.dropout = dropout

    def __call__(self, xs, ys):
        concat_outputs = self.predict(xs)
        concat_truths = F.concat(ys, axis=0)

        loss = F.softmax_cross_entropy(concat_outputs, concat_truths)
        accuracy = F.accuracy(concat_outputs, concat_truths)
        reporter.report({'loss': loss.data}, self)
        reporter.report({'accuracy': accuracy.data}, self)
        return loss

    def predict(self, xs, softmax=False, argmax=False):
        concat_encodings = F.dropout(self.encoder(xs), ratio=self.dropout)
        concat_outputs = self.output(concat_encodings)
        if softmax:
            return F.softmax(concat_outputs).data
        elif argmax:
            return self.xp.argmax(concat_outputs.data, axis=1)
        else:
            return concat_outputs


class RNNEncoder(chainer.Chain):

    def __init__(self, n_layers, n_vocab, n_units, dropout=0.1):
        super(RNNEncoder, self).__init__(
            embed=L.EmbedID(n_vocab, n_units),
            encoder=L.NStepLSTM(n_layers, n_units, n_units, dropout),
        )
        self.out_units = n_units
        self.dropout = dropout

    def __call__(self, xs):
        exs = sequence_embed(self.embed, xs, self.dropout)
        last_h, last_c, ys = self.encoder(None, None, exs)
        # last_h's shape = (n_layer, batchsize, n_units)
        concat_outputs = last_h[-1]
        return concat_outputs


class CNNEncoder(chainer.Chain):

    def __init__(self, n_layers, n_vocab, n_units, dropout=0.1):
        out_units = n_units // 3
        super(CNNEncoder, self).__init__(
            embed=L.EmbedID(n_vocab, n_units, ignore_label=-1),
            encoder_w2=L.Convolution2D(
                n_units, out_units, ksize=(2, 1), stride=1, pad=(1, 0),
                nobias=True),
            encoder_w3=L.Convolution2D(
                n_units, out_units, ksize=(3, 1), stride=1, pad=(2, 0),
                nobias=True),
            encoder_w4=L.Convolution2D(
                n_units, out_units, ksize=(4, 1), stride=1, pad=(3, 0),
                nobias=True),
            mlp=MLP(n_layers, out_units * 3, dropout)
        )
        self.out_units = out_units * 3
        self.dropout = dropout

    def __call__(self, xs):
        x_block = chainer.dataset.convert.concat_examples(xs, padding=-1)
        ex_block = block_embed(self.embed, x_block, self.dropout)
        h_w2 = F.max(self.encoder_w2(ex_block), axis=2)
        h_w3 = F.max(self.encoder_w3(ex_block), axis=2)
        h_w4 = F.max(self.encoder_w4(ex_block), axis=2)
        h = F.concat([h_w2, h_w3, h_w4], axis=1)
        h = F.relu(h)
        h = F.dropout(h, ratio=self.dropout)
        h = self.mlp(h)
        return h


class MLP(chainer.ChainList):

    def __init__(self, n_layers, n_units, dropout=0.1):
        super(MLP, self).__init__()
        for i in range(n_layers):
            self.add_link(L.Linear(None, n_units))
        self.dropout = dropout
        self.out_units = n_units

    def __call__(self, x):
        for i, link in enumerate(self.children()):
            x = F.dropout(x, ratio=self.dropout)
            x = F.relu(link(x))
        return x


class BOWEncoder(chainer.Chain):

    def __init__(self, n_layers, n_vocab, n_units, dropout=0.1):
        super(BOWEncoder, self).__init__(
            embed=L.EmbedID(n_vocab, n_units, ignore_label=-1),
        )
        self.out_units = n_units
        self.dropout = dropout

    def __call__(self, xs):
        x_block = chainer.dataset.convert.concat_examples(xs, padding=-1)
        ex_block = block_embed(self.embed, x_block)
        x_len = self.xp.array([len(x) for x in xs], 'i')[:, None, None]
        h = F.sum(ex_block, axis=2) / x_len
        return h


class BOWMLPEncoder(chainer.Chain):
    def __init__(self, n_layers, n_vocab, n_units, dropout=0.1):
        super(BOWMLPEncoder, self).__init__(
            bow_encoder=BOWEncoder(n_layers, n_vocab, n_units, dropout),
            mlp_encoder=MLP(n_layers, n_units, dropout)
        )
        self.out_units = n_units

    def __call__(self, xs):
        h = self.bow_encoder(xs)
        h = self.mlp_encoder(h)
        return h
