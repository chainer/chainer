#!/usr/bin/env python

import argparse
import datetime

from nltk.translate import bleu_score
import numpy
import progressbar
import six

import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions


UNK = 0
EOS = 1


def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_section = numpy.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0)
    return exs


def prepare_attention(variable_list):
    """Preprocess of attention.

    This function pads variables which have diffirent shapes
    such as ``(sequence_length, n_units)``.
    This function returns concatenation of the padded variables
    and an array which represents positions where pads do not exist.

    """
    xp = chainer.cuda.get_array_module(*variable_list)
    batch = len(variable_list)
    lengths = [v.shape[0] for v in variable_list]
    max_len = max(lengths)
    variable_concat = F.pad_sequence(
        variable_list, length=max_len, padding=0.0)
    pad_mask = xp.ones((batch, max_len), dtype='f')
    for i, l in enumerate(lengths):
        pad_mask[i, l:] = 0
    return variable_concat, pad_mask


def split_without_pads(V, lengths):
    """Postprocess of attention.

    This function un-pads a variable, that is,
    removes padded parts and split it into variables which have
    diffirent shapes such as ``(sequence_length, n_units)``.
    This function returns a list of variables.

    """
    batch, max_len, units = V.shape
    # get [len(seq1), len(pad1), len(seq2), len(pad2), ...]
    lengths = numpy.array(lengths)
    lengths_of_seq_and_pad = numpy.stack([lengths, max_len - lengths], axis=1)
    stitch = lengths_of_seq_and_pad.reshape(-1)
    # get indices to be split
    if stitch[-1] == 0:
        stitch = stitch[:-1]
    stitch_split_ids = numpy.cumsum(stitch)[:-1]
    # get [seq1, pad1, seq2, pad2, ...]
    # pick variables at even indices
    split_V = F.split_axis(V.reshape(batch * max_len, units),
                           stitch_split_ids, axis=0)[::2]
    return split_V


class AttentionMechanism(chainer.Chain):
    def __init__(self, n_units, n_att_units=None):
        super(AttentionMechanism, self).__init__()
        self.n_units = n_units
        if n_att_units:
            self.att_units = n_att_units
        else:
            self.att_units = n_units
        with self.init_scope():
            self.W_query = L.Linear(None, self.att_units)
            self.W_key = L.Linear(None, self.att_units)

    def __call__(self, qs, ks):
        """Applies attention mechanism.
        Args:
            qs (~chainer.Variable): Concatenated query vectors
                Its shape is (batchsize, n_query, query_units).
            ks (~chainer.Variable): Concatenated key vectors.
                Its shape is (batchsize, n_key, key_units).
        Returns:
            ~chainer.Variable: Weighted sum of `ks`.
                The weight is computed by a learned function
                of keys and queries.
        """
        concat_Q, q_pad_mask = prepare_attention(qs)
        batch, q_len, q_units = concat_Q.shape
        Q = self.W_query(concat_Q.reshape(batch * q_len, q_units))
        Q = Q.reshape(batch, q_len, self.att_units)
        assert Q.shape == (batch, q_len, self.att_units)

        concat_K, k_pad_mask = prepare_attention(ks)
        batchsize, k_len, k_units = concat_K.shape
        K = self.W_key(concat_K.reshape(batch * k_len, k_units))
        K = K.reshape(batch, k_len, self.att_units)
        assert K.shape == (batch, k_len, self.att_units)

        QK_dot = F.batch_matmul(Q, K, transb=True)
        assert QK_dot.shape == (batch, q_len, k_len)
        # ignore attention weights where padded values are used
        QK_pad_mask = q_pad_mask[:, :, None] * k_pad_mask[:, None, :]
        assert QK_pad_mask.shape == (batch, q_len, k_len)
        minus_infs = self.xp.full(QK_dot.shape, -1024., dtype='f')
        QK_dot = F.where(QK_pad_mask.astype('bool'),
                         QK_dot,
                         minus_infs)
        QK_weight = F.softmax(QK_dot, axis=2)
        assert QK_weight.shape == (batch, q_len, k_len)

        # broadcast weight to be multiplied to vector
        QK_weight = F.broadcast_to(
            QK_weight[:, :, :, None],
            (batch, q_len, k_len, k_units))
        V = F.broadcast_to(
            concat_K[:, None, :, :],
            (batch, q_len, k_len, k_units))
        weighted_V = F.sum(QK_weight * V, axis=2)
        assert weighted_V.shape == (batch, q_len, k_units)
        q_lengths = [q.shape[0] for q in qs]
        split_weighted_V = split_without_pads(weighted_V, q_lengths)
        assert all(v.shape == (l, k_units) for l, v
                   in zip(q_lengths, split_weighted_V))
        return split_weighted_V


class Seq2seq(chainer.Chain):

    def __init__(self, n_layers, n_source_vocab, n_target_vocab, n_units,
                 dropout=0.2, use_attention=False, use_bidirectional=False):
        super(Seq2seq, self).__init__()
        with self.init_scope():
            self.embed_x = L.EmbedID(n_source_vocab, n_units)
            self.embed_y = L.EmbedID(n_target_vocab, n_units)
            if use_bidirectional:
                self.encoder = L.NStepBiLSTM(
                    n_layers, n_units, n_units, dropout)
            else:
                self.encoder = L.NStepLSTM(
                    n_layers, n_units, n_units, dropout)
            self.decoder = L.NStepLSTM(
                n_layers, n_units, n_units, dropout)
            self.W = L.Linear(None, n_target_vocab)

            if use_attention:
                self.attention = AttentionMechanism(n_units)

        self.n_layers = n_layers
        self.n_units = n_units
        self.use_bidirectional = use_bidirectional

    def __call__(self, xs, ys):
        xs = [x[::-1] for x in xs]

        eos = self.xp.array([EOS], numpy.int32)
        ys_in = [F.concat([eos, y], axis=0) for y in ys]
        ys_out = [F.concat([y, eos], axis=0) for y in ys]

        # Both xs and ys_in are lists of arrays.
        exs = sequence_embed(self.embed_x, xs)
        eys = sequence_embed(self.embed_y, ys_in)

        batch = len(xs)
        # None represents a zero vector in an encoder.
        hx, cx, enc_os = self.encoder(None, None, exs)
        if self.use_bidirectional:
            # In NStepBiLSTM, cells of rightward LSTMs
            # are stored at odd indices
            hx = hx[::2]
            cx = cx[::2]
        _, _, os = self.decoder(hx, cx, eys)

        # It is faster to concatenate data before calculating loss
        # because only one matrix multiplication is called.
        concat_os = F.concat(os, axis=0)
        concat_ys_out = F.concat(ys_out, axis=0)

        if hasattr(self, 'attention'):
            vs = self.attention(os, enc_os)
            concat_vs = F.concat(vs, axis=0)
            concat_os = F.concat([concat_os, concat_vs], axis=1)

        loss = F.sum(F.softmax_cross_entropy(
            self.W(concat_os), concat_ys_out, reduce='no')) / batch

        chainer.report({'loss': loss.data}, self)
        n_words = concat_ys_out.shape[0]
        perp = self.xp.exp(loss.data * batch / n_words)
        chainer.report({'perp': perp}, self)
        return loss

    def translate(self, xs, max_length=100):
        batch = len(xs)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            xs = [x[::-1] for x in xs]
            exs = sequence_embed(self.embed_x, xs)
            h, c, enc_os = self.encoder(None, None, exs)
            if self.use_bidirectional:
                h = h[::2]
                c = c[::2]
            ys = self.xp.full(batch, EOS, numpy.int32)
            result = []
            for i in range(max_length):
                eys = self.embed_y(ys)
                eys = F.split_axis(eys, batch, 0)
                h, c, ys = self.decoder(h, c, eys)
                cys = F.concat(ys, axis=0)

                if hasattr(self, 'attention'):
                    vs = self.attention(ys, enc_os)
                    cvs = F.concat(vs, axis=0)
                    cys = F.concat([cys, cvs], axis=1)

                wy = self.W(cys)
                ys = self.xp.argmax(wy.data, axis=1).astype(numpy.int32)
                result.append(ys)

        # Using `xp.concatenate(...)` instead of `xp.stack(result)` here to
        # support NumPy 1.9.
        result = cuda.to_cpu(
            self.xp.concatenate([self.xp.expand_dims(x, 0) for x in result]).T)

        # Remove EOS taggs
        outs = []
        for y in result:
            inds = numpy.argwhere(y == EOS)
            if len(inds) > 0:
                y = y[:inds[0, 0]]
            outs.append(y)
        return outs


def convert(batch, device):
    def to_device_batch(batch):
        if device is None:
            return batch
        elif device < 0:
            return [chainer.dataset.to_device(device, x) for x in batch]
        else:
            xp = cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = numpy.cumsum([len(x)
                                     for x in batch[:-1]], dtype=numpy.int32)
            concat_dev = chainer.dataset.to_device(device, concat)
            batch_dev = cuda.cupy.split(concat_dev, sections)
            return batch_dev

    return {'xs': to_device_batch([x for x, _ in batch]),
            'ys': to_device_batch([y for _, y in batch])}


class CalculateBleu(chainer.training.Extension):

    trigger = 1, 'epoch'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(
            self, model, test_data, key, batch=100, device=-1, max_length=100):
        self.model = model
        self.test_data = test_data
        self.key = key
        self.batch = batch
        self.device = device
        self.max_length = max_length

    def __call__(self, trainer):
        with chainer.no_backprop_mode():
            references = []
            hypotheses = []
            for i in range(0, len(self.test_data), self.batch):
                sources, targets = zip(*self.test_data[i:i + self.batch])
                references.extend([[t.tolist()] for t in targets])

                sources = [
                    chainer.dataset.to_device(self.device, x) for x in sources]
                ys = [y.tolist()
                      for y in self.model.translate(sources, self.max_length)]
                hypotheses.extend(ys)

        bleu = bleu_score.corpus_bleu(
            references, hypotheses,
            smoothing_function=bleu_score.SmoothingFunction().method1)
        bleu *= 100
        chainer.report({self.key: bleu})


def count_lines(path):
    with open(path) as f:
        return sum([1 for _ in f])


def load_vocabulary(path):
    with open(path) as f:
        # +2 for UNK and EOS
        word_ids = {line.strip(): i + 2 for i, line in enumerate(f)}
    word_ids['<UNK>'] = 0
    word_ids['<EOS>'] = 1
    return word_ids


def load_data(vocabulary, path):
    n_lines = count_lines(path)
    bar = progressbar.ProgressBar()
    data = []
    print('loading...: %s' % path)
    with open(path) as f:
        for line in bar(f, max_value=n_lines):
            words = line.strip().split()
            array = numpy.array([vocabulary.get(w, UNK)
                                 for w in words], numpy.int32)
            data.append(array)
    return data


def load_data_using_dataset_api(
        src_vocab, src_path, target_vocab, target_path, filter_func):

    def _transform_line(vocabulary, line):
        words = line.strip().split()
        return numpy.array(
            [vocabulary.get(w, UNK) for w in words], numpy.int32)

    def _transform(example):
        source, target = example
        return (
            _transform_line(src_vocab, source),
            _transform_line(target_vocab, target)
        )

    return chainer.datasets.TransformDataset(
        chainer.datasets.TextDataset(
            [src_path, target_path],
            encoding='utf-8',
            filter_func=filter_func
        ), _transform)


def calculate_unknown_ratio(data):
    unknown = sum((s == UNK).sum() for s in data)
    total = sum(s.size for s in data)
    return unknown / total


def main():
    parser = argparse.ArgumentParser(description='Chainer example: seq2seq')
    parser.add_argument('SOURCE', help='source sentence list')
    parser.add_argument('TARGET', help='target sentence list')
    parser.add_argument('SOURCE_VOCAB', help='source vocabulary file')
    parser.add_argument('TARGET_VOCAB', help='target vocabulary file')
    parser.add_argument('--validation-source',
                        help='source sentence list for validation')
    parser.add_argument('--validation-target',
                        help='target sentence list for validation')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='number of sentence pairs in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', default='',
                        help='resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1024,
                        help='number of units')
    parser.add_argument('--layer', '-l', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--dropout', '-d', type=float, default=0.2,
                        help='ratio of dropout')
    parser.add_argument('--use-dataset-api', default=False,
                        action='store_true',
                        help='use TextDataset API to reduce CPU memory usage')
    parser.add_argument('--min-source-sentence', type=int, default=1,
                        help='minimium length of source sentence')
    parser.add_argument('--max-source-sentence', type=int, default=50,
                        help='maximum length of source sentence')
    parser.add_argument('--min-target-sentence', type=int, default=1,
                        help='minimium length of target sentence')
    parser.add_argument('--max-target-sentence', type=int, default=50,
                        help='maximum length of target sentence')
    parser.add_argument('--log-interval', type=int, default=200,
                        help='number of iteration to show log')
    parser.add_argument('--validation-interval', type=int, default=4000,
                        help='number of iteration to evlauate the model '
                        'with validation dataset')
    parser.add_argument('--out', '-o', default='result',
                        help='directory to output the result')
    parser.add_argument('--use-attention', default=False,
                        action='store_true',
                        help='use attention mechanism for decoder')
    parser.add_argument('--use-bidirectional', default=False,
                        action='store_true',
                        help='use bidirectional LSTM encoder')
    args = parser.parse_args()

    # Load pre-processed dataset
    print('[{}] Loading dataset... (this may take several minutes)'.format(
        datetime.datetime.now()))
    source_ids = load_vocabulary(args.SOURCE_VOCAB)
    target_ids = load_vocabulary(args.TARGET_VOCAB)

    if args.use_dataset_api:
        # By using TextDataset, you can avoid loading whole dataset on memory.
        # This significantly reduces the host memory usage.
        def _filter_func(s, t):
            sl = len(s.strip().split())  # number of words in source line
            tl = len(t.strip().split())  # number of words in target line
            return (
                args.min_source_sentence <= sl <= args.max_source_sentence and
                args.min_target_sentence <= tl <= args.max_target_sentence)

        train_data = load_data_using_dataset_api(
            source_ids, args.SOURCE,
            target_ids, args.TARGET,
            _filter_func,
        )
    else:
        # Load all records on memory.
        train_source = load_data(source_ids, args.SOURCE)
        train_target = load_data(target_ids, args.TARGET)
        assert len(train_source) == len(train_target)

        train_data = [
            (s, t)
            for s, t in six.moves.zip(train_source, train_target)
            if (args.min_source_sentence <= len(s) <= args.max_source_sentence
                and
                args.min_target_sentence <= len(t) <= args.max_target_sentence)
        ]
    print('[{}] Dataset loaded.'.format(datetime.datetime.now()))

    if not args.use_dataset_api:
        # Skip printing statistics when using TextDataset API, as it is slow.
        train_source_unknown = calculate_unknown_ratio(
            [s for s, _ in train_data])
        train_target_unknown = calculate_unknown_ratio(
            [t for _, t in train_data])

        print('Source vocabulary size: %d' % len(source_ids))
        print('Target vocabulary size: %d' % len(target_ids))
        print('Train data size: %d' % len(train_data))
        print('Train source unknown ratio: %.2f%%' % (
            train_source_unknown * 100))
        print('Train target unknown ratio: %.2f%%' % (
            train_target_unknown * 100))

    target_words = {i: w for w, i in target_ids.items()}
    source_words = {i: w for w, i in source_ids.items()}

    model = Seq2seq(args.layer, len(source_ids), len(target_ids), args.unit,
                    args.dropout, args.use_attention, args.use_bidirectional)

    if args.gpu >= 0:
        chainer.backends.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    # Setup optimizer
    optimizer = chainer.optimizers.Adam(alpha=1e-4)
    optimizer.setup(model)

    # Setup iterator
    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)

    # Setup updater and trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, converter=convert, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(extensions.LogReport(
        trigger=(args.log_interval, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss',
         'main/perp', 'validation/main/bleu',
         'elapsed_time']),
        trigger=(args.log_interval, 'iteration'))

    if args.validation_source and args.validation_target:
        test_source = load_data(source_ids, args.validation_source)
        test_target = load_data(target_ids, args.validation_target)
        assert len(test_source) == len(test_target)
        test_data = list(six.moves.zip(test_source, test_target))
        test_data = [(s, t) for s, t in test_data if 0 < len(s) and 0 < len(t)]
        test_source_unknown = calculate_unknown_ratio(
            [s for s, _ in test_data])
        test_target_unknown = calculate_unknown_ratio(
            [t for _, t in test_data])

        print('Validation data: %d' % len(test_data))
        print('Validation source unknown ratio: %.2f%%' %
              (test_source_unknown * 100))
        print('Validation target unknown ratio: %.2f%%' %
              (test_target_unknown * 100))

        @chainer.training.make_extension()
        def translate(trainer):
            source, target = test_data[numpy.random.choice(len(test_data))]
            result = model.translate([model.xp.array(source)])[0]

            source_sentence = ' '.join([source_words[x] for x in source])
            target_sentence = ' '.join([target_words[y] for y in target])
            result_sentence = ' '.join([target_words[y] for y in result])
            print('# source : ' + source_sentence)
            print('# result : ' + result_sentence)
            print('# expect : ' + target_sentence)

        trainer.extend(
            translate, trigger=(args.validation_interval, 'iteration'))
        trainer.extend(
            CalculateBleu(
                model, test_data, 'validation/main/bleu', device=args.gpu),
            trigger=(args.validation_interval, 'iteration'))

    print('start training')
    trainer.run()


if __name__ == '__main__':
    main()
