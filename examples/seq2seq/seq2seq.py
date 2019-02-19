#!/usr/bin/env python

import argparse
import datetime
import io

from nltk.translate import bleu_score
import numpy
import progressbar
import re
import six

import chainer
from chainer import backend
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import chainerx


UNK = 0
EOS = 1


def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_section = numpy.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0)
    return exs


class Seq2seq(chainer.Chain):

    def __init__(self, n_layers, n_source_vocab, n_target_vocab, n_units):
        super(Seq2seq, self).__init__()
        with self.init_scope():
            self.embed_x = L.EmbedID(n_source_vocab, n_units)
            self.embed_y = L.EmbedID(n_target_vocab, n_units)
            self.encoder = L.NStepLSTM(n_layers, n_units, n_units, 0.1)
            self.decoder = L.NStepLSTM(n_layers, n_units, n_units, 0.1)
            self.W = L.Linear(n_units, n_target_vocab)

        self.n_layers = n_layers
        self.n_units = n_units

    def forward(self, xs, ys):
        xs = [x[::-1] for x in xs]

        eos = self.xp.array([EOS], numpy.int32)
        ys_in = [F.concat([eos, y], axis=0) for y in ys]
        ys_out = [F.concat([y, eos], axis=0) for y in ys]

        # Both xs and ys_in are lists of arrays.
        exs = sequence_embed(self.embed_x, xs)
        eys = sequence_embed(self.embed_y, ys_in)

        batch = len(xs)
        # None represents a zero vector in an encoder.
        hx, cx, _ = self.encoder(None, None, exs)
        _, _, os = self.decoder(hx, cx, eys)

        # It is faster to concatenate data before calculating loss
        # because only one matrix multiplication is called.
        concat_os = F.concat(os, axis=0)
        concat_ys_out = F.concat(ys_out, axis=0)
        loss = F.sum(F.softmax_cross_entropy(
            self.W(concat_os), concat_ys_out, reduce='no')) / batch

        chainer.report({'loss': loss}, self)
        n_words = concat_ys_out.shape[0]
        perp = self.xp.exp(loss.array * batch / n_words)
        chainer.report({'perp': perp}, self)
        return loss

    def translate(self, xs, max_length=100):
        batch = len(xs)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            xs = [x[::-1] for x in xs]
            exs = sequence_embed(self.embed_x, xs)
            h, c, _ = self.encoder(None, None, exs)
            ys = self.xp.full(batch, EOS, numpy.int32)
            result = []
            for i in range(max_length):
                eys = self.embed_y(ys)
                eys = F.split_axis(eys, batch, 0)
                h, c, ys = self.decoder(h, c, eys)
                cys = F.concat(ys, axis=0)
                wy = self.W(cys)
                ys = self.xp.argmax(wy.array, axis=1).astype(numpy.int32)
                result.append(ys)

        # Using `xp.concatenate(...)` instead of `xp.stack(result)` here to
        # support NumPy 1.9.
        result = chainer.get_device(numpy).send(
            self.xp.concatenate([x[None, :] for x in result]).T)

        # Remove EOS taggs
        outs = []
        for y in result:
            inds = numpy.argwhere(y == EOS)
            if len(inds) > 0:
                y = y[:inds[0, 0]]
            outs.append(y)
        return outs


@chainer.dataset.converter()
def convert(batch, device):
    def to_device_batch(batch):
        if device is None:
            return batch
        src_xp = backend.get_array_module(*batch)
        xp = device.xp
        concat = src_xp.concatenate(batch, axis=0)
        sections = list(numpy.cumsum(
            [len(x) for x in batch[:-1]], dtype=numpy.int32))
        concat_dst = device.send(concat)
        batch_dst = xp.split(concat_dst, sections)
        return batch_dst

    return {'xs': to_device_batch([x for x, _ in batch]),
            'ys': to_device_batch([y for _, y in batch])}


class CalculateBleu(chainer.training.Extension):

    trigger = 1, 'epoch'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(
            self, model, test_data, key, device, batch=100, max_length=100):
        self.model = model
        self.test_data = test_data
        self.key = key
        self.batch = batch
        self.device = device
        self.max_length = max_length

    def __call__(self, trainer):
        device = self.device

        with chainer.no_backprop_mode():
            references = []
            hypotheses = []
            for i in range(0, len(self.test_data), self.batch):
                sources, targets = zip(*self.test_data[i:i + self.batch])
                references.extend([[t.tolist()] for t in targets])

                sources = [device.send(x) for x in sources]
                ys = [y.tolist()
                      for y in self.model.translate(sources, self.max_length)]
                hypotheses.extend(ys)

        bleu = bleu_score.corpus_bleu(
            references, hypotheses,
            smoothing_function=bleu_score.SmoothingFunction().method1)
        chainer.report({self.key: bleu})


def count_lines(path):
    with io.open(path, encoding='utf-8') as f:
        return sum([1 for _ in f])


def load_vocabulary(path):
    with io.open(path, encoding='utf-8') as f:
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
    with io.open(path, encoding='utf-8') as f:
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


def parse_device(args):
    gpu = None
    if args.gpu is not None:
        gpu = args.gpu
    elif re.match(r'(-|\+|)[0-9]+$', args.device):
        gpu = int(args.device)

    if gpu is not None:
        if gpu < 0:
            return chainer.get_device(numpy)
        else:
            import cupy
            return chainer.get_device((cupy, gpu))

    return chainer.get_device(args.device)


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
    parser.add_argument('--resume', '-r', default='',
                        help='resume the training from snapshot')
    parser.add_argument('--save', '-s', default='',
                        help='save a snapshot of the training')
    parser.add_argument('--unit', '-u', type=int, default=1024,
                        help='number of units')
    parser.add_argument('--layer', '-l', type=int, default=3,
                        help='number of layers')
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
    parser.add_argument('--device', '-d', type=str, default='-1',
                        help='Device specifier. Either ChainerX device '
                        'specifier or an integer. If non-negative integer, '
                        'CuPy arrays with specified device id are used. If '
                        'negative integer, NumPy arrays are used')
    parser.add_argument('--out', '-o', default='result',
                        help='directory to output the result')
    group = parser.add_argument_group('deprecated arguments')
    group.add_argument('--gpu', '-g', type=int, nargs='?', const=0,
                       help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    device = parse_device(args)

    print('Device: {}'.format(device))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # If the device is a ChainerX CUDA device, use the shared device memory
    # pool between ChainerX and CuPy.
    if device.xp is chainerx and device.device.backend.name == 'cuda':
        # TODO(niboshi): The API is provisional.
        chainerx._cuda.cupy_share_allocator()

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

    # Set the current device
    device.use()

    # Setup model
    model = Seq2seq(args.layer, len(source_ids), len(target_ids), args.unit)
    model.to_device(device)

    # Setup optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Setup iterator
    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)

    # Setup updater and trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, converter=convert, device=device)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(extensions.LogReport(
        trigger=(args.log_interval, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'validation/main/loss',
         'main/perp', 'validation/main/perp', 'validation/main/bleu',
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
                model, test_data, 'validation/main/bleu', device),
            trigger=(args.validation_interval, 'iteration'))

    print('start training')
    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

    if args.save:
        # Save a snapshot
        chainer.serializers.save_npz(args.save, trainer)


if __name__ == '__main__':
    main()
