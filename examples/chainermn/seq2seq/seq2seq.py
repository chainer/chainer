# encoding: utf-8

import argparse
import math
import os.path
import pickle
import re
import sys
import time

from nltk.translate import bleu_score
import numpy
import six

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import reporter
from chainer import training
from chainer.training import extensions
import chainermn

import europal


def cached_call(fname, func, *args):
    if os.path.exists(fname):
        with open(fname, 'rb') as f:
            return pickle.load(f)
    else:
        # not yet cached
        val = func(*args)
        with open(fname, 'wb') as f:
            pickle.dump(val, f)
        return val


def read_source(in_dir, cache=None):
    en_path = os.path.join(in_dir, 'giga-fren.release2.fixed.en')
    source_vocab = ['<eos>', '<unk>'] + europal.count_words(en_path)
    source_data = europal.make_dataset(en_path, source_vocab)

    return source_vocab, source_data


def read_target(in_dir, cahce=None):
    fr_path = os.path.join(in_dir, 'giga-fren.release2.fixed.fr')
    target_vocab = ['<eos>', '<unk>'] + europal.count_words(fr_path)
    target_data = europal.make_dataset(fr_path, target_vocab)

    return target_vocab, target_data


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

        xs = [x[::-1] for x in xs]

        eos = self.xp.zeros(1, self.xp.int32)
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

        reporter.report({'loss': loss.data}, self)
        n_words = concat_ys_out.shape[0]
        perp = self.xp.exp(loss.data * batch / n_words)
        reporter.report({'perp': perp}, self)
        return loss

    def translate(self, xs, max_length=100):
        batch = len(xs)
        with chainer.no_backprop_mode():
            with chainer.using_config('train', False):
                xs = [x[::-1] for x in xs]
                exs = sequence_embed(self.embed_x, xs)
                # Initial hidden variable and cell variable
                # zero = self.xp.zeros((self.n_layers, batch, self.n_units), self.xp.float32)  # NOQA
                # h, c, _ = self.encoder(zero, zero, exs, train=False)  # NOQA
                h, c, _ = self.encoder(None, None, exs)
                ys = self.xp.zeros(batch, self.xp.int32)
                result = []
                for i in range(max_length):
                    eys = self.embed_y(ys)
                    eys = chainer.functions.split_axis(
                        eys, batch, 0, force_tuple=True)
                    h, c, ys = self.decoder(h, c, eys)
                    cys = chainer.functions.concat(ys, axis=0)
                    wy = self.W(cys)
                    ys = self.xp.argmax(wy.data, axis=1).astype(self.xp.int32)
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
    def to_device_batch(batch):
        if device is None:
            return batch
        elif device < 0:
            return [chainer.dataset.to_device(device, x) for x in batch]
        else:
            xp = cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = numpy.cumsum(
                [len(x) for x in batch[:-1]], dtype=numpy.int32)
            concat_dev = chainer.dataset.to_device(device, concat)
            batch_dev = cuda.cupy.split(concat_dev, sections)
            return batch_dev

    return tuple(
        to_device_batch([x for x, _ in batch]) +
        to_device_batch([y for _, y in batch]))


class CalculateBleu(chainer.training.Extension):
    # priority = chainer.training.PRIORITY_WRITER
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
        reporter.report({self.key: bleu})


class BleuEvaluator(extensions.Evaluator):
    def __init__(self, model, test_data, device=-1, batch=100,
                 max_length=100, comm=None):
        super(BleuEvaluator, self).__init__({'main': None}, model)
        self.model = model
        self.test_data = test_data
        self.batch = batch
        self.device = device
        self.max_length = max_length
        self.comm = comm

    def evaluate(self):
        bt = time.time()
        with chainer.no_backprop_mode():
            references = []
            hypotheses = []
            observation = {}
            with reporter.report_scope(observation):
                for i in range(0, len(self.test_data), self.batch):
                    src, trg = zip(*self.test_data[i:i + self.batch])
                    references.extend([[t.tolist()] for t in trg])

                    src = [chainer.dataset.to_device(self.device, x)
                           for x in src]
                    ys = [y.tolist()
                          for y in self.model.translate(src, self.max_length)]
                    hypotheses.extend(ys)

                bleu = bleu_score.corpus_bleu(
                    references, hypotheses,
                    smoothing_function=bleu_score.SmoothingFunction().method1)
                reporter.report({'bleu': bleu}, self.model)
        et = time.time()

        if self.comm is not None:
            # This evaluator is called via chainermn.MultiNodeEvaluator
            for i in range(0, self.comm.size):
                print('BleuEvaluator::evaluate(): '
                      'took {:.3f} [s]'.format(et - bt))
                sys.stdout.flush()
                self.comm.mpi_comm.Barrier()
        else:
            # This evaluator is called from a conventional
            # Chainer exntension
            print('BleuEvaluator(single)::evaluate(): '
                  'took {:.3f} [s]'.format(et - bt))
            sys.stdout.flush()
        return observation


def create_optimizer(opt_arg):
    """Parse a string and get an optimizer.

    The syntax is:

        opt(params...)

    where
        opt := sgd | adam
        param := [float | key=val]...
    """
    m = re.match(r'(adam|sgd)\(([^)]*)\)', opt_arg, re.I)
    name = m.group(1).lower()
    args = m.group(2)

    names_dict = {
        'adadelta': chainer.optimizers.AdaDelta,
        'adagrad': chainer.optimizers.AdaGrad,
        'adam': chainer.optimizers.Adam,
        'momentumsgd': chainer.optimizers.MomentumSGD,
        'nesterovag': chainer.optimizers.NesterovAG,
        'rmsprop': chainer.optimizers.RMSprop,
        'rmspropgraves': chainer.optimizers.RMSpropGraves,
        'sgd': chainer.optimizers.SGD,
        'smorms3': chainer.optimizers.SMORMS3,
    }

    try:
        opt = names_dict[name]
    except KeyError:
        raise RuntimeError('Unknown optimizer: \'{}\' in \'{}\''.format(
            name, opt_arg))

    # positional arguments
    pos = []
    # keyword arguments
    kw = {}

    args = args.strip()

    if len(args) > 0:
        for a in re.split(r',\s*', args):
            if a.find('=') >= 0:
                key, val = a.split('=')
                kw[key] = float(val)
            else:
                pos.append(float(a))

    return opt(*pos, **kw)


def _get_num_split(excp):
    """Get the preferrable number of split from a DataSizeError error"""
    ps = excp.pickled_size
    mx = excp.max_size
    return (ps + mx - 1) // mx


def _slices(excp):
    """Get a list of slices that are expected to fit in a single send/recv."""
    ds = excp.dataset_size
    nsplit = _get_num_split(excp)
    size = math.ceil(ds / nsplit)

    return [(b, min(e, ds)) for b, e in
            ((i * size, (i + 1) * size) for i in range(0, nsplit))]


def main():
    parser = argparse.ArgumentParser(description='Chainer example: seq2seq')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--bleu', action='store_true', default=False,
                        help='Report BLEU score')
    parser.add_argument('--gpu', '-g', action='store_true',
                        help='Use GPU')
    parser.add_argument('--cache', '-c', default=None,
                        help='Directory to cache pre-processed dataset')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1024,
                        help='Number of units')
    parser.add_argument('--communicator', default='hierarchical',
                        help='Type of communicator')
    parser.add_argument('--stop', '-s', type=str, default='15e',
                        help='Stop trigger (ex. "500i", "15e")')
    parser.add_argument('--input', '-i', type=str, default='wmt',
                        help='Input directory')
    parser.add_argument('--optimizer', type=str, default='adam()',
                        help='Optimizer and its argument')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    args = parser.parse_args()

    # Prepare ChainerMN communicator
    if args.gpu:
        comm = chainermn.create_communicator('hierarchical')
        dev = comm.intra_rank
    else:
        comm = chainermn.create_communicator('naive')
        dev = -1

    if comm.rank == 0:
        print('==========================================')
        print('Num process (COMM_WORLD): {}'.format(comm.size))
        if args.gpu:
            print('Using GPUs')
        print('Using {} communicator'.format(args.communicator))
        print('Num unit: {}'.format(args.unit))
        print('Num Minibatch-size: {}'.format(args.batchsize))
        print('==========================================')

    # Rank 0 prepares all data
    if comm.rank == 0:
        if args.cache and not os.path.exists(args.cache):
            os.mkdir(args.cache)

        # Read source data
        bt = time.time()
        if args.cache:
            cache_file = os.path.join(args.cache, 'source.pickle')
            source_vocab, source_data = cached_call(cache_file,
                                                    read_source,
                                                    args.input, args.cache)
        else:
            source_vocab, source_data = read_source(args.input, args.cache)
        et = time.time()
        print('RD source done. {:.3f} [s]'.format(et - bt))
        sys.stdout.flush()

        # Read target data
        bt = time.time()
        if args.cache:
            cache_file = os.path.join(args.cache, 'target.pickle')
            target_vocab, target_data = cached_call(cache_file,
                                                    read_target,
                                                    args.input, args.cache)
        else:
            target_vocab, target_data = read_target(args.input, args.cache)
        et = time.time()
        print('RD target done. {:.3f} [s]'.format(et - bt))
        sys.stdout.flush()

        print('Original training data size: %d' % len(source_data))
        train_data = [(s, t)
                      for s, t in six.moves.zip(source_data, target_data)
                      if 0 < len(s) < 50 and 0 < len(t) < 50]
        print('Filtered training data size: %d' % len(train_data))

        en_path = os.path.join(args.input, 'dev', 'newstest2013.en')
        source_data = europal.make_dataset(en_path, source_vocab)
        fr_path = os.path.join(args.input, 'dev', 'newstest2013.fr')
        target_data = europal.make_dataset(fr_path, target_vocab)
        assert(len(source_data) == len(target_data))
        test_data = [(s, t) for s, t
                     in six.moves.zip(source_data, target_data)
                     if 0 < len(s) and 0 < len(t)]

        source_ids = {word: index
                      for index, word in enumerate(source_vocab)}
        target_ids = {word: index
                      for index, word in enumerate(target_vocab)}
    else:
        # target_data, source_data = None, None
        train_data, test_data = None, None
        target_ids, source_ids = None, None

    # Print GPU id
    for i in range(0, comm.size):
        if comm.rank == i:
            print('Rank {} GPU: {}'.format(comm.rank, dev))
        sys.stdout.flush()
        comm.mpi_comm.Barrier()

    # broadcast id- > word dictionary
    source_ids = comm.bcast_obj(source_ids, root=0)
    target_ids = comm.bcast_obj(target_ids, root=0)

    target_words = {i: w for w, i in target_ids.items()}
    source_words = {i: w for w, i in source_ids.items()}

    if comm.rank == 0:
        print('target_words : {}'.format(len(target_words)))
        print('source_words : {}'.format(len(source_words)))

    model = Seq2seq(3, len(source_ids), len(target_ids), args.unit)

    if dev >= 0:
        chainer.cuda.get_device_from_id(dev).use()
        model.to_gpu(dev)

    # determine the stop trigger
    m = re.match(r'^(\d+)e$', args.stop)
    if m:
        trigger = (int(m.group(1)), 'epoch')
    else:
        m = re.match(r'^(\d+)i$', args.stop)
        if m:
            trigger = (int(m.group(1)), 'iteration')
        else:
            if comm.rank == 0:
                sys.stderr.write('Error: unknown stop trigger: {}'.format(
                    args.stop))
            exit(-1)

    if comm.rank == 0:
        print('Trigger: {}'.format(trigger))

    optimizer = chainermn.create_multi_node_optimizer(
        create_optimizer(args.optimizer), comm)
    optimizer.setup(model)

    # Broadcast dataset
    # Sanity check of train_data
    train_data = chainermn.scatter_dataset(train_data, comm)

    test_data = chainermn.scatter_dataset(test_data, comm)

    train_iter = chainer.iterators.SerialIterator(train_data,
                                                  args.batchsize,
                                                  shuffle=False)
    updater = training.StandardUpdater(
        train_iter, optimizer, converter=convert, device=dev)
    trainer = training.Trainer(updater,
                               trigger,
                               out=args.out)

    trainer.extend(chainermn.create_multi_node_evaluator(
        BleuEvaluator(model, test_data, device=dev, comm=comm),
        comm))

    def translate_one(source, target):
        words = europal.split_sentence(source)
        print('# source : ' + ' '.join(words))
        x = model.xp.array(
            [source_ids.get(w, 1) for w in words], numpy.int32)
        ys = model.translate([x])[0]
        words = [target_words[y] for y in ys]
        print('#  result : ' + ' '.join(words))
        print('#  expect : ' + target)

    # @chainer.training.make_extension(trigger=(200, 'iteration'))
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
        source = ' '.join([source_words.get(i, '') for i in source])
        target = ' '.join([target_words.get(i, '') for i in target])
        translate_one(source, target)

    if comm.rank == 0:
        trainer.extend(extensions.LogReport(trigger=(1, 'epoch')),
                       trigger=(1, 'epoch'))

        report = extensions.PrintReport(['epoch',
                                         'iteration',
                                         'main/loss',
                                         'main/perp',
                                         'validation/main/bleu',
                                         'elapsed_time'])
        trainer.extend(report, trigger=(1, 'epoch'))

    comm.mpi_comm.Barrier()
    if comm.rank == 0:
        print('start training')
        sys.stdout.flush()

    trainer.run()


if __name__ == '__main__':
    main()
