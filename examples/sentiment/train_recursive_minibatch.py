import argparse

import numpy

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import reporter
from chainer import training
from chainer.training import extensions

import data
import thin_stack


def linearize_tree(vocab, root, xp=numpy):
    # Left node indexes for all parent nodes
    lefts = []
    # Right node indexes for all parent nodes
    rights = []
    # Parent node indexes
    dests = []
    # All labels to predict for all parent nodes
    labels = []

    # All words of leaf nodes
    words = []
    # Leaf labels
    leaf_labels = []

    # Current leaf node index
    leaf_index = [0]

    def traverse_leaf(exp):
        if len(exp) == 2:
            label, leaf = exp
            if leaf not in vocab:
                vocab[leaf] = len(vocab)
            words.append(vocab[leaf])
            leaf_labels.append(int(label))
            leaf_index[0] += 1
        elif len(exp) == 3:
            _, left, right = exp
            traverse_leaf(left)
            traverse_leaf(right)

    traverse_leaf(root)

    # Current internal node index
    node_index = leaf_index
    leaf_index = [0]

    def traverse_node(exp):
        if len(exp) == 2:
            leaf_index[0] += 1
            return leaf_index[0] - 1
        elif len(exp) == 3:
            label, left, right = exp
            l = traverse_node(left)
            r = traverse_node(right)

            lefts.append(l)
            rights.append(r)
            dests.append(node_index[0])
            labels.append(int(label))

            node_index[0] += 1
            return node_index[0] - 1

    traverse_node(root)
    assert len(lefts) == len(words) - 1

    return {
        'lefts': xp.array(lefts, 'i'),
        'rights': xp.array(rights, 'i'),
        'dests': xp.array(dests, 'i'),
        'words': xp.array(words, 'i'),
        'labels': xp.array(labels, 'i'),
        'leaf_labels': xp.array(leaf_labels, 'i'),
    }


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
        [to_device(d['lefts']) for d in batch] +
        [to_device(d['rights']) for d in batch] +
        [to_device(d['dests']) for d in batch] +
        [to_device(d['labels']) for d in batch] +
        [to_device(d['words']) for d in batch] +
        [to_device(d['leaf_labels']) for d in batch]
    )


class ThinStackRecursiveNet(chainer.Chain):

    def __init__(self, n_vocab, n_units, n_label):
        super(ThinStackRecursiveNet, self).__init__(
            embed=L.EmbedID(n_vocab, n_units),
            l=L.Linear(n_units * 2, n_units),
            w=L.Linear(n_units, n_label))
        self.n_units = n_units

    def leaf(self, x):
        return self.embed(x)

    def node(self, left, right):
        return F.tanh(self.l(F.concat((left, right))))

    def label(self, v):
        return self.w(v)

    def forward(self, *inputs):
        batch = len(inputs) // 6
        lefts = inputs[0: batch]
        rights = inputs[batch: batch * 2]
        dests = inputs[batch * 2: batch * 3]
        labels = inputs[batch * 3: batch * 4]
        sequences = inputs[batch * 4: batch * 5]
        leaf_labels = inputs[batch * 5: batch * 6]

        inds = numpy.argsort([-len(l) for l in lefts])
        # Sort all arrays in descending order and transpose them
        lefts = F.transpose_sequence([lefts[i] for i in inds])
        rights = F.transpose_sequence([rights[i] for i in inds])
        dests = F.transpose_sequence([dests[i] for i in inds])
        labels = F.transpose_sequence([labels[i] for i in inds])
        sequences = F.transpose_sequence([sequences[i] for i in inds])
        leaf_labels = F.transpose_sequence(
            [leaf_labels[i] for i in inds])

        batch = len(inds)
        maxlen = len(sequences)

        loss = 0
        count = 0
        correct = 0

        stack = self.xp.zeros((batch, maxlen * 2, self.n_units), 'f')
        for i, (word, label) in enumerate(zip(sequences, leaf_labels)):
            batch = word.shape[0]
            es = self.leaf(word)
            ds = self.xp.full((batch,), i, 'i')
            y = self.label(es)
            loss += F.softmax_cross_entropy(y, label, normalize=False) * batch
            count += batch
            predict = self.xp.argmax(y.data, axis=1)
            correct += (predict == label.data).sum()

            stack = thin_stack.thin_stack_set(stack, ds, es)

        for left, right, dest, label in zip(lefts, rights, dests, labels):
            l, stack = thin_stack.thin_stack_get(stack, left)
            r, stack = thin_stack.thin_stack_get(stack, right)
            o = self.node(l, r)
            y = self.label(o)
            batch = l.shape[0]
            loss += F.softmax_cross_entropy(y, label, normalize=False) * batch
            count += batch
            predict = self.xp.argmax(y.data, axis=1)
            correct += (predict == label.data).sum()

            stack = thin_stack.thin_stack_set(stack, dest, o)

        loss /= count
        reporter.report({'loss': loss}, self)
        reporter.report({'total': count}, self)
        reporter.report({'correct': correct}, self)
        return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epoch', '-e', default=400, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--unit', '-u', default=30, type=int,
                        help='number of units')
    parser.add_argument('--batchsize', '-b', type=int, default=25,
                        help='learning minibatch size')
    parser.add_argument('--label', '-l', type=int, default=5,
                        help='number of labels')
    parser.add_argument('--epocheval', '-p', type=int, default=5,
                        help='number of epochs per evaluation')
    parser.add_argument('--test', dest='test', action='store_true')
    parser.set_defaults(test=False)
    args = parser.parse_args()

    vocab = {}
    max_size = None
    train_trees = data.read_corpus('trees/train.txt', max_size)
    test_trees = data.read_corpus('trees/test.txt', max_size)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        xp = cuda.cupy
    else:
        xp = numpy

    train_data = [linearize_tree(vocab, t, xp) for t in train_trees]
    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    test_data = [linearize_tree(vocab, t, xp) for t in test_trees]
    test_iter = chainer.iterators.SerialIterator(
        test_data, args.batchsize, repeat=False, shuffle=False)

    model = ThinStackRecursiveNet(len(vocab), args.unit, args.label)

    if args.gpu >= 0:
        model.to_gpu()

    optimizer = chainer.optimizers.AdaGrad(0.1)
    optimizer.setup(model)

    updater = training.StandardUpdater(
        train_iter, optimizer, device=None, converter=convert)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'))
    trainer.extend(
        extensions.Evaluator(test_iter, model, converter=convert, device=None),
        trigger=(args.epocheval, 'epoch'))
    trainer.extend(extensions.LogReport())

    trainer.extend(extensions.MicroAverage(
        'main/correct', 'main/total', 'main/accuracy'))
    trainer.extend(extensions.MicroAverage(
        'validation/main/correct', 'validation/main/total',
        'validation/main/accuracy'))

    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    trainer.run()


if __name__ == '__main__':
    main()
