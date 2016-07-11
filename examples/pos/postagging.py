import collections
import itertools

from nltk.corpus import brown
import numpy

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
import chainer.optimizers as O


class CRF(chainer.Chain):

    def __init__(self, n_vocab, n_pos):
        super(CRF, self).__init__(
            embed=L.EmbedID(n_vocab, n_pos),
            crf=L.CRF1d(n_pos),
        )

    def __call__(self, xs, ys):
        hs = [self.embed(x) for x in xs]
        return self.crf(hs, ys)

    def argmax(self, xs):
        hs = [self.embed(x) for x in xs]
        return self.crf.argmax(hs)


vocab = collections.defaultdict(lambda: len(vocab))
pos_vocab = collections.defaultdict(lambda: len(pos_vocab))
data = []
for sentence in brown.tagged_sents():
    s = [(vocab[lex], pos_vocab[pos]) for lex, pos in sentence]
    data.append(s)
    if len(data) >= 1000:
        break

print('# of sentences: {}'.format(len(data)))
print('# of words: {}'.format(len(vocab)))
print('# of pos: {}'.format(len(pos_vocab)))

data.sort(key=len)
groups = []
for length, group in itertools.groupby(data, key=len):
    groups.append(list(group))


model = CRF(len(vocab), len(pos_vocab))
model.to_gpu()
xp = cuda.cupy
opt = O.Adam()
opt.setup(model)
#opt.add_hook(chainer.optimizer.WeightDecay(0.1))

n_epoch = 1000

for epoch in range(n_epoch):
    accum_loss = 0
    correct = 0
    total = 0

    for sentences in groups:
        length = len(sentences[0])
        xs = []
        ys = []
        for i in range(length):
            x_data = xp.array([s[i][0] for s in sentences], numpy.int32)
            y_data = xp.array([s[i][1] for s in sentences], numpy.int32)
            xs.append(chainer.Variable(x_data))
            ys.append(chainer.Variable(y_data))
        loss = model(xs, ys)
        accum_loss += loss.data
        model.zerograds()
        loss.backward()
        opt.update()

        _, path = model.argmax(xs)
        assert len(ys) == len(path)

        for y, p in zip(ys, path):
            correct += xp.sum(y.data == p)
            total += len(y.data)

    accuracy = float(correct) / total
    print('Accuracy: {}'.format(accuracy))
    print('Total loss: {}'.format(accum_loss))
