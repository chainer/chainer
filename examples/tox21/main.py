#!/usr/bin/env python
import argparse

import chainer
from chainer import datasets as D
from chainer import functions as F
from chainer import iterators as I
from chainer import links as L
from chainer import optimizers as O
from chainer import training
from chainer.training import extensions as E

import model as model_


parser = argparse.ArgumentParser(
    description='Multitask Learning with Tox21.')
parser.add_argument('--batchsize', '-b', type=int, default=128)
parser.add_argument('--gpu', '-g', type=int, default=0)
parser.add_argument('--out', '-o', type=str, default='result')
parser.add_argument('--epoch', '-e', type=int, default=10)
parser.add_argument('--unit-num', '-u', type=int, default=512)
args = parser.parse_args()


train, val, _ = D.get_tox21()
train_iter = I.SerialIterator(train, args.batchsize)
val_iter = I.SerialIterator(val, args.batchsize, repeat=False, shuffle=False)

C = len(D.tox21.tox21_tasks)
model = model_.Model(args.unit_num, C)

classifier = L.Classifier(model,
                          lossfun=F.sigmoid_cross_entropy,
                          accfun=F.binary_accuracy)
if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()
    classifier.to_gpu()

optimizer = O.Adam()
optimizer.setup(classifier)

updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)


def eval_mode(evalator):
    model = evalator.get_target('main')
    model.train = False


trainer.extend(E.Evaluator(val_iter, classifier, device=args.gpu))
trainer.extend(E.snapshot(), trigger=(args.epoch, 'epoch'))
trainer.extend(E.LogReport())
trainer.extend(E.PrintReport(['epoch', 'main/loss', 'main/accuracy',
                              'validation/main/loss',
                              'validation/main/accuracy',
                              'elapsed_time']))

trainer.run()
