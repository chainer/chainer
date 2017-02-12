#!/usr/bin/env python
import argparse

import chainer
from chainer import extensions as E
from chainer import iterators as I
from chainer import optimizers as O
from chainer import training
import numpy

import acc
import data
import loss
import model as model_


parser = argparse.ArgumentParser(
    description='Multitask Learning with Tox21.')
parser.add_argument('--batchsize', '-b', type=int, default=128)
parser.add_argument('--gpu', '-g', type=int, default=0)
parser.add_argument('--out', '-o', type=str, default='result')
parser.add_argument('--epoch', '-e', type=int, default=10)
args = parser.parse_args()


train, test, val = data.get_tox21()
train_iter = I.SerialIterator(train, batchsize)
test_iter = I.SerialIterator(test, batchsize, repeat=False, shuffle=False)
val_iter = I.SerialIterator(val, batchsize, repeat=False, shuffle=False)

model = model_.Model()
classifier = L.Classifier(model,
                          lossfun=loss.multitask_sce,
                          accfun=acc.multitask_acc)
if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()
    classifier.to_gpu()

optimizer = O.Adam()
optimizer.setup(classifier)

updater = training.StandardUpdater(train_iter, optimizer, device=gpu)
trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

trainer.extend(E.snapshot(), trigger=(args.epoch, 'epoch'))
trainer.extend(E.LogReport())
trainer.extend(E.PrintReport())
trainer.extend(E.Evaluator(test_iter, classifier), out=args.out)

trainer.run()
