import argparse

from chainer import cuda
import numpy

import alex

parser = argparse.ArgumentParser(
    description='Profiler')
parser.add_argument('--model', '-m', type=str, default='alex',
                    help='network architecture (alex, overfeat)')
parser.add_argument('--batchsize', '-b', type=int, default=128,
                    help='batchsize')
parser.add_argument('--iteration', '-i', type=int, default=10,
                    help='iteration')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='gpu to use')
args = parser.parse_args()


if args.model == 'alex':
    model = alex.Alex()
elif args.model == 'overfeat':
    raise NotImplementedError('Overfeat is not implemented yet')
else:
    raise ValueError('Invalid model name')

print('Architecture\t{}'.format(args.model))

if args.gpu >= 0:
    cuda.init(args.gpu)
    model.to_gpu()

insize = model.insize
in_channels = 3
outsize = 1000

for iteration in xrange(args.iteration):
    print('Iteration\t{}'.format(iteration))
    x_batch = numpy.random.uniform(-1, 1, (args.batchsize, in_channels, insize, insize)).astype(numpy.float32)
    y_batch = numpy.random.randint(outsize, size=(args.batchsize,)).astype(numpy.int32)
    if args.gpu >= 0:
        x_batch = cuda.to_gpu(x_batch)
        y_batch = cuda.to_gpu(y_batch)

    loss, accuracy = model.forward(x_batch, y_batch)
    loss.backward()


