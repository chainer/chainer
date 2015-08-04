import argparse

from chainer import cuda
import numpy


parser = argparse.ArgumentParser(
    description='Profiler')
parser.add_argument('--model', '-m', type=str, default='alex',
                    help='network architecture (alex|overfeat|vgg)')
parser.add_argument('--iteration', '-i', type=int, default=10,
                    help='iteration')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='gpu to use')
args = parser.parse_args()


if args.model == 'alex':
    import alex
    model = alex.Alex()
elif args.model == 'overfeat':
    raise NotImplementedError('Overfeat is not implemented yet')
else:
    raise ValueError('Invalid model name')

print('Architecture\t{}'.format(args.model))

if args.gpu >= 0:
    cuda.init(args.gpu)
    model.to_gpu()


for iteration in xrange(args.iteration):
    print('Iteration\t{}'.format(iteration))
    x_batch = numpy.random.uniform(-1, 1, (model.batchsize, model.in_channels, model.insize, model.insize)).astype(numpy.float32)
    y_batch = numpy.random.randint(model.outsize, size=(model.batchsize,)).astype(numpy.int32)
    if args.gpu >= 0:
        x_batch = cuda.to_gpu(x_batch)
        y_batch = cuda.to_gpu(y_batch)

    loss = model.forward(x_batch, y_batch)
    loss.backward()


