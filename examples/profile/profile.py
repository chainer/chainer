import argparse
import os
import shutil

import chainer
from chainer import cuda
from chainer import function_hooks
from chainer import links
from chainer import optimizers
import numpy
import six

import alex
import conv
import overfeat
import vgg


parser = argparse.ArgumentParser(description='Measurement of mean elapsed time of forward, '
                                 'backward and parameter update.')
parser.add_argument('--predictor', '-p', type=str, default='alex',
                    choices=('alex', 'overfeat', 'vgg', 'conv1', 'conv2',
                             'conv3', 'conv4', 'conv5'),
                    help='network architecture')
parser.add_argument('--seed', '-s', type=int, default=0,
                    help='random seed')
parser.add_argument('--iteration', '-i', type=int, default=10,
                    help='the number of iterations to be averaged over.')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU to use. Negative value to use CPU')
parser.add_argument('--cudnn', '-c', action='store_true',
                    help='True if using cudnn')
parser.add_argument('--cache-level', '-C', type=str, default=None,
                    choices=(None, 'memory', 'disk'),
                    help='This option determines the type of the kernel cache used.'
                    'By default, memory cache and disk cache are removed '
                    'at the beginning of every iteration. '
                    'Otherwise, elapsed times of each iteration are '
                    'measured with corresponding cache enabled. '
                    'If either cache is enabled, this script operates one additional '
                    'iteration for burn-in before measurement. '
                    'This iteration is not included in the mean elapsed time.')
parser.add_argument('--batchsize', '-b', type=int, default=None,
                    help='batchsize. If None, '
                    'batchsize is architecture-specific batchsize is used.')
parser.add_argument('--out-file', '-o', type=str, default=None,
                    help='path to output file')
parser.add_argument('--column-prefix', '-P', type=str, default='',
                    help='prefix of the name of columns of output matrix')
args = parser.parse_args()

numpy.random.seed(args.seed)
if args.gpu >= 0:
    cuda.cupy.random.seed(args.seed)


# model setup
if args.predictor == 'alex':
    predictor = alex.Alex(args.batchsize, args.cudnn)
elif args.predictor == 'overfeat':
    predictor = overfeat.Overfeat(args.batchsize, args.cudnn)
elif args.predictor == 'vgg':
    predictor = vgg.VGG(args.batchsize, args.cudnn)
elif args.predictor.startswith('conv'):
    number = args.predictor[4:]
    predictor = getattr(conv, 'Conv{}'.format(number))(args.batchsize, args.cudnn)
else:
    raise ValueError('Invalid predictor name')

model = links.Classifier(predictor)

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
xp = cuda.cupy if args.gpu >= 0 else numpy

optimizer = optimizers.SGD()
optimizer.setup(model)


print('Architecture\t{}'.format(args.predictor))
print('Iteration\t{}'.format(args.iteration))
print('Cache\t{}'.format(args.cache_level))
print('cuDNN\t{}'.format(cuda.cudnn_enabled and model.predictor.use_cudnn))
print('Batchsize\t{}'.format(model.predictor.batchsize))


def clear_cache(cache_level):
    if args.gpu < 0:
        return
    if cache_level is None:
        cache_dir = cuda.cupy.cuda.get_cache_dir()
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
    if cache_level is None or cache_level == 'disk':
        cuda.cupy.clear_memo()


if args.cache_level is None:
    start_iteration = 0
else:
    start_iteration = -1

forward_time = 0.0
backward_time = 0.0
update_time = 0.0

print('iteration\tforward\tbackward\tupdate (in seconds)')
for iteration in six.moves.range(start_iteration, args.iteration):
    clear_cache(args.cache_level)

    # data generation
    data = numpy.random.uniform(-1, 1,
                                (model.predictor.batchsize,
                                 model.predictor.in_channels,
                                 model.predictor.insize,
                                 model.predictor.insize)).astype(numpy.float32)
    data = chainer.Variable(xp.asarray(data))
    label = numpy.empty((args.batchsize,), dtype=numpy.int32)
    label.fill(0)
    label = chainer.Variable(xp.asarray(label))

    # forward
    with function_hooks.TimerHook() as h:
        loss = model(data, label)
    forward_time_one = h.total_time()

    # backward
    with function_hooks.TimerHook() as h:
        loss.backward()
    backward_time_one = h.total_time()

    # parameter update
    with function_hooks.TimerHook() as h:
        optimizer.update()
    update_time_one = h.pass_through_time

    if iteration < 0:
        print('Burn-in\t{}\t{}\t{}'.format(forward_time_one, backward_time_one, update_time_one))
    else:
        print('{}\t{}\t{}\t{}'.format(iteration, forward_time_one, backward_time_one, update_time_one))
        forward_time += forward_time_one
        backward_time += backward_time_one
        update_time += update_time_one

forward_time /= args.iteration
backward_time /= args.iteration
update_time /= args.iteration


print('Mean\t{}\t{}\t{}'.format(forward_time, backward_time, update_time))

# dump result
if args.out_file is not None:
    with open(args.out_file, 'w') as o:
        o.write(','.join('{}.{}'.format(args.column_prefix, c)
                         for c in ['forward', 'backward', 'update']))
        o.write('\n')
        o.write(','.join(str(v) for v in
                         [forward_time, backward_time, update_time]))
