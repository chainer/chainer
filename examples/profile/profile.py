import argparse

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


parser = argparse.ArgumentParser(description='ConvNet benchmark')
parser.add_argument('--predictor', '-p', type=str, default='alex',
                    choices=('alex', 'overfeat', 'vgg', 'conv1', 'conv2',
                             'conv3', 'conv4', 'conv5'),
                    help='network architecture')
parser.add_argument('--seed', '-s', type=int, default=0,
                    help='random seed')
parser.add_argument('--trial', '-t', type=int, default=10,
                    help='the number of trials to be averaged over.')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU to use. Negative value to use CPU')
parser.add_argument('--cudnn', '-c', action='store_true',
                    help='True if using cudnn')
parser.add_argument('--cached-kernel', '-k', action='store_true',
                    help='True if we clear cache at '
                    'the beginning of every iteration. '
                    'Otherwise, only measurements whose kernels are not cached are taken '
                    'into account to calculate elapsed time.')
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
print('Trial\t{}'.format(args.trial))
print('Cache\t{}'.format(not args.cached_kernel))
print('cuDNN\t{}'.format(cuda.cudnn_enabled and model.predictor.use_cudnn))
print('batchsize\t{}'.format(model.predictor.batchsize))


def clear_cache():
    pass

if args.cached_kernel:
    start_iteration = -1
else:
    start_iteration = 0

forward_time = 0.0
backward_time = 0.0
update_time = 0.0

print('iteration\tforward\tbackward\tupdate (in seconds)')
for iteration in six.moves.range(start_iteration, args.trial):
    if not args.cached_kernel:
        clear_cache()

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
        print('Trial{}\t{}\t{}\t{}'.format(iteration, forward_time_one, backward_time_one, update_time_one))
        forward_time += forward_time_one
        backward_time += backward_time_one
        update_time += update_time_one

forward_time /= args.trial
backward_time /= args.trial
update_time /= args.trial


print('Mean\t{}\t{}\t{}'.format(forward_time, backward_time, update_time))

# dump result
if args.out_file is not None:
    with open(args.out_file, 'w') as o:
        o.write(','.join('{}.{}'.format(args.column_prefix, c)
                         for c in ['forward', 'backward', 'update']))
        o.write('\n')
        o.write(','.join(str(v) for v in
                         [forward_time, backward_time, update_time]))
