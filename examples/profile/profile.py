import argparse

from chainer import cuda
from chainer import function_hooks
import numpy
import six


parser = argparse.ArgumentParser(description='ConvNet benchmark')
parser.add_argument('--model', '-m', type=str, default='alex',
                    choices=('alex', 'overfeat', 'vgg', 'conv1', 'conv2',
                             'conv3', 'conv4', 'conv5'),
                    help='network architecture (alex|overfeat|vgg|conv[1-5])')
parser.add_argument('--iteration', '-i', type=int, default=10,
                    help='iteration')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU to use. Negative value to use CPU')
parser.add_argument('--cudnn', '-c', action='store_true',
                    help='True if using cudnn')
parser.add_argument('--batchsize', '-b', type=int, default=None,
                    help='batchsize. If None, '
                    'batchsize is architecture-specific batchsize is used.')
args = parser.parse_args()


if args.model == 'alex':
    import alex
    model = alex.Alex(args.batchsize, args.cudnn)
elif args.model == 'overfeat':
    import overfeat
    model = overfeat.Overfeat(args.batchsize, args.cudnn)
elif args.model == 'vgg':
    import vgg
    model = vgg.VGG(args.batchsize, args.cudnn)
elif args.model.startswith('conv'):
    import conv
    number = args.model[4:]
    model = getattr(conv, 'Conv{}'.format(number))(args.batchsize, args.cudnn)
else:
    raise ValueError('Invalid model name')


print('Architecture\t{}'.format(args.model))
print('Iteration\t{}'.format(args.iteration))
print('cuDNN\t{}'.format(cuda.cudnn_enabled and model.use_cudnn))
print('batchsize\t{}'.format(model.batchsize))

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()


for iteration in six.moves.range(args.iteration):

    print('Iteratiron\t{}'.format(iteration + 1))
    x_batch = numpy.random.uniform(-1, 1,
                                   (model.batchsize,
                                    model.in_channels,
                                    model.insize,
                                    model.insize)).astype(numpy.float32)
    if args.gpu >= 0:
        x_batch = cuda.to_gpu(x_batch)

    with function_hooks.TimerHook() as t:
        y = model.forward(x_batch)
        print('forward total time\t{}\tms'.format(t.total_time()))

    if args.gpu >= 0:
        y.grad = cuda.ones_like(y.data)
    else:
        y.grad = numpy.ones_like(y.data)

    with function_hooks.TimerHook() as t:
        y.backward()
        print('backward total time\t{}\tms'.format(t.total_time()))
