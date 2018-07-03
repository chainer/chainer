import argparse
import pathlib

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass
import chainer
from chainer.training import extensions

from net import EncoderDecoderModel
from net import UpsampleNet
from net import WaveNet
from utils import Preprocess


parser = argparse.ArgumentParser(description='Chainer example: WaveNet')
parser.add_argument('--batchsize', '-b', type=int, default=4,
                    help='Numer of audio clips in each mini-batch')
parser.add_argument('--length', '-l', type=int, default=7680,
                    help='Number of samples in each audio clip')
parser.add_argument('--epoch', '-e', type=int, default=100,
                    help='Number of sweeps over the dataset to train')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--dataset', '-i', default='./VCTK-Corpus',
                    help='Directory of dataset')
parser.add_argument('--out', '-o', default='result',
                    help='Directory to output the result')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the training from snapshot')
parser.add_argument('--n_loop', type=int, default=4,
                    help='Number of residual blocks')
parser.add_argument('--n_layer', type=int, default=10,
                    help='Number of layers in each residual block')
parser.add_argument('--a_channels', type=int, default=256,
                    help='Number of channels in the output layers')
parser.add_argument('--r_channels', type=int, default=64,
                    help='Number of channels in residual layers and embedding')
parser.add_argument('--s_channels', type=int, default=256,
                    help='Number of channels in the skip layers')
parser.add_argument('--use_embed_tanh', type=bool, default=True,
                    help='Use tanh after an initial 2x1 convolution')
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed to split dataset into train and test')
parser.add_argument('--snapshot_interval', type=int, default=10000,
                    help='Interval of snapshot')
parser.add_argument('--display_interval', type=int, default=100,
                    help='Interval of displaying log to console')
parser.add_argument('--process', type=int, default=1,
                    help='Number of parallel processes')
parser.add_argument('--prefetch', type=int, default=8,
                    help='Number of prefetch samples')
args = parser.parse_args()

print('GPU: {}'.format(args.gpu))
print('# Minibatch-size: {}'.format(args.batchsize))
print('# epoch: {}'.format(args.epoch))
print('')

if args.gpu >= 0:
    chainer.global_config.autotune = True

# Datasets
paths = sorted([
    str(path) for path in pathlib.Path(args.dataset).glob('wav48/*/*.wav')])
preprocess = Preprocess(
    sr=16000, n_fft=1024, hop_length=256, n_mels=128, top_db=20,
    length=args.length, quantize=args.a_channels)
dataset = chainer.datasets.TransformDataset(paths, preprocess)
train, valid = chainer.datasets.split_dataset_random(
    dataset, int(len(dataset) * 0.9), args.seed)

# Networks
encoder = UpsampleNet(args.n_loop * args.n_layer, args.r_channels)
decoder = WaveNet(
    args.n_loop, args.n_layer,
    args.a_channels, args.r_channels, args.s_channels,
    args.use_embed_tanh)
model = chainer.links.Classifier(EncoderDecoderModel(encoder, decoder))

# Optimizer
optimizer = chainer.optimizers.Adam(1e-4)
optimizer.setup(model)

# Iterators
train_iter = chainer.iterators.MultiprocessIterator(
    train, args.batchsize,
    n_processes=args.process, n_prefetch=args.prefetch)
valid_iter = chainer.iterators.MultiprocessIterator(
    valid, args.batchsize, repeat=False, shuffle=False,
    n_processes=args.process, n_prefetch=args.prefetch)

# Updater and Trainer
updater = chainer.training.StandardUpdater(
    train_iter, optimizer, device=args.gpu)
trainer = chainer.training.Trainer(
    updater, (args.epoch, 'epoch'), out=args.out)

# Extensions
snapshot_interval = (args.snapshot_interval, 'iteration')
display_interval = (args.display_interval, 'iteration')
trainer.extend(extensions.Evaluator(valid_iter, model, device=args.gpu))
trainer.extend(extensions.dump_graph('main/loss'))
trainer.extend(extensions.snapshot(), trigger=snapshot_interval)
trainer.extend(extensions.LogReport(trigger=display_interval))
trainer.extend(extensions.PrintReport(
    ['epoch', 'iteration', 'main/loss', 'main/accuracy',
     'validation/main/loss', 'validation/main/accuracy']),
    trigger=display_interval)
trainer.extend(extensions.PlotReport(
    ['main/loss', 'validation/main/loss'],
    'iteration', file_name='loss.png', trigger=display_interval))
trainer.extend(extensions.PlotReport(
    ['main/accuracy', 'validation/main/accuracy'],
    'iteration', file_name='accuracy.png', trigger=display_interval))
trainer.extend(extensions.ProgressBar(update_interval=10))

# Resume
if args.resume:
    chainer.serializers.load_npz(args.resume, trainer)

# Run
trainer.run()
