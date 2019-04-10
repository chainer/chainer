import argparse

import chainer
import chainerx
import librosa
import numpy
import tqdm

from net import UpsampleNet
from net import WaveNet
from utils import MuLaw
from utils import Preprocess

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', required=True, help='input file')
parser.add_argument('--output', '-o', default='result.wav', help='output file')
parser.add_argument('--model', '-m', required=True,
                    help='snapshot of trained model')
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
parser.add_argument('--device', '-d', type=str, default='-1',
                    help='Device specifier. Either ChainerX device '
                    'specifier or an integer. If non-negative integer, '
                    'CuPy arrays with specified device id are used. If '
                    'negative integer, NumPy arrays are used')
group = parser.add_argument_group('deprecated arguments')
group.add_argument('--gpu', '-g', dest='device',
                   type=int, nargs='?', const=0,
                   help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

device = chainer.get_device(args.device)
device.use()

if device.xp is chainer.backends.cuda.cupy:
    chainer.global_config.autotune = True

# Preprocess
_, condition, _ = Preprocess(
    sr=16000, n_fft=1024, hop_length=256, n_mels=128, top_db=20,
    length=None, quantize=args.a_channels)(args.input)
x = numpy.zeros([1, args.a_channels, 1, 1], dtype=condition.dtype)
condition = numpy.expand_dims(condition, axis=0)

# Define networks
encoder = UpsampleNet(args.n_loop * args.n_layer, args.r_channels)
decoder = WaveNet(
    args.n_loop, args.n_layer,
    args.a_channels, args.r_channels, args.s_channels,
    args.use_embed_tanh)

# Load trained parameters
chainer.serializers.load_npz(
    args.model, encoder, 'updater/model:main/predictor/encoder/')
chainer.serializers.load_npz(
    args.model, decoder, 'updater/model:main/predictor/decoder/')

# Non-autoregressive generate
x = device.send(x)
condition = device.send(condition)
encoder.to_device(device)
decoder.to_device(device)
x = chainer.Variable(x)
condition = chainer.Variable(condition)
conditions = encoder(condition)
decoder.initialize(1)
output = decoder.xp.zeros(conditions.shape[3])


# A workaround for ChainerX, which does not have random.choice.
# TODO(niboshi): Implement it in ChainerX
def random_choice(device, a, size, p):
    if device.xp is chainerx:
        return device.send(
            numpy.random.choice(a, size=size, p=chainerx.to_numpy(p)))
    return device.xp.random.choice(a, size=size, p=p)


# Autoregressive generate
for i in tqdm.tqdm(range(len(output))):
    with chainer.no_backprop_mode():
        out = decoder.generate(x, conditions[:, :, :, i:i + 1]).array
    value = random_choice(
        device,
        args.a_channels, size=1,
        p=chainer.functions.softmax(out).array[0, :, 0, 0])[0]
    zeros = decoder.xp.zeros_like(x.array)
    zeros[:, value, :, :] = 1
    x = chainer.Variable(zeros)
    output[i] = value

# Save
output = chainer.get_device('@numpy').send(output)
wave = MuLaw(args.a_channels).itransform(output)
librosa.output.write_wav(args.output, wave, 16000)
