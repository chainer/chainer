import numpy as np

import chainer
import train_mnist
import argparse
import onnx_chainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unit', '-u', type=int, default=1000)
    parser.add_argument('--snapshot', '-s', type=str, default='result/snapshot_iter_600')
    args = parser.parse_args()

    model = train_mnist.MLP(args.unit, 10)
    chainer.serializers.load_npz(
        args.snapshot, model, path='updater/model:main/predictor/')

    # Pseudo input
    x = np.zeros((1, 784), dtype=np.float32)

    onnx_chainer.export(model, x, filename='result/mnist.onnx')


if __name__ == '__main__':
    main()
