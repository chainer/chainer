#!/usr/bin/env python
from __future__ import print_function
import argparse

import chainer
import chainer.links as L

import train_mnist

import copy

import numpy
import cupy
from cupy.cuda import device
from cupy.cuda import nccl
from cupy.cuda import profiler
from cupy.cuda import stream

from multiprocessing import Process
from multiprocessing import Pipe


def main():
    # This script is almost identical to train_mnist.py. The only difference is
    # that this script uses data-parallel computation on two GPUs.
    # See train_mnist.py for more details.
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=400,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu0', '-g', type=int, default=0,
                        help='First GPU ID')
    parser.add_argument('--gpu1', '-G', type=int, default=1,
                        help='Second GPU ID')
    parser.add_argument('--out', '-o', default='result_parallel',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()

    print('GPU: {}, {}'.format(args.gpu0, args.gpu1))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    model = L.Classifier(train_mnist.MLP(args.unit, 10))

    train, test = chainer.datasets.get_mnist()

    epoch = args.epoch
    batchsize = args.batchsize
    datasize = len(train)
    proc_num = 2

    conns = []
    procs = []

    # start slave process(es)
    for i in range(1, proc_num):
        conn, c_conn = Pipe()
        proc_id = i
        gpu_id = i
        proc = Process(target=run_training,
                       args=((c_conn,), proc_id, proc_num, gpu_id, model,
                             epoch, datasize, batchsize, train, test))
        proc.start()
        conns.append(conn)
        procs.append(proc)

    # start master process
    proc_id = 0
    gpu_id = 0
    run_training(conns, proc_id, proc_num, gpu_id, model,
                 epoch, datasize, batchsize, train, test)

    for proc in procs:
        proc.join()


def run_training(conns, proc_id, proc_num, gpuid, model_ref,
                 epoch, datasize, batchsize, train, test):

    with device.Device(gpuid):
        my_dev = chainer.cuda.get_device(gpuid)

        # NCCL: create communicator
        if proc_id == 0:
            # master process
            commId = nccl.NcclCommunicatorId()
            for conn in conns:
                conn.send(commId)
        else:
            # slave process(es)
            commId = conns[0].recv()
        comm = nccl.NcclCommunicator(proc_num, commId, proc_id)
        st = stream.Stream()

        if proc_id == 0:
            model = model_ref
        else:
            model = copy.deepcopy(model_ref)
        model.to_gpu()

        optimizer = chainer.optimizers.Adam()
        optimizer.setup(model)

        for epoch in range(epoch):

            if proc_id == 0:
                # master process
                print('epoch %d' % epoch)
                indexes = numpy.random.permutation(datasize)
                for conn in conns:
                    conn.send(indexes)
            else:
                # slave process(es)
                indexes = conns[0].recv()

            for i in range(0, datasize, batchsize):

                x_batch = train[indexes[i:i+batchsize]][0]
                y_batch = train[indexes[i:i+batchsize]][1]

                x = chainer.Variable(x_batch[proc_id::proc_num])
                y = chainer.Variable(y_batch[proc_id::proc_num])
                x.to_gpu()
                y.to_gpu()

                loss = model(x, y)

                model.cleargrads()
                loss.backward()

                gg = model.gather_grads()
                my_dev.synchronize()

                # NCCL: allreduce
                comm.allReduce(gg.data.ptr, gg.data.ptr, gg.size,
                               nccl.NCCL_FLOAT, nccl.NCCL_SUM, st.ptr)
                st.synchronize()

                model.scatter_grads(gg)

                optimizer.update()

        # NCCL: destroy communicator
        comm.destroy()
        profiler.stop()


if __name__ == '__main__':
    main()
