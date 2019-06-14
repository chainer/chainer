import os
try:
    import queue
except ImportError:
    import Queue as queue
import re
import signal
from subprocess import CalledProcessError
from subprocess import PIPE
from subprocess import Popen
import sys
import threading
import unittest

import numpy as np
import pytest

import chainer.testing
import chainer.testing.attr
import chainermn
from chainermn.communicators import _memory_utility


class _TimeoutThread(threading.Thread):
    def __init__(self, queue, rank):
        super(_TimeoutThread, self).__init__()
        self.queue = queue
        self.rank = rank

    def run(self):
        try:
            self.queue.get(timeout=60)
        except queue.Empty:
            # Show error message and information of the problem
            try:
                p = Popen(['ompi_info', '--all', '--parsable'], stdout=PIPE)
                out, err = p.communicate()
                if type(out) == bytes:
                    out = out.decode('utf-8')
                m = re.search(r'ompi:version:full:(\S+)', out)
                version = m.group(1)

                msg = "\n\n***** ERROR: " \
                      "It looks like you are using " \
                      "Open MPI version {}.\n" \
                      "***** It is known that the following Open MPI " \
                      "versions have a bug \n" \
                      "***** that cause MPI_Bcast() deadlock " \
                      "when GPUDirect is used: \n" \
                      "***** 3.0.0, 3.0.1, 3.0.2, 3.1.0, 3.1.1, 3.1.2\n"
                if self.rank == 1:
                    # Rank 1 prints the error message,
                    # Rank 0 may finish Bcast() immediately without deadlock,
                    # depending on the timing,
                    # because rank 0 is the root of Bcast().
                    print(msg.format(version))
                    sys.stdout.flush()

                os.kill(os.getpid(), signal.SIGKILL)
            except CalledProcessError:
                pass


class TestBcastDeadlock(unittest.TestCase):
    def setup(self, gpu):
        if gpu:
            self.communicator = chainermn.create_communicator('flat')
            self.device = self.communicator.intra_rank
            chainer.cuda.get_device_from_id(self.device).use()
        else:
            self.device = -1

        if self.communicator.size < 2:
            pytest.skip('This test is for at least two processes')

        self.queue = queue.Queue(maxsize=1)

    def teardown(self):
        pass

    @chainer.testing.attr.gpu
    def test_bcast_gpu_large_buffer_deadlock(self):
        """Regression test of Open MPI's issue #3972"""
        self.setup(True)
        buf_size = 10000
        mpi_comm = self.communicator.mpi_comm

        if self.communicator.rank == 0:
            array = np.arange(buf_size, dtype=np.float32)
        else:
            array = np.empty(buf_size, dtype=np.float32)

        array = chainer.cuda.to_gpu(array, device=self.device)

        ptr = _memory_utility.array_to_buffer_object(array)

        # This Bcast() cause deadlock if the underlying MPI has the bug.
        th = _TimeoutThread(self.queue, self.communicator.rank)
        th.start()
        mpi_comm.Bcast(ptr, root=0)
        mpi_comm.barrier()
        self.queue.put(True)
        assert True

        self.teardown()
