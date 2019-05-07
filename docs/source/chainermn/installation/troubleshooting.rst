.. -*- coding: utf-8; -*-

.. _troubleshooting:

Step-by-Step Troubleshooting
============================

This section is a step-by-step troubleshooting guide for ChainerMN.
Please follow these steps to identify and fix your problem.

We assume that you are using Linux or another Unix-like environment.

Single-node environment
-----------------------

Basic MPI installation
~~~~~~~~~~~~~~~~~~~~~~

Although ChainerMN stands for "Chainer MultiNode," it is good to start
from single-node execution. First of all, you need MPI. If MPI is
correctly installed, you will see the :command:`mpicc` and
:command:`mpiexec` commands in your PATH.

Below is an example of the output from Mvapich on Linux.::

    $ which mpicc
    /usr/local/bin/mpicc

    $ mpicc -show
    gcc -I/usr/local/include ...(snip)... -lmpi

    $ which mpiexec
    /usr/local/bin/mpiexec

    $ mpiexec --version
    HYDRA build details:
    Version:                                 3.1.4
    Release Date:                            Wed Sep  7 14:33:43 EDT 2016
    CC:                              gcc
    CXX:                             g++
    F77:
    F90:
    Configure options:  (snip)
    Process Manager:                         pmi
    Launchers available:                     ssh rsh fork slurm ll lsf sge manual persist
    Topology libraries available:            hwloc
    Resource management kernels available:   user slurm ll lsf sge pbs cobalt
    Checkpointing libraries available:
    Demux engines available:                 poll select

If you see any error in above commands, please go back to the
:ref:`mpi-install` and check your MPI installation.

.. _check-what-mpi:
     
Check what MPI you are using
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In :ref:`mpi-install`, we mention both of `Open MPI` and
`Mvapich`. If the MPI is provided by the system administrator and
you are not really sure which MPI you are using, check the output of
`mpiexec --version`.

- If the output contains `HYDRA`, then it's MVAPICH (or possibly MPICH).
- If the output contains `OpenRTE`, then it's Open MPI.

However, in such a case, you should make sure that the MPI is
`CUDA-aware`, as mentioned below.  We recommend to build your own MPI.


Check if MPI is CUDA-aware
~~~~~~~~~~~~~~~~~~~~~~~~~~

Your MPI must be configured as *CUDA-aware*. You can use the following
C program to check it.

.. code-block:: c

  /* check_cuda_aware.c */
  #include <assert.h>
  #include <stdio.h>
  #include <mpi.h>
  #include <cuda_runtime.h>

  #define CUDA_CALL(expr) do {                  \
    cudaError_t err;                            \
    err = expr;                                 \
    assert(err == cudaSuccess);                 \
  } while(0)

  int main(int argc, char **argv) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int *sendbuf_d = NULL;
    int *recvbuf_d = NULL;

    CUDA_CALL(cudaMalloc((void**)&sendbuf_d, sizeof(int)));
    CUDA_CALL(cudaMalloc((void**)&recvbuf_d, sizeof(int)));
    CUDA_CALL(cudaMemcpy(sendbuf_d, &rank, sizeof(int), cudaMemcpyDefault));

    MPI_Reduce(sendbuf_d, recvbuf_d, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
      int sum = -1;
      CUDA_CALL(cudaMemcpy(&sum, recvbuf_d, sizeof(int), cudaMemcpyDefault));
      if (sum == (size-1) * size / 2) {
        printf("OK.\n");
      } else {
        printf("Error.\n");
      }
    }

    cudaFree(sendbuf_d);
    cudaFree(recvbuf_d);

    MPI_Finalize();
  }

Save the code to a file named ``check_cuda_aware.c``. You can compile
and run it with the following command.::

    $ export MPICH_CC=nvcc  # if you use Mvapich
    $ export OMPI_CC=nvcc   # if you use Open MPI
    $ $(mpicc -show check_cuda_aware.c -arch sm_53 | sed -e 's/-Wl,/-Xlinker /g' | sed -e 's/-pthread/-Xcompiler -pthread/')
    $ ./a.out
    OK.

If the proglam prints `OK.`, your MPI is correctly configured.

Check mpi4py
~~~~~~~~~~~~

Next, let's check that mpi4py is correctly installed. You can use the following script to check it::

  # coding: utf-8
  import os
  from mpi4py import MPI

  comm = MPI.COMM_WORLD
  size = comm.Get_size()
  rank = comm.Get_rank()

  for i in range(size):
    if i == rank:
      print("{} {}".format(os.uname()[1], i))
    comm.Barrier()

Save the script into a file named :file:`check_mpi4py.py` and run it.
The output from the script should look like this.::

  $ mpiexec -np 4 python check_mpi4py.py
  host00 0
  host00 1
  host00 2
  host00 3

The script prints hostnames and ranks (process id in MPI) from
each MPI process in a sequential manner.
`host00` is the host name of the machine your are running the process.
If you get an output like below, it indicates something is wrong with
your installation.::

  # Wrong output !
  $ mpiexec -n 4 python check_mpi4py.py
  host00 0
  host00 0
  host00 0
  host00 0

A common problem is that the :command:`mpicc` used to build
:mod:`mpi4py` and :command:`mpiexec` used to run the script are from
different MPI installations.

Finally, run :command:`pytest` to check the single-node
configuration is ready.::

  $ git clone git@github.com:chainer/chainer.git
  Cloning into 'chainer'...
  remote: Enumerating objects: 7, done.
  remote: Counting objects: 100% (7/7), done.
  remote: Compressing objects: 100% (7/7), done.
  remote: Total 168242 (delta 1), reused 2 (delta 0), pack-reused 168235
  Receiving objects: 100% (168242/168242), 41.15 MiB | 1.65 MiB/s, done.
  Resolving deltas: 100% (123696/123696), done.
  Checking connectivity... done.
  $ cd chainer/
  $ pytest tests/chainermn_tests/
  ......S.S...S.S...S.S...S.S.........SS
  ----------------------------------------------------------------------
  Ran 38 tests in 63.083s

  OK (SKIP=10)

.. _check-nccl:

Check if NCCL is enabled in CuPy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CuPy requires NCCL to be enabled. You can check it with the following command.::
  
  $ python -c 'from cupy.cuda import nccl'

If you get an output like below, NCCL is not enabled in CuPy.
Please check the installation guide of CuPy.::

  Traceback (most recent call last):

   File "<string>", line 1, in <module>

  ImportError: cannot import name 'nccl'


Multi-node environment
-----------------------

.. _ssh-and-envvars:

Check SSH connection and environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use ChainerMN on multiple hosts, you need to connect to computing hosts,
including the one you are currently logged into, via ssh without
password authentication (and preferably without username).::

  $ ssh host00 'hostname'
  host00   # without hitting the password

  $ ssh host01 'hostname'
  host01   # without hitting the password

  ...

You may get a message like this::

  The authenticity of host 'host01 (xxx.xxx.xxx.xxx)' can't be established.
  ECDSA key fingerprint is SHA256:haGUMcCeC5A8lGh1lpjpwL5dF4xCglZArhhxxxxxxxxx.
  Are you sure you want to continue connecting (yes/no)?

This message appears when you log in a host for the first time. Just
type `yes` and the message won't appear again. You need to repeat this
process on all computing hosts.

Also, you need to pay attention to the environment variables on remote
hosts.  The MPI runtime connects to the remote hosts in *non-interactive*
mode, and environment variables may differ from your interactive login
sessions.::

  $ ssh host00 'env' | grep LD_LIBRARY_PATH
  # Check the values and compare it to the local value.

  $ ssh host01 'env' | grep LD_LIBRARY_PATH
  # Check the values and compare it to the local value.

  ...

In particular, check the following variables, which are critical to
executing MPI programs:

    * :envvar:`PATH`
    * :envvar:`LD_LIBRARY_PATH`
    * :envvar:`MV2_USE_CUDA` (if you use MVAPICH)
    * :envvar:`MV2_SMP_USE_CMA` (if you use MVAPICH)

Besides, you need to make sure the same :command:`mpiexec` binary is
used to run MPI programs.::

  $ ssh host00 'which mpiexec'
  /usr/local/bin/mpiexec
  
  $ ssh host01 'which mpiexec'
  /usr/local/bin/mpiexec

All the commands should give the same :command:`mpiexec` binary path.
  
Program files and data
~~~~~~~~~~~~~~~~~~~~~~

When you run MPI programs, all hosts must have the same Python binary
and script files in the same path. First, check that the python binary and
version are identical among hosts. Be careful if you are using `pyenv`
or `Anaconda`.::

  $ ssh host00 'which python; python --version'
  /home/username/.pyenv/shims/python
  Python 3.6.0 :: Anaconda 4.3.1 (64-bit)

  $ ssh host01 'which python'
  /home/username/.pyenv/shims/python
  Python 3.6.0 :: Anaconda 4.3.1 (64-bit)

  ...

Also, the script file (and possibly data files) must be in the same
path on each host. ::

  $ ls yourscript.py  # in the current directory
  yourscript.py

  $ ssh host00 "ls $PWD/yourscript.py"
  /home/username/your/dir/yourscript.py

  $ ssh host01 "ls $PWD/yourscript.py"
  /home/username/your/dir/yourscript.py

  ...

If you are using NFS, everything should be okay. If not, you need
to transfer all the necessary files manually.

In particular, when you run the ImageNet example in ChainerMN
repository, all data files must be available on all computing hosts.

hostfile
~~~~~~~~~~~~~~~~~~~~~~

The next step is to create a hostfile. A hostfile is a list of hosts on
which MPI processes run.::

  $ vi hostfile
  $ cat hostfile
  host00
  host01
  host02
  host03

Then, you can run your MPI program using the hostfile.  To check if
the MPI processes run over multiple hosts, save the following script
to a file and run it via :command:`mpiexec`::

  # print_rank.py
  import os

  from mpi4py import MPI

  comm = MPI.COMM_WORLD
  size = comm.Get_size()
  rank = comm.Get_rank()

  for i in range(size):
    if i == rank:
      print("{} {}".format(os.uname()[1], i))
    comm.Barrier()

If you get an output like below, it is working correctly.::

  $ mpiexec -n 4 --hostfile hostfile python print_rank.py
  host00 0
  host01 1
  host02 2
  host03 3

If you have multiple GPUs, you may want to run multiple processes on
each host.  You can modify hostfile and specify the number of
processes to run on each host.::

  # If you are using Mvapich:
  $ cat hostfile
  host00:4
  host01:4
  host02:4
  host03:4

  # If you are using Open MPI
  $ cat hostfile
  host00 cpu=4
  host01 cpu=4
  host02 cpu=4
  host03 cpu=4

With this hostfile, try running mpiexec again.::

  $ mpiexec -n 8 --hostfile hostfile python print_rank.py
  host00 0
  host00 1
  host00 2
  host00 3
  host01 4
  host01 5
  host01 6
  host01 7

You will find that the first 4 processes run on host00 and the latter
4 on host01.

You can also specify computing hosts and resource mapping/binding
using command line options of mpiexec. Please refer to the MPI manual
for the more advanced use of mpiexec command.

.. _troubleshooting_errors:

If you get runtime error:
~~~~~~~~~~~~~~~~~~~~~~~~~

If you get the following error messages, please check the specified
section of the troubleshooting or installation guide.

::
   
   [hostxxx:mpi_rank_0][MPIDI_CH3I_SMP_init] CMA is not available. Set MV2_SMP_USE_CMA=0 to disable CMA.
   [cli_0]: aborting job:
   Fatal error in PMPI_Init_thread:
   Other MPI error, error stack:
   MPIR_Init_thread(514)....:
   MPID_Init(365)...........: channel initialization failed
   MPIDI_CH3_Init(404)......:
   MPIDI_CH3I_SMP_Init(2132): process_vm_readv: Operation not permitted


   ===================================================================================
   =   BAD TERMINATION OF ONE OF YOUR APPLICATION PROCESSES
   =   PID 20327 RUNNING AT hostxxx
   =   EXIT CODE: 1
   =   CLEANING UP REMAINING PROCESSES
   =   YOU CAN IGNORE THE BELOW CLEANUP MESSAGES
   ===================================================================================

-> Check the value of :envvar:`MV2_SMP_USE_CMA` (see :ref:`mpi-install` and
:ref:`ssh-and-envvars`).


::

   [hostxx:mpi_rank_0][error_sighandler] Caught error: Segmentation fault (signal 11)

   ===================================================================================
   =   BAD TERMINATION OF ONE OF YOUR APPLICATION PROCESSES
   =   PID 20643 RUNNING AT hostxx
   =   EXIT CODE: 11
   =   CLEANING UP REMAINING PROCESSES
   =   YOU CAN IGNORE THE BELOW CLEANUP MESSAGES
   ===================================================================================
   YOUR APPLICATION TERMINATED WITH THE EXIT STRING: Segmentation fault (signal 11)
   This typically refers to a problem with your application.
   Please see the FAQ page for debugging suggestions

-> Check the value of :envvar:`MV2_USE_CUDA` (see :ref:`mpi-install` and :ref:`ssh-and-envvars`)


