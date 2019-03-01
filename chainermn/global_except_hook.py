import os
import sys
import warnings


_orig_except_hook = None


def _global_except_hook(exctype, value, traceback):
    """Catches an unhandled exception and call MPI_Abort()."""
    try:
        if _orig_except_hook:
            _orig_except_hook(exctype, value, traceback)
        else:
            sys.__excepthook__(exctype, value, traceback)

    finally:
        import mpi4py.MPI
        rank = mpi4py.MPI.COMM_WORLD.Get_rank()
        sys.stderr.write('\n')
        sys.stderr.write('******************************************\n')
        sys.stderr.write('ChainerMN: \n')
        sys.stderr.write('   Uncaught exception on rank {}. \n'.format(rank))
        sys.stderr.write('   Calling MPI_Abort() to shut down MPI...\n')
        sys.stderr.write('******************************************\n')
        sys.stderr.write('\n\n')
        sys.stderr.flush()

        try:
            import mpi4py.MPI
            mpi4py.MPI.COMM_WORLD.Abort(1)
        except Exception as e:
            # Something is completely broken...
            # There's nothing we can do any more
            sys.stderr.write(
                'Sorry, failed to stop MPI and the process may hang.\n')
            sys.stderr.flush()
            raise e


def _add_hook_if_enabled():
    # An MPI runtime is expected to kill all of its child processes
    # if one of them exits abnormally or without calling `MPI_Finalize()`.
    # However, when a Python program run on `mpi4py`, the MPI runtime
    # often fails to detect a process failure, and the rest of the processes
    # hang infinitely.
    # It is problematic especially when you run ChainerMN programs on a cloud
    # environment, on which you are charged on time basis.
    # See https://github.com/chainer/chainermn/issues/236 for more discussion.
    #
    # To activate this handler, set CHAINERMN_FORCE_ABORT_ON_EXCEPTION
    # to a non-empty value.
    # Note that you need to pass an argument to mpiexec (-x for Open MPI)
    # to activate the handler in all processes.
    var = os.environ.get('CHAINERMN_FORCE_ABORT_ON_EXCEPTION')
    if var is not None and len(var) > 0:
        add_hook()


def add_hook():
    """Add a global hook function that captures all unhandled exceptions.

    The function calls MPI_Abort() to force all processes abort.
    It is useful when you run your training script on a cloud platform.
    """
    global _orig_except_hook

    if _orig_except_hook is not None:
        warnings.warn('chainermn.global_except_hook.add_hook()'
                      'seems to be called multiple times. Ignoring.',
                      stacklevel=2)
        return

    _orig_except_hook = sys.excepthook
    sys.excepthook = _global_except_hook
