try:
    from cupy.cuda.nccl import get_build_version  # NOQA
    from cupy.cuda.nccl import get_unique_id  # NOQA
    from cupy.cuda.nccl import get_version  # NOQA
    from cupy.cuda.nccl import NCCL_FLOAT  # NOQA
    from cupy.cuda.nccl import NCCL_FLOAT16  # NOQA
    from cupy.cuda.nccl import NCCL_FLOAT32  # NOQA
    from cupy.cuda.nccl import NCCL_FLOAT64  # NOQA
    from cupy.cuda.nccl import NCCL_SUM  # NOQA
    from cupy.cuda.nccl import NcclCommunicator  # NOQA
    from cupy.cuda.nccl import NcclError  # NOQA
    _available = True
except Exception:
    _available = False
