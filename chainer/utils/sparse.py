import chainer
from chainer.backends import cuda


class CooMatrix(object):
    """A sparse matrix in COO format.

    Args:
        data (numpy.ndarray or cupy.ndarray): The entries of the matrix.
            The entries are usually non-zero-elements in the matrix.
        row (numpy.ndarray or cupy.ndarray): The row indices of the matrix
            entries.
        col (numpy.ndarray or cupy.ndarray): The column indices of the matrix
            entries.
        shape (tuple of int): The shape of the matrix in dense format.
        requires_grad (bool): If ``True``, gradient of this sparse matrix will
            be computed in back-propagation.
    """

    def __init__(self, data, row, col, shape, requires_grad=False):
        if not (1 <= data.ndim <= 2):
            raise ValueError('ndim of data must be 1 or 2.')
        if not (data.ndim == row.ndim == col.ndim):
            raise ValueError('ndim of data, row and col must be the same.')
        if len(shape) != 2:
            raise ValueError('length of shape must be 2.')
        if not (shape[0] > 0 and shape[1] > 0):
            raise ValueError('numbers in shape must be greater than 0.')
        self.data = chainer.Variable(data, requires_grad=requires_grad)
        self.row = row
        self.col = col
        self.shape = shape  # (row, col)

    def to_dense(self):
        """Returns a dense matrix format of this sparse matrix."""
        data = self.data.data
        xp = cuda.get_array_module(data)
        if data.ndim == 1:
            x = xp.zeros(self.shape, dtype=data.dtype)
            x[self.row, self.col] = data
            return x
        if data.ndim == 2:
            nb = data.shape[0]
            x = xp.zeros((nb, self.shape[0], self.shape[1]),
                         dtype=data.dtype)
            for i in range(nb):
                nnz = xp.count_nonzero(data[i])
                x[i, self.row[i, :nnz],
                  self.col[i, :nnz]] = data[i, :nnz]
            return x


def to_coo(x, ldnz=None, requires_grad=False):
    """Returns a sparse matrix in COO format of a dense matrix.

    Args:
        x (numpy.ndarray or cupy.ndarray): Input dense matrix. The ndim of
            ``x`` must be two or three. If ndim is two, it is treated as
            a single matrix. If three, it is treated as batched matrices.
        ldnz (int): Size of arrays for data, row index and column index to be
            created. The Actual size becomes max(nnz, ldnz) where nnz is number
            of non-zero elements in a input dense matrix.
        requires_grad (bool): If ``True``, gradient of sparse matrix will be
            computed in back-propagation.

    Returns:
        ~chainer.utils.CooMatrix: a sparse matrix or batched sparse matrices
        in COO format of a given dense matrix or batched dense matrices.
    """
    xp = cuda.get_array_module(x)
    if x.ndim == 2:
        _row, _col = xp.where(x != 0)
        nnz = len(_row)
        if ldnz is None or ldnz < nnz:
            ldnz = nnz
        data = xp.zeros((ldnz), dtype=x.dtype)
        row = xp.full((ldnz), -1, dtype=xp.int32)
        col = xp.full((ldnz), -1, dtype=xp.int32)
        data[:nnz] = x[_row, _col]
        row[:nnz] = xp.array(_row).astype(xp.int32)
        col[:nnz] = xp.array(_col).astype(xp.int32)
        shape = x.shape
        return CooMatrix(data, row, col, shape, requires_grad)
    elif x.ndim == 3:
        # first axis is batch axis
        nb = x.shape[0]
        if ldnz is None:
            ldnz = 0
        for i in range(nb):
            ldnz = max(ldnz, len(xp.where(x[i] != 0)[0]))
        data = xp.empty((nb, ldnz), dtype=x.dtype)
        row = xp.empty((nb, ldnz), dtype=xp.int32)
        col = xp.empty((nb, ldnz), dtype=xp.int32)
        for i in range(nb):
            coo = to_coo(x[i], ldnz)
            data[i] = coo.data.data
            row[i] = coo.row
            col[i] = coo.col
        shape = x.shape[1:]
        return CooMatrix(data, row, col, shape, requires_grad)
    else:
        raise ValueError('ndim of x must be 2 or 3.')
