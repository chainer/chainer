import numpy

from chainer import backend
from chainer.backends import cuda
from chainer import function_node
from chainer.utils import type_check


def _infer_lines(B, C, H, W, out_H, out_W, kH, kW):
    target_size = 2 ** 17
    line_size = B * C * (H * W // out_H + kH * kW * out_W)
    target_lines = target_size // line_size

    if target_lines < out_H:
        lines = 1
        while True:
            next_lines = lines * 2
            if next_lines > target_lines:
                break
            lines = next_lines
    else:
        lines = out_H

    return lines


def interpolate_bilinear_cpu(x, v, u, vw, uw):
    B, C, H, W = x.shape
    out_H, out_W = v.shape

    # Interpolation is done by each output panel (i.e. multi lines)
    # in order to better utilize CPU cache memory.
    lines = _infer_lines(B, C, H, W, out_H, out_W, 2, 2)

    vcol = numpy.empty((2, lines, out_W), dtype=v.dtype)
    ucol = numpy.empty((2, lines, out_W), dtype=u.dtype)
    wcol = numpy.empty((2, 2, lines, out_W), dtype=x.dtype)

    y = numpy.empty((B * C, out_H * out_W), dtype=x.dtype)

    for i in range(0, out_H, lines):
        l = min(lines, out_H - i)
        vcol = vcol[:, :l]
        ucol = ucol[:, :l]
        wcol = wcol[:, :, :l]
        i_end = i + l

        # indices
        vcol[0] = v[i:i_end]
        ucol[0] = u[i:i_end]
        numpy.add(vcol[0], 1, out=vcol[1])
        numpy.add(ucol[0], 1, out=ucol[1])
        numpy.minimum(vcol[1], H - 1, out=vcol[1])
        numpy.minimum(ucol[1], W - 1, out=ucol[1])

        # weights
        #   wcol[0, 0] = (1 - uw) * (1 - vw)
        #   wcol[0, 1] = uw * (1 - vw)
        #   wcol[1, 0] = (1 - uw) * vw
        #   wcol[1, 1] = uw * vw
        wcol[0, 1] = uw[i:i_end]
        numpy.subtract(1, wcol[0, 1], out=wcol[0, 0])
        numpy.multiply(wcol[0], vw[i:i_end], out=wcol[1])
        wcol[0] -= wcol[1]

        # packing to the panel whose shape is (B, C, 2, 2, l, out_W)
        panel = x[:, :, vcol[:, None], ucol[None, :]]

        # interpolation
        panel = panel.reshape((B * C, 4, l * out_W))
        weights = wcol.reshape((4, l * out_W))
        iout = i * out_W
        iout_end = i_end * out_W
        numpy.einsum('ijk,jk->ik', panel, weights, out=y[:, iout:iout_end])
        del panel, weights

    return y.reshape((B, C, out_H, out_W))


def interpolate_bilinear_gpu(x, v, u, vw, uw):
    B, C, H, W = x.shape
    out_H, out_W = v.shape
    y = cuda.cupy.empty((B, C, out_H, out_W), dtype=x.dtype)

    cuda.elementwise(
        'raw T x, S v, S u, T vw, T uw, S H, S W, S outsize', 'T y', '''
        // indices
        S v0 = v;
        S v1 = min(v + 1, (S)(H - 1));
        S u0 = u;
        S u1 = min(u + 1, (S)(W - 1));
        // weights
        T w0 = (1 - vw) * (1 - uw);
        T w1 = (1 - vw) * uw;
        T w2 = vw * (1 - uw);
        T w3 = vw * uw;
        // fetch
        S offset = i / outsize * H * W;
        T px0 = x[offset + v0 * W + u0];
        T px1 = x[offset + v0 * W + u1];
        T px2 = x[offset + v1 * W + u0];
        T px3 = x[offset + v1 * W + u1];
        // interpolate
        y = (w0 * px0 + w1 * px1) + (w2 * px2 + w3 * px3);
        ''', 'resize_images_interpolate_bilinear'
    )(x, v, u, vw, uw, H, W, out_H * out_W, y)
    return y


def interpolate_grad_bilinear_cpu(gy, v, u, vw, uw, H, W):
    B, C, out_H, out_W = gy.shape

    # indices
    vcol = numpy.empty((2, out_H, out_W), dtype=v.dtype)
    ucol = numpy.empty((2, out_H, out_W), dtype=u.dtype)
    vcol[0] = v
    ucol[0] = u
    numpy.add(vcol[0], 1, out=vcol[1])
    numpy.add(ucol[0], 1, out=ucol[1])
    numpy.minimum(vcol[1], H - 1, out=vcol[1])
    numpy.minimum(ucol[1], W - 1, out=ucol[1])

    # weights
    wcol = numpy.empty((2, 2, out_H, out_W), dtype=gy.dtype)
    wcol[0, 1] = uw
    numpy.subtract(1, wcol[0, 1], out=wcol[0, 0])
    numpy.multiply(wcol[0], vw, out=wcol[1])
    wcol[0] -= wcol[1]

    # grad
    gycol = gy.reshape((B * C, 1, 1, out_H, out_W)) * wcol

    # ravel everything and use `bincount`
    indices = (vcol[:, None] * W + ucol[None, :]).ravel()
    offsets = numpy.arange(0, B * C * H * W, H * W, dtype=v.dtype)
    indices = (offsets[:, None] + indices).ravel()
    gx = numpy.bincount(indices, weights=gycol.ravel(),
                        minlength=(B * C * H * W))
    gx = gx.astype(gy.dtype, copy=False)

    return gx.reshape((B, C, H, W))


def interpolate_grad_bilinear_gpu(gy, v, u, vw, uw, H, W):
    B, C, out_H, out_W = gy.shape
    gx = cuda.cupy.zeros((B * C, H, W), dtype=gy.dtype)

    cuda.elementwise(
        'T gy, S v, S u, T vw, T uw, S H, S W, S outsize', 'raw T gx', '''
        // indices
        S v0 = v;
        S v1 = min(v + 1, (S)(H - 1));
        S u0 = u;
        S u1 = min(u + 1, (S)(W - 1));
        // weights
        T w0 = (1 - vw) * (1 - uw);
        T w1 = (1 - vw) * uw;
        T w2 = vw * (1 - uw);
        T w3 = vw * uw;
        // scatter
        S offset = i / outsize * H * W;
        atomicAdd(&gx[offset + v0 * W + u0], w0 * gy);
        atomicAdd(&gx[offset + v0 * W + u1], w1 * gy);
        atomicAdd(&gx[offset + v1 * W + u0], w2 * gy);
        atomicAdd(&gx[offset + v1 * W + u1], w3 * gy);
        ''', 'resize_images_interpolate_grad_bilinear'
    )(gy, v, u, vw, uw, H, W, out_H * out_W, gx)

    return gx.reshape((B, C, H, W))


class ResizeImages(function_node.FunctionNode):

    def __init__(self, output_shape):
        self.out_H = output_shape[0]
        self.out_W = output_shape[1]

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))

        x_type = in_types[0]
        type_check.expect(
            x_type.dtype == numpy.float32,
            x_type.ndim == 4
        )

    def forward(self, inputs):
        x, = inputs
        xp = backend.get_array_module(x)

        _, C, H, W = x.shape
        out_H, out_W = self.out_H, self.out_W

        # Compute indices and weights.
        v = xp.linspace(0, H - 1, num=out_H, dtype=numpy.float)
        u = xp.linspace(0, W - 1, num=out_W, dtype=numpy.float)
        vw, v = xp.modf(v)
        uw, u = xp.modf(u)
        v = v.astype(numpy.intp)
        u = u.astype(numpy.intp)
        vw = vw.astype(x.dtype)
        uw = uw.astype(x.dtype)

        # Meshgrid-like operation. Meshgrid can cause
        # performance loss due to memory consumption.
        # Note that numpy 1.9 doesn't support broadcast_to method.
        v, u, vw, uw = xp.broadcast_arrays(
            v[:, None], u[None, :], vw[:, None], uw[None, :])

        if xp is numpy:
            y = interpolate_bilinear_cpu(x, v, u, vw, uw)
        else:
            y = interpolate_bilinear_gpu(x, v, u, vw, uw)
        return y,

    def backward(self, indexes, grad_outputs):
        return ResizeImagesGrad(
            self.inputs[0].shape, (self.out_H, self.out_W)).apply(grad_outputs)


class ResizeImagesGrad(function_node.FunctionNode):

    def __init__(self, input_shape, output_shape):
        self.out_H = output_shape[0]
        self.out_W = output_shape[1]
        self.input_shape = input_shape

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('gy',))

        gy_type = in_types[0]
        type_check.expect(
            gy_type.dtype == numpy.float32,
            gy_type.ndim == 4
        )

    def forward(self, inputs):
        gy, = inputs
        xp = backend.get_array_module(gy)

        _, C, H, W = self.input_shape
        out_H, out_W = self.out_H, self.out_W

        # Compute indices and weights.
        v = xp.linspace(0, H - 1, num=out_H, dtype=numpy.float)
        u = xp.linspace(0, W - 1, num=out_W, dtype=numpy.float)
        vw, v = xp.modf(v)
        uw, u = xp.modf(u)
        v = v.astype(numpy.intp)
        u = u.astype(numpy.intp)
        vw = vw.astype(gy.dtype)
        uw = uw.astype(gy.dtype)

        # Meshgrid-like operation. Meshgrid can cause
        # performance loss due to memory consumption.
        # Note that numpy 1.9 doesn't support broadcast_to method.
        v, u, vw, uw = xp.broadcast_arrays(
            v[:, None], u[None, :], vw[:, None], uw[None, :])

        if xp is numpy:
            gx = interpolate_grad_bilinear_cpu(gy, v, u, vw, uw, H, W)
        else:
            gx = interpolate_grad_bilinear_gpu(gy, v, u, vw, uw, H, W)
        return gx,

    def backward(self, indexes, grad_outputs):
        return ResizeImages((self.out_H, self.out_W)).apply(grad_outputs)


def resize_images(x, output_shape):
    """Resize images to the given shape.

    This function resizes 2D data to :obj:`output_shape`.
    Currently, only bilinear interpolation is supported as the sampling method.

    Notation: here is a notation for dimensionalities.

    - :math:`n` is the batch size.
    - :math:`c_I` is the number of the input channels.
    - :math:`h` and :math:`w` are the height and width of the input image,
      respectively.
    - :math:`h_O` and :math:`w_O` are the height and width of the output
      image.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input variable of shape :math:`(n, c_I, h, w)`.
        output_shape (tuple): This is a tuple of length 2 whose values are
            :obj:`(h_O, w_O)`. Note that the order of height and width is
            opposite of the one in OpenCV.

    Returns:
        ~chainer.Variable: Resized image whose shape is \
            :math:`(n, c_I, h_O, w_O)`.

    """
    return ResizeImages(output_shape).apply((x,))[0]
