# Modified work:
# -----------------------------------------------------------------------------
# Copyright (c) 2018 Preferred Infrastructure, Inc.
# Copyright (c) 2018 Preferred Networks, Inc.
# -----------------------------------------------------------------------------

# Original work:
# -----------------------------------------------------------------------------
# Copyright (c) 2015 by Contributors
# \file roi_pooling.cu
# \brief roi pooling operator
# \author Ross Girshick, Kye-Hyeon Kim, Jian Guo
# \changed to roi_align by Elaine Bao
# \file roi_align.cu
# \roi align operator described in Mask RCNN
# -----------------------------------------------------------------------------

import numbers
import numpy
import six

import chainer
from chainer.backends import cuda
from chainer import function
from chainer.functions.pooling.roi_average_align_2d \
    import _GET_BILINEAR_INTERP_KERNEL
from chainer.functions.pooling.roi_average_align_2d \
    import _get_bilinear_interp_params
from chainer.functions.pooling.roi_average_align_2d import _get_bounds
from chainer import utils
from chainer.utils import type_check


def _pair(x):
    if isinstance(x, chainer.utils.collections_abc.Iterable):
        return x
    return x, x


class ROIMaxAlign2D(function.Function):

    """ROI max align over a set of 2d planes."""

    def __init__(self, outsize, spatial_scale, sampling_ratio=None):
        outh, outw = _pair(outsize)
        if not (isinstance(outh, numbers.Integral) and outh > 0):
            raise TypeError(
                'outsize[0] must be positive integer: {}, {}'
                .format(type(outh), outh))
        if not (isinstance(outw, numbers.Integral) and outw > 0):
            raise TypeError(
                'outsize[1] must be positive integer: {}, {}'
                .format(type(outw), outw))
        if isinstance(spatial_scale, numbers.Integral):
            spatial_scale = float(spatial_scale)
        if not (isinstance(spatial_scale, numbers.Real) and
                spatial_scale > 0):
            raise TypeError(
                'spatial_scale must be a positive float number: {}, {}'
                .format(type(spatial_scale), spatial_scale))
        sampling_ratio = _pair(sampling_ratio)
        if not all((isinstance(s, numbers.Integral) and s >= 1) or
                   s is None for s in sampling_ratio):
            raise TypeError(
                'sampling_ratio must be integer >= 1 or a pair of it: {}'
                .format(sampling_ratio))

        self.outh, self.outw = outh, outw
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)

        x_type, roi_type, roi_index_type = in_types
        type_check.expect(
            x_type.dtype == numpy.float32,
            x_type.ndim == 4,
            roi_type.dtype == numpy.float32,
            roi_type.ndim == 2,
            roi_type.shape[1] == 4,
            roi_index_type.dtype == numpy.int32,
            roi_index_type.ndim == 1,
            roi_type.shape[0] == roi_index_type.shape[0],
        )

    def forward_cpu(self, inputs):
        self.retain_inputs((1, 2))
        self._bottom_data_shape = inputs[0].shape

        bottom_data, bottom_rois, bottom_roi_indices = inputs
        channels, height, width = bottom_data.shape[1:]
        n_rois = bottom_rois.shape[0]
        top_data = numpy.empty((n_rois, channels, self.outh,
                                self.outw), dtype=bottom_data.dtype)
        self.argmax_data = numpy.empty(top_data.shape, numpy.int32)

        pooled_width, pooled_height = self.outw, self.outh
        spatial_scale = self.spatial_scale

        for i in six.moves.range(top_data.size):
            pw = i % pooled_width
            ph = int(i / pooled_width) % pooled_height
            c = int(i / pooled_width / pooled_height) % channels
            n = int(i / pooled_width / pooled_height / channels)

            roi_batch_ind = bottom_roi_indices[n]
            roi_start_h = bottom_rois[n, 0] * spatial_scale
            roi_start_w = bottom_rois[n, 1] * spatial_scale
            roi_end_h = bottom_rois[n, 2] * spatial_scale
            roi_end_w = bottom_rois[n, 3] * spatial_scale

            roi_width = max(roi_end_w - roi_start_w, 1.)
            roi_height = max(roi_end_h - roi_start_h, 1.)
            bin_size_h = roi_height / pooled_height
            bin_size_w = roi_width / pooled_width

            if self.sampling_ratio[0] is None:
                roi_bin_grid_h = int(numpy.ceil(roi_height / pooled_height))
            else:
                roi_bin_grid_h = self.sampling_ratio[0]
            if self.sampling_ratio[1] is None:
                roi_bin_grid_w = int(numpy.ceil(roi_width / pooled_width))
            else:
                roi_bin_grid_w = self.sampling_ratio[1]

            max_val = - numpy.inf
            max_index = -1
            for iy in six.moves.range(roi_bin_grid_h):
                y = roi_start_h + ph * bin_size_h + \
                    (iy + .5) * bin_size_h / roi_bin_grid_h
                y, y_low, y_high = _get_bounds(y, height)
                if y is None or y_low is None or y_high is None:
                    continue
                for ix in six.moves.range(roi_bin_grid_w):
                    x = roi_start_w + pw * bin_size_w + \
                        (ix + .5) * bin_size_w / roi_bin_grid_w
                    x, x_low, x_high = _get_bounds(x, width)
                    if x is None or x_low is None or x_high is None:
                        continue
                    # bilinear interpolation {{

                    w1, w2, w3, w4 = _get_bilinear_interp_params(
                        y, x, y_low, x_low, y_high, x_high)

                    tmp_val = 0.
                    if w1 > 0 and y_low >= 0 and x_low >= 0:
                        v1 = bottom_data[roi_batch_ind, c, y_low, x_low]
                        tmp_val += w1 * v1

                    if w2 > 0 and y_low >= 0 and x_high <= width - 1:
                        v2 = bottom_data[roi_batch_ind, c, y_low, x_high]
                        tmp_val += w2 * v2

                    if w3 > 0 and y_high <= height - 1 and x_low >= 0:
                        v3 = bottom_data[roi_batch_ind, c, y_high, x_low]
                        tmp_val += w3 * v3

                    if w4 > 0 and y_high <= height - 1 and x_high <= width - 1:
                        v4 = bottom_data[roi_batch_ind, c, y_high, x_high]
                        tmp_val += w4 * v4

                    tmp_index = iy * roi_bin_grid_w + ix
                    if tmp_val > max_val:
                        max_val = tmp_val
                        max_index = tmp_index

                    # }}
            top_data[n, c, ph, pw] = max_val
            self.argmax_data[n, c, ph, pw] = max_index

        return top_data,

    def forward_gpu(self, inputs):
        self.retain_inputs((1, 2))
        self._bottom_data_shape = inputs[0].shape

        bottom_data, bottom_rois, bottom_roi_indices = inputs
        channels, height, width = bottom_data.shape[1:]
        n_rois = bottom_rois.shape[0]
        top_data = cuda.cupy.empty((n_rois, channels, self.outh,
                                    self.outw), dtype=bottom_data.dtype)
        self.argmax_data = cuda.cupy.empty(top_data.shape, numpy.int32)

        if self.sampling_ratio[0] is None:
            sampling_ratio_h = 0
        else:
            sampling_ratio_h = self.sampling_ratio[0]
        if self.sampling_ratio[1] is None:
            sampling_ratio_w = 0
        else:
            sampling_ratio_w = self.sampling_ratio[1]
        cuda.elementwise(
            '''
            raw T bottom_data, T spatial_scale, int32 channels,
            int32 height, int32 width, int32 pooled_height, int32 pooled_width,
            int32 sampling_ratio_h, int32 sampling_ratio_w,
            raw T bottom_rois, raw int32 bottom_roi_indices
            ''',
            'T top_data, int32 argmax_data',
            '''
            int pw = i % pooled_width;
            int ph = (i / pooled_width) % pooled_height;
            int c = (i / pooled_width / pooled_height) % channels;
            int n = i / pooled_width / pooled_height / channels;

            int roi_batch_ind = bottom_roi_indices[n];

            T roi_start_h = bottom_rois[n * 4 + 0] * spatial_scale;
            T roi_start_w = bottom_rois[n * 4 + 1] * spatial_scale;
            T roi_end_h = bottom_rois[n * 4 + 2] * spatial_scale;
            T roi_end_w = bottom_rois[n * 4 + 3] * spatial_scale;

            // Force malformed ROIs to be 1x1
            T roi_width = max(roi_end_w - roi_start_w, (T)1.);
            T roi_height = max(roi_end_h - roi_start_h, (T)1.);
            T bin_size_h = static_cast<T>(roi_height)
                            / static_cast<T>(pooled_height);
            T bin_size_w = static_cast<T>(roi_width)
                            / static_cast<T>(pooled_width);

            int bottom_data_offset =
                (roi_batch_ind * channels + c) * height * width;

            // We use roi_bin_grid to sample the grid and mimic integral
            int roi_bin_grid_h = (sampling_ratio_h > 0)
                ? sampling_ratio_h
                : ceil(roi_height / pooled_height);  // e.g. = 2
            int roi_bin_grid_w = (sampling_ratio_w > 0)
                ? sampling_ratio_w
                : ceil(roi_width / pooled_width);

            T max_val = - (T) (1.0 / 0.0);
            int max_index = -1;
            for (int iy = 0; iy < roi_bin_grid_h; iy++)  // e.g. iy = 0, 1
            {
                T y = roi_start_h + ph * bin_size_h +
                    static_cast<T>(iy + .5f) * bin_size_h /
                        static_cast<T>(roi_bin_grid_h);  // e.g. 0.5, 1.5
                int y_low, y_high;
                bool y_ret = get_bounds(y, height, y_low, y_high);
                if (!y_ret) continue;
                for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                    T x = roi_start_w + pw * bin_size_w +
                        static_cast<T>(ix + .5f) * bin_size_w /
                            static_cast<T>(roi_bin_grid_w);

                    int x_low, x_high;
                    bool x_ret = get_bounds(x, width, x_low, x_high);
                    if (!x_ret) continue;
                    // bilinear_interpolation {{
                    T w1, w2, w3, w4;
                    get_bilinear_interp_params(
                        y, x, y_low, x_low, y_high, x_high, w1, w2, w3, w4);

                    T tmp_val = 0.;
                    if (w1 > 0 && y_low >= 0 && x_low >= 0) {
                        T v1 = bottom_data[
                            bottom_data_offset + y_low * width + x_low];
                        tmp_val += w1 * v1;
                    }
                    if (w2 > 0 && y_low >= 0 && x_high <= width - 1) {
                        T v2 = bottom_data[
                            bottom_data_offset + y_low * width + x_high];
                        tmp_val += w2 * v2;
                    }
                    if (w3 > 0 && y_high <= height - 1 && x_low >= 0) {
                        T v3 = bottom_data[
                            bottom_data_offset + y_high * width + x_low];
                        tmp_val += w3 * v3;
                    }
                    if (w4 > 0 && y_high <= height - 1 &&
                            x_high <= width - 1) {
                        T v4 = bottom_data[
                            bottom_data_offset + y_high * width + x_high];
                        tmp_val += w4 * v4;
                    }

                    int tmp_index = iy * roi_bin_grid_w + ix;
                    if (tmp_val > max_val) {
                        max_val = tmp_val;
                        max_index = tmp_index;
                    }

                    // }}
                }
            }

            top_data = max_val;
            argmax_data = max_index;
            ''',
            'roi_max_align_2d_fwd',
            preamble=_GET_BILINEAR_INTERP_KERNEL,
        )(bottom_data, self.spatial_scale, channels, height, width,
          self.outh, self.outw, sampling_ratio_h, sampling_ratio_w,
          bottom_rois, bottom_roi_indices, top_data, self.argmax_data)

        return top_data,

    def backward_cpu(self, inputs, gy):
        bottom_rois, bottom_roi_indices = inputs[1:]
        channels, height, width = self._bottom_data_shape[1:]
        bottom_diff = numpy.zeros(self._bottom_data_shape, gy[0].dtype)

        spatial_scale = self.spatial_scale
        pooled_height = self.outh
        pooled_width = self.outw
        top_diff = gy[0]

        for i in six.moves.range(top_diff.size):
            pw = i % pooled_width
            ph = int(i / pooled_width) % pooled_height
            c = int(i / pooled_width / pooled_height) % channels
            n = int(i / pooled_width / pooled_height / channels)

            roi_batch_ind = bottom_roi_indices[n]
            roi_start_h = bottom_rois[n, 0] * spatial_scale
            roi_start_w = bottom_rois[n, 1] * spatial_scale
            roi_end_h = bottom_rois[n, 2] * spatial_scale
            roi_end_w = bottom_rois[n, 3] * spatial_scale

            roi_height = max(roi_end_h - roi_start_h, 1.)
            roi_width = max(roi_end_w - roi_start_w, 1.)
            bin_size_h = roi_height / pooled_height
            bin_size_w = roi_width / pooled_width

            top_diff_this_bin = top_diff[n, c, ph, pw]
            max_index = self.argmax_data[n, c, ph, pw]

            if max_index != -1:
                if self.sampling_ratio[0] is None:
                    roi_bin_grid_h = numpy.ceil(roi_height / pooled_height)
                else:
                    roi_bin_grid_h = self.sampling_ratio[0]
                if self.sampling_ratio[1] is None:
                    roi_bin_grid_w = numpy.ceil(roi_width / pooled_width)
                else:
                    roi_bin_grid_w = self.sampling_ratio[1]

                iy = int(max_index / roi_bin_grid_w)
                ix = max_index % roi_bin_grid_w

                y = roi_start_h + ph * bin_size_h + \
                    (iy + .5) * bin_size_h / roi_bin_grid_h
                x = roi_start_w + pw * bin_size_w + \
                    (ix + .5) * bin_size_w / roi_bin_grid_w

                # bilinear_interpolation_gradient {{

                y, y_low, y_high = _get_bounds(y, height)
                if y is None or y_low is None or y_high is None:
                    continue
                x, x_low, x_high = _get_bounds(x, width)
                if x is None or x_low is None or x_high is None:
                    continue
                w1, w2, w3, w4 = _get_bilinear_interp_params(
                    y, x, y_low, x_low, y_high, x_high)

                if w1 > 0 and y_low >= 0 and x_low >= 0:
                    g1 = top_diff_this_bin * w1
                    bottom_diff[roi_batch_ind, c, y_low, x_low] += g1

                if w2 > 0 and y_low >= 0 and x_high <= width - 1:
                    g2 = top_diff_this_bin * w2
                    bottom_diff[roi_batch_ind, c, y_low, x_high] += g2

                if w3 > 0 and y_high <= height - 1 and x_low >= 0:
                    g3 = top_diff_this_bin * w3
                    bottom_diff[roi_batch_ind, c, y_high, x_low] += g3

                if w4 > 0 and y_high <= height - 1 and x_high <= width - 1:
                    g4 = top_diff_this_bin * w4
                    bottom_diff[roi_batch_ind, c, y_high, x_high] += g4

                # }}

        return bottom_diff, None, None

    def backward_gpu(self, inputs, gy):
        utils.nondeterministic('atomicAdd')
        bottom_rois, bottom_roi_indices = inputs[1:]
        channels, height, width = self._bottom_data_shape[1:]
        bottom_diff = cuda.cupy.zeros(self._bottom_data_shape, gy[0].dtype)

        if self.sampling_ratio[0] is None:
            sampling_ratio_h = 0
        else:
            sampling_ratio_h = self.sampling_ratio[0]
        if self.sampling_ratio[1] is None:
            sampling_ratio_w = 0
        else:
            sampling_ratio_w = self.sampling_ratio[1]

        cuda.elementwise(
            '''
            raw T top_diff, T spatial_scale,
            int32 channels, int32 height, int32 width,
            int32 pooled_height, int32 pooled_width,
            int32 sampling_ratio_h, int32 sampling_ratio_w,
            raw T bottom_rois, raw int32 bottom_roi_indices
            ''',
            'raw T bottom_diff, raw int32 argmax_data',
            '''
            // (n, c, h, w) coords in bottom data
            int pw = i % pooled_width;
            int ph = (i / pooled_width) % pooled_height;
            int c = (i / pooled_width / pooled_height) % channels;
            int n = i / pooled_width / pooled_height / channels;

            // Do not using rounding; this implementation detail is critical
            int roi_batch_ind = bottom_roi_indices[n];
            T roi_start_h = bottom_rois[n * 4 + 0] * spatial_scale;
            T roi_start_w = bottom_rois[n * 4 + 1] * spatial_scale;
            T roi_end_h = bottom_rois[n * 4 + 2] * spatial_scale;
            T roi_end_w = bottom_rois[n * 4 + 3] * spatial_scale;

            // Force malformed ROIs to be 1x1
            T roi_width = max(roi_end_w - roi_start_w, (T)1.);
            T roi_height = max(roi_end_h - roi_start_h, (T)1.);
            T bin_size_h = static_cast<T>(roi_height) /
                static_cast<T>(pooled_height);
            T bin_size_w = static_cast<T>(roi_width) /
                static_cast<T>(pooled_width);

            int bottom_diff_offset =
                (roi_batch_ind * channels + c) * height * width;

            int top_offset = (n * channels + c) * pooled_height * pooled_width;
            int max_index = argmax_data[top_offset + ph * pooled_width + pw];

            if (max_index != -1) {
                T top_diff_this_bin =
                    top_diff[top_offset + ph * pooled_width + pw];

                // We use roi_bin_grid to sample the grid and mimic integral
                int roi_bin_grid_h = (sampling_ratio_h > 0)
                    ? sampling_ratio_h
                    : ceil(roi_height / pooled_height); // e.g. = 2
                int roi_bin_grid_w = (sampling_ratio_w > 0)
                    ? sampling_ratio_w
                    : ceil(roi_width / pooled_width);

                int iy = max_index / roi_bin_grid_w;
                int ix = max_index % roi_bin_grid_w;

                T y = roi_start_h + ph * bin_size_h +
                    static_cast<T>(iy + .5f) * bin_size_h /
                        static_cast<T>(roi_bin_grid_h);  // e.g. 0.5, 1.5
                T x = roi_start_w + pw * bin_size_w +
                    static_cast<T>(ix + .5f) * bin_size_w /
                        static_cast<T>(roi_bin_grid_w);

                // bilinear_interpolation_gradient {{
                int y_low, x_low, y_high, x_high;
                T w1, w2, w3, w4;
                bool y_ret = get_bounds(y, height, y_low, y_high);
                bool x_ret = get_bounds(x, width, x_low, x_high);
                if (!x_ret || !y_ret) continue;
                get_bilinear_interp_params(
                    y, x, y_low, x_low, y_high, x_high, w1, w2, w3, w4);

                if (w1 > 0 && y_low >= 0 && x_low >= 0) {
                    T g1 = top_diff_this_bin * w1;
                    atomicAdd(&bottom_diff[
                        bottom_diff_offset + y_low * width + x_low], g1);
                }
                if (w2 > 0 && y_low >= 0 && x_high <= width - 1) {
                    T g2 = top_diff_this_bin * w2;
                    atomicAdd(&bottom_diff[
                        bottom_diff_offset + y_low * width + x_high], g2);
                }
                if (w3 > 0 && y_high <= height - 1 && x_low >= 0) {
                    T g3 = top_diff_this_bin * w3;
                    atomicAdd(&bottom_diff[
                        bottom_diff_offset + y_high * width + x_low], g3);
                }
                if (w4 > 0 && y_high <= height - 1 && x_high <= width - 1) {
                    T g4 = top_diff_this_bin * w4;
                    atomicAdd(&bottom_diff[
                        bottom_diff_offset + y_high * width + x_high], g4);
                }
            }
            // }}
            ''',
            'roi_max_align_2d_bwd',
            preamble=_GET_BILINEAR_INTERP_KERNEL,
        )(gy[0], self.spatial_scale, channels, height, width,
          self.outh, self.outw, sampling_ratio_h, sampling_ratio_w,
          bottom_rois, bottom_roi_indices, bottom_diff, self.argmax_data,
          size=gy[0].size)

        return bottom_diff, None, None


def roi_max_align_2d(
        x, rois, roi_indices, outsize, spatial_scale, sampling_ratio=None
):
    """Spatial Region of Interest (ROI) max align function.

    This function acts similarly to
    :func:`~chainer.functions.roi_max_pooling_2d`, but it computes maximum
    of input spatial patch with bilinear interpolation for each channel with
    the region of interest.

    Args:
        x (~chainer.Variable): Input variable. The shape is expected to be
            4 dimensional: ``(n: batch, c: channel, h, height, w: width)``.
        rois (~chainer.Variable): Input roi variable. The shape is expected to
            be ``(n: data size, 4)``, and each datum is set as below:
            ``(y_min, x_min, y_max, x_max)``.
        roi_indices (~chainer.Variable): Input roi variable. The shape is
            expected to be ``(n: data size, )``.
        outsize ((int, int) or int): Expected output size after pooled
            (height, width). ``outsize=o`` and ``outsize=(o, o)``
            are equivalent.
        spatial_scale (float): Scale of the roi is resized.
        sampling_ratio ((int, int) or int): Sampling step for the alignment.
            It must be an integer over :math:`1` or :obj:`None`, and the value
            is automatically decided when :obj:`None` is passed.  Use of
            different ratio in height and width axis is also supported by
            passing tuple of int as ``(sampling_ratio_h, sampling_ratio_w)``.
            ``sampling_ratio=s`` and ``sampling_ratio=(s, s)`` are equivalent.

    Returns:
        ~chainer.Variable: Output variable.

    See the original paper proposing ROIAlign:
    `Mask R-CNN <https://arxiv.org/abs/1703.06870>`_.

    """
    return ROIMaxAlign2D(outsize, spatial_scale, sampling_ratio)(
        x, rois, roi_indices)
