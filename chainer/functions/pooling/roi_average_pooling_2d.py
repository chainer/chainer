# Modified work:
# -----------------------------------------------------------------------------
# Copyright (c) 2015 Preferred Infrastructure, Inc.
# Copyright (c) 2015 Preferred Networks, Inc.
# -----------------------------------------------------------------------------

# Original work of _roi_pooling_slice, forward_cpu and backward_cpu:
# -----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------------

# Original work of forward_gpu and backward_gpu:
# -----------------------------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see fast-rcnn/LICENSE for details]
# Written by Ross Girshick
# -----------------------------------------------------------------------------

import numbers
import numpy
import six

from chainer.backends import cuda
from chainer import function
from chainer.functions.pooling.roi_pooling_2d import _roi_pooling_slice
from chainer import utils
from chainer.utils import collections_abc
from chainer.utils import type_check


def _pair(x):
    if isinstance(x, collections_abc.Iterable):
        return x
    return x, x


class ROIAveragePooling2D(function.Function):

    """RoI average pooling over a set of 2d planes."""

    def __init__(self, outsize, spatial_scale):
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

        self.outh, self.outw = outh, outw
        self.spatial_scale = spatial_scale

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)

        x_type, roi_type, roi_index_type = in_types
        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim == 4,
            x_type.dtype == roi_type.dtype,
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
        top_data = numpy.zeros((n_rois, channels, self.outh, self.outw),
                               dtype=bottom_data.dtype)

        for i_roi in six.moves.range(n_rois):
            idx = bottom_roi_indices[i_roi]
            ymin, xmin, ymax, xmax = bottom_rois[i_roi]
            ymin = int(round(ymin * self.spatial_scale))
            xmin = int(round(xmin * self.spatial_scale))
            ymax = int(round(ymax * self.spatial_scale))
            xmax = int(round(xmax * self.spatial_scale))
            roi_height = max(ymax - ymin, 1)
            roi_width = max(xmax - xmin, 1)
            strideh = 1. * roi_height / self.outh
            stridew = 1. * roi_width / self.outw

            for outh in six.moves.range(self.outh):
                sliceh, lenh = _roi_pooling_slice(
                    outh, strideh, height, ymin)
                if sliceh.stop <= sliceh.start:
                    continue
                for outw in six.moves.range(self.outw):
                    slicew, lenw = _roi_pooling_slice(
                        outw, stridew, width, xmin)
                    if slicew.stop <= slicew.start:
                        continue
                    roi_data = bottom_data[int(idx), :, sliceh, slicew]\
                        .reshape(channels, -1)
                    top_data[i_roi, :, outh, outw] =\
                        numpy.average(roi_data, axis=1)

        return top_data,

    def forward_gpu(self, inputs):
        self.retain_inputs((1, 2))
        self._bottom_data_shape = inputs[0].shape

        bottom_data, bottom_rois, bottom_roi_indices = inputs
        channels, height, width = bottom_data.shape[1:]
        n_rois = bottom_rois.shape[0]
        top_data = cuda.cupy.empty((n_rois, channels, self.outh,
                                    self.outw), dtype=bottom_data.dtype)
        cuda.elementwise(
            '''
            raw T bottom_data, raw T bottom_rois, raw int32 bottom_roi_indices,
            T spatial_scale, int32 channels, int32 height, int32 width,
            int32 pooled_height, int32 pooled_width
            ''',
            'T top_data',
            '''
            // pos in output filter
            int pw = i % pooled_width;
            int ph = (i / pooled_width) % pooled_height;
            int c = (i / pooled_width / pooled_height) % channels;
            int n = i / pooled_width / pooled_height / channels;

            int roi_batch_ind = bottom_roi_indices[n];
            int roi_start_h = round(bottom_rois[n * 4 + 0] * spatial_scale);
            int roi_start_w = round(bottom_rois[n * 4 + 1] * spatial_scale);
            int roi_end_h = round(bottom_rois[n * 4 + 2] * spatial_scale);
            int roi_end_w = round(bottom_rois[n * 4 + 3] * spatial_scale);

            // Force malformed ROIs to be 1x1
            int roi_height = max(roi_end_h - roi_start_h, 1);
            int roi_width = max(roi_end_w - roi_start_w, 1);
            T bin_size_h = static_cast<T>(roi_height)
                           / static_cast<T>(pooled_height);
            T bin_size_w = static_cast<T>(roi_width)
                           / static_cast<T>(pooled_width);

            int hstart = static_cast<int>(floor(static_cast<T>(ph)
                                          * bin_size_h));
            int wstart = static_cast<int>(floor(static_cast<T>(pw)
                                          * bin_size_w));
            int hend = static_cast<int>(ceil(static_cast<T>(ph + 1)
                                        * bin_size_h));
            int wend = static_cast<int>(ceil(static_cast<T>(pw + 1)
                                        * bin_size_w));

            // Add roi offsets and clip to input boundaries
            hstart = min(max(hstart + roi_start_h, 0), height);
            hend = min(max(hend + roi_start_h, 0), height);
            wstart = min(max(wstart + roi_start_w, 0), width);
            wend = min(max(wend + roi_start_w, 0), width);
            bool is_empty = (hend <= hstart) || (wend <= wstart);

            // Define an empty pooling region to be zero
            T sumval = 0.;
            T count = (hend - hstart) * (wend - wstart);
            int data_offset = (roi_batch_ind * channels + c) * height * width;
            for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                    int bottom_index = h * width + w;
                    sumval += bottom_data[data_offset + bottom_index];
                }
            }
            top_data = is_empty ? 0. : sumval / count;
            ''', 'roi_average_pooling_2d_fwd'
        )(bottom_data, bottom_rois, bottom_roi_indices, self.spatial_scale,
          channels, height, width, self.outh, self.outw, top_data)

        return top_data,

    def backward_cpu(self, inputs, gy):
        bottom_rois, bottom_roi_indices = inputs[1:]
        channels, height, width = self._bottom_data_shape[1:]
        n_rois = bottom_rois.shape[0]
        bottom_diff = numpy.zeros(self._bottom_data_shape, gy[0].dtype)

        for i_roi in six.moves.range(n_rois):
            idx = bottom_roi_indices[i_roi]
            ymin, xmin, ymax, xmax = bottom_rois[i_roi]
            ymin = int(round(ymin * self.spatial_scale))
            xmin = int(round(xmin * self.spatial_scale))
            ymax = int(round(ymax * self.spatial_scale))
            xmax = int(round(xmax * self.spatial_scale))
            roi_height = max(ymax - ymin, 1)
            roi_width = max(xmax - xmin, 1)
            strideh = 1. * roi_height / self.outh
            stridew = 1. * roi_width / self.outw

            for outh in six.moves.range(self.outh):
                sliceh, lenh = _roi_pooling_slice(
                    outh, strideh, height, ymin)
                if sliceh.stop <= sliceh.start:
                    continue
                for outw in six.moves.range(self.outw):
                    slicew, lenw = _roi_pooling_slice(
                        outw, stridew, width, xmin)
                    if slicew.stop <= slicew.start:
                        continue
                    diff_val = gy[0][i_roi, :, outh, outw]\
                        .reshape(channels, 1, 1)
                    diff_val = diff_val / lenh / lenw
                    bottom_diff[int(idx), :, sliceh, slicew] \
                        += diff_val

        return bottom_diff, None, None

    def backward_gpu(self, inputs, gy):
        utils.nondeterministic('atomicAdd')
        bottom_rois, bottom_roi_indices = inputs[1:]
        channels, height, width = self._bottom_data_shape[1:]
        bottom_diff = cuda.cupy.zeros(
            self._bottom_data_shape, gy[0].dtype)

        cuda.elementwise(
            '''
            raw T top_diff, raw T bottom_rois, raw int32 bottom_roi_indices,
            T spatial_scale, int32 channels, int32 height, int32 width,
            int32 pooled_height, int32 pooled_width
            ''',
            'raw T bottom_diff',
            '''
            // pos in output filter
            int pw = i % pooled_width;
            int ph = (i / pooled_width) % pooled_height;
            int c = (i / pooled_width / pooled_height) % channels;
            int n = i / pooled_width / pooled_height / channels;

            int roi_batch_ind = bottom_roi_indices[n];
            int roi_start_h = round(bottom_rois[n * 4 + 0] * spatial_scale);
            int roi_start_w = round(bottom_rois[n * 4 + 1] * spatial_scale);
            int roi_end_h = round(bottom_rois[n * 4 + 2] * spatial_scale);
            int roi_end_w = round(bottom_rois[n * 4 + 3] * spatial_scale);

            // Force malformed ROIs to be 1x1
            int roi_height = max(roi_end_h - roi_start_h, 1);
            int roi_width = max(roi_end_w - roi_start_w, 1);
            T bin_size_h = static_cast<T>(roi_height)
                           / static_cast<T>(pooled_height);
            T bin_size_w = static_cast<T>(roi_width)
                           / static_cast<T>(pooled_width);

            int hstart = static_cast<int>(floor(static_cast<T>(ph)
                                          * bin_size_h));
            int wstart = static_cast<int>(floor(static_cast<T>(pw)
                                          * bin_size_w));
            int hend = static_cast<int>(ceil(static_cast<T>(ph + 1)
                                        * bin_size_h));
            int wend = static_cast<int>(ceil(static_cast<T>(pw + 1)
                                        * bin_size_w));

            // Add roi offsets and clip to input boundaries
            hstart = min(max(hstart + roi_start_h, 0), height);
            hend = min(max(hend + roi_start_h, 0), height);
            wstart = min(max(wstart + roi_start_w, 0), width);
            wend = min(max(wend + roi_start_w, 0), width);
            bool is_empty = (hend <= hstart) || (wend <= wstart);

            int bottom_diff_offset =
                (roi_batch_ind * channels + c) * height * width;
            int top_offset =
                (n * channels + c) * pooled_height * pooled_width;

            T count = (hend - hstart) * (wend - wstart);
            T diff_val = is_empty ? 0. :
                top_diff[top_offset + ph * pooled_width + pw] / count;
            for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                    int bottom_index = h * width + w;
                    atomicAdd(
                        &bottom_diff[bottom_diff_offset + bottom_index],
                        diff_val);
                }
            }
            ''', 'roi_average_pooling_2d_bwd'
        )(gy[0], bottom_rois, bottom_roi_indices, self.spatial_scale,
          channels, height, width, self.outh, self.outw,
          bottom_diff, size=gy[0].size)

        return bottom_diff, None, None


def roi_average_pooling_2d(x, rois, roi_indices, outsize, spatial_scale):
    """Spatial Region of Interest (ROI) average pooling function.

    This function acts similarly to
    :func:`~chainer.functions.average_pooling_2d`, but it computes the average
    of input spatial patch for each channel with the region of interest.

    Args:
        x (~chainer.Variable): Input variable. The shape is expected to be
            4 dimensional: (n: batch, c: channel, h, height, w: width).
        rois (~chainer.Variable): Input roi variable. The shape is expected to
            be (n: data size, 4), and each datum is set as below:
            (y_min, x_min, y_max, x_max).
        roi_indices (~chainer.Variable): Input roi variable. The shape is
            expected to be (n: data size, ).
        outsize ((int, int) or int): Expected output size after pooled
            (height, width). ``outsize=o`` and ``outsize=(o, o)``
            are equivalent.
        spatial_scale (float): Scale of the roi is resized.

    Returns:
        ~chainer.Variable: Output variable.

    See the original paper proposing ROIPooling:
    `Fast R-CNN <https://arxiv.org/abs/1504.08083>`_.

    """
    return ROIAveragePooling2D(outsize, spatial_scale)(x, rois, roi_indices)
