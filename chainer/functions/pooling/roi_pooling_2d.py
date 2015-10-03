import numpy

import collections

from chainer import cuda
from chainer import function
from chainer.utils import type_check

def _pair(x):
    if isinstance(x, collections.Iterable):
        return x
    return (x, x)

class ROIPooling2D(function.Function):

    """RoI pooling over a set of 2d planes."""

    def __init__(self, pooled_size=7, spatial_scale=0.0625):
        self.ph, self.pw = _pair(pooled_size)
        self.spatial_scale = spatial_scale

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 2,
            in_types[0].dtype == numpy.float32,
            in_types[0].ndim == 4,
            in_types[1].dtype == numpy.float32,
            in_types[1].ndim == 2,
            in_types[1].shape[1] == 5
        )

    def forward_cpu(self, x):
        raise NotImplementedError()

    def forward_gpu(self, x):
        n, c, h, w = x[0].shape
        n_roi      = x[1].shape[0]

        y                = cuda.empty((n_roi, c, self.ph, self.pw), dtype=x[0].dtype)
        self.argmax_data = cuda.empty((n_roi, c, self.ph, self.pw), dtype=numpy.int32)

        cuda.elementwise(
            'raw T bottom_data, T spatial_scale, int32 channels, int32 height, int32 width,'
            'int32 pooled_height, int32 pooled_width, raw T bottom_rois',
            'T top_data, S argmax_data',
            '''
               int pw = i % pooled_width;
               int ph = (i / pooled_width) % pooled_height;
               int c = (i / pooled_width / pooled_height) % channels;
               int this_n = i / pooled_width / pooled_height / channels;

               int roi_batch_ind = bottom_rois[this_n * 5];
               int roi_start_w = round(bottom_rois[this_n * 5 + 1] * spatial_scale);
               int roi_start_h = round(bottom_rois[this_n * 5 + 2] * spatial_scale);
               int roi_end_w = round(bottom_rois[this_n * 5 + 3] * spatial_scale);
               int roi_end_h = round(bottom_rois[this_n * 5 + 4] * spatial_scale);

               int roi_width = max(roi_end_w - roi_start_w + 1, 1);
               int roi_height = max(roi_end_h - roi_start_h + 1, 1);
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

               T maxval = is_empty ? 0 : -3.402823466E+38F;
               int maxidx = -1;
               int bd_ind = (roi_batch_ind * channels + c) * height * width;
               for (int h = hstart; h < hend; ++h) {
                 for (int w = wstart; w < wend; ++w) {
                   int bottom_i = h * width + w + bd_ind;
                   if (bottom_data[bottom_i] > maxval) {
                     maxval = bottom_data[bottom_i];
                     maxidx = bottom_i;
                   }
                 }
               }
               top_data = maxval;
               argmax_data = maxidx;
            ''', 'roi_pool_fwd')(x[0].reduced_view(), self.spatial_scale,
                                 c, h, w, self.ph, self.pw, x[1].reduced_view(),
                                 y, self.argmax_data)

        return y,

    def backward_cpu(self, x, gy):
        raise NotImplementedError()

    def backward_gpu(self, x, gy):
        n, c, h, w = x[0].shape
        n_roi      = x[1].shape[0]

        gx = cuda.empty_like(x[0])
        groi_dummy = cuda.empty_like(x[1])

        cuda.elementwise(
            'raw T top_diff, raw S argmax_data, int32 num_rois, T spatial_scale,'
            'int32 channels, int32 height, int32 width, int32 pooled_height,'
            'int32 pooled_width, raw T bottom_rois',
            'T bottom_diff',
            '''
               int w = i % width;
               int h = (i / width) % height;
               int c = (i / width / height) % channels;
               int this_n = i / width / height / channels;

               T gradient = 0;
               for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
                 int roi_batch_ind = bottom_rois[roi_n * 5];

                 if (this_n != roi_batch_ind) {
                   continue;
                 }

                 int roi_start_w = round(bottom_rois[1 + roi_n * 5] * spatial_scale);
                 int roi_start_h = round(bottom_rois[2 + roi_n * 5] * spatial_scale);
                 int roi_end_w = round(bottom_rois[3 + roi_n * 5] * spatial_scale);
                 int roi_end_h = round(bottom_rois[4 + roi_n * 5] * spatial_scale);


                 const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                                      h >= roi_start_h && h <= roi_end_h);
                 if (!in_roi) {
                   continue;
                 }

                 int offset = (roi_n * channels + c) * pooled_height * pooled_width;

                 int roi_width = max(roi_end_w - roi_start_w + 1, 1);
                 int roi_height = max(roi_end_h - roi_start_h + 1, 1);

                 T bin_size_h = static_cast<T>(roi_height)
                                    / static_cast<T>(pooled_height);
                 T bin_size_w = static_cast<T>(roi_width)
                                    / static_cast<T>(pooled_width);

                 int phstart = floor(static_cast<T>(h - roi_start_h) / bin_size_h);
                 int phend = ceil(static_cast<T>(h - roi_start_h + 1) / bin_size_h);
                 int pwstart = floor(static_cast<T>(w - roi_start_w) / bin_size_w);
                 int pwend = ceil(static_cast<T>(w - roi_start_w + 1) / bin_size_w);

                 phstart = min(max(phstart, 0), pooled_height);
                 phend = min(max(phend, 0), pooled_height);
                 pwstart = min(max(pwstart, 0), pooled_width);
                 pwend = min(max(pwend, 0), pooled_width);

                 for (int ph = phstart; ph < phend; ++ph) {
                   for (int pw = pwstart; pw < pwend; ++pw) {
                     if (argmax_data[ph * pooled_width + pw + offset] == (h * width + w)) {
                       gradient += top_diff[ph * pooled_width + pw + offset];
                     }
                   }
                 }
               }
               bottom_diff = gradient;
            ''',
            'roi_pool_bwd')(gy[0].reduced_view(), self.argmax_data.reduced_view(),
                            n_roi, self.spatial_scale, c, h, w,
                            self.ph, self.pw, x[1].reduced_view(),
                            gx)
        return gx, groi_dummy

def roi_pooling_2d(x, rois, pooled_size=7, spatial_scale=0.0625):
    return ROIPooling2D(pooled_size, spatial_scale)(x, rois)
