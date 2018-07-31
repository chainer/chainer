#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>

#include "xchainer/array.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"
#include "xchainer/macro.h"

namespace xchainer {

// Argument to reduction kernels.
//
// It contains five data:
//
// - Input array
// - Output array
// - Input indexer (using the input shape)
// - Output indexer (using the output shape)
// - Reduction indexer (using the input shape only at reduction axes)
//
// Input and output arrays are transposed so that the reduction axes come last. Axes of length 1 are also removed.
//
// Any instance of this struct can be passed directly to a kernel function (including CUDA __global__ function).
template <typename In, typename Out, int8_t InNdim = kDynamicNdim, int8_t OutNdim = kDynamicNdim, int8_t ReduceNdim = kDynamicNdim>
struct ReductionKernelArg {
    IndexableArray<const In, InNdim> in;
    IndexableArray<Out, OutNdim> out;
    Indexer<InNdim> in_indexer;
    Indexer<OutNdim> out_indexer;
    Indexer<ReduceNdim> reduce_indexer;
};

template <typename In, typename Out, int8_t InNdim, int8_t OutNdim, int8_t ReduceNdim>
ReductionKernelArg<In, Out, InNdim, OutNdim, ReduceNdim> MakeReductionKernelArg(const ReductionKernelArg<In, Out>& arg) {
    // TODO(sonots): Reduce # of copies by avoiding creation of Strides and Shape.
    return ReductionKernelArg<In, Out, InNdim, OutNdim, ReduceNdim>{
            IndexableArray<const In, InNdim>{arg.in.data(), Strides{gsl::make_span(arg.in.strides(), arg.in.ndim())}},
            IndexableArray<Out, OutNdim>{arg.out.data(), Strides{gsl::make_span(arg.out.strides(), arg.out.ndim())}},
            Indexer<InNdim>{Shape{gsl::make_span(arg.in_indexer.shape(), arg.in_indexer.ndim())}},
            Indexer<OutNdim>{Shape{gsl::make_span(arg.out_indexer.shape(), arg.out_indexer.ndim())}},
            Indexer<ReduceNdim>{Shape{gsl::make_span(arg.reduce_indexer.shape(), arg.reduce_indexer.ndim())}}};
}

template <typename In, typename Out>
ReductionKernelArg<In, Out> MakeReductionKernelArg(const Array& in, const Axes& axis, const Array& out) {
    // True if some axes are reduced but kept in output as 1-dim axes.
    // Corresponding to keepdim argument in Array::Sum().
    bool has_kept_dims = out.ndim() + static_cast<int64_t>(axis.size()) != in.ndim();

    // Prepare axis mappings
    Shape reduce_shape{};  // Reduction dimensions
    Axes out_axis_map{};  // Mapping from effective output indices to actual output indices
    Shape new_out_shape{};
    // (Here "effective output indices" means source indices minus reduction indices.)

    // Example (in the case of has_kept_dims=false):
    // - in.shape():      (12, 13, 14, 15, 16)
    // - axis:             (1, 3)
    // - out.shape():      (12, 14, 16)
    // - reduce_shape:     (13, 15)
    // - out_axis_map:     (0, 1, 2)
    // - new_out_shape:    (12, 14, 16)
    // Example (in the case of has_kept_dims=true):
    // - in.shape():      (12, 13, 14, 15, 16)
    // - axis:             (1, 3)
    // - out.shape():      (12, 1, 14, 1, 16)
    // - reduce_shape:     (13, 15)
    // - out_axis_map:     (0, 2, 4)
    // - new_out_shape:    (12, 14, 16)

    {
        size_t i_axis = 0;
        size_t i_out_axis = 0;
        for (int8_t i = 0; i < in.shape().ndim(); ++i) {
            if (i_axis < axis.size() && i == axis[i_axis]) {
                // i is to be reduced
                int64_t in_dim = in.shape()[i];
                if (in_dim != 1) {
                    reduce_shape.emplace_back(in_dim);
                }
                ++i_axis;
                if (has_kept_dims) {
                    ++i_out_axis;
                }
            } else {
                // i is not to be reduced
                int64_t out_dim = out.shape()[i_out_axis];
                if (out_dim != 1) {
                    out_axis_map.emplace_back(static_cast<int8_t>(i_out_axis));
                    new_out_shape.emplace_back(out_dim);
                }
                ++i_out_axis;
            }
        }
        assert(i_out_axis == out.shape().size());
        assert(i_axis == axis.size());
    }
    // Inequality because 1-dim axes are eliminated.
    assert(reduce_shape.size() <= axis.size());
    assert(out_axis_map.size() <= in.shape().size() - axis.size());
    assert(out_axis_map.size() == new_out_shape.size());

    // Calculate source axis permutation
    Axes axis_permutes{};
    {
        size_t i_reduce = 0;
        for (int8_t i = 0; i < in.ndim(); ++i) {
            if (i_reduce < axis.size() && i == axis[i_reduce]) {
                ++i_reduce;
            } else {
                if (in.shape()[i] != 1) {
                    axis_permutes.emplace_back(i);
                }
            }
        }
    }
    for (int8_t i : axis) {
        if (in.shape()[i] != 1) {
            axis_permutes.emplace_back(i);
        }
    }
    assert(axis_permutes.size() <= in.shape().size());  // Inequality because 1-dim axes are eliminated.

    // Calculate new source shape
    Shape new_in_shape{};
    for (int8_t i : axis_permutes) {
        new_in_shape.emplace_back(in.shape()[i]);
    }

    // 1-dim axes must be eliminated
    assert(std::find(new_in_shape.begin(), new_in_shape.end(), 1) == new_in_shape.end());
    assert(std::find(new_out_shape.begin(), new_out_shape.end(), 1) == new_out_shape.end());

    return ReductionKernelArg<In, Out>{IndexableArray<const In>{in}.Permute(axis_permutes),
                                       IndexableArray<Out>{out}.Permute(out_axis_map),
                                       Indexer<>{new_in_shape},
                                       Indexer<>{new_out_shape},
                                       Indexer<>{reduce_shape}};
}

}  // namespace xchainer
