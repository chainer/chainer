#pragma once

#include <cudnn.h>

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>

#include "xchainer/array.h"
#include "xchainer/cuda/cudnn.h"
#include "xchainer/dtype.h"
#include "xchainer/shape.h"
#include "xchainer/stack_vector.h"

namespace xchainer {
namespace cuda {

namespace internal {

struct ConvAlgoCacheKey {
    Shape x_shape;
    Shape w_shape;
    Shape y_shape;
    StackVector<int64_t, kMaxNdim> pad;
    StackVector<int64_t, kMaxNdim> stride;
    Dtype dtype;
    size_t max_workspace_size;

    bool operator==(const ConvAlgoCacheKey& other) const {
        return x_shape == other.x_shape && w_shape == other.w_shape && y_shape == other.y_shape && pad == other.pad &&
               stride == other.stride && dtype == other.dtype && max_workspace_size == other.max_workspace_size;
    }

    bool operator!=(const ConvAlgoCacheKey& other) const { return !operator==(other); }
};

struct ConvAlgoCacheKeyHash {
    using result_type = std::size_t;
    std::size_t operator()(const ConvAlgoCacheKey& key) const;
};

using ConvFwdAlgoCacheMap = std::unordered_map<ConvAlgoCacheKey, std::pair<cudnnConvolutionFwdAlgo_t, size_t>, ConvAlgoCacheKeyHash>;
using ConvBwdDataAlgoCacheMap =
        std::unordered_map<ConvAlgoCacheKey, std::pair<cudnnConvolutionBwdDataAlgo_t, size_t>, ConvAlgoCacheKeyHash>;
using ConvBwdFilterAlgoCacheMap =
        std::unordered_map<ConvAlgoCacheKey, std::pair<cudnnConvolutionBwdFilterAlgo_t, size_t>, ConvAlgoCacheKeyHash>;

class ConvAlgo {
public:
    std::pair<cudnnConvolutionFwdAlgo_t, size_t> FindConvolutionForwardAlgorithm(
            cudnnHandle_t handle,
            const TensorDescriptor& x_desc,
            const Array& x,
            const FilterDescriptor& filter_desc,
            const Array& w,
            const ConvolutionDescriptor& conv_desc,
            const TensorDescriptor& y_desc,
            const Array& y,
            size_t max_workspace_size,
            const StackVector<int64_t, kMaxNdim>& pad,
            const StackVector<int64_t, kMaxNdim>& stride);
    std::pair<cudnnConvolutionBwdDataAlgo_t, size_t> FindConvolutionBackwardDataAlgorithm(
            cudnnHandle_t handle,
            const FilterDescriptor& filter_desc,
            const Array& w,
            const TensorDescriptor& x_desc,
            const Array& x,
            const ConvolutionDescriptor& conv_desc,
            const TensorDescriptor& y_desc,
            const Array& y,
            size_t max_workspace_size,
            const StackVector<int64_t, kMaxNdim>& pad,
            const StackVector<int64_t, kMaxNdim>& stride);
    std::pair<cudnnConvolutionBwdFilterAlgo_t, size_t> FindConvolutionBackwardFilterAlgorithm(
            cudnnHandle_t handle,
            const TensorDescriptor& x_desc,
            const Array& x,
            const TensorDescriptor& gy_desc,
            const Array& gy,
            const ConvolutionDescriptor& conv_desc,
            const FilterDescriptor& gw_desc,
            const Array& gw,
            size_t max_workspace_size,
            const StackVector<int64_t, kMaxNdim>& pad,
            const StackVector<int64_t, kMaxNdim>& stride);

private:
    ConvFwdAlgoCacheMap conv_fwd_algo_cache_map_{};
    ConvBwdDataAlgoCacheMap conv_bwd_data_algo_cache_map_{};
    ConvBwdFilterAlgoCacheMap conv_bwd_filter_algo_cache_map_{};
};

}  // namespace internal
}  // namespace cuda
}  // namespace xchainer
