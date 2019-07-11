#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <utility>

#include <absl/types/optional.h>
#include <cudnn.h>

#include "chainerx/array.h"
#include "chainerx/constant.h"
#include "chainerx/cuda/cudnn.h"
#include "chainerx/dims.h"
#include "chainerx/dtype.h"
#include "chainerx/shape.h"

namespace chainerx {
namespace cuda {

class CudaDevice;

namespace cuda_internal {

class CudaConvTest;  // for unit-tests

// All the public operations in this class are guaranteed to be thread safe.
class CudaConv {
public:
    Array Conv(
            CudaDevice& device,
            const Array& x,
            const Array& w,
            const absl::optional<Array>& b,
            const Dims& stride,
            const Dims& pad,
            bool cover_all,
            Dtype out_dtype);
    Array ConvTranspose(
            CudaDevice& device,
            const Array& x,
            const Array& w,
            const absl::optional<Array>& b,
            const Dims& stride,
            const Dims& pad,
            const Dims& out_size,
            Dtype out_dtype);
    Array ConvGradWeight(
            CudaDevice& device,
            Dtype w_dtype,
            const Shape& w_shape,
            const Array& x,
            const Array& gy,
            const Dims& stride,
            const Dims& pad,
            bool cover_all);

private:
    void AddBias(CudnnHandle& handle, const CudnnTensorDescriptor& y_desc, const Array& y, const Array& b);

    std::tuple<cudnnConvolutionFwdAlgo_t, size_t, cudnnMathType_t> FindConvolutionForwardAlgorithm(
            CudnnHandle& handle,
            const CudnnTensorDescriptor& x_desc,
            const Array& x,
            const CudnnFilterDescriptor& filter_desc,
            const Array& w,
            const CudnnConvolutionDescriptor& conv_desc,
            const CudnnTensorDescriptor& y_desc,
            const Array& y,
            size_t max_workspace_size,
            const Dims& pad,
            const Dims& stride);
    std::tuple<cudnnConvolutionBwdDataAlgo_t, size_t, cudnnMathType_t> FindConvolutionBackwardDataAlgorithm(
            CudnnHandle& handle,
            const CudnnFilterDescriptor& filter_desc,
            const Array& w,
            const CudnnTensorDescriptor& x_desc,
            const Array& x,
            const CudnnConvolutionDescriptor& conv_desc,
            const CudnnTensorDescriptor& y_desc,
            const Array& y,
            size_t max_workspace_size,
            const Dims& pad,
            const Dims& stride);
    std::tuple<cudnnConvolutionBwdFilterAlgo_t, size_t, cudnnMathType_t> FindConvolutionBackwardFilterAlgorithm(
            CudnnHandle& handle,
            const CudnnTensorDescriptor& x_desc,
            const Array& x,
            const CudnnTensorDescriptor& gy_desc,
            const Array& gy,
            const CudnnConvolutionDescriptor& conv_desc,
            const CudnnFilterDescriptor& gw_desc,
            const Array& gw,
            size_t max_workspace_size,
            const Dims& pad,
            const Dims& stride);

    struct AlgoCacheKey {
        Shape x_shape;
        Shape w_shape;
        Shape y_shape;
        Dims pad;
        Dims stride;
        Dtype dtype;
        size_t max_workspace_size;

        bool operator==(const AlgoCacheKey& other) const {
            return x_shape == other.x_shape && w_shape == other.w_shape && y_shape == other.y_shape && pad == other.pad &&
                   stride == other.stride && dtype == other.dtype && max_workspace_size == other.max_workspace_size;
        }

        bool operator!=(const AlgoCacheKey& other) const { return !operator==(other); }
    };

    struct AlgoCacheKeyHash {
        using result_type = std::size_t;
        std::size_t operator()(const AlgoCacheKey& key) const;
    };

    using FwdAlgoCacheMap =
            std::unordered_map<AlgoCacheKey, std::tuple<cudnnConvolutionFwdAlgo_t, size_t, cudnnMathType_t>, AlgoCacheKeyHash>;
    using BwdDataAlgoCacheMap =
            std::unordered_map<AlgoCacheKey, std::tuple<cudnnConvolutionBwdDataAlgo_t, size_t, cudnnMathType_t>, AlgoCacheKeyHash>;
    using BwdFilterAlgoCacheMap =
            std::unordered_map<AlgoCacheKey, std::tuple<cudnnConvolutionBwdFilterAlgo_t, size_t, cudnnMathType_t>, AlgoCacheKeyHash>;

    friend class CudaConvTest;  // for unit-tests

    std::mutex fwd_algo_cache_mutex_;
    FwdAlgoCacheMap fwd_algo_cache_map_{};
    std::mutex bwd_data_algo_cache_mutex_;
    BwdDataAlgoCacheMap bwd_data_algo_cache_map_{};
    std::mutex bwd_filter_algo_cache_mutex_;
    BwdFilterAlgoCacheMap bwd_filter_algo_cache_map_{};
};

}  // namespace cuda_internal
}  // namespace cuda
}  // namespace chainerx
