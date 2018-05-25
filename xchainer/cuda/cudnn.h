#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include <cudnn.h>
#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/device.h"
#include "xchainer/error.h"

namespace xchainer {
namespace cuda {

class CudnnError : public XchainerError {
public:
    explicit CudnnError(cudnnStatus_t status);
    cudnnStatus_t error() const noexcept { return status_; }

private:
    cudnnStatus_t status_;
};

void CheckCudnnError(cudnnStatus_t status);

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

using ConvAlgoCacheMap = std::unordered_map<ConvAlgoCacheKey, std::pair<cudnnConvolutionFwdAlgo_t, size_t>, ConvAlgoCacheKeyHash>;

class Cudnn {
public:
    explicit Cudnn(int device_index) : device_index_{device_index} {}
    ~Cudnn();

    void ConvolutionForward(
            const Array& x,
            const Array& w,
            const nonstd::optional<Array>& b,
            const Array& y,
            const StackVector<int64_t, kMaxNdim>& pad,
            const StackVector<int64_t, kMaxNdim>& stride,
            const nonstd::optional<StackVector<int64_t, kMaxNdim>>& dilation,
            int groups);

    cudnnHandle_t handle();

private:
    std::pair<cudnnConvolutionFwdAlgo_t, size_t> FindConvolutionForwardAlgorithm(
            const std::shared_ptr<cudnnTensorStruct>& x_desc,
            const Array& x,
            const std::shared_ptr<cudnnFilterStruct>& filter_desc,
            const Array& w,
            const std::shared_ptr<cudnnConvolutionStruct>& conv_desc,
            const std::shared_ptr<cudnnTensorStruct>& y_desc,
            const Array& y,
            size_t max_workspace_size,
            const StackVector<int64_t, kMaxNdim>& pad,
            const StackVector<int64_t, kMaxNdim>& stride);

    int device_index_;
    cudnnHandle_t handle_{};
    ConvAlgoCacheMap conv_fwd_algo_cache_map_{};
};

}  // namespace internal
}  // namespace cuda
}  // namespace xchainer
