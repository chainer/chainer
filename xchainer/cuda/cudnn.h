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

using ConvFwdAlgoCacheMap = std::unordered_map<ConvAlgoCacheKey, std::pair<cudnnConvolutionFwdAlgo_t, size_t>, ConvAlgoCacheKeyHash>;
using ConvBwdDataAlgoCacheMap =
        std::unordered_map<ConvAlgoCacheKey, std::pair<cudnnConvolutionBwdDataAlgo_t, size_t>, ConvAlgoCacheKeyHash>;
using ConvBwdFilterAlgoCacheMap =
        std::unordered_map<ConvAlgoCacheKey, std::pair<cudnnConvolutionBwdFilterAlgo_t, size_t>, ConvAlgoCacheKeyHash>;

class CudnnContext {
public:
    explicit CudnnContext(int device_index);
    ~CudnnContext();

    void ConvolutionForward(
            const Array& x,
            const Array& w,
            const nonstd::optional<Array>& b,
            const Array& y,
            const StackVector<int64_t, kMaxNdim>& pad,
            const StackVector<int64_t, kMaxNdim>& stride,
            const nonstd::optional<StackVector<int64_t, kMaxNdim>>& dilation,
            int groups);
    void ConvolutionBackwardData(
            const Array& w,
            const Array& x,
            const nonstd::optional<Array>& b,
            const Array& y,
            const StackVector<int64_t, kMaxNdim>& pad,
            const StackVector<int64_t, kMaxNdim>& stride,
            const nonstd::optional<StackVector<int64_t, kMaxNdim>>& dilation,
            int groups);
    void ConvolutionBackwardFilter(
            const Array& x,
            const Array& gy,
            const Array& gw,
            const StackVector<int64_t, kMaxNdim>& pad,
            const StackVector<int64_t, kMaxNdim>& stride,
            const nonstd::optional<StackVector<int64_t, kMaxNdim>>& dilation,
            int groups);

    cudnnHandle_t handle() { return handle_; }

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
    std::pair<cudnnConvolutionBwdDataAlgo_t, size_t> FindConvolutionBackwardDataAlgorithm(
            const std::shared_ptr<cudnnFilterStruct>& filter_desc,
            const Array& w,
            const std::shared_ptr<cudnnTensorStruct>& x_desc,
            const Array& x,
            const std::shared_ptr<cudnnConvolutionStruct>& conv_desc,
            const std::shared_ptr<cudnnTensorStruct>& y_desc,
            const Array& y,
            size_t max_workspace_size,
            const StackVector<int64_t, kMaxNdim>& pad,
            const StackVector<int64_t, kMaxNdim>& stride);
    std::pair<cudnnConvolutionBwdFilterAlgo_t, size_t> FindConvolutionBackwardFilterAlgorithm(
            const std::shared_ptr<cudnnTensorStruct>& x_desc,
            const Array& x,
            const std::shared_ptr<cudnnTensorStruct>& gy_desc,
            const Array& gy,
            const std::shared_ptr<cudnnConvolutionStruct>& conv_desc,
            const std::shared_ptr<cudnnFilterStruct>& gw_desc,
            const Array& gw,
            size_t max_workspace_size,
            const StackVector<int64_t, kMaxNdim>& pad,
            const StackVector<int64_t, kMaxNdim>& stride);
    void AddBias(const std::shared_ptr<cudnnTensorStruct>& y_desc, const Array& y, const Array& b);

    int device_index_;
    cudnnHandle_t handle_{};
    ConvFwdAlgoCacheMap conv_fwd_algo_cache_map_{};
    ConvBwdDataAlgoCacheMap conv_bwd_data_algo_cache_map_{};
    ConvBwdFilterAlgoCacheMap conv_bwd_filter_algo_cache_map_{};
};

}  // namespace internal
}  // namespace cuda
}  // namespace xchainer
