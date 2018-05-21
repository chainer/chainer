#pragma once

#include <cudnn.h>

#include <memory>

#include "xchainer/array.h"
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

std::shared_ptr<cudnnTensorStruct> CreateTensorDescriptor(const Array& arr, cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW);
std::shared_ptr<cudnnFilterStruct> CreateFilterDescriptor(const Array& arr, cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW);
std::shared_ptr<cudnnConvolutionStruct> CreateConvolutionDescriptor(
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        Dtype dtype,
        cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION,
        const nonstd::optional<StackVector<int64_t, kMaxNdim>>& dilation = nonstd::nullopt,
        bool use_tensor_core = false,
        int groups = 1);

}  // namespace internal
}  // namespace cuda
}  // namespace xchainer
