#pragma once

#include <cudnn.h>
#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/macro.h"
#include "chainerx/stack_vector.h"

namespace chainerx {
namespace cuda {

class CudnnError : public XchainerError {
public:
    using XchainerError::XchainerError;

    explicit CudnnError(cudnnStatus_t status);
    cudnnStatus_t error() const noexcept { return status_; }

private:
    cudnnStatus_t status_;
};

void CheckCudnnError(cudnnStatus_t status);

namespace cuda_internal {

// Returns a pointer to a value of given type, allocated on the static storage.
template <int kValue>
const void* GetValuePtr(Dtype dtype) {
    static const float kFloat32Value = kValue;
    static const double kFloat64Value = kValue;

    switch (dtype) {
        case Dtype::kFloat64:
            return &kFloat64Value;
        case Dtype::kFloat32:
            return &kFloat32Value;
        default:
            XCHAINER_NEVER_REACH();
    }
}

class CudnnTensorDescriptor {
public:
    explicit CudnnTensorDescriptor(const Array& arr);
    ~CudnnTensorDescriptor();

    cudnnTensorDescriptor_t descriptor() const { return desc_; }
    cudnnTensorDescriptor_t operator*() const { return desc_; }

private:
    CudnnTensorDescriptor();
    cudnnTensorDescriptor_t desc_{};
};

class CudnnFilterDescriptor {
public:
    explicit CudnnFilterDescriptor(const Array& w);
    ~CudnnFilterDescriptor();

    cudnnFilterDescriptor_t descriptor() const { return desc_; }
    cudnnFilterDescriptor_t operator*() const { return desc_; }

private:
    CudnnFilterDescriptor();
    cudnnFilterDescriptor_t desc_{};
};

class CudnnConvolutionDescriptor {
public:
    explicit CudnnConvolutionDescriptor(
            Dtype dtype,
            const StackVector<int64_t, kMaxNdim>& pad,
            const StackVector<int64_t, kMaxNdim>& stride,
            const nonstd::optional<StackVector<int64_t, kMaxNdim>>& dilation,
            int groups);
    ~CudnnConvolutionDescriptor();

    cudnnConvolutionDescriptor_t descriptor() const { return desc_; }
    cudnnConvolutionDescriptor_t operator*() const { return desc_; }

private:
    CudnnConvolutionDescriptor();
    cudnnConvolutionDescriptor_t desc_{};
};

class CudnnPoolingDescriptor {
public:
    explicit CudnnPoolingDescriptor(
            cudnnPoolingMode_t mode,
            cudnnNanPropagation_t max_pooling_nan_opt,
            const StackVector<int64_t, kMaxNdim>& kernel_size,
            const StackVector<int64_t, kMaxNdim>& pad,
            const StackVector<int64_t, kMaxNdim>& stride);
    ~CudnnPoolingDescriptor();

    cudnnPoolingDescriptor_t descriptor() const { return desc_; }
    cudnnPoolingDescriptor_t operator*() const { return desc_; }

private:
    CudnnPoolingDescriptor();
    cudnnPoolingDescriptor_t desc_{};
};

}  // namespace cuda_internal
}  // namespace cuda
}  // namespace chainerx
