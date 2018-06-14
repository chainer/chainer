#pragma once

#include <cudnn.h>
#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/dtype.h"
#include "xchainer/error.h"
#include "xchainer/macro.h"
#include "xchainer/stack_vector.h"

namespace xchainer {
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

namespace internal {

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

class TensorDescriptor {
public:
    explicit TensorDescriptor(const Array& arr);
    ~TensorDescriptor();

    cudnnTensorDescriptor_t descriptor() const { return desc_; }
    cudnnTensorDescriptor_t operator*() const { return desc_; }

private:
    TensorDescriptor();
    cudnnTensorDescriptor_t desc_{};
};

class FilterDescriptor {
public:
    explicit FilterDescriptor(const Array& w);
    ~FilterDescriptor();

    cudnnFilterDescriptor_t descriptor() const { return desc_; }
    cudnnFilterDescriptor_t operator*() const { return desc_; }

private:
    FilterDescriptor();
    cudnnFilterDescriptor_t desc_{};
};

class ConvolutionDescriptor {
public:
    explicit ConvolutionDescriptor(
            Dtype dtype,
            const StackVector<int64_t, kMaxNdim>& pad,
            const StackVector<int64_t, kMaxNdim>& stride,
            const nonstd::optional<StackVector<int64_t, kMaxNdim>>& dilation,
            int groups);
    ~ConvolutionDescriptor();

    cudnnConvolutionDescriptor_t descriptor() const { return desc_; }
    cudnnConvolutionDescriptor_t operator*() const { return desc_; }

private:
    ConvolutionDescriptor();
    cudnnConvolutionDescriptor_t desc_{};
};

class PoolingDescriptor {
public:
    explicit PoolingDescriptor(
            cudnnPoolingMode_t mode,
            cudnnNanPropagation_t max_pooling_nan_opt,
            const StackVector<int64_t, kMaxNdim>& kernel_size,
            const StackVector<int64_t, kMaxNdim>& pad,
            const StackVector<int64_t, kMaxNdim>& stride);
    ~PoolingDescriptor();

    cudnnPoolingDescriptor_t descriptor() const { return desc_; }
    cudnnPoolingDescriptor_t operator*() const { return desc_; }

private:
    PoolingDescriptor();
    cudnnPoolingDescriptor_t desc_{};
};

}  // namespace internal
}  // namespace cuda
}  // namespace xchainer
