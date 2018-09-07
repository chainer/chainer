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

class CudnnError : public ChainerxError {
public:
    using ChainerxError::ChainerxError;

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
            CHAINERX_NEVER_REACH();
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

// cuDNN API calls using same handle is not thread-safe.
// This class ensures that the API calls are serialized using mutex lock.
//
// This class is a thin wrapper of cuDNN APIs, we follow cuDNN naming convensions rather than ChainerX naming convensions.
class Cudnn {
public:
    explicit Cudnn(int index) : index_{index} {}
    ~Cudnn();

    void CudnnAddTensor(
            const void* alpha,
            const cudnnTensorDescriptor_t aDesc,
            const void* A,
            const void* beta,
            const cudnnTensorDescriptor_t cDesc,
            void* C) {
        std::lock_guard<std::mutex> lock{handle_mutex_};
        CheckCudnnError(cudnnAddTensor(handle(), alpha, aDesc, A, beta, cDesc, C));
    }

    void CudnnFindConvolutionForwardAlgorithmEx(
            const cudnnTensorDescriptor_t xDesc,
            const void* x,
            const cudnnFilterDescriptor_t wDesc,
            const void* w,
            const cudnnConvolutionDescriptor_t convDesc,
            const cudnnTensorDescriptor_t yDesc,
            void* y,
            const int requestedAlgoCount,
            int* returnedAlgoCount,
            cudnnConvolutionFwdAlgoPerf_t* perfResults,
            void* workSpace,
            size_t workSpaceSizeInBytes) {
        std::lock_guard<std::mutex> lock{handle_mutex_};
        CheckCudnnError(cudnnFindConvolutionForwardAlgorithmEx(
                handle(),
                xDesc,
                x,
                wDesc,
                w,
                convDesc,
                yDesc,
                y,
                requestedAlgoCount,
                returnedAlgoCount,
                perfResults,
                workSpace,
                workSpaceSizeInBytes));
    }

    void CudnnFindConvolutionBackwardDataAlgorithmEx(
            const cudnnFilterDescriptor_t wDesc,
            const void* w,
            const cudnnTensorDescriptor_t dyDesc,
            const void* dy,
            const cudnnConvolutionDescriptor_t convDesc,
            const cudnnTensorDescriptor_t dxDesc,
            void* dx,
            const int requestedAlgoCount,
            int* returnedAlgoCount,
            cudnnConvolutionBwdDataAlgoPerf_t* perfResults,
            void* workSpace,
            size_t workSpaceSizeInBytes) {
        std::lock_guard<std::mutex> lock{handle_mutex_};
        CheckCudnnError(cudnnFindConvolutionBackwardDataAlgorithmEx(
                handle(),
                wDesc,
                w,
                dyDesc,
                dy,
                convDesc,
                dxDesc,
                dx,
                requestedAlgoCount,
                returnedAlgoCount,
                perfResults,
                workSpace,
                workSpaceSizeInBytes));
    }

    void CudnnFindConvolutionBackwardFilterAlgorithmEx(
            const cudnnTensorDescriptor_t xDesc,
            const void* x,
            const cudnnTensorDescriptor_t dyDesc,
            const void* dy,
            const cudnnConvolutionDescriptor_t convDesc,
            const cudnnFilterDescriptor_t dwDesc,
            void* dw,
            const int requestedAlgoCount,
            int* returnedAlgoCount,
            cudnnConvolutionBwdFilterAlgoPerf_t* perfResults,
            void* workSpace,
            size_t workSpaceSizeInBytes) {
        std::lock_guard<std::mutex> lock{handle_mutex_};
        CheckCudnnError(cudnnFindConvolutionBackwardFilterAlgorithmEx(
                handle(),
                xDesc,
                x,
                dyDesc,
                dy,
                convDesc,
                dwDesc,
                dw,
                requestedAlgoCount,
                returnedAlgoCount,
                perfResults,
                workSpace,
                workSpaceSizeInBytes));
    }

    void CudnnConvolutionForward(
            const void* alpha,
            const cudnnTensorDescriptor_t xDesc,
            const void* x,
            const cudnnFilterDescriptor_t wDesc,
            const void* w,
            const cudnnConvolutionDescriptor_t convDesc,
            cudnnConvolutionFwdAlgo_t algo,
            void* workSpace,
            size_t workSpaceSizeInBytes,
            const void* beta,
            const cudnnTensorDescriptor_t yDesc,
            void* y) {
        std::lock_guard<std::mutex> lock{handle_mutex_};
        CheckCudnnError(cudnnConvolutionForward(
                handle(), alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y));
    }

    void CudnnConvolutionBackwardData(
            const void* alpha,
            const cudnnFilterDescriptor_t wDesc,
            const void* w,
            const cudnnTensorDescriptor_t dyDesc,
            const void* dy,
            const cudnnConvolutionDescriptor_t convDesc,
            cudnnConvolutionBwdDataAlgo_t algo,
            void* workSpace,
            size_t workSpaceSizeInBytes,
            const void* beta,
            const cudnnTensorDescriptor_t dxDesc,
            void* dx) {
        std::lock_guard<std::mutex> lock{handle_mutex_};
        CheckCudnnError(cudnnConvolutionBackwardData(
                handle(), alpha, wDesc, w, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dxDesc, dx));
    }

    void CudnnConvolutionBackwardFilter(
            const void* alpha,
            const cudnnTensorDescriptor_t xDesc,
            const void* x,
            const cudnnTensorDescriptor_t dyDesc,
            const void* dy,
            const cudnnConvolutionDescriptor_t convDesc,
            cudnnConvolutionBwdFilterAlgo_t algo,
            void* workSpace,
            size_t workSpaceSizeInBytes,
            const void* beta,
            const cudnnFilterDescriptor_t dwDesc,
            void* dw) {
        std::lock_guard<std::mutex> lock{handle_mutex_};
        CheckCudnnError(cudnnConvolutionBackwardFilter(
                handle(), alpha, xDesc, x, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dwDesc, dw));
    }

    void CudnnBatchNormalizationForwardTraining(
            cudnnBatchNormMode_t mode,
            const void* alpha,
            const void* beta,
            const cudnnTensorDescriptor_t xDesc,
            const void* x,
            const cudnnTensorDescriptor_t yDesc,
            void* y,
            const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
            const void* bnScale,
            const void* bnBias,
            double exponentialAverageFactor,
            void* resultRunningMean,
            void* resultRunningVariance,
            double epsilon,
            void* resultSaveMean,
            void* resultSaveInvVariance) {
        std::lock_guard<std::mutex> lock{handle_mutex_};
        CheckCudnnError(cudnnBatchNormalizationForwardTraining(
                handle(),
                mode,
                alpha,
                beta,
                xDesc,
                x,
                yDesc,
                y,
                bnScaleBiasMeanVarDesc,
                bnScale,
                bnBias,
                exponentialAverageFactor,
                resultRunningMean,
                resultRunningVariance,
                epsilon,
                resultSaveMean,
                resultSaveInvVariance));
    }

    void CudnnBatchNormalizationBackward(
            cudnnBatchNormMode_t mode,
            const void* alphaDataDiff,
            const void* betaDataDiff,
            const void* alphaParamDiff,
            const void* betaParamDiff,
            const cudnnTensorDescriptor_t xDesc,
            const void* x,
            const cudnnTensorDescriptor_t dyDesc,
            const void* dy,
            const cudnnTensorDescriptor_t dxDesc,
            void* dx,
            const cudnnTensorDescriptor_t bnScaleBiasDiffDesc,
            const void* bnScale,
            void* resultBnScaleDiff,
            void* resultBnBiasDiff,
            double epsilon,
            const void* savedMean,
            const void* savedInvVariance) {
        std::lock_guard<std::mutex> lock{handle_mutex_};
        CheckCudnnError(cudnnBatchNormalizationBackward(
                handle(),
                mode,
                alphaDataDiff,
                betaDataDiff,
                alphaParamDiff,
                betaParamDiff,
                xDesc,
                x,
                dyDesc,
                dy,
                dxDesc,
                dx,
                bnScaleBiasDiffDesc,
                bnScale,
                resultBnScaleDiff,
                resultBnBiasDiff,
                epsilon,
                savedMean,
                savedInvVariance));
    }

    void CudnnBatchNormalizationForwardInference(
            cudnnBatchNormMode_t mode,
            const void* alpha,
            const void* beta,
            const cudnnTensorDescriptor_t xDesc,
            const void* x,
            const cudnnTensorDescriptor_t yDesc,
            void* y,
            const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
            const void* bnScale,
            const void* bnBias,
            const void* estimatedMean,
            const void* estimatedVariance,
            double epsilon) {
        std::lock_guard<std::mutex> lock{handle_mutex_};
        CheckCudnnError(cudnnBatchNormalizationForwardInference(
                handle(),
                mode,
                alpha,
                beta,
                xDesc,
                x,
                yDesc,
                y,
                bnScaleBiasMeanVarDesc,
                bnScale,
                bnBias,
                estimatedMean,
                estimatedVariance,
                epsilon));
    }

    void CudnnPoolingForward(
            const cudnnPoolingDescriptor_t poolingDesc,
            const void* alpha,
            const cudnnTensorDescriptor_t xDesc,
            const void* x,
            const void* beta,
            const cudnnTensorDescriptor_t yDesc,
            void* y) {
        std::lock_guard<std::mutex> lock{handle_mutex_};
        CheckCudnnError(cudnnPoolingForward(handle(), poolingDesc, alpha, xDesc, x, beta, yDesc, y));
    }

    void CudnnPoolingBackward(
            const cudnnPoolingDescriptor_t poolingDesc,
            const void* alpha,
            const cudnnTensorDescriptor_t yDesc,
            const void* y,
            const cudnnTensorDescriptor_t dyDesc,
            const void* dy,
            const cudnnTensorDescriptor_t xDesc,
            const void* xData,
            const void* beta,
            const cudnnTensorDescriptor_t dxDesc,
            void* dx) {
        std::lock_guard<std::mutex> lock{handle_mutex_};
        CheckCudnnError(cudnnPoolingBackward(handle(), poolingDesc, alpha, yDesc, y, dyDesc, dy, xDesc, xData, beta, dxDesc, dx));
    }

private:
    cudnnHandle_t handle();

    int index_;
    std::mutex handle_mutex_{};
    cudnnHandle_t handle_{};
};

}  // namespace cuda_internal
}  // namespace cuda
}  // namespace chainerx
