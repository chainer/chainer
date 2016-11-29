// This file is a stub header file of cudnn for Read the Docs.

#ifndef INCLUDE_GUARD_CUPY_CUDNN_H
#define INCLUDE_GUARD_CUPY_CUDNN_H

#include "cupy_cuda.h"

#ifndef CUPY_NO_CUDA
#include <cudnn.h>

extern "C" {

#if CUDNN_VERSION < 3000

// ***_v3 functions are not declared in cuDNN v4.
// Following definitions are for compatibility with cuDNN v2 and v3.

typedef int cudnnConvolutionBwdDataAlgo_t;
typedef int cudnnConvolutionBwdDataPreference_t;
typedef int cudnnConvolutionBwdFilterAlgo_t;
typedef int cudnnConvolutionBwdFilterPreference_t;

cudnnStatus_t cudnnAddTensor_v3(
        cudnnHandle_t handle, const void* alpha,
        const cudnnTensorDescriptor_t bDesc, const void* b, const void* beta,
        cudnnTensorDescriptor_t yDesc, void* y) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnConvolutionBackwardFilter_v3(
        cudnnHandle_t handle, const void* alpha,
        const cudnnTensorDescriptor_t xDesc, const void* x,
        const cudnnTensorDescriptor_t dyDesc, const void* dy,
        const cudnnConvolutionDescriptor_t convDesc,
        cudnnConvolutionBwdFilterAlgo_t algo,
        void* workSpace, size_t workSpaceSizeInBytes, const void* beta,
        const cudnnFilterDescriptor_t dwDesc, void* dw) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize(
        cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
        const cudnnTensorDescriptor_t dyDesc,
        const cudnnConvolutionDescriptor_t convDes,
        const cudnnFilterDescriptor_t gradDes,
        cudnnConvolutionBwdFilterAlgo_t algo, size_t* sizeInBytes) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm(
        cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
        const cudnnTensorDescriptor_t dyDesc,
        const cudnnConvolutionDescriptor_t convDesc,
        const cudnnFilterDescriptor_t dwDesc,
        cudnnConvolutionBwdFilterPreference_t preference,
        size_t memoryLimitInBytes, cudnnConvolutionBwdFilterAlgo_t* algo) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnConvolutionBackwardData_v3(
        cudnnHandle_t handle, const void* alpha,
        const cudnnFilterDescriptor_t wDesc, const void* w,
        const cudnnTensorDescriptor_t dyDesc, const void* dy,
        const cudnnConvolutionDescriptor_t convDesc,
        cudnnConvolutionBwdDataAlgo_t algo, void* workSpace,
        size_t workSpaceSizeInBytes, const void* beta,
        const cudnnTensorDescriptor_t dxDesc, void* dx) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm(
        cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc,
        const cudnnTensorDescriptor_t dyDesc,
        const cudnnConvolutionDescriptor_t convDesc,
        const cudnnTensorDescriptor_t dxDesc,
        cudnnConvolutionBwdDataPreference_t preference,
        size_t memoryLimitInBytes, cudnnConvolutionBwdDataAlgo_t* algo) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc,
        const cudnnTensorDescriptor_t dyDesc,
        const cudnnConvolutionDescriptor_t convDesc,
        const cudnnTensorDescriptor_t dxDesc,
        cudnnConvolutionBwdDataAlgo_t algo, size_t* sizeInBytes) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnSetConvolutionNdDescriptor_v3(
        cudnnConvolutionDescriptor_t convDesc, int arrayLength,
        const int padA[], const int filterStrideA[], const int upscaleA[],
        cudnnConvolutionMode_t mode, cudnnDataType_t dataType) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

#endif // CUDNN_VERSION < 3000

#if CUDNN_VERSION < 4000

// ***_v2 functions are not declared in cuDNN v2 and v3.
// Following definitions are for compatibility with cuDNN v4.

#define cudnnAddTensor_v2 cudnnAddTensor
#define cudnnConvolutionBackwardData_v2 cudnnConvolutionBackwardData
#define cudnnConvolutionBackwardFilter_v2 cudnnConvolutionBackwardFilter
#define cudnnSetConvolutionNdDescriptor_v2 cudnnSetConvolutionNdDescriptor

typedef int cudnnBatchNormMode_t;

cudnnStatus_t cudnnDeriveBNTensorDescriptor(
         cudnnTensorDescriptor_t derivedBnDesc,
         const cudnnTensorDescriptor_t xDesc,
         cudnnBatchNormMode_t mode) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}


cudnnStatus_t cudnnBatchNormalizationForwardTraining(
        cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alpha,
        const void *beta, const cudnnTensorDescriptor_t xDesc, const void *x,
        const cudnnTensorDescriptor_t yDesc, void *y,
        const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
        const void *bnScale, const void *bnBias,
        double exponentialAverageFactor, void *resultRunningMean,
        void *resultRunningVariance, double epsilon, void *resultSaveMean,
        void *resultSaveInvVariance) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnBatchNormalizationForwardInference(
        cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alpha,
        const void *beta, const cudnnTensorDescriptor_t xDesc, const void *x,
        const cudnnTensorDescriptor_t yDesc, void *y,
        const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
        const void *bnScale, const void *bnBias, const void *estimatedMean,
        const void *estimatedVariance, double epsilon) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnBatchNormalizationBackward(
        cudnnHandle_t handle, cudnnBatchNormMode_t mode,
        const void *alphaDataDiff, const void *betaDataDiff,
        const void *alphaParamDiff, const void *betaParamDiff,
        const cudnnTensorDescriptor_t xDesc, const void *x,
        const cudnnTensorDescriptor_t dyDesc, const void *dy,
        const cudnnTensorDescriptor_t dxDesc, void *dx,
        const cudnnTensorDescriptor_t dBnScaleBiasDesc, const void *bnScale,
        void *dBnScaleResult, void *dBnBiasResult, double epsilon,
        const void *savedMean, const void *savedInvVariance) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}


#endif // #if CUDNN_VERSION < 4000

#if CUDNN_VERSION < 5000

// ***_v3 functions are not declared in cuDNN v2, v3 and v4.
// Following definitions are for compatibility with cuDNN v5.

#define cudnnActivationForward_v3 cudnnActivationForward
#define cudnnActivationBackward_v3 cudnnActivationBackward
#define cudnnSetFilter4dDescriptor_v3 cudnnSetFilter4dDescriptor
#define cudnnSetFilterNdDescriptor_v3 cudnnSetFilterNdDescriptor
#define cudnnSetPooling2dDescriptor_v3 cudnnSetPooling2dDescriptor
#define cudnnSetPoolingNdDescriptor_v3 cudnnSetPoolingNdDescriptor

cudnnStatus_t cudnnGetFilterNdDescriptor_v5(
        const cudnnFilterDescriptor_t filterDesc, int nbDimsRequested,
        cudnnDataType_t* dataType, cudnnTensorFormat_t* format, int* nbDims,
        int filterDimA[]) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

typedef void* cudnnDropoutDescriptor_t;

cudnnStatus_t cudnnCreateDropoutDescriptor(
        cudnnDropoutDescriptor_t* dropoutDesc) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnDestroyDropoutDescriptor(
        cudnnDropoutDescriptor_t dropoutDesc) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnDropoutGetStatesSize(
        cudnnHandle_t handle, size_t * sizeInBytes) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnDropoutGetReserveSpaceSize(
        cudnnTensorDescriptor_t xdesc, size_t * sizeInBytes) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnSetDropoutDescriptor(
        cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle,
        float dropout, void* states, size_t stateSizeInBytes,
        unsigned long long seed) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnDropoutForward(
        cudnnHandle_t handle, const cudnnDropoutDescriptor_t dropoutDesc,
        const cudnnTensorDescriptor_t xdesc, const void* x,
        const cudnnTensorDescriptor_t ydesc, void* y,
        void* reserveSpace, size_t reserveSpaceSizeInBytes) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnDropoutBackward(
        cudnnHandle_t handle, const cudnnDropoutDescriptor_t dropoutDesc,
        const cudnnTensorDescriptor_t dydesc, const void* dy,
        const cudnnTensorDescriptor_t dxdesc, void* dx,
        void* reserveSpace, size_t reserveSpaceSizeInBytes) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

typedef int cudnnRNNMode_t;
typedef int cudnnDirectionMode_t;
typedef int cudnnRNNInputMode_t;
typedef void* cudnnRNNDescriptor_t;

cudnnStatus_t cudnnCreateRNNDescriptor(cudnnRNNDescriptor_t* rnnDesc) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t rnnDesc) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnSetRNNDescriptor(
        cudnnRNNDescriptor_t rnnDesc, int hiddenSize, int numLayers,
        cudnnDropoutDescriptor_t dropoutDesc, cudnnRNNInputMode_t inputMode,
        cudnnDirectionMode_t direction, cudnnRNNMode_t mode, 
        cudnnDataType_t dataType) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetRNNWorkspaceSize(
        cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
        const int seqLength, const cudnnTensorDescriptor_t* xDesc,
        size_t* sizeInBytes) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}
                                                      
cudnnStatus_t cudnnGetRNNTrainingReserveSize(
        cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,  
        const int seqLength, const cudnnTensorDescriptor_t* xDesc,
        size_t* sizeInBytes) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}
                                                    
cudnnStatus_t cudnnGetRNNParamsSize(
        cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
        const cudnnTensorDescriptor_t xDesc, size_t* sizeInBytes,
        cudnnDataType_t dataType) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetRNNLinLayerMatrixParams(
        cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
        const int layer, const cudnnTensorDescriptor_t xDesc,
        const cudnnFilterDescriptor_t wDesc, const void* w,
        const int linLayerID, cudnnFilterDescriptor_t linLayerMatDesc, 
        void** linLayerMat) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetRNNLinLayerBiasParams(
        cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
        const int layer, const cudnnTensorDescriptor_t xDesc,
        const cudnnFilterDescriptor_t wDesc, const void* w,
        const int linLayerID, cudnnFilterDescriptor_t linLayerBiasDesc,
        void** linLayerBias) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnRNNForwardInference(
        cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, 
        const int seqLength, 
        const cudnnTensorDescriptor_t* xDesc, const void* x,
        const cudnnTensorDescriptor_t hxDesc, const void* hx,
        const cudnnTensorDescriptor_t cxDesc, const void* cx,
        const cudnnFilterDescriptor_t wDesc, const void* w,
        const cudnnTensorDescriptor_t* yDesc, void* y,
        const cudnnTensorDescriptor_t hyDesc, void* hy,
        const cudnnTensorDescriptor_t cyDesc, void* cy,
        void* workspace, size_t workSpaceSizeInBytes) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnRNNForwardTraining(
        cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, 
        const int seqLength, 
        const cudnnTensorDescriptor_t* xDesc, const void* x, 
        const cudnnTensorDescriptor_t hxDesc, const void* hx, 
        const cudnnTensorDescriptor_t cxDesc, const void* cx, 
        const cudnnFilterDescriptor_t wDesc, const void* w, 
        const cudnnTensorDescriptor_t* yDesc, void* y, 
        const cudnnTensorDescriptor_t hyDesc, void* hy, 
        const cudnnTensorDescriptor_t cyDesc, void* cy, 
        void* workspace, size_t workSpaceSizeInBytes,
        void* reserveSpace, size_t reserveSpaceSizeInBytes) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnRNNBackwardData(
        cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
        const int seqLength,
        const cudnnTensorDescriptor_t* yDesc, const void * y,
        const cudnnTensorDescriptor_t* dyDesc, const void * dy,
        const cudnnTensorDescriptor_t dhyDesc, const void * dhy, 
        const cudnnTensorDescriptor_t dcyDesc, const void * dcy, 
        const cudnnFilterDescriptor_t wDesc, const void * w, 
        const cudnnTensorDescriptor_t hxDesc, const void * hx,
        const cudnnTensorDescriptor_t cxDesc, const void * cx,
        const cudnnTensorDescriptor_t* dxDesc, void* dx,
        const cudnnTensorDescriptor_t dhxDesc, void* dhx,
        const cudnnTensorDescriptor_t dcxDesc, void* dcx,
        void * workspace, size_t workSpaceSizeInBytes,
        const void* reserveSpace, size_t reserveSpaceSizeInBytes) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnRNNBackwardWeights(
         cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
         const int seqLength, const cudnnTensorDescriptor_t* xDesc,
         const void* x, const cudnnTensorDescriptor_t hxDesc,
         const void* hx, const cudnnTensorDescriptor_t* yDesc,
         const void* y, const void* workspace, size_t workSpaceSizeInBytes, 
         const cudnnFilterDescriptor_t dwDesc, void* dw,
         const void* reserveSpace, size_t reserveSpaceSizeInBytes) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}


#endif // #if CUDNN_VERSION < 5000

#if CUDNN_VERSION >= 5000

#define cudnnAddTensor_v3 cudnnAddTensor
#define cudnnConvolutionBackwardData_v3 cudnnConvolutionBackwardData
#define cudnnConvolutionBackwardFilter_v3 cudnnConvolutionBackwardFilter
#define cudnnSetConvolutionNdDescriptor_v3 cudnnSetConvolutionNdDescriptor

#define cudnnGetFilterNdDescriptor_v5 cudnnGetFilterNdDescriptor

typedef enum {} cudnnAddMode_t;


cudnnStatus_t cudnnSetConvolutionNdDescriptor_v2(
        cudnnConvolutionDescriptor_t convDesc, int arrayLength,
        const int padA[], const int filterStrideA[], const int upscaleA[],
        cudnnConvolutionMode_t mode) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetConvolutionNdDescriptor_v2(
        const cudnnConvolutionDescriptor_t convDesc, int arrayLengthRequested,
        int* arrayLength, int padA[], int strideA[], int upscaleA[],
        cudnnConvolutionMode_t* mode) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnAddTensor_v2(
        cudnnHandle_t handle, cudnnAddMode_t mode, const void* alpha,
        const cudnnTensorDescriptor_t biasDesc, const void* biasData,
        const void* beta, cudnnTensorDescriptor_t srcDestDesc,
        void* srcDestData) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnConvolutionBackwardFilter_v2(
        cudnnHandle_t handle, const void* alpha,
        const cudnnTensorDescriptor_t srcDesc, const void* srcData,
        const cudnnTensorDescriptor_t diffDesc, const void* diffData,
        const cudnnConvolutionDescriptor_t convDesc, const void* beta,
        const cudnnFilterDescriptor_t gradDesc, void* gradData) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnConvolutionBackwardData_v2(
        cudnnHandle_t handle, const void* alpha,
        const cudnnFilterDescriptor_t filterDesc, const void* filterData,
        const cudnnTensorDescriptor_t diffDesc, const void* diffData,
        const cudnnConvolutionDescriptor_t convDesc, const void* beta,
        const cudnnTensorDescriptor_t gradDesc, void *gradData) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}


#endif // CUDNN_VERSION >= 5000

} // extern "C"

#else // #ifndef CUPY_NO_CUDA

extern "C" {

///////////////////////////////////////////////////////////////////////////////
// cuda.h
///////////////////////////////////////////////////////////////////////////////

typedef int cudnnActivationMode_t;
typedef int cudnnAddMode_t;
typedef int cudnnBatchNormMode_t;
typedef int cudnnConvolutionBwdDataAlgo_t;
typedef int cudnnConvolutionBwdDataPreference_t;
typedef int cudnnConvolutionBwdFilterAlgo_t;
typedef int cudnnConvolutionBwdFilterPreference_t;
typedef int cudnnConvolutionFwdAlgo_t;
typedef int cudnnConvolutionFwdPreference_t;
typedef int cudnnConvolutionMode_t;
typedef int cudnnDataType_t;
typedef int cudnnDirectionMode_t;
typedef int cudnnNanPropagation_t;
typedef int cudnnPoolingMode_t;
typedef int cudnnRNNInputMode_t;
typedef int cudnnRNNMode_t;
typedef int cudnnSoftmaxAlgorithm_t;
typedef int cudnnSoftmaxMode_t;
typedef int cudnnStatus_t;
typedef int cudnnTensorFormat_t;


typedef int ActivationMode;
typedef int AddMode;
typedef int BatchNormMode;
typedef int ConvolutionBwdDataAlgo;
typedef int ConvolutionBwdDataPreference;
typedef int ConvolutionBwdFilterAlgo;
typedef int ConvolutionBwdFilterPreference;
typedef int ConvolutionFwdAlgo;
typedef int ConvolutionFwdPreference;
typedef int ConvolutionMode;
typedef int DataType;
typedef int DirectionMode;
typedef int PoolingMode;
typedef int RNNInputMode;
typedef int SoftmaxAlgorithm;
typedef int SoftmaxMode;
typedef int Status;
typedef int TensorFormat;


typedef void* cudnnConvolutionDescriptor_t;
typedef void* cudnnDropoutDescriptor_t;
typedef void* cudnnFilterDescriptor_t;
typedef void* cudnnHandle_t;
typedef void* cudnnPoolingDescriptor_t;
typedef void* cudnnRNNDescriptor_t;
typedef void* cudnnTensorDescriptor_t;

typedef void* ConvolutionDescriptor;
typedef void* FilterDescriptor;
typedef void* Handle;
typedef void* PoolingDescriptor;
typedef void* TensorDescriptor;


// Error handling
const char* cudnnGetErrorString(Status status) {
    return NULL;
}

// Version
size_t cudnnGetVersion() {
    return 0;
}

// Initialization and CUDA cooperation
int cudnnCreate(Handle* handle) {
    return 0;
}

int cudnnDestroy(Handle handle) {
    return 0;
}

int cudnnSetStream(Handle handle, cudaStream_t stream) {
    return 0;
}

int cudnnGetStream(Handle handle, cudaStream_t* stream) {
    return 0;
}


// Tensor manipulation
int cudnnCreateTensorDescriptor(TensorDescriptor* descriptor) {
    return 0;
}

int cudnnSetTensor4dDescriptor(
        TensorDescriptor tensorDesc, TensorFormat format,
        DataType dataType, int n, int c, int h, int w) {
    return 0;
}

int cudnnSetTensor4dDescriptorEx(
        TensorDescriptor tensorDesc, DataType dataType,
        int n, int c, int h, int w,
        int nStride, int cStride, int hStride, int wStride) {
    return 0;
}

int cudnnSetTensorNdDescriptor(
        TensorDescriptor tensorDesc, DataType dataType, int nbDims,
        int* dimA, int* strideA) {
    return 0;
}

int cudnnDestroyTensorDescriptor(TensorDescriptor tensorDesc) {
    return 0;
}

int cudnnAddTensor_v2(
        Handle handle, AddMode mode, void* alpha,
        TensorDescriptor biasDesc, void* biasData, void* beta,
        TensorDescriptor srcDestDesc, void* srcDestData) {
    return 0;
}

int cudnnAddTensor_v3(
        Handle handle, void* alpha, TensorDescriptor bDesc,
        void* b, void* beta, TensorDescriptor yDesc, void* y) {
    return 0;
}

// Filter manipulation
int cudnnCreateFilterDescriptor(FilterDescriptor* filterDesc) {
    return 0;
}

int cudnnSetFilter4dDescriptor_v3(
        FilterDescriptor filterDesc, DataType dataType,
        int n, int c, int h, int w) {
    return 0;
}

int cudnnSetFilterNdDescriptor_v3(
        FilterDescriptor filterDesc, DataType dataType, int nbDims,
        int* filterDimA) {
    return 0;
}


cudnnStatus_t cudnnGetFilterNdDescriptor_v5(
        const cudnnFilterDescriptor_t filterDesc, int nbDimsRequested,
        cudnnDataType_t* dataType, cudnnTensorFormat_t* format, int* nbDims,
        int filterDimA[]) {
    return 0;
}

int cudnnDestroyFilterDescriptor(FilterDescriptor filterDesc) {
    return 0;
}


// Convolution
int cudnnCreateConvolutionDescriptor(ConvolutionDescriptor* convDesc) {
    return 0;
}

int cudnnSetConvolution2dDescriptor(
        ConvolutionDescriptor convDesc, int pad_h, int pad_w, int u,
        int v, int upscalex, int upscaley, ConvolutionMode mode) {
    return 0;
}

int cudnnSetConvolutionNdDescriptor_v2(
        ConvolutionDescriptor convDesc, int arrayLength, int* padA,
        int* filterStrideA, int* upscaleA, ConvolutionMode mode) {
    return 0;
}

int cudnnSetConvolutionNdDescriptor_v3(
        ConvolutionDescriptor convDesc, int arrayLength, int* padA,
        int* filterStrideA, int* upscaleA, ConvolutionMode mode,
        DataType dataType) {
    return 0;
}

int cudnnDestroyConvolutionDescriptor(ConvolutionDescriptor conDesc) {
    return 0;
}

int cudnnGetConvolutionForwardAlgorithm(
        Handle handle, TensorDescriptor srcDesc,
        FilterDescriptor filterDesc, ConvolutionDescriptor convDesc,
        TensorDescriptor destDesc, ConvolutionFwdPreference preference,
        size_t memoryLimitInbytes, ConvolutionFwdAlgo* algo) {
    return 0;
}

int cudnnGetConvolutionForwardWorkspaceSize(
        Handle handle, TensorDescriptor srcDesc,
        FilterDescriptor filterDesc, ConvolutionDescriptor convDesc,
        TensorDescriptor destDesc, ConvolutionFwdAlgo algo,
        size_t* sizeInBytes) {
    return 0;
}

int cudnnConvolutionForward(
        Handle handle, void* alpha, TensorDescriptor srcDesc,
        void* srcData, FilterDescriptor filterDesc, void* filterData,
        ConvolutionDescriptor convDesc, ConvolutionFwdAlgo algo,
        void* workSpace, size_t workSpaceSizeInBytes, void* beta,
        TensorDescriptor destDesc, void* destData) {
    return 0;
}

int cudnnConvolutionBackwardBias(
        Handle handle, void* alpha,
        TensorDescriptor srcDesc, void* srcData, void* beta,
        TensorDescriptor destDesc, void* destData) {
    return 0;
}

int cudnnGetConvolutionBackwardFilterAlgorithm(
        Handle handle, TensorDescriptor srcDesc,
        FilterDescriptor filterDesc, ConvolutionDescriptor convDesc,
        TensorDescriptor destDesc, ConvolutionBwdFilterPreference preference,
        size_t memoryLimitInbytes, ConvolutionBwdFilterAlgo* algo) {
    return 0;
}

int cudnnGetConvolutionBackwardFilterWorkspaceSize(
        Handle handle, TensorDescriptor srcDesc,
        FilterDescriptor filterDesc, ConvolutionDescriptor convDesc,
        FilterDescriptor destDesc, ConvolutionBwdFilterAlgo algo,
        size_t* sizeInBytes) {
    return 0;
}

int cudnnConvolutionBackwardFilter_v2(
        Handle handle, void* alpha,
        TensorDescriptor srcDesc, void* srcData,
        TensorDescriptor diffDesc, void* diffData,
        ConvolutionDescriptor convDesc, void* beta,
        FilterDescriptor gradDesc, void* gradData) {
    return 0;
}

int cudnnConvolutionBackwardFilter_v3(
         Handle handle, void* alpha,
         TensorDescriptor srcDesc, void* srcData,
         TensorDescriptor diffDesc, void* diffData,
         ConvolutionDescriptor convDesc, ConvolutionBwdFilterAlgo algo,
         void* workSpace, size_t workSpaceSizeInBytes, void* beta,
         FilterDescriptor gradDesc, void* gradData) {
     return 0;
 }

int cudnnGetConvolutionBackwardDataAlgorithm(
        Handle handle, FilterDescriptor filterDesc,
        TensorDescriptor diffDesc, ConvolutionDescriptor convDesc,
        TensorDescriptor gradDesc, ConvolutionBwdDataPreference preference,
        size_t memoryLimitInbytes, ConvolutionBwdDataAlgo* algo) {
    return 0;
}

int cudnnGetConvolutionBackwardDataWorkspaceSize(
        Handle handle, FilterDescriptor filterDesc,
        TensorDescriptor diffDesc, ConvolutionDescriptor convDesc,
        TensorDescriptor gradDesc, ConvolutionBwdDataAlgo algo,
        size_t* sizeInBytes) {
    return 0;
}

int cudnnConvolutionBackwardData_v2(
        Handle handle, void* alpha,
        FilterDescriptor filterDesc, void* filterData,
        TensorDescriptor diffDesc, void* diffData,
        ConvolutionDescriptor convDesc, void* beta,
        TensorDescriptor gradDesc, void* gradData) {
    return 0;
}

int cudnnConvolutionBackwardData_v3(
         Handle handle, void* alpha,
         FilterDescriptor filterDesc, void* filterData,
         TensorDescriptor diffDesc, void* diffData,
         ConvolutionDescriptor convDesc, ConvolutionBwdDataAlgo algo,
         void* workSpace, size_t workSpaceSizeInBytes, void* beta,
         TensorDescriptor gradDesc, void* gradData) {
     return 0;
 }

// Pooling
int cudnnCreatePoolingDescriptor(PoolingDescriptor* desc) {
    return 0;
}

int cudnnSetPooling2dDescriptor_v3(
        PoolingDescriptor poolingDesc, PoolingMode mode,
        int windowHeight, int windowWidth,
        int verticalPadding, int horizontalPadding,
        int verticalStride, int horizontalStride) {
    return 0;
}

int cudnnSetPoolingNdDescriptor_v3(
        PoolingDescriptor poolingDesc, PoolingMode mode, int nbDims,
        int* windowDimA, int* paddingA, int* strideA) {
    return 0;
}

int cudnnDestroyPoolingDescriptor(PoolingDescriptor poolingDesc) {
    return 0;
}

int cudnnPoolingForward(
        Handle handle, PoolingDescriptor poolingDesc, void* alpha,
        TensorDescriptor srcDesc, void* srcData, void* beta,
        TensorDescriptor dstDesc, void* dstData) {
    return 0;
}

int cudnnPoolingBackward(
        Handle handle, PoolingDescriptor poolingDesc, void* alpha,
        TensorDescriptor srcDesc, void* srcData,
        TensorDescriptor srcDiffDesc, void* srcDiffData,
        TensorDescriptor destDesc, void* destData, void* beta,
        TensorDescriptor destDiffDesc, void* destDiffData) {
    return 0;
}


// Activation
int cudnnSoftmaxForward(
        Handle handle, SoftmaxAlgorithm algorithm, SoftmaxMode mode,
        void* alpha, TensorDescriptor srcDesc, void* srcData,
        void* beta, TensorDescriptor dstDesc, void* dstData) {
    return 0;
}

int cudnnSoftmaxBackward(
        Handle handle, SoftmaxAlgorithm algorithm, SoftmaxMode mode,
        void* alpha, TensorDescriptor srcDesc, void* srcData,
        TensorDescriptor srcDiffDesc, void* srcDiffData, void* beta,
        TensorDescriptor destDiffDesc, void* destDiffData) {
    return 0;
}

int cudnnActivationForward_v3(
        Handle handle, ActivationMode mode, void* alpha,
        TensorDescriptor srcDesc, void* srcData, void* beta,
        TensorDescriptor dstDesc, void* dstData) {
    return 0;
}

int cudnnActivationBackward_v3(
        Handle handle, ActivationMode mode, void* alpha,
        TensorDescriptor srcDesc, void* srcData,
        TensorDescriptor srcDiffDesc, void* srcDiffData,
        TensorDescriptor destDesc, void* destData, void* beta,
        TensorDescriptor destDiffDesc, void* destDiffData) {
    return 0;
}

// Batch normalization
int cudnnDeriveBNTensorDescriptor(
    	TensorDescriptor derivedBnDesc, TensorDescriptor xDesc,
	    BatchNormMode mode) {
    return 0;
}

int cudnnBatchNormalizationForwardTraining(
        Handle handle, BatchNormMode mode,
	void* alpha, void* beta, TensorDescriptor xDesc,
	void* x, TensorDescriptor yDesc, void* y,
	TensorDescriptor bnScaleBiasMeanVarDesc, void* bnScale,
	void* bnBias, double exponentialAverageFactor,
	void* resultRunningMean, void* resultRunningVariance,
	double epsilon, void* resultSaveMean, void* resultSaveInvVariance) {
    return 0;
}

int cudnnBatchNormalizationForwardInference(
        Handle handle, BatchNormMode mode,
	void* alpha, void* beta, TensorDescriptor xDesc,
	void* x, TensorDescriptor yDesc, void* y,
	TensorDescriptor bnScaleBiasMeanVarDesc, void* bnScale,
	void* bnBias, void* estimatedMean, void* estimatedVariance,
	double epsilon) {
    return 0;
}

int cudnnBatchNormalizationBackward(
        Handle handle, BatchNormMode mode,
	void* alphaDataDiff, void* betaDataDiff,
	void* alphaParamDiff, void* betaParamDiff,
	TensorDescriptor xDesc, void* x,
	TensorDescriptor dyDesc, void* dy,
	TensorDescriptor dxDesc, void* dx,
	TensorDescriptor dBnScaleBiasDesc, void* bnScale,
	void* dBnScaleResult, void* dBnBiasResult,
	double epsilon, void* savedMean, void* savedInvVariance) {
    return 0;
}


cudnnStatus_t cudnnCreateDropoutDescriptor(
        cudnnDropoutDescriptor_t* dropoutDesc) {
    return 0;
}

cudnnStatus_t cudnnDestroyDropoutDescriptor(
        cudnnDropoutDescriptor_t dropoutDesc) {
    return 0;
}

cudnnStatus_t cudnnDropoutGetStatesSize(
        cudnnHandle_t handle, size_t * sizeInBytes) {
    return 0;
}

cudnnStatus_t cudnnDropoutGetReserveSpaceSize(
        cudnnTensorDescriptor_t xdesc, size_t * sizeInBytes) {
    return 0;
}

cudnnStatus_t cudnnSetDropoutDescriptor(
        cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle,
        float dropout, void* states, size_t stateSizeInBytes,
        unsigned long long seed) {
    return 0;
}

cudnnStatus_t cudnnDropoutForward(
        cudnnHandle_t handle, const cudnnDropoutDescriptor_t dropoutDesc,
        const cudnnTensorDescriptor_t xdesc, const void* x,
        const cudnnTensorDescriptor_t ydesc, void* y,
        void* reserveSpace, size_t reserveSpaceSizeInBytes) {
    return 0;
}

cudnnStatus_t cudnnDropoutBackward(
        cudnnHandle_t handle, const cudnnDropoutDescriptor_t dropoutDesc,
        const cudnnTensorDescriptor_t dydesc, const void* dy,
        const cudnnTensorDescriptor_t dxdesc, void* dx,
        void* reserveSpace, size_t reserveSpaceSizeInBytes) {
    return 0;
}

cudnnStatus_t cudnnCreateRNNDescriptor(cudnnRNNDescriptor_t* rnnDesc) {
    return 0;
}

cudnnStatus_t cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t rnnDesc) {
    return 0;
}

cudnnStatus_t cudnnSetRNNDescriptor(
        cudnnRNNDescriptor_t rnnDesc, int hiddenSize, int numLayers,
        cudnnDropoutDescriptor_t dropoutDesc, cudnnRNNInputMode_t inputMode,
        cudnnDirectionMode_t direction, cudnnRNNMode_t mode,
        cudnnDataType_t dataType) {
    return 0;
}

cudnnStatus_t cudnnGetRNNWorkspaceSize(
        cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
        const int seqLength, const cudnnTensorDescriptor_t* xDesc,
        size_t* sizeInBytes) {
    return 0;
}

cudnnStatus_t cudnnGetRNNTrainingReserveSize(
        cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
        const int seqLength, const cudnnTensorDescriptor_t* xDesc,
        size_t* sizeInBytes) {
    return 0;
}

cudnnStatus_t cudnnGetRNNParamsSize(
        cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
        const cudnnTensorDescriptor_t xDesc, size_t* sizeInBytes,
        cudnnDataType_t dataType) {
    return 0;
}

cudnnStatus_t cudnnGetRNNLinLayerMatrixParams(
        cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
        const int layer, const cudnnTensorDescriptor_t xDesc,
        const cudnnFilterDescriptor_t wDesc, const void* w,
        const int linLayerID, cudnnFilterDescriptor_t linLayerMatDesc,
        void** linLayerMat) {
    return 0;
}

cudnnStatus_t cudnnGetRNNLinLayerBiasParams(
        cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
        const int layer, const cudnnTensorDescriptor_t xDesc,
        const cudnnFilterDescriptor_t wDesc, const void* w,
        const int linLayerID, cudnnFilterDescriptor_t linLayerBiasDesc,
        void** linLayerBias) {
    return 0;
}

cudnnStatus_t cudnnRNNForwardInference(
        cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
        const int seqLength,
        const cudnnTensorDescriptor_t* xDesc, const void* x,
        const cudnnTensorDescriptor_t hxDesc, const void* hx,
        const cudnnTensorDescriptor_t cxDesc, const void* cx,
        const cudnnFilterDescriptor_t wDesc, const void* w,
        const cudnnTensorDescriptor_t* yDesc, void* y,
        const cudnnTensorDescriptor_t hyDesc, void* hy,
        const cudnnTensorDescriptor_t cyDesc, void* cy,
        void* workspace, size_t workSpaceSizeInBytes) {
    return 0;
}

cudnnStatus_t cudnnRNNForwardTraining(
        cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
        const int seqLength,
        const cudnnTensorDescriptor_t* xDesc, const void* x,
        const cudnnTensorDescriptor_t hxDesc, const void* hx,
        const cudnnTensorDescriptor_t cxDesc, const void* cx,
        const cudnnFilterDescriptor_t wDesc, const void* w,
        const cudnnTensorDescriptor_t* yDesc, void* y,
        const cudnnTensorDescriptor_t hyDesc, void* hy,
        const cudnnTensorDescriptor_t cyDesc, void* cy,
        void* workspace, size_t workSpaceSizeInBytes,
        void* reserveSpace, size_t reserveSpaceSizeInBytes) {
    return 0;
}

cudnnStatus_t cudnnRNNBackwardData(
        cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
        const int seqLength,
        const cudnnTensorDescriptor_t* yDesc, const void * y,
        const cudnnTensorDescriptor_t* dyDesc, const void * dy,
        const cudnnTensorDescriptor_t dhyDesc, const void * dhy,
        const cudnnTensorDescriptor_t dcyDesc, const void * dcy,
        const cudnnFilterDescriptor_t wDesc, const void * w,
        const cudnnTensorDescriptor_t hxDesc, const void * hx,
        const cudnnTensorDescriptor_t cxDesc, const void * cx,
        const cudnnTensorDescriptor_t* dxDesc, void* dx,
        const cudnnTensorDescriptor_t dhxDesc, void* dhx,
        const cudnnTensorDescriptor_t dcxDesc, void* dcx,
        void * workspace, size_t workSpaceSizeInBytes,
        const void* reserveSpace, size_t reserveSpaceSizeInBytes) {
    return 0;
}

cudnnStatus_t cudnnRNNBackwardWeights(
         cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
         const int seqLength, const cudnnTensorDescriptor_t* xDesc,
         const void* x, const cudnnTensorDescriptor_t hxDesc,
         const void* hx, const cudnnTensorDescriptor_t* yDesc,
         const void* y, const void* workspace, size_t workSpaceSizeInBytes,
         const cudnnFilterDescriptor_t dwDesc, void* dw,
         const void* reserveSpace, size_t reserveSpaceSizeInBytes) {
    return 0;
}

} // extern "C"

#endif // #ifndef CUPY_NO_CUDA
#endif // #ifndef INCLUDE_GUARD_CUPY_CUDNN_H
