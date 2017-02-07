// This file is a stub header file of cuda for Read the Docs.

#ifndef INCLUDE_GUARD_CUPY_CUDA_H
#define INCLUDE_GUARD_CUPY_CUDA_H

#include "cupy_stdint.h"

#ifndef CUPY_NO_CUDA
#include <cuda.h>
#endif

#ifdef __APPLE__
#if CUDA_VERSION == 7050
// To avoid redefinition error of cudaDataType_t
// caused by including library_types.h.
// https://github.com/pfnet/chainer/issues/1700
// https://github.com/pfnet/chainer/pull/1819
#define __LIBRARY_TYPES_H__
#endif // #if CUDA_VERSION == 7050
#endif // #ifdef __APPLE__

#ifndef CUPY_NO_CUDA
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusolverDn.h>
#ifndef _WIN32
#include <nvToolsExt.h>
#endif

extern "C" {

#if CUDA_VERSION < 8000
#if CUDA_VERSION >= 7050
typedef cublasDataType_t cudaDataType;
#else
enum cudaDataType_t {};
typedef enum cudaDataType_t cudaDataType;
#endif // #if CUDA_VERSION >= 7050
#endif // #if CUDA_VERSION < 8000

#if CUDA_VERSION < 7050
cublasStatus_t cublasSgemmEx(...) {
    return CUBLAS_STATUS_NOT_SUPPORTED;
}
#endif // #if CUDA_VERSION < 7050

} // extern "C"

#else // #ifndef CUPY_NO_CUDA

extern "C" {

///////////////////////////////////////////////////////////////////////////////
// cuda.h
///////////////////////////////////////////////////////////////////////////////

typedef int CUdevice;
typedef enum {
    CUDA_SUCCESS = 0,
} CUresult;


typedef void* CUdeviceptr;
struct CUevent_st;
struct CUfunc_st;
struct CUmod_st;
struct CUstream_st;

typedef struct CUevent_st* cudaEvent_t;
typedef struct CUfunc_st* CUfunction;
typedef struct CUmod_st* CUmodule;
typedef struct CUstream_st* cudaStream_t;


// Error handling
CUresult cuGetErrorName(...) {
    return CUDA_SUCCESS;
}

CUresult cuGetErrorString(...) {
    return CUDA_SUCCESS;
}


// Module load and kernel execution
CUresult cuModuleLoad(...) {
    return CUDA_SUCCESS;
}

CUresult cuModuleLoadData(...) {
    return CUDA_SUCCESS;
}

CUresult cuModuleUnload(...) {
    return CUDA_SUCCESS;
}

CUresult cuModuleGetFunction(...) {
    return CUDA_SUCCESS;
}

CUresult cuModuleGetGlobal(...) {
    return CUDA_SUCCESS;
}

CUresult cuLaunchKernel(...) {
    return CUDA_SUCCESS;
}


///////////////////////////////////////////////////////////////////////////////
// cuda_runtime.h
///////////////////////////////////////////////////////////////////////////////

typedef enum {
    cudaSuccess = 0,
} cudaError_t;
typedef enum {} cudaDataType;
enum cudaDeviceAttr {};
enum cudaMemcpyKind {};


typedef void (*cudaStreamCallback_t)(
    cudaStream_t stream, cudaError_t status, void* userData);

typedef cudaStreamCallback_t StreamCallback;


struct cudaPointerAttributes{
    int device;
    void* devicePointer;
    void* hostPointer;
    int isManaged;
    int memoryType;
};

typedef cudaPointerAttributes _PointerAttributes;


// Error handling
const char* cudaGetErrorName(...) {
    return NULL;
}

const char* cudaGetErrorString(...) {
    return NULL;
}


// Initialization
cudaError_t cudaDriverGetVersion(...) {
    return cudaSuccess;
}

cudaError_t cudaRuntimeGetVersion(...) {
    return cudaSuccess;
}


// CUdevice operations
cudaError_t cudaGetDevice(...) {
    return cudaSuccess;
}

cudaError_t cudaDeviceGetAttribute(...) {
    return cudaSuccess;
}

cudaError_t cudaGetDeviceCount(...) {
    return cudaSuccess;
}

cudaError_t cudaSetDevice(...) {
    return cudaSuccess;
}

cudaError_t cudaDeviceSynchronize() {
    return cudaSuccess;
}

cudaError_t cudaDeviceCanAccessPeer(...) {
    return cudaSuccess;
}

cudaError_t cudaDeviceEnablePeerAccess(...) {
    return cudaSuccess;
}


// Memory management
cudaError_t cudaMalloc(...) {
    return cudaSuccess;
}

cudaError_t cudaHostAlloc(...) {
    return cudaSuccess;
}

int cudaFree(...) {
    return cudaSuccess;
}

cudaError_t cudaFreeHost(...) {
    return cudaSuccess;
}

int cudaMemGetInfo(...) {
    return cudaSuccess;
}

cudaError_t cudaMemcpy(...) {
    return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(...) {
    return cudaSuccess;
}

cudaError_t cudaMemcpyPeer(...) {
    return cudaSuccess;
}

cudaError_t cudaMemcpyPeerAsync(...) {
    return cudaSuccess;
}

cudaError_t cudaMemset(...) {
    return cudaSuccess;
}

cudaError_t cudaMemsetAsync(...) {
    return cudaSuccess;
}

cudaError_t cudaPointerGetAttributes(...) {
    return cudaSuccess;
}


// Stream and Event
cudaError_t cudaStreamCreate(...) {
    return cudaSuccess;
}

cudaError_t cudaStreamCreateWithFlags(...) {
    return cudaSuccess;
}

cudaError_t cudaStreamDestroy(...) {
    return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(...) {
    return cudaSuccess;
}

cudaError_t cudaStreamAddCallback(...) {
    return cudaSuccess;
}

cudaError_t cudaStreamQuery(...) {
    return cudaSuccess;
}

cudaError_t cudaStreamWaitEvent(...) {
    return cudaSuccess;
}

cudaError_t cudaEventCreate(...) {
    return cudaSuccess;
}

cudaError_t cudaEventCreateWithFlags(...) {
    return cudaSuccess;
}

cudaError_t cudaEventDestroy(...) {
    return cudaSuccess;
}

cudaError_t cudaEventElapsedTime(...) {
    return cudaSuccess;
}

cudaError_t cudaEventQuery(...) {
    return cudaSuccess;
}

cudaError_t cudaEventRecord(...) {
    return cudaSuccess;
}

cudaError_t cudaEventSynchronize(...) {
    return cudaSuccess;
}


///////////////////////////////////////////////////////////////////////////////
// cublas_v2.h
///////////////////////////////////////////////////////////////////////////////

typedef void* cublasHandle_t;

typedef enum {} cublasOperation_t;
typedef enum {} cublasPointerMode_t;
typedef enum {} cublasSideMode_t;
typedef enum {
    CUBLAS_STATUS_SUCCESS=0,
} cublasStatus_t;


// Context
cublasStatus_t cublasCreate(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDestroy(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetVersion(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetPointerMode(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetPointerMode(...) {
    return CUBLAS_STATUS_SUCCESS;
}

// Stream
cublasStatus_t cublasSetStream(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetStream(...) {
    return CUBLAS_STATUS_SUCCESS;
}

// BLAS Level 1
cublasStatus_t cublasIsamax(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasIsamin(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSasum(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSaxpy(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDaxpy(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSdot(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDdot(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSnrm2(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSscal(...) {
    return CUBLAS_STATUS_SUCCESS;
}


// BLAS Level 2
cublasStatus_t cublasSgemv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgemv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSger(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDger(...) {
    return CUBLAS_STATUS_SUCCESS;
}

// BLAS Level 3
cublasStatus_t cublasSgemm(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgemm(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgemmBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgemmEx(...) {
    return CUBLAS_STATUS_SUCCESS;
}


// BLAS extension
cublasStatus_t cublasSgeam(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgeam(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSdgmm(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgetrfBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgetriBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}


///////////////////////////////////////////////////////////////////////////////
// curand.h
///////////////////////////////////////////////////////////////////////////////

typedef enum {} curandOrdering_t;
typedef enum {} curandRngType_t;
typedef enum {
    CURAND_STATUS_SUCCESS = 0,
} curandStatus_t;

typedef void* curandGenerator_t;


// curandGenerator_t
curandStatus_t curandCreateGenerator(...) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandDestroyGenerator(...) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGetVersion(...) {
    return CURAND_STATUS_SUCCESS;
}


// Stream
curandStatus_t curandSetStream(...) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandSetPseudoRandomGeneratorSeed(...) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandSetGeneratorOffset(...) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandSetGeneratorOrdering(...) {
    return CURAND_STATUS_SUCCESS;
}


// Generation functions
curandStatus_t curandGenerate(...) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateLongLong(...) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateUniform(...) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateUniformDouble(...) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateNormal(...) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateNormalDouble(...) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateLogNormal(...) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateLogNormalDouble(...) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGeneratePoisson(...) {
    return CURAND_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
// cusolver.h
///////////////////////////////////////////////////////////////////////////////

int cusolverDnCreate(Handle* handle) {
    return 0;
}

int cusolverDnDestroy(Handle handle) {
    return 0;
}

int cusolverDnGetStream(Handle handle, Stream* streamId) {
    return 0;
}

int cusolverDnSetStream(Handle handle, Stream streamId) {
    return 0;
}

int cusolverDnSpotrf_bufferSize(Handle handle, FillMode uplo,
                                int n, float* A, int lda, int* Lwork) {
    return 0;
}

int cusolverDnDpotrf_bufferSize(Handle handle, FillMode uplo,
                                int n, double* A, int lda, int* Lwork) {
    return 0;
}

int cusolverDnSpotrf(Handle handle, FillMode uplo, int n, float* A, int lda,
                     float* Workspace, int Lwork, int* devInfo) {
    return 0;
}

int cusolverDnDpotrf(Handle handle, FillMode uplo, int n, double *A, int lda,
                     double* Workspace, int Lwork, int* devInfo) {
    return 0;
}

int cusolverDnSpotrs(Handle handle, FillMode uplo, int n, int nrhs,
                     const float* A, int lda, float* B, int ldb,
                     int* devInfo) {
    return 0;
}

int cusolverDnDpotrs(Handle handle, FillMode uplo, int n, int nrhs,
                     const double* A, int lda, double* B, int ldb,
                     int *devInfo) {
    return 0;
}


int cusolverDnSgetrf(Handle handle, int m, int n, float* A, int lda,
                     float* Workspace, int* devIpiv, int* devInfo) {
    return 0;
}

int cusolverDnDgetrf(Handle handle, int m, int n, double* A, int lda,
                     double* Workspace, int* devIpiv, int* devInfo) {
    return 0;
}

int cusolverDnSgetrs(Handle handle, Operation trans,
                     int n, int nrhs, const float* A, int lda,
                     const int* devIpiv, float* B, int ldb, int* devInfo) {
    return 0;
}

int cusolverDnDgetrs(Handle handle, Operation trans,
                     int n, int nrhs, const double* A, int lda,
                     const int* devIpiv, double* B, int ldb, int* devInfo) {
    return 0;
}

int cusolverDnSgeqrf_bufferSize(Handle handle, int m, int n,
                                float* A, int lda, int* Lwork) {
    return 0;
}

int cusolverDnDgeqrf_bufferSize(Handle handle, int m, int n,
                                double* A, int lda, int* Lwork) {
    return 0;
}

int cusolverDnSgeqrf(Handle handle, int m, int n, float* A, int lda,
                     float* TAU, float* Workspace, int Lwork, int* devInfo) {
    return 0;
}

int cusolverDnDgeqrf(Handle handle, int m, int n, double* A, int lda,
                     double* TAU, double* Workspace, int Lwork, int* devInfo) {
    return 0;
}

int cusolverDnSormqr(Handle handle, SideMode side, Operation trans,
                     int m, int n, int k, const float* A, int lda,
                     const float* tau, float* C, int ldc, float* work,
                     int lwork, int* devInfo) {
    return 0;
}

int cusolverDnDormqr(Handle handle, SideMode side, Operation trans,
                     int m, int n, int k, const double* A, int lda,
                     const double* tau, double* C, int ldc, double* work,
                     int lwork, int* devInfo) {
    return 0;
}

int cusolverDnSsytrf(Handle handle, FillMode uplo, int n, float *A, int lda,
                     int *ipiv, float *work, int lwork, int *devInfo) {
    return 0;
}

int cusolverDnDsytrf(Handle handle, FillMode uplo, int n, double* A, int lda,
                     int* ipiv, double* work, int lwork, int* devInfo) {
    return 0;
}

int cusolverDnSgebrd(Handle handle, int m, int n, float* A, int lda,
                     float* D, float* E, float* TAUQ, float* TAUP,
                     float* Work, int Lwork, int* devInfo) {
    return 0;
}

int cusolverDnDgebrd(Handle handle, int m, int n, double* A, int lda,
                     double* D, double* E, double* TAUQ, double* TAUP,
                     double* Work, int Lwork, int* devInfo) {
    return 0;
}

int cusolverDnSgesvd_bufferSize(Handle handle, int m, int n, int* Lwork) {
    return 0;
}

int cusolverDnDgesvd_bufferSize(Handle handle, int m, int n, int* Lwork) {
    return 0;
}

int cusolverDnSgesvd(Handle handle, char jobu, char jobvt, int m, int n,
                     float* A, int lda, float* S, float* U, int ldu,
                     float* VT, int ldvt, float* Work, int Lwork,
                     float* rwork, int* devInfo) {
    return 0;
}
                     
int cusolverDnDgesvd(Handle handle, char jobu, char jobvt, int m, int n,
                     double* A, int lda, double* S, double* U, int ldu,
                     double* VT, int ldvt, double* Work, int Lwork,
                     double* rwork, int* devInfo) {
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
// cuda_profiler_api.h
///////////////////////////////////////////////////////////////////////////////

typedef enum {} cudaOutputMode_t;

cudaError_t cudaProfilerInitialize(...) {
  return cudaSuccess;
}

cudaError_t cudaProfilerStart() {
  return cudaSuccess;
}

cudaError_t cudaProfilerStop() {
  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////
// nvToolsExt.h
///////////////////////////////////////////////////////////////////////////////

#define NVTX_VERSION 1

typedef enum nvtxColorType_t
{
    NVTX_COLOR_UNKNOWN  = 0,
    NVTX_COLOR_ARGB     = 1
} nvtxColorType_t;

typedef enum nvtxMessageType_t
{
    NVTX_MESSAGE_UNKNOWN          = 0,
    NVTX_MESSAGE_TYPE_ASCII       = 1,
    NVTX_MESSAGE_TYPE_UNICODE     = 2,
} nvtxMessageType_t;

typedef union nvtxMessageValue_t
{
    const char* ascii;
    const wchar_t* unicode;
} nvtxMessageValue_t;

typedef struct nvtxEventAttributes_v1
{
    uint16_t version;
    uint16_t size;
    uint32_t category;
    int32_t colorType;
    uint32_t color;
    int32_t payloadType;
    int32_t reserved0;
    union payload_t
    {
        uint64_t ullValue;
        int64_t llValue;
        double dValue;
    } payload;
    int32_t messageType;
    nvtxMessageValue_t message;
} nvtxEventAttributes_v1;

typedef nvtxEventAttributes_v1 nvtxEventAttributes_t;

void nvtxMarkA(...) {
}

void nvtxMarkEx(...) {
}

int nvtxRangePushA(...) {
    return 0;
}

int nvtxRangePushEx(...) {
    return 0;
}

int nvtxRangePop() {
    return 0;
}

} // extern "C"

#endif // #ifndef CUPY_NO_CUDA
#endif // #ifndef INCLUDE_GUARD_CUPY_CUDA_H
