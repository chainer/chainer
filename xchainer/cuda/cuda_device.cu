#include "xchainer/cuda/cuda_device.h"

#include <cublas_v2.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>

#include <cuda_runtime.h>

#include "xchainer/array.h"
#include "xchainer/axes.h"
#include "xchainer/backend_util.h"
#include "xchainer/cuda/cast.cuh"
#include "xchainer/cuda/cublas.h"
#include "xchainer/cuda/cuda_runtime.h"
#include "xchainer/cuda/cudnn.h"
#include "xchainer/cuda/elementwise.cuh"
#include "xchainer/cuda/reduce.cuh"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/enum.h"
#include "xchainer/error.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"
#include "xchainer/native/native_device.h"
#include "xchainer/numeric_limits.h"
#include "xchainer/reduction_kernel_arg.h"
#include "xchainer/routines/connection.h"
#include "xchainer/routines/creation.h"
#include "xchainer/scalar.h"
#include "xchainer/shape.h"

namespace xchainer {
namespace cuda {

CudaDevice::~CudaDevice() {
    if (cublas_handle_) {
        cudaSetDevice(index());
        cublasDestroy(cublas_handle_);
    }
    if (cudnn_handle_) {
        cudaSetDevice(index());
        cudnnDestroy(cudnn_handle_);
    }
}

cublasHandle_t CudaDevice::cublas_handle() {
    if (!cublas_handle_) {
        CheckCudaError(cudaSetDevice(index()));
        CheckCublasError(cublasCreate(&cublas_handle_));
    }
    return cublas_handle_;
}

cudnnHandle_t CudaDevice::cudnn_handle() {
    if (!cudnn_handle_) {
        CheckCudaError(cudaSetDevice(index()));
        CheckCudnnError(cudnnCreate(&cudnn_handle_));
    }
    return cudnn_handle_;
}

std::shared_ptr<void> CudaDevice::Allocate(size_t bytesize) {
    CheckCudaError(cudaSetDevice(index()));
    void* ptr = memory_pool_.Malloc(bytesize);
    return std::shared_ptr<void>{ptr, [this](void* ptr) { memory_pool_.Free(ptr); }};
}

std::shared_ptr<void> CudaDevice::MakeDataFromForeignPointer(const std::shared_ptr<void>& data) {
    // check memory validity
    void* ptr = data.get();
    cudaPointerAttributes attr{};
    cudaError_t status = cudaPointerGetAttributes(&attr, ptr);
    switch (status) {
        case cudaSuccess:
            if (attr.isManaged == 0) {
                throw XchainerError{"CUDA memory: ", ptr, " must be a managed (unified) memory"};
            }
            if (attr.device != index()) {
                throw XchainerError{"CUDA memory: ", ptr, " must reside on the device: ", index()};
            }
            break;
        case cudaErrorInvalidValue:
            throw XchainerError{"Memory: ", ptr, " is not a CUDA memory"};
        default:
            Throw(status);
    }
    return data;
}

void CudaDevice::MemoryCopyFrom(void* dst, const void* src, size_t bytesize, Device& src_device) {
    assert(bytesize == 0 || IsPointerCudaMemory(dst));
    if (bytesize == 0) {
        return;
    }
    CheckCudaError(cudaSetDevice(index()));
    if (&src_device == this || nullptr != dynamic_cast<CudaDevice*>(&src_device)) {
        // Copy between CUDA devices
        CheckCudaError(cudaMemcpy(dst, src, bytesize, cudaMemcpyDeviceToDevice));
    } else {
        assert(nullptr != dynamic_cast<native::NativeDevice*>(&src_device) &&
               "CudaDevice only supports copy between cuda or native devices.");
        // Copy from native device
        CheckCudaError(cudaMemcpy(dst, src, bytesize, cudaMemcpyHostToDevice));
    }
}

void CudaDevice::MemoryCopyTo(void* dst, const void* src, size_t bytesize, Device& dst_device) {
    assert(bytesize == 0 || src == nullptr || IsPointerCudaMemory(src));
    if (bytesize == 0) {
        return;
    }
    CheckCudaError(cudaSetDevice(index()));
    if (&dst_device == this || nullptr != dynamic_cast<CudaDevice*>(&dst_device)) {
        // Copy between CUDA devices
        CheckCudaError(cudaMemcpy(dst, src, bytesize, cudaMemcpyDeviceToDevice));
    } else {
        assert(nullptr != dynamic_cast<native::NativeDevice*>(&dst_device) &&
               "CudaDevice only supports copy between cuda or native devices.");
        // Copy to native device
        CheckCudaError(cudaMemcpy(dst, src, bytesize, cudaMemcpyDeviceToHost));
    }
}

std::shared_ptr<void> CudaDevice::TransferDataFrom(
        Device& src_device, const std::shared_ptr<void>& src_ptr, size_t offset, size_t bytesize) {
    std::shared_ptr<void> dst_ptr = Allocate(bytesize);
    MemoryCopyFrom(dst_ptr.get(), &(static_cast<int8_t*>(src_ptr.get())[offset]), bytesize, src_device);
    return dst_ptr;
}

std::shared_ptr<void> CudaDevice::TransferDataTo(Device& dst_device, const std::shared_ptr<void>& src_ptr, size_t offset, size_t bytesize) {
    std::shared_ptr<void> dst_ptr = dst_device.Allocate(bytesize);
    MemoryCopyTo(dst_ptr.get(), &(static_cast<int8_t*>(src_ptr.get())[offset]), bytesize, dst_device);
    return dst_ptr;
}

std::shared_ptr<void> CudaDevice::FromHostMemory(const std::shared_ptr<void>& src_ptr, size_t bytesize) {
    std::shared_ptr<void> dst_ptr = Allocate(bytesize);
    CheckCudaError(cudaMemcpy(dst_ptr.get(), src_ptr.get(), bytesize, cudaMemcpyHostToDevice));
    return dst_ptr;
}

namespace {

template <typename T>
struct FillImpl {
    __device__ void operator()(int64_t /*i*/, T& out) { out = value; }
    T value;
};

}  // namespace

void CudaDevice::Fill(const Array& out, Scalar value) {
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<T>(FillImpl<T>{static_cast<T>(value)}, out);
    });
}

namespace {

template <typename T>
struct ArangeImpl {
    __device__ void operator()(int64_t i, T& out) { out = start + step * i; }
    T start;
    T step;
};

}  // namespace

void CudaDevice::Arange(Scalar start, Scalar step, const Array& out) {
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<T>(ArangeImpl<T>{static_cast<T>(start), static_cast<T>(step)}, out);
    });
}

namespace {

template <typename T>
struct ArgMaxImpl {
    struct MaxAndArgMax {
        T max;
        int64_t argmax;
    };
    __device__ MaxAndArgMax Identity() { return {T{}, -1}; }
    __device__ MaxAndArgMax MapIn(T in, int64_t index) { return {in, index}; }
    __device__ void Reduce(MaxAndArgMax next, MaxAndArgMax& accum) {
        if (accum.argmax < 0 || accum.max < next.max) {
            accum = next;
        }
    }
    __device__ int64_t MapOut(MaxAndArgMax accum) { return accum.argmax; }
};

}  // namespace

void CudaDevice::ArgMax(const Array& a, const Axes& axis, const Array& out) {
    CheckDevicesCompatible(a, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(a.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Reduce(MakeReductionKernelArg<T, int64_t>(a, axis, out), ArgMaxImpl<T>{});
    });
}

namespace {

template <typename T>
struct SumImpl {
    __device__ T Identity() { return T{0}; }
    __device__ T MapIn(T in, int64_t /*index*/) { return in; }
    __device__ void Reduce(T next, T& accum) { accum += next; }
    __device__ T MapOut(T accum) { return accum; }
};

}  // namespace

void CudaDevice::Sum(const Array& a, const Axes& axis, const Array& out) {
    assert(internal::IsValidReductionShape(a.shape(), axis, out.shape(), true));
    CheckDevicesCompatible(a, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Reduce(MakeReductionKernelArg<T, T>(a, axis, out), SumImpl<T>{});
    });
}

namespace {
template <typename T>
__device__ bool IsNan(T /*value*/) {
    return false;
}
__device__ bool IsNan(double value) { return ::isnan(value); }
__device__ bool IsNan(float value) { return ::isnan(value); }

template <typename T>
struct AMaxImpl {
    __device__ T Identity() { return NumericLimits<T>::LowestOrInf(); }
    __device__ T MapIn(T in, int64_t /*index*/) { return in; }
    __device__ void Reduce(T next, T& accum) {
        if (IsNan(next) || accum < next) {
            accum = next;
        }
    }
    __device__ T MapOut(T accum) { return accum; }
};
}  // namespace

void CudaDevice::AMax(const Array& a, const Axes& axis, const Array& out) {
    assert(internal::IsValidReductionShape(a.shape(), axis, out.shape(), true));
    CheckDevicesCompatible(a, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Reduce(MakeReductionKernelArg<T, T>(a, axis, out), AMaxImpl<T>{});
    });
}

namespace {

template <typename T>
struct CopyImpl {
    __device__ void operator()(int64_t /*i*/, T a, T& out) { out = a; }
};

}  // namespace

void CudaDevice::Copy(const Array& a, const Array& out) {
    CheckDevicesCompatible(a, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, T>(CopyImpl<T>{}, a, out);
    });
}

namespace {

template <typename InT, typename OutT>
struct AsTypeImpl {
    __device__ void operator()(int64_t /*i*/, InT a, OutT& out) { out = static_cast<OutT>(a); }
};

}  // namespace

void CudaDevice::AsType(const Array& a, const Array& out) {
    CheckDevicesCompatible(a, out);
    CheckCudaError(cudaSetDevice(index()));
    auto do_astype = [&](auto in_pt, auto out_pt) {
        using InT = typename decltype(in_pt)::type;
        using OutT = typename decltype(out_pt)::type;
        Elementwise<const InT, OutT>(AsTypeImpl<InT, OutT>{}, a, out);
    };
    VisitDtype(out.dtype(), [&](auto out_pt) { VisitDtype(a.dtype(), do_astype, out_pt); });
}

namespace {

template <typename T>
struct EqualImpl {
    __device__ void operator()(int64_t /*i*/, T x1, T x2, bool& out) { out = x1 == x2; }
};

}  // namespace

void CudaDevice::Equal(const Array& x1, const Array& x2, const Array& out) {
    CheckDevicesCompatible(x1, x2, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(x1.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, const T, bool>(EqualImpl<T>{}, x1, x2, out);
    });
}

namespace {

template <typename T>
struct AddImpl {
    __device__ void operator()(int64_t /*i*/, T x1, T x2, T& out) { out = x1 + x2; }
};

}  // namespace

// TODO(sonots): support stream
void CudaDevice::Add(const Array& x1, const Array& x2, const Array& out) {
    CheckDevicesCompatible(x1, x2, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, const T, T>(AddImpl<T>{}, x1, x2, out);
    });
}

namespace {

template <typename T>
struct AddASImpl {
    __device__ void operator()(int64_t /*i*/, T x1, T& out) { out = x1 + x2; }
    T x2;
};

}  // namespace

void CudaDevice::AddAS(const Array& x1, Scalar x2, const Array& out) {
    CheckDevicesCompatible(x1, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, T>(AddASImpl<T>{static_cast<T>(x2)}, x1, out);
    });
}

namespace {

template <typename T>
struct SubtractImpl {
    __device__ void operator()(int64_t /*i*/, T x1, T x2, T& out) { out = x1 - x2; }
};

}  // namespace

void CudaDevice::Subtract(const Array& x1, const Array& x2, const Array& out) {
    CheckDevicesCompatible(x1, x2, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, const T, T>(SubtractImpl<T>{}, x1, x2, out);
    });
}

namespace {

template <typename T>
struct SubtractASImpl {
    __device__ void operator()(int64_t /*i*/, T x1, T& out) { out = x1 - x2; }
    T x2;
};

}  // namespace

void CudaDevice::SubtractAS(const Array& x1, Scalar x2, const Array& out) {
    CheckDevicesCompatible(x1, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, T>(SubtractASImpl<T>{static_cast<T>(x2)}, x1, out);
    });
}

namespace {

template <typename T>
struct MultiplyImpl {
    __device__ void operator()(int64_t /*i*/, T x1, T x2, T& out) { out = x1 * x2; }
};

}  // namespace

// TODO(sonots): support stream
void CudaDevice::Multiply(const Array& x1, const Array& x2, const Array& out) {
    CheckDevicesCompatible(x1, x2, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, const T, T>(MultiplyImpl<T>{}, x1, x2, out);
    });
}

namespace {

template <typename T>
struct MultiplyASImpl {
    __device__ void operator()(int64_t /*i*/, T x1, T& out) { out = x1 * x2; }
    T x2;
};

}  // namespace

void CudaDevice::MultiplyAS(const Array& x1, Scalar x2, const Array& out) {
    CheckDevicesCompatible(x1, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, T>(MultiplyASImpl<T>{static_cast<T>(x2)}, x1, out);
    });
}

namespace {

template <typename T>
struct DivideImpl {
    __device__ void operator()(int64_t /*i*/, T x1, T x2, T& out) { out = x1 / x2; }
};

}  // namespace

void CudaDevice::Divide(const Array& x1, const Array& x2, const Array& out) {
    CheckDevicesCompatible(x1, x2, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, const T, T>(DivideImpl<T>{}, x1, x2, out);
    });
}

namespace {

template <typename T>
struct DivideASImpl {
    __device__ void operator()(int64_t /*i*/, T x1, T& out) { out = x1 / x2; }
    T x2;
};

}  // namespace

void CudaDevice::DivideAS(const Array& x1, Scalar x2, const Array& out) {
    CheckDevicesCompatible(x1, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, T>(DivideASImpl<T>{static_cast<T>(x2)}, x1, out);
    });
}

namespace {

template <typename T>
struct IfLessElseASSAImpl {
    __device__ void operator()(int64_t /*i*/, T x1, T neg, T& out) { out = x1 < x2 ? pos : neg; }
    T x2;
    T pos;
};

}  // namespace

void CudaDevice::IfLessElseASSA(const Array& x1, Scalar x2, Scalar pos, const Array& neg, const Array& out) {
    CheckDevicesCompatible(x1, neg, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, const T, T>(IfLessElseASSAImpl<T>{static_cast<T>(x2), static_cast<T>(pos)}, x1, neg, out);
    });
}

namespace {

// Dispatch gemm routines based on the element type T
template <typename T>
struct Gemm;

template <>
struct Gemm<float> {
    template <typename... Args>
    void operator()(Args&&... args) const {
        CheckCublasError(cublasSgemm(std::forward<Args>(args)...));
    }
};

template <>
struct Gemm<double> {
    template <typename... Args>
    void operator()(Args&&... args) const {
        CheckCublasError(cublasDgemm(std::forward<Args>(args)...));
    }
};

struct GemmInputLayout {
    int64_t ld = 0;
    cublasOperation_t trans = CUBLAS_OP_T;

    // Makes the array C or Fortran contiguous and configure leading dimension and transposition accordingly.
    Array Configure(const Array& a) {
        assert(a.ndim() == 2);
        if (a.strides()[0] == a.item_size() && a.strides()[0] * a.shape()[0] == a.strides()[1]) {
            // Fortran contiguous
            ld = a.shape()[0];
            return a;
        }
        // Force C contiguous
        ld = a.shape()[1];
        trans = CUBLAS_OP_N;  // transposed
        return a.IsContiguous() ? a : a.AsConstant(CopyKind::kCopy);
    }
};

}  // namespace

void CudaDevice::Dot(const Array& a, const Array& b, const Array& out) {
    CheckDevicesCompatible(a, b, out);
    CheckCudaError(cudaSetDevice(index()));

    assert(a.ndim() == 2);
    assert(b.ndim() == 2);
    assert(out.ndim() == 2);

    int64_t m = a.shape()[0];
    int64_t k = a.shape()[1];
    int64_t n = b.shape()[1];
    assert(b.shape()[0] == k);
    assert(out.shape()[0] == m);
    assert(out.shape()[1] == n);

    if (m == 1 && n == 1) {
        // TODO(beam2d): Write a custom reduction kernel.
        Array l = a.AsConstant();
        Array r = b.AsConstant();
        Array o = out.AsConstant();
        Sum(l.Reshape({k}) * r.Reshape({k}), {0}, o.Reshape({}));
        return;
    }

    bool is_out_contiguous = out.IsContiguous();
    Array out_contiguous = is_out_contiguous ? out : EmptyLike(out, *this);

    auto gemm_impl = [&](auto pt) {
        using T = typename decltype(pt)::type;

        // Note that cuBLAS uses Fortran order.
        // To compute out = a x b, we use cuBLAS to compute out^T = b^T x a^T (here x is the matrix product).

        GemmInputLayout a_layout;
        GemmInputLayout b_layout;
        Array a_config = a_layout.Configure(a);
        Array b_config = b_layout.Configure(b);

        const T one = 1;
        const T zero = 0;
        const T* a_ptr = xchainer::internal::GetRawOffsetData<const T>(a_config);
        const T* b_ptr = xchainer::internal::GetRawOffsetData<const T>(b_config);
        T* out_ptr = xchainer::internal::GetRawOffsetData<T>(out_contiguous);
        Gemm<T>{}(
                cublas_handle(), b_layout.trans, a_layout.trans, n, m, k, &one, b_ptr, b_layout.ld, a_ptr, a_layout.ld, &zero, out_ptr, n);
    };

    if (a.dtype() == Dtype::kFloat32) {
        gemm_impl(PrimitiveType<float>{});
    } else if (a.dtype() == Dtype::kFloat64) {
        gemm_impl(PrimitiveType<double>{});
    } else {
        throw NotImplementedError("dot is not implemented for non-float types in CUDA");
    }

    if (!is_out_contiguous) {
        Copy(out_contiguous, out);
    }
}

namespace {

template <typename T>
struct ExpImpl {
    __device__ void operator()(int64_t /*i*/, T x, T& out) { out = std::exp(x); }
};

}  // namespace

void CudaDevice::Exp(const Array& x, const Array& out) {
    CheckDevicesCompatible(x, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, T>(ExpImpl<T>{}, x, out);
    });
}

namespace {

template <typename T>
struct LogImpl {
    __device__ void operator()(int64_t /*i*/, T x, T& out) { out = std::log(x); }
};

}  // namespace

void CudaDevice::Log(const Array& x, const Array& out) {
    CheckDevicesCompatible(x, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, T>(LogImpl<T>{}, x, out);
    });
}

namespace {

// Makes axes for permutation that moves [first_axis, last_axis) to the head.
Axes MakeRollingPermutation(int8_t first_axis, int8_t last_axis, int8_t ndim) {
    assert(0 <= first_axis);
    assert(first_axis < last_axis);
    assert(last_axis <= ndim);

    Axes permutation{};
    permutation.resize(ndim);
    auto head_end = permutation.begin() + (last_axis - first_axis);
    auto last = permutation.begin() + last_axis;
    std::iota(permutation.begin(), head_end, first_axis);
    std::iota(head_end, last, int8_t{0});
    std::iota(last, permutation.end(), last_axis);
    return permutation;
}

template <typename T>
__global__ void TakeKernel(
        IndexableArray<const T> a_iarray,
        IndexableArray<T> out_iarray,
        IndexableArray<const int64_t> indices_iarray,
        Indexer<> a_indexer,
        Indexer<> out_indexer,
        Indexer<> indices_indexer,
        int64_t common_total_size,
        int64_t axis_dim) {
    for (auto it = out_indexer.It(blockIdx.x * blockDim.x + threadIdx.x, blockDim.x * gridDim.x); it; ++it) {
        int64_t indices_pos = it.raw_index() / common_total_size;
        int64_t common_pos = it.raw_index() % common_total_size;

        int64_t index = indices_iarray[indices_indexer.It(indices_pos)];
        if (index < 0) {
            index = axis_dim - ((-index + axis_dim - 1) % axis_dim + 1);
        } else {
            index = index % axis_dim;
        }
        assert(0 <= index);
        assert(index < axis_dim);

        out_iarray[it] = a_iarray[a_indexer.It(index * common_total_size + common_pos)];
    }
}

template <typename T>
__global__ void AddAtKernel(
        IndexableArray<const T> a_iarray,
        IndexableArray<const T> b_iarray,
        IndexableArray<T> out_iarray,
        IndexableArray<const int64_t> indices_iarray,
        Indexer<> b_indexer,
        Indexer<> out_indexer,
        Indexer<> indices_indexer,
        int64_t common_total_size,
        int64_t axis_dim) {
    for (auto it = out_indexer.It(blockIdx.x * blockDim.x + threadIdx.x, blockDim.x * gridDim.x); it; ++it) {
        int64_t axis_pos = it.raw_index() / common_total_size;
        int64_t common_pos = it.raw_index() % common_total_size;

        T out_value = a_iarray[it];

        for (auto it_indices = indices_indexer.It(0); it_indices; ++it_indices) {
            int64_t index = indices_iarray[it_indices];

            if (index < 0) {
                index = axis_dim - ((-index + axis_dim - 1) % axis_dim + 1);
            } else {
                index = index % axis_dim;
            }
            assert(0 <= index);
            assert(index < axis_dim);

            if (index == axis_pos) {
                out_value += b_iarray[b_indexer.It(it_indices.raw_index() * common_total_size + common_pos)];
            }
        }

        out_iarray[it] = out_value;
    }
}

}  // namespace

void CudaDevice::Take(const Array& a, const Array& indices, int8_t axis, const Array& out) {
    CheckDevicesCompatible(a, indices, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;

        // a and out are transposed as follows.
        // a:       (Ni..., N,     Nj...) => (N,     Ni..., Nj...)
        // out:     (Ni..., Nk..., Nj...) => (Nk..., Ni..., Nj...)
        //
        // indices is used as is.
        // indices: (Nk...)

        IndexableArray<const T> a_iarray{a};
        Axes a_perm = MakeRollingPermutation(axis, axis + 1, a.ndim());
        a_iarray.Permute(a_perm);
        Shape a_shape = internal::TransposeShape(a.shape(), a_perm);
        Indexer<> a_indexer{a_shape};

        IndexableArray<T> out_iarray{out};
        Axes out_perm = MakeRollingPermutation(axis, axis + indices.ndim(), out.ndim());
        out_iarray.Permute(out_perm);
        Shape out_shape = internal::TransposeShape(out.shape(), out_perm);
        Indexer<> out_indexer{out_shape};

        IndexableArray<const int64_t> indices_iarray{indices};
        Indexer<> indices_indexer{indices.shape()};

        // size of (Ni..., Nj...) part
        int64_t common_total_size = a_indexer.total_size() / a_shape[0];

        static const int kMaxBlockSize = CudaOccupancyMaxPotentialBlockSize(&TakeKernel<T>).block_size;
        int64_t total_size = out_indexer.total_size();
        int64_t grid_size = (total_size + kMaxBlockSize - 1) / kMaxBlockSize;
        int64_t block_size = std::min<int64_t>(total_size, kMaxBlockSize);

        TakeKernel<<<grid_size, block_size>>>(
                a_iarray, out_iarray, indices_iarray, a_indexer, out_indexer, indices_indexer, common_total_size, a_shape[0]);
    });
}

void CudaDevice::AddAt(const Array& a, const Array& indices, int8_t axis, const Array& b, const Array& out) {
    // TODO(niboshi): Current implementation only distributes output elements in respective threads. Summation on the indices is performed
    // serially in each thread. This implementation can be improved by distributing indices as well, possibly using atomicAdd.

    assert(a.shape() == out.shape());
    CheckDevicesCompatible(a, indices, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;

        // b and out are transposed as follows.
        // a:       (Ni..., N,     Nj...) => (N,     Ni..., Nj...)
        // b:       (Ni..., Nk..., Nj...) => (Nk..., Ni..., Nj...)
        // out:     (Ni..., N    , Nj...) => (N    , Ni..., Nj...)
        //
        // indices is used as is.
        // indices: (Nk...)

        IndexableArray<const T> a_iarray{a};
        Axes a_perm = MakeRollingPermutation(axis, axis + 1, a.ndim());
        a_iarray.Permute(a_perm);
        Shape a_shape = internal::TransposeShape(a.shape(), a_perm);
        Indexer<> a_indexer{a_shape};

        IndexableArray<const T> b_iarray{b};
        Axes b_perm = MakeRollingPermutation(axis, axis + indices.ndim(), b.ndim());
        b_iarray.Permute(b_perm);
        Shape b_shape = internal::TransposeShape(b.shape(), b_perm);
        Indexer<> b_indexer{b_shape};

        IndexableArray<T> out_iarray{out};
        Axes out_perm = MakeRollingPermutation(axis, axis + 1, out.ndim());
        out_iarray.Permute(out_perm);
        Shape out_shape = internal::TransposeShape(out.shape(), out_perm);
        Indexer<> out_indexer{out_shape};

        IndexableArray<const int64_t> indices_iarray{indices};
        Indexer<> indices_indexer{indices.shape()};

        // size of (Ni..., Nj...) part
        int64_t common_total_size = a_indexer.total_size() / a_shape[0];

        static const int kMaxBlockSize = CudaOccupancyMaxPotentialBlockSize(&AddAtKernel<T>).block_size;
        int64_t total_size = out_indexer.total_size();
        int64_t grid_size = (total_size + kMaxBlockSize - 1) / kMaxBlockSize;
        int64_t block_size = std::min<int64_t>(total_size, kMaxBlockSize);

        AddAtKernel<<<grid_size, block_size>>>(
                a_iarray, b_iarray, out_iarray, indices_iarray, b_indexer, out_indexer, indices_indexer, common_total_size, a_shape[0]);
    });
}

namespace {

template <typename T>
struct IdentityImpl {
    explicit IdentityImpl(int64_t n) : n_plus_one{n + 1} {}
    __device__ void operator()(int64_t i, T& out) { out = i % n_plus_one == 0 ? T{1} : T{0}; }
    int64_t n_plus_one;
};

}  // namespace

void CudaDevice::Identity(const Array& out) {
    assert(out.ndim() == 2);
    assert(out.shape()[0] == out.shape()[1]);

    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<T>(IdentityImpl<T>{out.shape()[0]}, out);
    });
}

namespace {

template <typename T>
struct EyeImpl {
    EyeImpl(int64_t m, int64_t k) : start{k < 0 ? -k * m : k}, stop{m * (m - k)}, step{m + 1} {}
    __device__ void operator()(int64_t i, T& out) { out = start <= i && i < stop && (i - start) % step == 0 ? T{1} : T{0}; }
    int64_t start;
    int64_t stop;
    int64_t step;
};

}  // namespace

void CudaDevice::Eye(int64_t k, const Array& out) {
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [k, &out](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<T>(EyeImpl<T>{out.shape()[1], k}, out);
    });
}

namespace {

template <typename T>
__global__ void SetVecInMat(
        IndexableArray<const T, 1> vec_iarray,
        IndexableArray<T, 2> mat_iarray,
        Indexer<1> vec_indexer,
        Indexer<1> mat_row_indexer,
        Indexer<1> mat_col_indexer,
        Indexer<2> mat_indexer,
        int64_t mat_row_start,
        int64_t mat_col_start) {
    for (auto vec_it = vec_indexer.It(blockIdx.x * blockDim.x + threadIdx.x, blockDim.x * gridDim.x); vec_it; ++vec_it) {
        auto mat_row_it = mat_row_indexer.It(mat_row_start + vec_it.raw_index());
        auto mat_col_it = mat_col_indexer.It(mat_col_start + vec_it.raw_index());
        auto mat_it = mat_indexer.At(mat_row_it, mat_col_it);
        mat_iarray[mat_it] = vec_iarray[vec_it];
    }
}

}  // namespace

void CudaDevice::Diagflat(const Array& v, int64_t k, const Array& out) {
    assert(v.ndim() == 1);
    assert(out.ndim() == 2);

    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;

        // Start indices for the 2-D array axes with applied offset k.
        int64_t row_start{0};
        int64_t col_start{0};

        if (k >= 0) {
            col_start += k;
        } else {
            row_start -= k;
        }

        // Initialize all elements to 0 first instead of conditionally filling in the diagonal.
        Fill(out, T{0});

        IndexableArray<const T, 1> v_iarray{v};
        IndexableArray<T, 2> out_iarray{out};
        Indexer<1> v_indexer{v.shape()};
        Indexer<1> out_row_indexer{Shape{out.shape()[0]}};
        Indexer<1> out_col_indexer{Shape{out.shape()[1]}};
        Indexer<2> out_indexer{out.shape()};

        static const int kMaxBlockSize = CudaOccupancyMaxPotentialBlockSize(&SetVecInMat<T>).block_size;
        int64_t total_size = out_indexer.total_size();
        int64_t grid_size = (total_size + kMaxBlockSize - 1) / kMaxBlockSize;
        int64_t block_size = std::min<int64_t>(total_size, kMaxBlockSize);

        SetVecInMat<<<grid_size, block_size>>>(
                v_iarray, out_iarray, v_indexer, out_row_indexer, out_col_indexer, out_indexer, row_start, col_start);
    });
}

namespace {

template <typename T>
struct LinspaceImpl {
    __device__ void operator()(int64_t i, T& out) {
        double value = n == 1 ? start : (start * (n - 1 - i) + stop * i) / (n - 1);
        out = cuda_numeric_cast<T>(value);
    }
    int64_t n;
    double start;
    double stop;
};

}  // namespace

void CudaDevice::Linspace(double start, double stop, const Array& out) {
    assert(out.ndim() == 1);
    assert(out.shape()[0] > 0);

    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        int64_t n = out.shape()[0];
        Elementwise<T>(LinspaceImpl<T>{n, start, stop}, out);
    });
}

namespace {

// def get_conv_outsize(size, k, s, p, cover_all=False, d=1):
//     """Calculates output size of convolution.
//
//     This function takes the size of input feature map, kernel, stride, and
//     pooling of one particular dimension, then calculates the output feature
//     map size of that dimension.
//
//     .. seealso:: :func:`~chainer.utils.get_deconv_outsize`
//
//     Args:
//         size (int): The size of input feature map. It usually is the length of
//             a side of feature map.
//         k (int): The size of convolution kernel.
//         s (int): The size of stride.
//         p (int): The size of padding.
//         cover_all (bool): Use ``cover_all`` option or not.
//         d (int): The size of dilation.
//
//     Returns:
//         int: The expected output size of the convolution operation.
//
//     """
//     dk = k + (k - 1) * (d - 1)
//     if cover_all:
//         return (size + p * 2 - dk + s - 1) // s + 1
//     else:
//         return (size + p * 2 - dk) // s + 1

// TODO(sonots): Use same codes of NativeDevice
int64_t GetConvOutDim(int64_t in_dim, int64_t kernel_size, int64_t stride, int64_t pad, bool cover_all) {
    if (cover_all) {
        return (in_dim + pad * 2 - kernel_size + stride - 1) / stride + 1;
    }
    return (in_dim + pad * 2 - kernel_size) / stride + 1;
}

// def convolution_forward(
//         core.ndarray x, core.ndarray W, core.ndarray b, core.ndarray y,
//         tuple pad, tuple stride, tuple dilation, int groups, *,
//         bint auto_tune, str tensor_core):
// TODO(sonots): Support tensor core
void ConvolutionForward(
        CudaDevice& device,
        const Array& x,
        const Array& w,
        const nonstd::optional<Array>& b,
        Array& y,
        const StackVector<int64_t, kMaxNdim>& pad,
        const StackVector<int64_t, kMaxNdim>& stride,
        const nonstd::optional<StackVector<int64_t, kMaxNdim>>& dilation = nonstd::nullopt,
        int groups = 1) {
    // cdef int dev_id = x.data.device.id
    // assert dev_id == W.data.device.id
    // assert dev_id == y.data.device.id
    assert(&device == &x.device());
    assert(&device == &y.device());
    assert(&device == &w.device());

    // cdef float float_zero = 0, float_one = 1
    // cdef double double_zero = 0, double_one = 1
    // cdef size_t zero, one
    // if x.dtype == 'd':
    //     zero = <size_t>&double_zero
    //     one = <size_t>&double_one
    // else:
    //     zero = <size_t>&float_zero
    //     one = <size_t>&float_one
    //
    // cdef bint use_tensor_core = _should_use_tensor_core(tensor_core, x.dtype)
    // cdef tuple conv_param = (pad, stride, x.dtype)
    //
    // # cuDNN 7 supports dilation only in *_FWD_ALGO_IMPLICIT_GEMM, but
    // # it supports Tensor Cores only in *_FWD_ALGO_IMPLICIT_PRECOMP_GEMM.
    // if use_tensor_core:
    //     for i in dilation:
    //         if i > 1:
    //             use_tensor_core = False
    //             break

    auto zero = x.dtype() == Dtype::kFloat64 ? double{0} : float{0};
    auto one = x.dtype() == Dtype::kFloat64 ? double{1} : float{1};

    // handle = get_handle()
    // x = core.ascontiguousarray(x)
    // W = core.ascontiguousarray(W)
    cudnnHandle_t handle = device.cudnn_handle();
    Array x_ = AsContiguousArray(x);
    Array w_ = AsContiguousArray(w);
    assert(y.IsContiguous());

    // # TODO(okuta) check performance
    // cdef size_t x_desc = cudnn.createTensorDescriptor()
    // cdef size_t y_desc = cudnn.createTensorDescriptor()
    // cdef size_t b_desc = cudnn.createTensorDescriptor()
    // cdef size_t filter_desc = cudnn.createFilterDescriptor()
    // cdef size_t conv_desc = cudnn.createConvolutionDescriptor()
    //
    // cdef int algo
    // cdef size_t max_workspace_size = get_max_workspace_size()
    // cdef size_t workspace_size = 0
    // try:
    //     _create_tensor_nd_descriptor(x_desc, x, -1)
    //     _create_tensor_nd_descriptor(y_desc, y, -1)
    //     _create_filter_descriptor(filter_desc, W, cudnn.CUDNN_TENSOR_NCHW)
    //     _create_convolution_descriptor(
    //         conv_desc, pad, stride, dilation, groups, x.dtype,
    //         cudnn.CUDNN_CROSS_CORRELATION, use_tensor_core)
    std::shared_ptr<cudnnTensorStruct> x_desc = CreateTensorDescriptor(x_);
    std::shared_ptr<cudnnTensorStruct> y_desc = CreateTensorDescriptor(y);
    std::shared_ptr<cudnnFilterStruct> filter_desc = CreateFilterDescriptor(w_, CUDNN_TENSOR_NCHW);
    std::shared_ptr<cudnnConvolutionStruct> conv_desc =
            CreateConvolutionDescriptor(stride, pad, x.dtype(), CUDNN_CROSS_CORRELATION, dilation, groups);
    size_t max_workspace_size = device.max_workspace_size();

    // if auto_tune and _cudnn_version >= 5000:
    //     algo, workspace_size = _find_algorithm_fwd(
    //         x, W, y, conv_param, handle, x_desc, filter_desc,
    //         conv_desc, y_desc, max_workspace_size)
    // else:
    //     algo, workspace_size = _get_algorithm_fwd(
    //         handle, x_desc, filter_desc, conv_desc, y_desc,
    //         max_workspace_size, use_tensor_core)

    // auto_tune
    std::tuple<cudnnConvolutionFwdAlgo_t, size_t> algo_workspace_size =
            FindConvolutionForwardAlgorithm(handle, x_desc, x_, filter_desc, w_, conv_desc, y_desc, y, max_workspace_size);
    // max_workspace_size = max(max_workspace_size, workspace_size)
    // # TODO(okuta): allocate best size memory
    // workspace = memory.alloc(max_workspace_size)
    cudnnConvolutionFwdAlgo_t algo = std::get<0>(algo_workspace_size);
    size_t workspace_size = std::max(max_workspace_size, std::get<1>(algo_workspace_size));
    std::shared_ptr<void> workspace = device.Allocate(workspace_size);

    // cudnn.convolutionForward(
    //     handle, one, x_desc, x.data.ptr, filter_desc, W.data.ptr,
    //     conv_desc, algo, workspace.ptr, max_workspace_size, zero, y_desc,
    //     y.data.ptr)
    CheckCudnnError(cudnnConvolutionForward(
            handle,
            &one,
            x_desc.get(),
            x_.data().get(),
            filter_desc.get(),
            w_.data().get(),
            conv_desc.get(),
            algo,
            workspace.get(),
            workspace_size,
            &zero,
            y_desc.get(),
            y.data().get()));

    // if b is not None:
    //     assert dev_id == b.data.device.id
    //     ndim = x.ndim - 2
    //     b = core.ascontiguousarray(b).reshape((1, -1) + (1,) * ndim)
    //     _create_tensor_nd_descriptor(b_desc, b, -1)
    //     cudnn.addTensor_v3(handle, one, b_desc,
    //                        b.data.ptr, one, y_desc, y.data.ptr)
    if (b) {
        assert(&device == &b->device());
        int8_t ndim = x.ndim() - 2;
        Shape new_shape;
        new_shape.emplace_back(1);
        new_shape.emplace_back(-1);
        for (int8_t idim = 0; idim < ndim; ++idim) {
            new_shape.emplace_back(1);
        }
        Array b_contig = AsContiguousArray(*b).Reshape(new_shape);
        std::shared_ptr<cudnnTensorStruct> b_desc = CreateTensorDescriptor(*b);
        CheckCudnnError(cudnnAddTensor(handle, &one, b_desc.get(), b->data().get(), &one, y_desc.get(), y.data().get()));
    }
}

}  // namespace

// chainer/functions/connection/convolution_nd.py
// def _forward_cudnn(self, x, W, b):
//     out_c = W.shape[0]      # (c_O, _, k_1, k_2, ..., k_N)
//     ksize = W.shape[2:]
//     n, c = x.shape[:2]      # (n, c_I, d_1, d_2, ..., d_N)
//     dims = x.shape[2:]
//     stride = self.stride
//     pad = self.pad

//     # Make empty array for result.
//     outs = tuple(
//         conv.get_conv_outsize(d, k, s, p, cover_all=self.cover_all)
//         for (d, k, s, p) in zip(dims, ksize, stride, pad))
//     assert all(out > 0 for out in outs), 'Output sizes should be positive.'
//     y_shape = (n, out_c) + outs  # (n, c_O, out_1, out_2, ..., out_N)
//     y = cuda.cupy.empty(y_shape, dtype=x.dtype)
//     dilation = (1,) * self.ndim
//     groups = 1
//     auto_tune = configuration.config.autotune
//     tensor_core = configuration.config.use_cudnn_tensor_core
//     cuda.cudnn.convolution_forward(
//         x, W, b, y, pad, stride, dilation, groups,
//         auto_tune=auto_tune, tensor_core=tensor_core)
//     return y,

Array CudaDevice::Conv(
        const Array& x,
        const Array& w,
        const nonstd::optional<Array>& b,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        bool cover_all) {
    int8_t ndim = w.ndim() - 2;
    assert(ndim > 0);

    if (cover_all) {
        throw XchainerError{"CUDA convolution does not support cover_all"};
    }

    // out_c = W.shape[0]      # (c_O, _, k_1, k_2, ..., k_N)
    int64_t out_c = w.shape()[0];
    // ksize = W.shape[2:]
    // Get the kernel size from the weight array as w.shape[2:]
    StackVector<int64_t, kMaxNdim> ksize;
    std::copy_n(w.shape().begin() + 2, ndim, std::back_inserter(ksize));
    // n, c = x.shape[:2]      # (n, c_I, d_1, d_2, ..., d_N)
    int64_t n = x.shape()[0];
    int64_t c = x.shape()[1];
    // dims = x.shape[2:]
    StackVector<int64_t, kMaxNdim> dims;
    std::copy_n(x.shape().begin() + 2, ndim, std::back_inserter(dims));

    // # Make empty array for result.
    // outs = tuple(
    //     conv.get_conv_outsize(d, k, s, p, cover_all=self.cover_all)
    //     for (d, k, s, p) in zip(dims, ksize, stride, pad))
    // assert all(out > 0 for out in outs), 'Output sizes should be positive.'
    // y_shape = (n, out_c) + outs  # (n, c_O, out_1, out_2, ..., out_N)
    // y = cuda.cupy.empty(y_shape, dtype=x.dtype)

    // Create the output array.
    StackVector<int64_t, kMaxNdim> out_dims;  // Number of patches along each axis
    for (int8_t i = 0; i < ndim; ++i) {
        out_dims.emplace_back(GetConvOutDim(x.shape()[i + 2], ksize[i], stride[i], pad[i], cover_all));
        assert(out_dims.back() > 0);
    }

    int64_t batch_size = x.shape()[0];
    int64_t channels = x.shape()[1];

    Shape out_shape{batch_size, channels};
    std::copy(ksize.begin(), ksize.end(), std::back_inserter(out_shape));
    std::copy(out_dims.begin(), out_dims.end(), std::back_inserter(out_shape));
    Array y = Empty(out_shape, x.dtype(), *this);

    //     dilation = (1,) * self.ndim
    //     groups = 1
    //     auto_tune = configuration.config.autotune
    //     tensor_core = configuration.config.use_cudnn_tensor_core
    //     cuda.cudnn.convolution_forward(
    //         x, W, b, y, pad, stride, dilation, groups,
    //         auto_tune=auto_tune, tensor_core=tensor_core)
    //     return y,

    ConvolutionForward(*this, x, w, b, y, pad, stride);

<<<<<<< da720f1f552e4ad65580a64c98f4175f0b1910db
    // return y,
>>>>>>> CudaDevice: Convolution (wip)
}

Array CudaDevice::ConvTranspose(
        const Array& /*x*/,
        const Array& /*w*/,
        const nonstd::optional<Array>& /*b*/,
        const StackVector<int64_t, kMaxNdim>& /*stride*/,
        const StackVector<int64_t, kMaxNdim>& /*pad*/,
        const nonstd::optional<StackVector<int64_t, kMaxNdim>>& /*out_size*/) {
    // TODO(hvy): Implement it
    throw NotImplementedError{""};
=======
    return y;
>>>>>>> ConvolutionForward
}

void CudaDevice::Synchronize() {
    CheckCudaError(cudaSetDevice(index()));
    CheckCudaError(cudaDeviceSynchronize());
}

}  // namespace cuda
}  // namespace xchainer
