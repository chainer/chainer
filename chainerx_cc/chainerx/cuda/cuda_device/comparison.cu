#include "chainerx/cuda/cuda_device.h"

#include <cstdint>

#include <cuda_runtime.h>

#include "chainerx/array.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/cuda_set_device_scope.h"
#include "chainerx/cuda/data_type.cuh"
#include "chainerx/cuda/elementwise.cuh"
#include "chainerx/cuda/op_regist.h"
#include "chainerx/cuda/reduce.cuh"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/routines/logic.h"

namespace chainerx {
namespace cuda {
namespace {

template <typename T>
struct EqualImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x1, CudaType x2, bool& out) { out = x1 == x2; }
};

class CudaEqualOp : public EqualOp {
public:
    void Call(const Array& x1, const Array& x2, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, x2, out);
        Dtype dtype = PromoteTypes(x1.dtype(), x2.dtype());
        const Array& x1_cast = x1.dtype() == dtype ? x1 : x1.AsType(dtype);
        const Array& x2_cast = x2.dtype() == dtype ? x2 : x2.AsType(dtype);
        CudaSetDeviceScope scope{device.index()};
        VisitDtype(dtype, [&](auto pt) {
            using T = typename decltype(pt)::type;
            Elementwise<const T, const T, bool>(EqualImpl<T>{}, x1_cast, x2_cast, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_OP(EqualOp, CudaEqualOp);

template <typename T>
struct NotEqualImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x1, CudaType x2, bool& out) { out = x1 != x2; }
};

class CudaNotEqualOp : public NotEqualOp {
public:
    void Call(const Array& x1, const Array& x2, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, x2, out);
        Dtype dtype = PromoteTypes(x1.dtype(), x2.dtype());
        const Array& x1_cast = x1.dtype() == dtype ? x1 : x1.AsType(dtype);
        const Array& x2_cast = x2.dtype() == dtype ? x2 : x2.AsType(dtype);
        CudaSetDeviceScope scope{device.index()};
        VisitDtype(dtype, [&](auto pt) {
            using T = typename decltype(pt)::type;
            Elementwise<const T, const T, bool>(NotEqualImpl<T>{}, x1_cast, x2_cast, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_OP(NotEqualOp, CudaNotEqualOp);

template <typename T>
struct GreaterImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x1, CudaType x2, bool& out) { out = x1 > x2; }
};

class CudaGreaterOp : public GreaterOp {
public:
    void Call(const Array& x1, const Array& x2, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, x2, out);
        Dtype dtype = PromoteTypes(x1.dtype(), x2.dtype());
        const Array& x1_cast = x1.dtype() == dtype ? x1 : x1.AsType(dtype);
        const Array& x2_cast = x2.dtype() == dtype ? x2 : x2.AsType(dtype);
        CudaSetDeviceScope scope{device.index()};
        VisitDtype(dtype, [&](auto pt) {
            using T = typename decltype(pt)::type;
            Elementwise<const T, const T, bool>(GreaterImpl<T>{}, x1_cast, x2_cast, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_OP(GreaterOp, CudaGreaterOp);

template <typename T>
struct GreaterEqualImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x1, CudaType x2, bool& out) { out = x1 >= x2; }
};

class CudaGreaterEqualOp : public GreaterEqualOp {
public:
    void Call(const Array& x1, const Array& x2, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, x2, out);
        Dtype dtype = PromoteTypes(x1.dtype(), x2.dtype());
        const Array& x1_cast = x1.dtype() == dtype ? x1 : x1.AsType(dtype);
        const Array& x2_cast = x2.dtype() == dtype ? x2 : x2.AsType(dtype);
        CudaSetDeviceScope scope{device.index()};
        VisitDtype(dtype, [&](auto pt) {
            using T = typename decltype(pt)::type;
            Elementwise<const T, const T, bool>(GreaterEqualImpl<T>{}, x1_cast, x2_cast, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_OP(GreaterEqualOp, CudaGreaterEqualOp);

template <typename T>
struct LogicalNotImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x, bool& out) { out = !x; }
};

class CudaLogicalNotOp : public LogicalNotOp {
public:
    void Call(const Array& x, const Array& out) override {
        Device& device = x.device();
        device.CheckDevicesCompatible(x, out);
        CudaSetDeviceScope scope{device.index()};
        VisitDtype(x.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            Elementwise<const T, bool>(LogicalNotImpl<T>{}, x, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_OP(LogicalNotOp, CudaLogicalNotOp);

template <typename T>
struct LogicalAndImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x1, CudaType x2, bool& out) { out = x1 && x2; }
};

class CudaLogicalAndOp : public LogicalAndOp {
public:
    void Call(const Array& x1, const Array& x2, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, x2, out);
        Dtype dtype = PromoteTypes(x1.dtype(), x2.dtype());
        const Array& x1_cast = x1.dtype() == dtype ? x1 : x1.AsType(dtype);
        const Array& x2_cast = x2.dtype() == dtype ? x2 : x2.AsType(dtype);
        CudaSetDeviceScope scope{device.index()};
        VisitDtype(dtype, [&](auto pt) {
            using T = typename decltype(pt)::type;
            Elementwise<const T, const T, bool>(LogicalAndImpl<T>{}, x1_cast, x2_cast, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_OP(LogicalAndOp, CudaLogicalAndOp);

template <typename T>
struct LogicalOrImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x1, CudaType x2, bool& out) { out = x1 || x2; }
};

class CudaLogicalOrOp : public LogicalOrOp {
public:
    void Call(const Array& x1, const Array& x2, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, x2, out);
        Dtype dtype = PromoteTypes(x1.dtype(), x2.dtype());
        const Array& x1_cast = x1.dtype() == dtype ? x1 : x1.AsType(dtype);
        const Array& x2_cast = x2.dtype() == dtype ? x2 : x2.AsType(dtype);
        CudaSetDeviceScope scope{device.index()};
        VisitDtype(dtype, [&](auto pt) {
            using T = typename decltype(pt)::type;
            Elementwise<const T, const T, bool>(LogicalOrImpl<T>{}, x1_cast, x2_cast, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_OP(LogicalOrOp, CudaLogicalOrOp);

template <typename In>
struct AllImpl {
    using InCudaType = cuda_internal::DataType<In>;
    __device__ bool Identity() { return true; }
    __device__ bool MapIn(InCudaType in, int64_t /*index*/) { return static_cast<bool>(in); }
    __device__ void Reduce(bool next, bool& accum) { accum = accum && next; }
    __device__ bool MapOut(bool accum) { return accum; }
};

class CudaAllOp : public AllOp {
public:
    void Call(const Array& a, const Axes& axis, const Array& out) {
        CHAINERX_ASSERT(internal::IsValidReductionShape(a.shape(), axis, out.shape(), true));
        Device& device = a.device();
        device.CheckDevicesCompatible(a, out);
        CudaSetDeviceScope scope{device.index()};
        const Array& a_cast = a.dtype() == out.dtype() ? a : a.AsType(out.dtype());
        auto do_all = [&a_cast, &axis, &out](auto in_pt) {
            using In = typename decltype(in_pt)::type;
            Reduce<In, bool>(a_cast, axis, out, AllImpl<In>{});
        };

        VisitDtype(out.dtype(), do_all);
    }
};

CHAINERX_CUDA_REGISTER_OP(AllOp, CudaAllOp);

template <typename In>
struct AnyImpl {
    using InCudaType = cuda_internal::DataType<In>;
    __device__ bool Identity() { return false; }
    __device__ bool MapIn(InCudaType in, int64_t /*index*/) { return static_cast<bool>(in); }
    __device__ void Reduce(bool next, bool& accum) { accum = accum || next; }
    __device__ bool MapOut(bool accum) { return accum; }
};

class CudaAnyOp : public AnyOp {
public:
    void Call(const Array& a, const Axes& axis, const Array& out) {
        CHAINERX_ASSERT(internal::IsValidReductionShape(a.shape(), axis, out.shape(), true));
        Device& device = a.device();
        device.CheckDevicesCompatible(a, out);
        CudaSetDeviceScope scope{device.index()};
        const Array& a_cast = a.dtype() == out.dtype() ? a : a.AsType(out.dtype());
        auto do_any = [&a_cast, &axis, &out](auto in_pt) {
            using In = typename decltype(in_pt)::type;
            Reduce<In, bool>(a_cast, axis, out, AnyImpl<In>{});
        };

        VisitDtype(out.dtype(), do_any);
    }
};

CHAINERX_CUDA_REGISTER_OP(AnyOp, CudaAnyOp);

}  // namespace
}  // namespace cuda
}  // namespace chainerx
