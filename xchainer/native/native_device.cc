#include "xchainer/native/native_device.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <tuple>
#include <vector>

#include <gsl/gsl>

#include "xchainer/array.h"
#include "xchainer/dtype.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"
#include "xchainer/native/reduce.h"
#include "xchainer/reduction_kernel_arg.h"
#include "xchainer/scalar.h"

namespace xchainer {
namespace native {

std::shared_ptr<void> NativeDevice::Allocate(size_t bytesize) { return std::make_unique<uint8_t[]>(bytesize); }

void NativeDevice::MemoryCopyFrom(void* dst, const void* src, size_t bytesize, Device& src_device) {
    assert(nullptr != dynamic_cast<NativeDevice*>(&src_device) && "Native device only supports copy between native devices");
    std::memcpy(dst, src, bytesize);
}

void NativeDevice::MemoryCopyTo(void* dst, const void* src, size_t bytesize, Device& dst_device) {
    assert(nullptr != dynamic_cast<NativeDevice*>(&dst_device) && "Native device only supports copy between native devices");
    std::memcpy(dst, src, bytesize);
}

std::shared_ptr<void> NativeDevice::TransferDataFrom(
        Device& src_device, const std::shared_ptr<void>& src_ptr, size_t offset, size_t bytesize) {
    std::shared_ptr<void> dst_ptr = Allocate(bytesize);
    MemoryCopyFrom(dst_ptr.get(), &(static_cast<int8_t*>(src_ptr.get())[offset]), bytesize, src_device);
    return dst_ptr;
}

std::shared_ptr<void> NativeDevice::TransferDataTo(
        Device& dst_device, const std::shared_ptr<void>& src_ptr, size_t offset, size_t bytesize) {
    return dst_device.TransferDataFrom(*this, src_ptr, offset, bytesize);
}

std::shared_ptr<void> NativeDevice::FromHostMemory(const std::shared_ptr<void>& src_ptr, size_t bytesize) {
    (void)bytesize;  // unused
    return src_ptr;
}

void NativeDevice::Fill(const Array& out, Scalar value) {
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        T c_value{value};

        IndexableArray<T> out_iarray{out};
        Indexer indexer{out.shape()};
        for (int64_t i = 0; i < indexer.total_size(); ++i) {
            indexer.Set(i);
            out_iarray[indexer] = c_value;
        }
    });
}

void NativeDevice::ArgMax(const Array& src, const std::vector<int8_t>& axis, const Array& out) {
    assert(src.GetTotalSize() > 0);
    assert(out.GetTotalSize() > 0);
    assert(out.ndim() == src.ndim() - static_cast<int64_t>(axis.size()));
    CheckDevicesCompatible(src, out);

    VisitDtype(src.dtype(), [&src, &axis, &out](auto pt) {
        using T = typename decltype(pt)::type;
        struct ArgMaxImpl {
            struct MaxAndArgMax {
                T max;
                int64_t argmax;
            };

            MaxAndArgMax Identity() { return {T{}, -1}; }
            MaxAndArgMax MapIn(T in, int64_t index) { return {in, index}; }
            void Reduce(MaxAndArgMax next, MaxAndArgMax& accum) {
                if (accum.argmax < 0 || accum.max < next.max) {
                    accum = next;
                }
            }
            int64_t MapOut(MaxAndArgMax accum) { return accum.argmax; }
        };
        Reduce(MakeReductionKernelArg<T, int64_t>(src, axis, out), ArgMaxImpl{});
    });
}

void NativeDevice::Sum(const Array& src, const std::vector<int8_t>& axis, const Array& out) {
    // keepdims denotes the corresponding argument in Array::Sum().
    assert(out.ndim() == src.ndim() - static_cast<int64_t>(axis.size()) ||  // keepdims=false
           out.ndim() == src.ndim());                                       // keepdims=true
    CheckDevicesCompatible(src, out);

    VisitDtype(src.dtype(), [&src, &axis, &out](auto pt) {
        using T = typename decltype(pt)::type;
        struct SumImpl {
            T Identity() { return T{0}; }
            T MapIn(T in, int64_t /*index*/) { return in; }
            void Reduce(T next, T& accum) { accum += next; }
            T MapOut(T accum) { return accum; }
        };
        Reduce(MakeReductionKernelArg<T, T>(src, axis, out), SumImpl{});
    });
}

void NativeDevice::Copy(const Array& src, const Array& out) {
    CheckDevicesCompatible(src, out);
    VisitDtype(src.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> src_iarray{src};
        IndexableArray<T> out_iarray{out};
        Indexer indexer{src.shape()};

        for (int64_t i = 0; i < indexer.total_size(); ++i) {
            indexer.Set(i);
            out_iarray[indexer] = src_iarray[indexer];
        }
    });
}

void NativeDevice::Equal(const Array& lhs, const Array& rhs, const Array& out) {
    CheckDevicesCompatible(lhs, rhs, out);
    VisitDtype(lhs.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> lhs_iarray{lhs};
        IndexableArray<const T> rhs_iarray{rhs};
        IndexableArray<bool> out_iarray{out};
        Indexer indexer{lhs.shape()};

        for (int64_t i = 0; i < indexer.total_size(); ++i) {
            indexer.Set(i);
            out_iarray[indexer] = lhs_iarray[indexer] == rhs_iarray[indexer];
        }
    });
}

void NativeDevice::Add(const Array& lhs, const Array& rhs, const Array& out) {
    CheckDevicesCompatible(lhs, rhs, out);
    VisitDtype(lhs.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> lhs_iarray{lhs};
        IndexableArray<const T> rhs_iarray{rhs};
        IndexableArray<T> out_iarray{out};
        Indexer indexer{lhs.shape()};

        for (int64_t i = 0; i < indexer.total_size(); ++i) {
            indexer.Set(i);
            out_iarray[indexer] = lhs_iarray[indexer] + rhs_iarray[indexer];
        }
    });
}

void NativeDevice::Subtract(const Array& lhs, const Array& rhs, const Array& out) {
    CheckDevicesCompatible(lhs, rhs, out);
    VisitDtype(lhs.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> lhs_iarray{lhs};
        IndexableArray<const T> rhs_iarray{rhs};
        IndexableArray<T> out_iarray{out};
        Indexer indexer{lhs.shape()};

        for (int64_t i = 0; i < indexer.total_size(); ++i) {
            indexer.Set(i);
            out_iarray[indexer] = lhs_iarray[indexer] - rhs_iarray[indexer];
        }
    });
}

void NativeDevice::Mul(const Array& lhs, Scalar rhs, const Array& out) {
    CheckDevicesCompatible(lhs, out);
    VisitDtype(lhs.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> lhs_iarray{lhs};
        IndexableArray<T> out_iarray{out};
        Indexer indexer{lhs.shape()};

        for (int64_t i = 0; i < indexer.total_size(); ++i) {
            indexer.Set(i);
            out_iarray[indexer] = lhs_iarray[indexer] * static_cast<T>(rhs);
        }
    });
}

void NativeDevice::Mul(const Array& lhs, const Array& rhs, const Array& out) {
    CheckDevicesCompatible(lhs, rhs, out);
    VisitDtype(lhs.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> lhs_iarray{lhs};
        IndexableArray<const T> rhs_iarray{rhs};
        IndexableArray<T> out_iarray{out};
        Indexer indexer{lhs.shape()};

        for (int64_t i = 0; i < indexer.total_size(); ++i) {
            indexer.Set(i);
            out_iarray[indexer] = lhs_iarray[indexer] * rhs_iarray[indexer];
        }
    });
}

void NativeDevice::IfLessElse(const Array& lhs, Scalar rhs, Scalar pos, const Array& neg, const Array& out) {
    CheckDevicesCompatible(lhs, neg, out);
    VisitDtype(lhs.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> lhs_iarray{lhs};
        IndexableArray<const T> neg_iarray{neg};
        IndexableArray<T> out_iarray{out};
        Indexer indexer{lhs.shape()};
        T rhs_value{rhs};
        T pos_value{pos};

        for (int64_t i = 0; i < indexer.total_size(); ++i) {
            indexer.Set(i);
            out_iarray[indexer] = lhs_iarray[indexer] < rhs_value ? pos_value : neg_iarray[indexer];
        }
    });
}

void NativeDevice::Dot(const Array& lhs, const Array& rhs, const Array& out) {
    CheckDevicesCompatible(lhs, rhs, out);
    VisitDtype(lhs.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> lhs_iarray{lhs};
        IndexableArray<const T> rhs_iarray{rhs};
        IndexableArray<T> out_iarray{out};

        // These asserts have to check iarray instead of the original array, otherwise clang-tidy fails bound-checking.
        assert(lhs_iarray.ndim() == 2);
        assert(rhs_iarray.ndim() == 2);
        assert(out_iarray.ndim() == 2);

        int64_t m = lhs.shape()[0];
        int64_t k = lhs.shape()[1];
        int64_t n = rhs.shape()[1];
        assert(rhs.shape()[0] == k);
        assert(out.shape()[0] == m);
        assert(out.shape()[1] == n);

        // TODO(beam2d): Use BLAS.
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                int64_t out_i[] = {i, j};
                T& out_value = out_iarray[out_i];
                out_value = 0;
                for (int64_t l = 0; l < k; ++l) {
                    int64_t lhs_i[] = {i, l};
                    int64_t rhs_i[] = {l, j};
                    out_value += lhs_iarray[lhs_i] * rhs_iarray[rhs_i];
                }
            }
        }
    });
}

void NativeDevice::Log(const Array& x, const Array& out) {
    CheckDevicesCompatible(x, out);
    VisitDtype(x.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> x_iarray{x};
        IndexableArray<T> out_iarray{out};
        Indexer indexer{x.shape()};

        for (int64_t i = 0; i < indexer.total_size(); ++i) {
            indexer.Set(i);
            out_iarray[indexer] = std::log(x_iarray[indexer]);
        }
    });
}

void NativeDevice::Synchronize() {}

}  // namespace native
}  // namespace xchainer
