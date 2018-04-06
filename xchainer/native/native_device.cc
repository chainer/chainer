#include "xchainer/native/native_device.h"

#include <algorithm>
#include <cassert>
#include <cmath>
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

void NativeDevice::ArgMax(const Array& a, const std::vector<int8_t>& axis, const Array& out) {
    assert(a.GetTotalSize() > 0);
    assert(out.GetTotalSize() > 0);
    assert(out.ndim() == a.ndim() - static_cast<int64_t>(axis.size()));
    CheckDevicesCompatible(a, out);

    VisitDtype(a.dtype(), [&a, &axis, &out](auto pt) {
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
        Reduce(MakeReductionKernelArg<T, int64_t>(a, axis, out), ArgMaxImpl{});
    });
}

void NativeDevice::Sum(const Array& a, const std::vector<int8_t>& axis, const Array& out) {
    // keepdims denotes the corresponding argument in Array::Sum().
    assert(out.ndim() == a.ndim() - static_cast<int64_t>(axis.size()) ||  // keepdims=false
           out.ndim() == a.ndim());                                       // keepdims=true
    CheckDevicesCompatible(a, out);

    VisitDtype(out.dtype(), [&a, &axis, &out](auto pt) {
        using T = typename decltype(pt)::type;
        struct SumImpl {
            T Identity() { return T{0}; }
            T MapIn(T in, int64_t /*index*/) { return in; }
            void Reduce(T next, T& accum) { accum += next; }
            T MapOut(T accum) { return accum; }
        };
        Reduce(MakeReductionKernelArg<T, T>(a, axis, out), SumImpl{});
    });
}

void NativeDevice::Copy(const Array& a, const Array& out) {
    CheckDevicesCompatible(a, out);
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> a_iarray{a};
        IndexableArray<T> out_iarray{out};
        Indexer indexer{out.shape()};

        for (int64_t i = 0; i < indexer.total_size(); ++i) {
            indexer.Set(i);
            out_iarray[indexer] = a_iarray[indexer];
        }
    });
}

void NativeDevice::Equal(const Array& x1, const Array& x2, const Array& out) {
    CheckDevicesCompatible(x1, x2, out);
    VisitDtype(x1.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> x1_iarray{x1};
        IndexableArray<const T> x2_iarray{x2};
        IndexableArray<bool> out_iarray{out};
        Indexer indexer{out.shape()};

        for (int64_t i = 0; i < indexer.total_size(); ++i) {
            indexer.Set(i);
            out_iarray[indexer] = x1_iarray[indexer] == x2_iarray[indexer];
        }
    });
}

void NativeDevice::Add(const Array& x1, const Array& x2, const Array& out) {
    CheckDevicesCompatible(x1, x2, out);
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> x1_iarray{x1};
        IndexableArray<const T> x2_iarray{x2};
        IndexableArray<T> out_iarray{out};
        Indexer indexer{out.shape()};

        for (int64_t i = 0; i < indexer.total_size(); ++i) {
            indexer.Set(i);
            out_iarray[indexer] = x1_iarray[indexer] + x2_iarray[indexer];
        }
    });
}

void NativeDevice::Subtract(const Array& x1, const Array& x2, const Array& out) {
    CheckDevicesCompatible(x1, x2, out);
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> x1_iarray{x1};
        IndexableArray<const T> x2_iarray{x2};
        IndexableArray<T> out_iarray{out};
        Indexer indexer{out.shape()};

        for (int64_t i = 0; i < indexer.total_size(); ++i) {
            indexer.Set(i);
            out_iarray[indexer] = x1_iarray[indexer] - x2_iarray[indexer];
        }
    });
}

void NativeDevice::Multiply(const Array& x1, const Array& x2, const Array& out) {
    CheckDevicesCompatible(x1, x2, out);
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> x1_iarray{x1};
        IndexableArray<const T> x2_iarray{x2};
        IndexableArray<T> out_iarray{out};
        Indexer indexer{out.shape()};

        for (int64_t i = 0; i < indexer.total_size(); ++i) {
            indexer.Set(i);
            out_iarray[indexer] = x1_iarray[indexer] * x2_iarray[indexer];
        }
    });
}

void NativeDevice::MultiplyAS(const Array& x1, Scalar x2, const Array& out) {
    CheckDevicesCompatible(x1, out);
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> x1_iarray{x1};
        IndexableArray<T> out_iarray{out};
        Indexer indexer{out.shape()};

        for (int64_t i = 0; i < indexer.total_size(); ++i) {
            indexer.Set(i);
            out_iarray[indexer] = x1_iarray[indexer] * static_cast<T>(x2);
        }
    });
}

void NativeDevice::IfLessElseASSA(const Array& x1, Scalar x2, Scalar pos, const Array& neg, const Array& out) {
    CheckDevicesCompatible(x1, neg, out);
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> x1_iarray{x1};
        IndexableArray<const T> neg_iarray{neg};
        IndexableArray<T> out_iarray{out};
        Indexer indexer{out.shape()};
        T x2_value{x2};
        T pos_value{pos};

        for (int64_t i = 0; i < indexer.total_size(); ++i) {
            indexer.Set(i);
            out_iarray[indexer] = x1_iarray[indexer] < x2_value ? pos_value : neg_iarray[indexer];
        }
    });
}

void NativeDevice::Dot(const Array& a, const Array& b, const Array& out) {
    CheckDevicesCompatible(a, b, out);
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> a_iarray{a};
        IndexableArray<const T> b_iarray{b};
        IndexableArray<T> out_iarray{out};

        // These asserts have to check iarray instead of the original array, otherwise clang-tidy fails bound-checking.
        assert(a_iarray.ndim() == 2);
        assert(b_iarray.ndim() == 2);
        assert(out_iarray.ndim() == 2);

        int64_t m = a.shape()[0];
        int64_t k = a.shape()[1];
        int64_t n = b.shape()[1];
        assert(b.shape()[0] == k);
        assert(out.shape()[0] == m);
        assert(out.shape()[1] == n);

        // TODO(beam2d): Use BLAS.
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                int64_t out_i[] = {i, j};
                T& out_value = out_iarray[out_i];
                out_value = 0;
                for (int64_t l = 0; l < k; ++l) {
                    int64_t a_i[] = {i, l};
                    int64_t b_i[] = {l, j};
                    out_value += a_iarray[a_i] * b_iarray[b_i];
                }
            }
        }
    });
}

void NativeDevice::Exp(const Array& x, const Array& out) {
    CheckDevicesCompatible(x, out);
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> x_iarray{x};
        IndexableArray<T> out_iarray{out};
        Indexer indexer{out.shape()};

        for (int64_t i = 0; i < indexer.total_size(); ++i) {
            indexer.Set(i);
            out_iarray[indexer] = std::exp(x_iarray[indexer]);
        }
    });
}

void NativeDevice::Log(const Array& x, const Array& out) {
    CheckDevicesCompatible(x, out);
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> x_iarray{x};
        IndexableArray<T> out_iarray{out};
        Indexer indexer{out.shape()};

        for (int64_t i = 0; i < indexer.total_size(); ++i) {
            indexer.Set(i);
            out_iarray[indexer] = std::log(x_iarray[indexer]);
        }
    });
}

void NativeDevice::Synchronize() {}

}  // namespace native
}  // namespace xchainer
