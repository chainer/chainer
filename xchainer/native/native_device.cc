#include "xchainer/native/native_device.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <tuple>
#include <type_traits>

#include <gsl/gsl>

#include "xchainer/array.h"
#include "xchainer/dtype.h"
#include "xchainer/elementwise_kernel_arg.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"
#include "xchainer/native/elementwise.h"
#include "xchainer/native/reduce.h"
#include "xchainer/ndim_vector.h"
#include "xchainer/numeric_limits.h"
#include "xchainer/reduction_kernel_arg.h"
#include "xchainer/scalar.h"
#include "xchainer/shape.h"

namespace xchainer {
namespace native {
namespace {

template <typename T>
struct FillImpl {
    void operator()(int64_t /*i*/, T& out) { out = value; }
    T value;
};

template <typename T>
struct CopyImpl {
    void operator()(int64_t /*i*/, T a, T& out) { out = a; }
};

template <typename T>
struct ArangeImpl {
    void operator()(int64_t i, T& out) { out = start + step * i; }
    T start;
    T step;
};

template <typename InT, typename OutT>
struct AsTypeImpl {
    void operator()(int64_t /*i*/, InT a, OutT& out) { out = static_cast<OutT>(a); }
};

template <typename T>
struct EqualImpl {
    void operator()(int64_t /*i*/, T x1, T x2, bool& out) { out = x1 == x2; }
};

template <typename T>
struct AddImpl {
    void operator()(int64_t /*i*/, T x1, T x2, T& out) { out = x1 + x2; }
};

template <typename T>
struct SubtractImpl {
    void operator()(int64_t /*i*/, T x1, T x2, T& out) { out = x1 - x2; }
};

template <typename T>
struct MultiplyImpl {
    void operator()(int64_t /*i*/, T x1, T x2, T& out) { out = x1 * x2; }
};

template <typename T>
struct MultiplyASImpl {
    void operator()(int64_t /*i*/, T x1, T& out) { out = x1 * x2; }
    T x2;
};

template <typename T>
struct DivideImpl {
    void operator()(int64_t /*i*/, T lhs, T rhs, T& out) { out = lhs / rhs; }
};

template <typename T>
struct ExpImpl {
    void operator()(int64_t /*i*/, T x, T& out) { out = std::exp(x); }
};

template <typename T>
struct LogImpl {
    void operator()(int64_t /*i*/, T x, T& out) { out = std::log(x); }
};

template <typename T>
struct IfLessElseASSAImpl {
    void operator()(int64_t /*i*/, T x1, T neg, T& out) { out = x1 < x2 ? pos : neg; }
    T x2;
    T pos;
};

}  // namespace

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
        Elementwise(MakeElementwiseKernelArg<T>(out), FillImpl<T>{static_cast<T>(value)});
    });
}

void NativeDevice::Arange(Scalar start, Scalar step, const Array& out) {
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise(MakeElementwiseKernelArg<T>(out), ArangeImpl<T>{static_cast<T>(start), static_cast<T>(step)});
    });
}

void NativeDevice::ArgMax(const Array& a, const NdimVector<int8_t>& axis, const Array& out) {
    assert(std::all_of(axis.begin(), axis.end(), [&a](int8_t i) { return a.shape()[i] > 0; }));
    assert(internal::IsValidReductionShape(a.shape(), axis, out.shape(), false));
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

void NativeDevice::Sum(const Array& a, const NdimVector<int8_t>& axis, const Array& out) {
    assert(internal::IsValidReductionShape(a.shape(), axis, out.shape(), true));
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

void NativeDevice::AMax(const Array& a, const NdimVector<int8_t>& axis, const Array& out) {
    assert(internal::IsValidReductionShape(a.shape(), axis, out.shape(), true));
    CheckDevicesCompatible(a, out);

    VisitDtype(a.dtype(), [&a, &axis, &out](auto pt) {
        using T = typename decltype(pt)::type;
        struct AMaxImpl {
            T Identity() { return NumericLimits<T>::LowestOrInf(); }
            T MapIn(T in, int64_t /*index*/) { return in; }
            void Reduce(T next, T& accum) {
                if (std::isnan(next) || accum < next) {
                    accum = next;
                }
            }
            T MapOut(T accum) { return accum; }
        };
        Reduce(MakeReductionKernelArg<T, T>(a, axis, out), AMaxImpl{});
    });
}

void NativeDevice::Copy(const Array& a, const Array& out) {
    CheckDevicesCompatible(a, out);
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise(MakeElementwiseKernelArg<const T, T>(a, out), CopyImpl<T>{});
    });
}

void NativeDevice::AsType(const Array& a, const Array& out) {
    CheckDevicesCompatible(a, out);
    auto do_astype = [&](auto in_pt, auto out_pt) {
        using InT = typename decltype(in_pt)::type;
        using OutT = typename decltype(out_pt)::type;
        Elementwise(MakeElementwiseKernelArg<const InT, OutT>(a, out), AsTypeImpl<InT, OutT>{});
    };
    VisitDtype(out.dtype(), [&](auto out_pt) { VisitDtype(a.dtype(), do_astype, out_pt); });
}

void NativeDevice::Equal(const Array& x1, const Array& x2, const Array& out) {
    CheckDevicesCompatible(x1, x2, out);
    VisitDtype(x1.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise(MakeElementwiseKernelArg<const T, const T, bool>(x1, x2, out), EqualImpl<T>{});
    });
}

void NativeDevice::Add(const Array& x1, const Array& x2, const Array& out) {
    CheckDevicesCompatible(x1, x2, out);
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise(MakeElementwiseKernelArg<const T, const T, T>(x1, x2, out), AddImpl<T>{});
    });
}

void NativeDevice::Subtract(const Array& x1, const Array& x2, const Array& out) {
    CheckDevicesCompatible(x1, x2, out);
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise(MakeElementwiseKernelArg<const T, const T, T>(x1, x2, out), SubtractImpl<T>{});
    });
}

void NativeDevice::Multiply(const Array& x1, const Array& x2, const Array& out) {
    CheckDevicesCompatible(x1, x2, out);
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise(MakeElementwiseKernelArg<const T, const T, T>(x1, x2, out), MultiplyImpl<T>{});
    });
}

void NativeDevice::MultiplyAS(const Array& x1, Scalar x2, const Array& out) {
    CheckDevicesCompatible(x1, out);
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise(MakeElementwiseKernelArg<const T, T>(x1, out), MultiplyASImpl<T>{static_cast<T>(x2)});
    });
}

void NativeDevice::Divide(const Array& lhs, const Array& rhs, const Array& out) {
    CheckDevicesCompatible(lhs, rhs, out);
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise(MakeElementwiseKernelArg<const T, const T, T>(lhs, rhs, out), DivideImpl<T>{});
    });
}

void NativeDevice::IfLessElseASSA(const Array& x1, Scalar x2, Scalar pos, const Array& neg, const Array& out) {
    CheckDevicesCompatible(x1, neg, out);
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise(
                MakeElementwiseKernelArg<const T, const T, T>(x1, neg, out),
                IfLessElseASSAImpl<T>{static_cast<T>(x2), static_cast<T>(pos)});
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
    VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise(MakeElementwiseKernelArg<const T, T>(x, out), ExpImpl<T>{});
    });
}

void NativeDevice::Log(const Array& x, const Array& out) {
    CheckDevicesCompatible(x, out);
    VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise(MakeElementwiseKernelArg<const T, T>(x, out), LogImpl<T>{});
    });
}

void NativeDevice::Take(const Array& a, const Array& indices, int8_t axis, const Array& out) {
    CheckDevicesCompatible(a, indices, out);
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;

        IndexableArray<const T> a_iarray{a};
        IndexableArray<T> out_iarray{out};
        IndexableArray<const int64_t> indices_iarray{indices};
        Indexer a_indexer{a.shape()};
        Indexer out_indexer{out.shape()};
        Indexer indices_indexer{indices.shape()};

        int64_t axis_dim = a.shape()[axis];

        // left: set of input dimensions lower than the axis
        // right: set of input dimensions higher than the axis
        Shape left_shape{a.shape().begin(), a.shape().begin() + axis};
        Shape right_shape{a.shape().begin() + (axis + 1), a.shape().end()};
        Shape axis_shape{axis_dim};  // always ndim==1
        Indexer left_indexer{left_shape};
        Indexer right_indexer{right_shape};
        Indexer axis_indexer{axis_shape};

        for (auto it = indices_indexer.It(0); it; ++it) {
            int64_t index = indices_iarray[it];
            if (index < 0) {
                index = axis_dim - ((-index + axis_dim - 1) % axis_dim + 1);
            } else {
                index = index % axis_dim;
            }
            assert(0 <= index);
            assert(index < axis_dim);
            auto it_axis = axis_indexer.It(index);

            for (auto it_left = left_indexer.It(0); it_left; ++it_left) {
                for (auto it_right = right_indexer.It(0); it_right; ++it_right) {
                    auto it_out = out_indexer.It(it_left, it, it_right);
                    auto it_a = a_indexer.It(it_left, it_axis, it_right);
                    out_iarray[it_out] = a_iarray[it_a];
                }
            }
        }
    });
}

void NativeDevice::AddAt(const Array& a, const Array& indices, int8_t axis, const Array& b, const Array& out) {
    CheckDevicesCompatible(a, indices, b);
    assert(a.shape() == out.shape());
    VisitDtype(a.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;

        IndexableArray<const T> a_iarray{a};
        IndexableArray<const T> b_iarray{b};
        IndexableArray<const int64_t> indices_iarray{indices};
        IndexableArray<T> out_iarray{out};
        Indexer b_indexer{b.shape()};
        Indexer indices_indexer{indices.shape()};
        Indexer out_indexer{out.shape()};  // indexer for both out_iarray and a_array

        int64_t axis_dim = a.shape()[axis];

        // left: set of input dimensions lower than the axis
        // right: set of input dimensions higher than the axis
        Shape left_shape{a.shape().begin(), a.shape().begin() + axis};
        Shape right_shape{a.shape().begin() + (axis + 1), a.shape().end()};
        Shape axis_shape{axis_dim};  // always ndim==1
        Indexer left_indexer{left_shape};
        Indexer right_indexer{right_shape};
        Indexer axis_indexer{axis_shape};

        // Copy
        for (auto it = out_indexer.It(0); it; ++it) {
            out_iarray[it] = a_iarray[it];
        }

        // Add
        for (auto it = indices_indexer.It(0); it; ++it) {
            int64_t index = indices_iarray[it];
            if (index < 0) {
                index = axis_dim - ((-index + axis_dim - 1) % axis_dim + 1);
            } else {
                index = index % axis_dim;
            }
            assert(0 <= index);
            assert(index < axis_dim);
            auto it_axis = axis_indexer.It(index);

            for (auto it_left = left_indexer.It(0); it_left; ++it_left) {
                for (auto it_right = right_indexer.It(0); it_right; ++it_right) {
                    auto it_out = out_indexer.It(it_left, it_axis, it_right);
                    auto it_b = b_indexer.It(it_left, it, it_right);
                    out_iarray[it_out] += b_iarray[it_b];
                }
            }
        }
    });
}

void NativeDevice::Synchronize() {}

}  // namespace native
}  // namespace xchainer
