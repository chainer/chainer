#include "chainerx/numerical_gradient.h"

#include <algorithm>
#include <functional>
#include <vector>

#include "chainerx/array.h"
#include "chainerx/array_repr.h"
#include "chainerx/backprop_mode.h"
#include "chainerx/context.h"
#include "chainerx/device.h"
#include "chainerx/error.h"
#include "chainerx/indexable_array.h"
#include "chainerx/indexer.h"
#include "chainerx/native/data_type.h"
#include "chainerx/native/native_backend.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/manipulation.h"
#include "chainerx/scalar.h"

namespace chainerx {
namespace numerical_gradient_internal {
namespace {

void SynchronizeArrays(const Arrays& arrays) {
    std::for_each(arrays.begin(), arrays.end(), [](const Array& array) { array.device().Synchronize(); });
}

}  // namespace

Scalar Norm(const Array& x) {
    Scalar s = AsScalar((x * x).Sum());
    return Scalar{std::sqrt(static_cast<double>(s))};
}

void Set(const Array& out, int64_t flat_index, Scalar value) {
    Device& native_device = out.device().context().GetNativeBackend().GetDevice(0);

    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<T> iarray{out};
        Indexer<> indexer{out.shape()};
        T& dst = native::StorageToDataType<T>(iarray[indexer.It(flat_index)]);
        auto src = static_cast<T>(value);
        Device& device = out.device();
        device.MemoryCopyFrom(&dst, &src, sizeof(T), native_device);
        // TODO(hvy): Avoid having to synchronize for each Set.
        device.Synchronize();
    });
}

Scalar Get(const Array& out, int64_t flat_index) {
    Device& native_device = out.device().context().GetNativeBackend().GetDevice(0);

    return VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> iarray{out};
        Indexer<> indexer{out.shape()};
        const T& src = native::StorageToDataType<const T>(iarray[indexer.It(flat_index)]);
        T dst{};
        out.device().MemoryCopyTo(&dst, &src, sizeof(T), native_device);
        return Scalar{dst};
    });
}

Arrays CalculateNumericalGradient(
        std::function<Arrays(const Arrays&)> func, const Arrays& inputs, const Arrays& grad_outputs, const Arrays& eps) {
    // TODO(niboshi): Currently only elementwise functions are supported.
    // TODO(niboshi): Implement arithmetic operations and avoid manual synchronize
    NoBackpropModeScope scope{};

    SynchronizeArrays(inputs);
    SynchronizeArrays(grad_outputs);
    SynchronizeArrays(eps);

    const int nin = inputs.size();
    const int nout = grad_outputs.size();

    std::vector<Array> xs;
    xs.reserve(inputs.size());
    std::transform(inputs.begin(), inputs.end(), std::back_inserter(xs), [](const Array& x) { return x.MakeView(); });

    if (eps.size() != static_cast<size_t>(nin)) {
        throw ChainerxError{"Invalid number of eps arrays where number of inputs: ", nin, ", eps: ", eps.size()};
    }

    for (int i = 0; i < nin; ++i) {
        if (xs.at(i).shape() != eps.at(i).shape()) {
            throw ChainerxError{"Invalid eps shape"};
        }
        if (xs.at(i).dtype() != eps.at(i).dtype()) {
            throw ChainerxError{"Invalid eps dtype"};
        }
        // TODO(niboshi): Check: eps must not contain zeros.
    }

    auto eval = [&func, &xs](int i_in, int64_t in_flat_index, Scalar eps_scalar, float multiplier) mutable -> Arrays {
        Arrays xs_copy = xs;  // shallow copy
        Array& xi = xs_copy.at(i_in);
        // Only the target array is deeply copied
        xi = xi.Copy();
        // Give displacement and evaluate
        Set(xi, in_flat_index, Get(xi, in_flat_index) + Scalar{static_cast<float>(eps_scalar) * multiplier});
        return func(xs_copy);
    };

    Arrays grads;
    for (int i = 0; i < nin; ++i) {
        Array grad_i = ZerosLike(xs.at(i));
        Dtype dtype = grad_i.dtype();
        int64_t size = grad_i.GetTotalSize();

        for (int64_t in_flat_index = 0; in_flat_index < size; ++in_flat_index) {
            Scalar eps_scalar = Get(eps.at(i), in_flat_index);
            Arrays ys0 = eval(i, in_flat_index, eps_scalar, -1);
            Arrays ys1 = eval(i, in_flat_index, eps_scalar, 1);

            for (int j = 0; j < nout; ++j) {
                Array dy = ys1.at(j) - ys0.at(j);
                Array denom = FullLike(dy, eps_scalar) * FullLike(dy, Scalar{2, GetKind(dtype)});

                Array slope = (ys1.at(j) - ys0.at(j)) / denom;
                Scalar g = AsScalar((slope * grad_outputs.at(j)).Sum().AsType(dtype));
                Scalar g_ij = Get(grad_i, in_flat_index) + g;
                Set(grad_i, in_flat_index, g_ij);
            }
        }
        grads.push_back(grad_i);
    }

    return grads;
}

}  // namespace numerical_gradient_internal
}  // namespace chainerx
