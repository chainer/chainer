#include "xchainer/gradient_check.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <vector>

#ifdef XCHAINER_ENABLE_CUDA
#include <cuda_runtime.h>
#endif  // XCHAINER_ENABLE_CUDA

#include "xchainer/array.h"
#include "xchainer/array_repr.h"
#include "xchainer/backend.h"
#include "xchainer/error.h"
#ifdef XCHAINER_ENABLE_CUDA
#include "xchainer/cuda/cuda_runtime.h"
#endif  // XCHAINER_ENABLE_CUDA
#include "xchainer/device_id.h"

namespace xchainer {
namespace gradient_internal {

void Synchronize() {
#ifdef XCHAINER_ENABLE_CUDA
    DeviceId device_id = GetDefaultDeviceId();
    // TODO(sonots): Use device_id.backend->Synchronize()
    if (device_id.backend()->GetName() == "cuda") {
        cuda::CheckError(cudaDeviceSynchronize());
    }
#endif  // XCHAINER_ENABLE_CUDA
}

Array& Subtract(const Array& lhs, const Array& rhs, Array& out) {
    VisitDtype(lhs.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Synchronize();
        auto* ldata = static_cast<const T*>(lhs.data().get());
        auto* rdata = static_cast<const T*>(rhs.data().get());
        auto* odata = static_cast<T*>(out.data().get());
        int64_t total_size = lhs.GetTotalSize();
        for (int64_t i = 0; i < total_size; ++i) {
            odata[i] = ldata[i] - rdata[i];
        }
    });
    return out;
}

Array& Divide(const Array& lhs, const Array& rhs, Array& out) {
    VisitDtype(lhs.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Synchronize();
        auto* ldata = static_cast<const T*>(lhs.data().get());
        auto* rdata = static_cast<const T*>(rhs.data().get());
        auto* odata = static_cast<T*>(out.data().get());
        int64_t total_size = lhs.GetTotalSize();
        for (int64_t i = 0; i < total_size; ++i) {
            odata[i] = ldata[i] / rdata[i];
        }
    });
    return out;
}

Array operator-(const Array& lhs, const Array& rhs) {
    Array out = Array::EmptyLike(lhs);
    Subtract(lhs, rhs, out);
    return out;
}

Array operator/(const Array& lhs, const Array& rhs) {
    Array out = Array::EmptyLike(lhs);
    Divide(lhs, rhs, out);
    return out;
}

template <typename T>
T SumImpl(const Array& array) {
    int64_t size = array.GetTotalSize();
    T s = 0;
    for (int64_t i = 0; i < size; ++i) {
        s += static_cast<const T*>(array.data().get())[i];
    }
    return s;
}

Scalar Sum(const Array& x) {
    if (x.dtype() == Dtype::kFloat32) {
        return Scalar(SumImpl<float>(x));
    } else if (x.dtype() == Dtype::kFloat64) {
        return Scalar(SumImpl<double>(x));
    } else {
        assert(false);
    }
}

Scalar Norm(const Array& x) {
    Scalar s = Sum(x * x);
    return Scalar(std::sqrt(static_cast<double>(s)), x.dtype());
}

Scalar VectorDot(const Array& x, const Array& y) { return Sum(x * y); }

void Set(Array& out, int64_t flat_index, Scalar value) {
    if (out.dtype() == Dtype::kFloat32) {
        static_cast<float*>(out.data().get())[flat_index] = static_cast<float>(value);
    } else if (out.dtype() == Dtype::kFloat64) {
        static_cast<double*>(out.data().get())[flat_index] = static_cast<double>(value);
    } else {
        assert(false);
    }
}

Scalar Get(const Array& out, int64_t flat_index) {
    if (out.dtype() == Dtype::kFloat32) {
        return static_cast<const float*>(out.data().get())[flat_index];
    } else if (out.dtype() == Dtype::kFloat64) {
        return static_cast<const double*>(out.data().get())[flat_index];
    } else {
        assert(false);
    }
    return 0;
}

Arrays CalculateNumericalGradient(std::function<Arrays(const Arrays&)> func, const Arrays& inputs, const Arrays& grad_outputs,
                                  const Arrays& eps) {
    // TODO(niboshi): Currently only elementwise functions are supported.
    // TODO(niboshi): Implement arithmetic operations and avoid manual synchronize
    const int nin = inputs.size();
    const int nout = grad_outputs.size();

    if (eps.size() != static_cast<size_t>(nin)) {
        throw XchainerError("Invalid number of eps arrays");
    }

    for (int i = 0; i < nin; ++i) {
        if (inputs.at(i).shape() != eps.at(i).shape()) {
            throw XchainerError("Invalid eps shape");
        }
        if (inputs.at(i).dtype() != eps.at(i).dtype()) {
            throw XchainerError("Invalid eps dtype");
        }
        // TODO(niboshi): Check: eps must not contain zeros.
    }

    Dtype dtype = inputs[0].dtype();

    auto eval = [&](int i_in, int64_t in_flat_index, Scalar eps_scalar, float multiplier) -> Arrays {
        // TODO(niboshi): In this deep copy, currently xs and inputs are connected with the computational graph.
        // We should avoid this connection.
        Arrays xs;
        std::transform(inputs.begin(), inputs.end(), std::back_inserter(xs), [](const Array& x) { return x.Copy(); });

        Set(xs.at(i_in), in_flat_index, Get(xs.at(i_in), in_flat_index) + Scalar(static_cast<float>(eps_scalar) * multiplier, dtype));
        return func(xs);
    };

    Arrays grads;
    for (int i = 0; i < nin; ++i) {
        Array grad_i = Array::ZerosLike(inputs.at(i));
        int64_t size = grad_i.GetTotalSize();

        for (int64_t in_flat_index = 0; in_flat_index < size; ++in_flat_index) {
            Scalar eps_scalar = Get(eps.at(i), in_flat_index);
            Arrays ys0 = eval(i, in_flat_index, eps_scalar, -1);
            Arrays ys1 = eval(i, in_flat_index, eps_scalar, 1);

            Array denom = Array::FullLike(eps.at(i), Get(eps.at(i), in_flat_index)) * Array::FullLike(eps.at(i), Scalar(2, dtype));

            for (int j = 0; j < nout; ++j) {
                Array dy = ys1.at(j) - ys0.at(j);
                Scalar g = VectorDot((ys1.at(j) - ys0.at(j)) / denom, grad_outputs.at(j));
                Scalar g_ij = Get(grad_i, in_flat_index) + g;
                Set(grad_i, in_flat_index, g_ij);
            }
        }
        grads.push_back(grad_i);
    }

    return grads;
}

}  // namespace gradient_internal
}  // namespace xchainer
