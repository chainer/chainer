#include "xchainer/numerical_gradient.h"

#include <algorithm>
#include <functional>
#include <vector>

#include "xchainer/array.h"
#include "xchainer/array_repr.h"
#include "xchainer/device.h"
#include "xchainer/error.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"
#include "xchainer/routines/manipulation.h"

namespace xchainer {
namespace numerical_gradient_internal {

// TODO(niboshi): These temporary implementation for primitive operations depend on that the data in arrays can be accessed directly
// (e.g. with unified memory). In order for numerical gradient calculation to work corretly on general devices, They should be replaced with
// full-featured operations.

Array& Divide(const Array& lhs, const Array& rhs, Array& out) {
    lhs.device().Synchronize();
    rhs.device().Synchronize();
    out.device().Synchronize();

    VisitDtype(lhs.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> lhs_iarray{lhs};
        IndexableArray<const T> rhs_iarray{rhs};
        IndexableArray<T> out_iarray{out};
        Indexer indexer{out.shape()};

        for (int64_t i = 0; i < indexer.total_size(); ++i) {
            indexer.Set(i);
            out_iarray[indexer] = lhs_iarray[indexer] / rhs_iarray[indexer];
        }
    });
    return out;
}

Array operator/(const Array& lhs, const Array& rhs) {
    Array out = Array::EmptyLike(lhs);
    Divide(lhs, rhs, out);
    return out;
}

Scalar Norm(const Array& x) {
    Scalar s = AsScalar((x * x).Sum());
    return Scalar(std::sqrt(static_cast<double>(s)), x.dtype());
}

void Set(Array& out, int64_t flat_index, Scalar value) {
    out.device().Synchronize();

    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<T> iarray{out};
        Indexer indexer{out.shape()};
        indexer.Set(flat_index);
        iarray[indexer] = static_cast<T>(value);
    });
}

Scalar Get(const Array& out, int64_t flat_index) {
    out.device().Synchronize();

    return VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> iarray{out};
        Indexer indexer{out.shape()};
        indexer.Set(flat_index);
        return Scalar{iarray[indexer]};
    });
}

Arrays CalculateNumericalGradient(
        std::function<Arrays(const Arrays&)> func,
        const Arrays& inputs,
        const Arrays& grad_outputs,
        const Arrays& eps,
        const GraphId& graph_id) {
    // TODO(niboshi): Currently only elementwise functions are supported.
    // TODO(niboshi): Implement arithmetic operations and avoid manual synchronize
    const int nin = inputs.size();
    const int nout = grad_outputs.size();

    if (eps.size() != static_cast<size_t>(nin)) {
        throw XchainerError(
                "Invalid number of eps arrays where number of inputs: " + std::to_string(nin) + ", eps: " + std::to_string(eps.size()));
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

    auto eval = [&, graph_id](int i_in, int64_t in_flat_index, Scalar eps_scalar, float multiplier) -> Arrays {
        Arrays xs;
        std::transform(inputs.begin(), inputs.end(), std::back_inserter(xs), [graph_id](const Array& x) {
            return x.AsConstant(CopyKind::kCopy).RequireGrad(graph_id);
        });

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

            for (int j = 0; j < nout; ++j) {
                Array dy = ys1.at(j) - ys0.at(j);
                Array denom = Array::FullLike(dy, eps_scalar) * Array::FullLike(dy, Scalar(2, dtype));

                Array slope = (ys1.at(j) - ys0.at(j)) / denom;
                Scalar g = AsScalar((slope * grad_outputs.at(j)).Sum());
                Scalar g_ij = Get(grad_i, in_flat_index) + g;
                Set(grad_i, in_flat_index, g_ij);
            }
        }
        grads.push_back(grad_i);
    }

    return grads;
}

}  // namespace numerical_gradient_internal
}  // namespace xchainer
