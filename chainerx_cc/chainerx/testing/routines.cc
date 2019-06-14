#include "chainerx/testing/routines.h"

#include <functional>
#include <sstream>
#include <vector>

#include <gsl/gsl>

#include "chainerx/array.h"
#include "chainerx/array_body_leak_detection.h"
#include "chainerx/context.h"
#include "chainerx/numeric.h"
#include "chainerx/testing/threading.h"

namespace chainerx {
namespace testing {
namespace {

void CheckOutputArraysEqual(const std::vector<Array>& expected, const std::vector<Array>& actual, double atol, double rtol) {
    // Number of outputs
    if (expected.size() != actual.size()) {
        throw RoutinesCheckError{"Number of output arrays does not match."};
    }

    // Output array properties
    for (size_t i = 0; i < expected.size(); ++i) {
        const Array& e = expected[i];
        const Array& a = actual[i];
        if (e.shape() != a.shape()) {
            throw RoutinesCheckError{"Shape of output array ",
                                     i,
                                     " of ",
                                     expected.size(),
                                     " is incorrect. Actual ",
                                     a.shape(),
                                     " != Expected ",
                                     e.shape(),
                                     "."};
        }
        if (e.dtype() != a.dtype()) {
            throw RoutinesCheckError{"Dtype of output array ",
                                     i,
                                     " of ",
                                     expected.size(),
                                     " is incorrect. Actual ",
                                     GetDtypeName(a.dtype()),
                                     " != Expected ",
                                     GetDtypeName(e.dtype()),
                                     "."};
        }
        if (&e.device() != &a.device()) {
            throw RoutinesCheckError{"Device of output array ",
                                     i,
                                     " of ",
                                     expected.size(),
                                     " is incorrect. Actual ",
                                     a.device().name(),
                                     " != Expected ",
                                     e.device().name(),
                                     "."};
        }
    }

    // Numerical check
    std::vector<size_t> failed_input_indices;
    for (size_t i = 0; i < expected.size(); ++i) {
        const Array& e = expected[i];
        const Array& a = actual[i];
        if (!AllClose(e, a, atol, rtol, true /*equal_nan*/)) {
            failed_input_indices.emplace_back(i);
        }
    }
    if (!failed_input_indices.empty()) {
        std::ostringstream os;
        os << "Numerical error in forward outputs (out of " << expected.size() << "): ";
        for (size_t i : failed_input_indices) {
            if (i != 0) {
                os << ", ";
            }
            os << i;
        }
        os << "\n";
        os << "Atol: " << atol << "  Rtol: " << rtol << "\n";
        for (size_t i : failed_input_indices) {
            const Array& e = expected[i];
            const Array& a = actual[i];
            os << "Error[" << i << "]:\n"
               << e - a << "\n"  // TODO(niboshi): Use abs
               << "Actual output[" << i << "]:\n"
               << a << "\n"
               << "Expected output[" << i << "]:\n"
               << e << "\n";
        }
        throw RoutinesCheckError{os.str()};
    }
}

}  // namespace

// TODO(niboshi): Check array nodes of output arrays to ensure the implementation takes backprop mode into account
void CheckForward(
        const std::function<std::vector<Array>(const std::vector<Array>&)>& func,
        const std::vector<Array>& inputs,
        const std::vector<Array>& expected_outputs,
        double atol,
        double rtol) {
    Context& context = inputs.front().context();
    CHAINERX_ASSERT(std::all_of(inputs.begin(), inputs.end(), [&context](const Array& array) { return &array.context() == &context; }));
    CHAINERX_ASSERT(std::all_of(
            expected_outputs.begin(), expected_outputs.end(), [&context](const Array& array) { return &array.context() == &context; }));

    // Use thread local or global default context if it is set. Else, use the context of the given arrays.
    if (Context* default_context = chainerx::internal::GetDefaultContextNoExcept()) {
        chainerx::SetDefaultContext(default_context);
    } else {
        chainerx::SetDefaultContext(&context);
    }

    std::vector<Array> outputs = func(inputs);
    CheckOutputArraysEqual(expected_outputs, outputs, atol, rtol);
}

}  // namespace testing
}  // namespace chainerx
