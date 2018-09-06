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

void CheckAllArrayBodiesFreed(chainerx::internal::ArrayBodyLeakTracker& tracker) {
    std::ostringstream os;
    if (!tracker.IsAllArrayBodiesFreed(os)) {
        throw RoutinesCheckError{os.str()};
    }
}

}  // namespace

// TODO(niboshi): Check array nodes of output arrays to ensure the implementation takes backprop mode into account
void CheckForward(
        const std::function<std::vector<Array>(const std::vector<Array>&)>& func,
        const std::vector<Array>& inputs,
        const std::vector<Array>& expected_outputs,
        size_t concurrent_check_thread_count,
        double atol,
        double rtol) {
    CHAINERX_ASSERT(concurrent_check_thread_count != 1 && "concurrent_check_thread_count == 1 is meaningless.");

    // Run single-shot test
    {
        chainerx::internal::ArrayBodyLeakTracker tracker{};
        {
            chainerx::internal::ArrayBodyLeakDetectionScope scope{tracker};
            std::vector<Array> outputs = func(inputs);
            CheckOutputArraysEqual(expected_outputs, outputs, atol, rtol);
        }
        CheckAllArrayBodiesFreed(tracker);
    }

    // Run thread safety check
    if (concurrent_check_thread_count > 0) {
        chainerx::internal::ArrayBodyLeakTracker tracker{};
        {
            chainerx::internal::ArrayBodyLeakDetectionScope scope{tracker};
            Context& context = chainerx::GetDefaultContext();

            RunThreads(concurrent_check_thread_count, [&func, &inputs, &expected_outputs, &atol, &rtol, &context]() {
                chainerx::SetDefaultContext(&context);
                std::vector<Array> outputs = func(inputs);
                CheckOutputArraysEqual(expected_outputs, outputs, atol, rtol);
            });
        }
        CheckAllArrayBodiesFreed(tracker);
    }
}

}  // namespace testing
}  // namespace chainerx
