#pragma once

#include <cstddef>
#include <functional>
#include <vector>

#include "chainerx/array.h"
#include "chainerx/error.h"

namespace chainerx {
namespace testing {

class RoutinesCheckError : public ChainerxError {
public:
    using ChainerxError::ChainerxError;
};

// Checks forward implementation of a routine.
// If concurrent_check_thread_count is nonzero, this function calls RunThreads() for concurrency test.
void CheckForward(
        const std::function<std::vector<Array>(const std::vector<Array>&)>& func,
        const std::vector<Array>& inputs,
        const std::vector<Array>& expected_outputs,
        double atol = 1e-5,
        double rtol = 1e-4);

}  // namespace testing
}  // namespace chainerx
