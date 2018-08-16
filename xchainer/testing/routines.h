#pragma once

#include <cstddef>
#include <functional>
#include <vector>

#include "xchainer/array.h"
#include "xchainer/error.h"

namespace xchainer {
namespace testing {

class RoutinesCheckError : public XchainerError {
public:
    using XchainerError::XchainerError;
};

// Checks forward implementation of a routine.
// If concurrent_check_repeat_count is nonzero, this function calls CheckThreadSafety() for concurrency test.
void CheckForward(
        const std::function<std::vector<Array>(const std::vector<Array>&)>& func,
        const std::vector<Array>& inputs,
        const std::vector<Array>& expected_outputs,
        size_t concurrent_check_repeat_count = 0U,
        size_t concurrent_check_thread_count = 2U,
        double atol = 1e-5,
        double rtol = 1e-4);

}  // namespace testing
}  // namespace xchainer
