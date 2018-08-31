#include "chainerx/testing/array_check.h"

#include <memory>

#include <gtest/gtest.h>

#include "chainerx/array.h"
#include "chainerx/array_body.h"
#include "chainerx/array_node.h"
#include "chainerx/numeric.h"

namespace chainerx {
namespace testing {
namespace testing_internal {
namespace {

inline bool IsAllClose(const Array& a, const Array& b, double rtol, double atol, bool equal_nan) {
    return &a.device() == &b.device() && AllClose(a, b, rtol, atol, equal_nan);
}

}  // namespace

::testing::AssertionResult IsEqual(const char* a_expr, const char* b_expr, const Array& a, const Array& b) {
    if (IsAllClose(a, b, 0., 0., true)) {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << "Expected " << a_expr << ": " << a << "\nTo be equal to " << b_expr << ": " << b << "\n"
                                         << "Error: " << a - b;
}

::testing::AssertionResult IsAllClose(
        const char* a_expr,
        const char* b_expr,
        const char* /*rtol_expr*/,
        const char* /*atol_expr*/,
        const char* /*equal_nan_expr*/,
        const Array& a,
        const Array& b,
        double rtol,
        double atol,
        bool equal_nan) {
    if (IsAllClose(a, b, rtol, atol, equal_nan)) {
        return ::testing::AssertionSuccess();
    }
    // clang-format off
    return ::testing::AssertionFailure() << "Expected " << a_expr << ": " << a << "\n"
                                         << "To be close to " << b_expr << ": " << b << "\n"
                                         << "Error: " << a - b << "\n"
                                         << "With rtol: " << rtol << "\n"
                                         << "     atol: " << atol << "\n"
                                         << "equal_nan: " << equal_nan;
    // clang-format on
}

::testing::AssertionResult HaveDistinctArrayNodes(const char* a_expr, const char* b_expr, const Array& a, const Array& b) {
    // No array nodes should be shared.
    for (const std::shared_ptr<internal::ArrayNode>& array_node_a : internal::GetArrayBody(a)->nodes()) {
        for (const std::shared_ptr<internal::ArrayNode>& array_node_b : internal::GetArrayBody(b)->nodes()) {
            if (array_node_a == array_node_b) {
                return ::testing::AssertionFailure()
                       << "Expected " << a_expr << " including: " << array_node_a << "(" << array_node_a->backprop_id() << ")\n"
                       << "Actual" << b_expr << " including: " << array_node_b << "(" << array_node_b->backprop_id() << ")";
            }
        }
    }
    return ::testing::AssertionSuccess();
}

}  // namespace testing_internal
}  // namespace testing
}  // namespace chainerx
