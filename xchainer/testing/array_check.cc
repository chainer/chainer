#include "xchainer/testing/array_check.h"

#include <memory>

#include <gtest/gtest.h>

#include "xchainer/array.h"
#include "xchainer/array_body.h"
#include "xchainer/array_node.h"
#include "xchainer/numeric.h"

namespace xchainer {
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

::testing::AssertionResult IsCopy(const char* orig_expr, const char* copy_expr, const Array& orig, const Array& copy) {
    if (!copy.IsContiguous()) {
        return ::testing::AssertionFailure() << "Expected contiguity of copy " << copy_expr << ": " << copy.IsContiguous() << "\n"
                                             << "To be : true";
    }
    if (copy.offset() != 0) {
        return ::testing::AssertionFailure() << "Expected offset of copy " << copy_expr << ": " << copy.offset() << "\n"
                                             << "To be : 0";
    }
    if (orig.data().get() == copy.data().get()) {  // Deep copy, therefore assert different addresses to data
        return ::testing::AssertionFailure() << "Expected data address of copy " << copy_expr << ": " << copy.data().get() << "\n"
                                             << "To be different from that of " << orig_expr << ": " << orig.data().get();
    }
    return IsEqual(copy_expr, orig_expr, copy, orig);
}

::testing::AssertionResult IsView(const char* orig_expr, const char* view_expr, const Array& orig, const Array& view) {
    if (orig.IsContiguous() != view.IsContiguous()) {
        return ::testing::AssertionFailure() << "Expected contiguity of view " << view_expr << ": " << view.IsContiguous() << "\n"
                                             << "To be equal to that of " << orig_expr << ": " << orig.IsContiguous();
    }
    if (orig.offset() != view.offset()) {
        return ::testing::AssertionFailure() << "Expected offset of view " << view_expr << ": " << view.offset() << "\n"
                                             << "To be equal to that of " << orig_expr << ": " << orig.offset();
    }
    if (orig.data().get() != view.data().get()) {  // Shallow copy, therefore assert the same address to data.
        return ::testing::AssertionFailure() << "Expected data address of view " << view_expr << ": " << view.data().get() << "\n"
                                             << "To be equal to that of " << orig_expr << ": " << orig.data().get();
    }
    if (internal::GetArrayBody(orig) == internal::GetArrayBody(view)) {  // Views should have different array bodies.
        return ::testing::AssertionFailure() << "Expected body of view " << view_expr << ": " << internal::GetArrayBody(view) << "\n"
                                             << "To be different from that of " << orig_expr << ": " << internal::GetArrayBody(orig);
    }

    // No array nodes should be shared.
    for (const std::shared_ptr<internal::ArrayNode>& array_node_orig : internal::GetArrayBody(orig)->nodes()) {
        for (const std::shared_ptr<internal::ArrayNode>& array_node_view : internal::GetArrayBody(view)->nodes()) {
            if (array_node_orig == array_node_view) {
                return ::testing::AssertionFailure() << "Expected all array nodes of view " << view_expr
                                                     << " including: " << array_node_view << "(" << array_node_view->backprop_id() << ")\n"
                                                     << "To to not be shared with that of " << orig_expr
                                                     << " including: " << array_node_orig << "(" << array_node_orig->backprop_id() << ")";
            }
        }
    }

    return IsEqual(view_expr, orig_expr, view, orig);
}

}  // namespace testing_internal
}  // namespace testing
}  // namespace xchainer
