#include "chainerx/array_body_leak_detection.h"

#include <cstddef>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "chainerx/array.h"
#include "chainerx/array_body.h"
#include "chainerx/dtype.h"
#include "chainerx/routines/creation.h"
#include "chainerx/shape.h"
#include "chainerx/testing/context_session.h"

namespace chainerx {
namespace {

TEST(ArrayBodyLeakDetectionTest, NoLeak) {
    testing::ContextSession context_session{};

    internal::ArrayBodyLeakTracker tracker{};
    {
        internal::ArrayBodyLeakDetectionScope scope{tracker};
        Ones({3, 2}, Dtype::kFloat32);
    }
    ASSERT_TRUE(tracker.GetAliveArrayBodies().empty());
}

TEST(ArrayBodyLeakDetectionTest, Leak) {
    testing::ContextSession context_session{};
    std::shared_ptr<internal::ArrayBody> leaked_body{};

    internal::ArrayBodyLeakTracker tracker{};
    {
        internal::ArrayBodyLeakDetectionScope scope{tracker};
        Array a = Ones({3, 2}, Dtype::kFloat32);
        leaked_body = internal::GetArrayBody(a);
    }
    std::vector<std::shared_ptr<internal::ArrayBody>> alive_arr_bodies = tracker.GetAliveArrayBodies();
    ASSERT_EQ(size_t{1}, alive_arr_bodies.size());
    ASSERT_EQ(leaked_body, alive_arr_bodies[0]);
}

}  // namespace
}  // namespace chainerx
