#include "xchainer/native_backend.h"

#include <gtest/gtest.h>

namespace xchainer {
namespace {

TEST(NativeBackendTest, GetName) { EXPECT_EQ("native", NativeBackend().GetName()); }

}  // namespace
}  // namespace xchainer
