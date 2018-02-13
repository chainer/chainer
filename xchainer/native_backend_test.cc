#include "xchainer/native_backend.h"

#include <gtest/gtest.h>

namespace xchainer {
namespace {

TEST(NativeBackendTest, name) { EXPECT_EQ("native", NativeBackend().name()); }

}  // namespace
}  // namespace xchainer
