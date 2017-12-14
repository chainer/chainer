#include "xchainer/scalar.h"

#include <gtest/gtest.h>

namespace xchainer {
namespace {

TEST(ScalarTest, Dtype) {
    EXPECT_EQ(Scalar(int8_t(1)).dtype(), Dtype::kInt8);
    EXPECT_EQ(Scalar(int16_t(3)).dtype(), Dtype::kInt16);
}

}  // namespace
}  // namespace xchainer
