#include "chainerx/dims.h"

#include <cstdint>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "chainerx/constant.h"
#include "chainerx/stack_vector.h"

namespace chainerx {
namespace {

std::string ToString(const StackVector<int64_t, kMaxNdim>& dims) {
    std::ostringstream os;
    os << DimsFormatter{dims};
    return os.str();
}

TEST(DimsTest, DimsFormatterTest) {
    {
        StackVector<int64_t, kMaxNdim> vec{};
        EXPECT_EQ("[]", ToString(vec));
    }
    {
        StackVector<int64_t, kMaxNdim> vec{1};
        EXPECT_EQ("[1]", ToString(vec));
    }
    {
        StackVector<int64_t, kMaxNdim> vec{1, 2, 3};
        EXPECT_EQ("[1, 2, 3]", ToString(vec));
    }
}

}  // namespace
}  // namespace chainerx
