#include "chainerx/dims.h"

#include <cstdint>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

namespace chainerx {
namespace {

std::string ToString(const Dims& dims) {
    std::ostringstream os;
    os << DimsFormatter{dims};
    return os.str();
}

TEST(DimsTest, DimsFormatterTest) {
    {
        Dims vec{};
        EXPECT_EQ("[]", ToString(vec));
    }
    {
        Dims vec{1};
        EXPECT_EQ("[1]", ToString(vec));
    }
    {
        Dims vec{1, 2, 3};
        EXPECT_EQ("[1, 2, 3]", ToString(vec));
    }
}

}  // namespace
}  // namespace chainerx
