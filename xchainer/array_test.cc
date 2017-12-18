#include "xchainer/array.h"

#include <gtest/gtest.h>
#include <array>
#include <cstddef>
#include <initializer_list>

namespace xchainer {
namespace {

class ArrayTest : public ::testing::Test {
public:
    void SetUp() override { dtype_ = Dtype::kFloat32; }

    Array MakeArray(std::initializer_list<int64_t> shape, std::shared_ptr<void> data) const { return {shape, dtype_, data}; }

    Dtype dtype() const { return dtype_; }

private:
    Dtype dtype_;
};

TEST_F(ArrayTest, Ctor) {
    std::shared_ptr<void> data = std::unique_ptr<float[]>(new float[2 * 3 * 4]);
    Array x = MakeArray({2, 3, 4}, data);
    EXPECT_EQ(dtype(), x.dtype());
    EXPECT_EQ(3, x.ndim());
    EXPECT_EQ(2 * 3 * 4, x.total_size());
    EXPECT_EQ(4, x.element_bytes());
    EXPECT_EQ(2 * 3 * 4 * 4, x.total_bytes());
    EXPECT_EQ(data, x.data());
    EXPECT_EQ(0, x.offset());
    EXPECT_TRUE(x.is_contiguous());
}

}  // namespace
}  // namespace xchainer
