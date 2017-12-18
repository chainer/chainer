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
    ASSERT_EQ(dtype(), x.dtype());
    ASSERT_EQ(3, x.ndim());
    ASSERT_EQ(2 * 3 * 4, x.total_size());
    ASSERT_EQ(4, x.element_bytes());
    ASSERT_EQ(2 * 3 * 4 * 4, x.total_bytes());
    const std::shared_ptr<void> x_data = x.data();
    ASSERT_EQ(data, x_data);
    ASSERT_EQ(0, x.offset());
    ASSERT_TRUE(x.is_contiguous());
}

TEST_F(ArrayTest, ConstArray) {
    std::shared_ptr<void> data = std::unique_ptr<float[]>(new float[2 * 3 * 4]);
    const Array x = MakeArray({2, 3, 4}, data);
    std::shared_ptr<const void> x_data = x.data();
    ASSERT_EQ(data, x_data);
}

TEST_F(ArrayTest, iadd) {
    std::shared_ptr<void> data_a = std::unique_ptr<float[]>(new float[3]{1, 1, 1});
    std::shared_ptr<void> data_b = std::unique_ptr<float[]>(new float[3]{1, 1, 1});
    std::shared_ptr<void> data_e = std::unique_ptr<float[]>(new float[3]{2, 2, 2});
    Array a = MakeArray({3, 1}, data_a);
    Array b = MakeArray({3, 1}, data_b);
    Array e = MakeArray({3, 1}, data_e);
    a.iadd(b);
    ASSERT_NO_THROW(CheckEqual(e, a));
}

TEST_F(ArrayTest, imul) {
    std::shared_ptr<void> data_a = std::unique_ptr<float[]>(new float[3]{2, 2, 2});
    std::shared_ptr<void> data_b = std::unique_ptr<float[]>(new float[3]{2, 2, 2});
    std::shared_ptr<void> data_e = std::unique_ptr<float[]>(new float[3]{4, 4, 4});
    Array a = MakeArray({3, 1}, data_a);
    Array b = MakeArray({3, 1}, data_b);
    Array e = MakeArray({3, 1}, data_e);
    a.imul(b);
    ASSERT_NO_THROW(CheckEqual(e, a));
}

TEST_F(ArrayTest, add) {
    std::shared_ptr<void> data_a = std::unique_ptr<float[]>(new float[3]{1, 1, 1});
    std::shared_ptr<void> data_b = std::unique_ptr<float[]>(new float[3]{1, 1, 1});
    std::shared_ptr<void> data_e = std::unique_ptr<float[]>(new float[3]{2, 2, 2});
    Array a = MakeArray({3, 1}, data_a);
    Array b = MakeArray({3, 1}, data_b);
    Array e = MakeArray({3, 1}, data_e);
    Array o = a.add(b);
    ASSERT_NO_THROW(CheckEqual(e, o));
}

TEST_F(ArrayTest, mul) {
    std::shared_ptr<void> data_a = std::unique_ptr<float[]>(new float[3]{2, 2, 2});
    std::shared_ptr<void> data_b = std::unique_ptr<float[]>(new float[3]{2, 2, 2});
    std::shared_ptr<void> data_e = std::unique_ptr<float[]>(new float[3]{4, 4, 4});
    Array a = MakeArray({3, 1}, data_a);
    Array b = MakeArray({3, 1}, data_b);
    Array e = MakeArray({3, 1}, data_e);
    Array o = a.mul(b);
    ASSERT_NO_THROW(CheckEqual(e, o));
}


}  // namespace
}  // namespace xchainer
