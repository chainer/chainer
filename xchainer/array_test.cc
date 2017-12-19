#include "xchainer/array.h"

#include <gtest/gtest.h>
#include <array>
#include <cstddef>
#include <initializer_list>

namespace xchainer {
namespace {

class ArrayTest : public ::testing::Test {
public:
    template <typename T>
    Array MakeArray(std::initializer_list<int64_t> shape, std::shared_ptr<void> data) {
        return {shape, TypeToDtype<T>, data};
    }
};

TEST_F(ArrayTest, Ctor) {
    std::shared_ptr<void> data = std::unique_ptr<float[]>(new float[2 * 3 * 4]);
    Array x = MakeArray<float>({2, 3, 4}, data);
    ASSERT_EQ(TypeToDtype<float>, x.dtype());
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
    const Array x = MakeArray<float>({2, 3, 4}, data);
    std::shared_ptr<const void> x_data = x.data();
    ASSERT_EQ(data, x_data);
}

TEST_F(ArrayTest, IAdd) {
    {
        Array a = MakeArray<bool>({4, 1}, std::unique_ptr<bool[]>(new bool[4]{true, true, false, false}));
        Array b = MakeArray<bool>({4, 1}, std::unique_ptr<bool[]>(new bool[4]{true, false, true, false}));
        Array e = MakeArray<bool>({4, 1}, std::unique_ptr<bool[]>(new bool[4]{true, true, true, false}));
        a.IAdd(b);
        ASSERT_NO_THROW(CheckEqual(e, a));
    }
    {
        Array a = MakeArray<int8_t>({3, 1}, std::unique_ptr<int8_t[]>(new int8_t[3]{1, 2, 3}));
        Array b = MakeArray<int8_t>({3, 1}, std::unique_ptr<int8_t[]>(new int8_t[3]{1, 2, 3}));
        Array e = MakeArray<int8_t>({3, 1}, std::unique_ptr<int8_t[]>(new int8_t[3]{2, 4, 6}));
        a.IAdd(b);
        ASSERT_NO_THROW(CheckEqual(e, a));
    }
    {
        Array a = MakeArray<float>({3, 1}, std::unique_ptr<float[]>(new float[3]{1, 2, 3}));
        Array b = MakeArray<float>({3, 1}, std::unique_ptr<float[]>(new float[3]{1, 2, 3}));
        Array e = MakeArray<float>({3, 1}, std::unique_ptr<float[]>(new float[3]{2, 4, 6}));
        a.IAdd(b);
        ASSERT_NO_THROW(CheckEqual(e, a));
    }
}

TEST_F(ArrayTest, IMul) {
    {
        Array a = MakeArray<bool>({4, 1}, std::unique_ptr<bool[]>(new bool[4]{true, true, false, false}));
        Array b = MakeArray<bool>({4, 1}, std::unique_ptr<bool[]>(new bool[4]{true, false, true, false}));
        Array e = MakeArray<bool>({4, 1}, std::unique_ptr<bool[]>(new bool[4]{true, false, false, false}));
        a.IMul(b);
        ASSERT_NO_THROW(CheckEqual(e, a));
    }
    {
        Array a = MakeArray<int8_t>({3, 1}, std::unique_ptr<int8_t[]>(new int8_t[3]{1, 2, 3}));
        Array b = MakeArray<int8_t>({3, 1}, std::unique_ptr<int8_t[]>(new int8_t[3]{1, 2, 3}));
        Array e = MakeArray<int8_t>({3, 1}, std::unique_ptr<int8_t[]>(new int8_t[3]{1, 4, 9}));
        a.IMul(b);
        ASSERT_NO_THROW(CheckEqual(e, a));
    }
    {
        Array a = MakeArray<float>({3, 1}, std::unique_ptr<float[]>(new float[3]{1, 2, 3}));
        Array b = MakeArray<float>({3, 1}, std::unique_ptr<float[]>(new float[3]{1, 2, 3}));
        Array e = MakeArray<float>({3, 1}, std::unique_ptr<float[]>(new float[3]{1, 4, 9}));
        a.IMul(b);
        ASSERT_NO_THROW(CheckEqual(e, a));
    }
}

TEST_F(ArrayTest, Add) {
    {
        Array a = MakeArray<bool>({4, 1}, std::unique_ptr<bool[]>(new bool[4]{true, true, false, false}));
        Array b = MakeArray<bool>({4, 1}, std::unique_ptr<bool[]>(new bool[4]{true, false, true, false}));
        Array e = MakeArray<bool>({4, 1}, std::unique_ptr<bool[]>(new bool[4]{true, true, true, false}));
        Array o = a.Add(b);
        ASSERT_NO_THROW(CheckEqual(e, o));
    }
    {
        Array a = MakeArray<int8_t>({3, 1}, std::unique_ptr<int8_t[]>(new int8_t[3]{1, 2, 3}));
        Array b = MakeArray<int8_t>({3, 1}, std::unique_ptr<int8_t[]>(new int8_t[3]{1, 2, 3}));
        Array e = MakeArray<int8_t>({3, 1}, std::unique_ptr<int8_t[]>(new int8_t[3]{2, 4, 6}));
        Array o = a.Add(b);
        ASSERT_NO_THROW(CheckEqual(e, o));
    }
    {
        Array a = MakeArray<float>({3, 1}, std::unique_ptr<float[]>(new float[3]{1, 2, 3}));
        Array b = MakeArray<float>({3, 1}, std::unique_ptr<float[]>(new float[3]{1, 2, 3}));
        Array e = MakeArray<float>({3, 1}, std::unique_ptr<float[]>(new float[3]{2, 4, 6}));
        Array o = a.Add(b);
        ASSERT_NO_THROW(CheckEqual(e, o));
    }
}

TEST_F(ArrayTest, Mul) {
    {
        Array a = MakeArray<bool>({4, 1}, std::unique_ptr<bool[]>(new bool[4]{true, true, false, false}));
        Array b = MakeArray<bool>({4, 1}, std::unique_ptr<bool[]>(new bool[4]{true, false, true, false}));
        Array e = MakeArray<bool>({4, 1}, std::unique_ptr<bool[]>(new bool[4]{true, false, false, false}));
        Array o = a.Mul(b);
        ASSERT_NO_THROW(CheckEqual(e, o));
    }
    {
        Array a = MakeArray<int8_t>({3, 1}, std::unique_ptr<int8_t[]>(new int8_t[3]{1, 2, 3}));
        Array b = MakeArray<int8_t>({3, 1}, std::unique_ptr<int8_t[]>(new int8_t[3]{1, 2, 3}));
        Array e = MakeArray<int8_t>({3, 1}, std::unique_ptr<int8_t[]>(new int8_t[3]{1, 4, 9}));
        Array o = a.Mul(b);
        ASSERT_NO_THROW(CheckEqual(e, o));
    }
    {
        Array a = MakeArray<float>({3, 1}, std::unique_ptr<float[]>(new float[3]{1, 2, 3}));
        Array b = MakeArray<float>({3, 1}, std::unique_ptr<float[]>(new float[3]{1, 2, 3}));
        Array e = MakeArray<float>({3, 1}, std::unique_ptr<float[]>(new float[3]{1, 4, 9}));
        Array o = a.Mul(b);
        ASSERT_NO_THROW(CheckEqual(e, o));
    }
}

}  // namespace
}  // namespace xchainer
