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

    template <typename T>
    Array MakeArray(std::initializer_list<int64_t> shape, std::initializer_list<T> data) {
        auto a = std::unique_ptr<T[]>(new T[data.size()]);
        size_t i = 0;
        for (const auto& e : data) {
            a[i] = e;
            i++;
        }
        return {shape, TypeToDtype<T>, std::move(a)};
    }

    template <typename T>
    void AssertEqual(const Array& lhs, const Array& rhs) {
        ASSERT_NO_THROW(CheckEqual(lhs.dtype(), rhs.dtype()));
        ASSERT_NO_THROW(CheckEqual(lhs.shape(), rhs.shape()));
        auto total_size = lhs.shape().total_size();
        const T* ldata = static_cast<const T*>(lhs.data().get());
        const T* rdata = static_cast<const T*>(rhs.data().get());
        for (decltype(total_size) i = 0; i < total_size; i++) {
            ASSERT_EQ(ldata[i], rdata[i]);
        }
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
        Array a = MakeArray<bool>({4, 1}, {true, true, false, false});
        Array b = MakeArray<bool>({4, 1}, {true, false, true, false});
        Array e = MakeArray<bool>({4, 1}, {true, true, true, false});
        a.IAdd(b);
        AssertEqual<bool>(e, a);
    }
    {
        Array a = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array b = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array e = MakeArray<int8_t>({3, 1}, {2, 4, 6});
        a.IAdd(b);
        AssertEqual<int8_t>(e, a);
    }
    {
        Array a = MakeArray<float>({3, 1}, {1, 2, 3});
        Array b = MakeArray<float>({3, 1}, {1, 2, 3});
        Array e = MakeArray<float>({3, 1}, {2, 4, 6});
        a.IAdd(b);
        AssertEqual<float>(e, a);
    }
}

TEST_F(ArrayTest, IMul) {
    {
        Array a = MakeArray<bool>({4, 1}, {true, true, false, false});
        Array b = MakeArray<bool>({4, 1}, {true, false, true, false});
        Array e = MakeArray<bool>({4, 1}, {true, false, false, false});
        a.IMul(b);
        AssertEqual<bool>(e, a);
    }
    {
        Array a = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array b = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array e = MakeArray<int8_t>({3, 1}, {1, 4, 9});
        a.IMul(b);
        AssertEqual<int8_t>(e, a);
    }
    {
        Array a = MakeArray<float>({3, 1}, {1, 2, 3});
        Array b = MakeArray<float>({3, 1}, {1, 2, 3});
        Array e = MakeArray<float>({3, 1}, {1, 4, 9});
        a.IMul(b);
        AssertEqual<float>(e, a);
    }
}

TEST_F(ArrayTest, Add) {
    {
        Array a = MakeArray<bool>({4, 1}, {true, true, false, false});
        Array b = MakeArray<bool>({4, 1}, {true, false, true, false});
        Array e = MakeArray<bool>({4, 1}, {true, true, true, false});
        Array o = a.Add(b);
        AssertEqual<bool>(e, o);
    }
    {
        Array a = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array b = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array e = MakeArray<int8_t>({3, 1}, {2, 4, 6});
        Array o = a.Add(b);
        AssertEqual<int8_t>(e, o);
    }
    {
        Array a = MakeArray<float>({3, 1}, {1, 2, 3});
        Array b = MakeArray<float>({3, 1}, {1, 2, 3});
        Array e = MakeArray<float>({3, 1}, {2, 4, 6});
        Array o = a.Add(b);
        AssertEqual<float>(e, o);
    }
}

TEST_F(ArrayTest, Mul) {
    {
        Array a = MakeArray<bool>({4, 1}, {true, true, false, false});
        Array b = MakeArray<bool>({4, 1}, {true, false, true, false});
        Array e = MakeArray<bool>({4, 1}, {true, false, false, false});
        Array o = a.Mul(b);
        AssertEqual<bool>(e, o);
    }
    {
        Array a = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array b = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array e = MakeArray<int8_t>({3, 1}, {1, 4, 9});
        Array o = a.Mul(b);
        AssertEqual<int8_t>(e, o);
    }
    {
        Array a = MakeArray<float>({3, 1}, {1, 2, 3});
        Array b = MakeArray<float>({3, 1}, {1, 2, 3});
        Array e = MakeArray<float>({3, 1}, {1, 4, 9});
        Array o = a.Mul(b);
        AssertEqual<float>(e, o);
    }
}

}  // namespace
}  // namespace xchainer
