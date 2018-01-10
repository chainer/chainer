#include "xchainer/numeric.h"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "xchainer/array.h"
#include "xchainer/error.h"
#include "xchainer/scalar.h"
#include "xchainer/shape.h"

namespace xchainer {
namespace testing {

class NumericTest : public ::testing::Test {
public:
    template <typename T>
    Array MakeArray(const Shape& shape, std::initializer_list<T> data) {
        auto a = std::make_unique<T[]>(data.size());
        std::copy(data.begin(), data.end(), a.get());
        return {shape, TypeToDtype<T>, std::move(a)};
    }

    template <typename T>
    void CheckAllClose(const Shape& shape, std::initializer_list<T> adata, std::initializer_list<T> bdata, double rtol, double atol) {
        Array a = MakeArray<T>(shape, adata);
        Array b = MakeArray<T>(shape, bdata);
        ASSERT_TRUE(AllClose(a, b, rtol, atol));
    }

    template <typename T>
    void CheckNotAllClose(const Shape& shape, std::initializer_list<T> adata, std::initializer_list<T> bdata, double rtol, double atol) {
        Array a = MakeArray<T>(shape, adata);
        Array b = MakeArray<T>(shape, bdata);
        ASSERT_FALSE(AllClose(a, b, rtol, atol));
    }

    template <typename T, typename U>
    void CheckAllCloseThrow(const Shape& shape, std::initializer_list<T> adata, std::initializer_list<U> bdata, double rtol, double atol) {
        Array a = MakeArray<T>(shape, adata);
        Array b = MakeArray<U>(shape, bdata);
        ASSERT_THROW(AllClose(a, b, rtol, atol), DtypeError);
    }
};

TEST_F(NumericTest, AllClose) {
    CheckAllClose<bool>({2}, {true, false}, {true, false}, 0., 0.);
    CheckAllClose<bool>({2}, {true, false}, {false, true}, 0., 1.);
    CheckAllClose<bool>({2}, {false, false}, {true, true}, 1., 0.);
    CheckAllClose<bool>({2}, {false, true}, {true, false}, 1., 1.);
    CheckNotAllClose<bool>({2}, {true, false}, {false, true}, 0., 0.);
    CheckNotAllClose<bool>({2}, {true, false}, {false, true}, 1., 0.);

    CheckAllClose<int8_t>({2}, {1, 2}, {1, 2}, 0., 0.);
    CheckAllClose<int8_t>({2}, {1, 2}, {3, 4}, 0., 2.);
    CheckAllClose<int8_t>({2}, {1, 2}, {3, 4}, 2., 0.);
    CheckAllClose<int8_t>({2}, {1, 2}, {3, 4}, 1., 1.);
    CheckNotAllClose<int8_t>({2}, {1, 2}, {3, 4}, 0., 0.);
    CheckNotAllClose<int8_t>({2}, {1, 2}, {3, 4}, 0., 1.);
    CheckNotAllClose<int8_t>({2}, {3, 4}, {1, 2}, 1., 0.);

    CheckAllClose<int16_t>({2}, {1, 2}, {1, 2}, 0., 0.);
    CheckAllClose<int16_t>({2}, {1, 2}, {3, 4}, 0., 2.);
    CheckAllClose<int16_t>({2}, {1, 2}, {3, 4}, 2., 0.);
    CheckAllClose<int16_t>({2}, {1, 2}, {3, 4}, 1., 1.);
    CheckNotAllClose<int16_t>({2}, {1, 2}, {3, 4}, 0., 0.);
    CheckNotAllClose<int16_t>({2}, {1, 2}, {3, 4}, 0., 1.);
    CheckNotAllClose<int16_t>({2}, {3, 4}, {1, 2}, 1., 0.);

    CheckAllClose<int32_t>({2}, {1, 2}, {1, 2}, 0., 0.);
    CheckAllClose<int32_t>({2}, {-1, 0}, {1, 2}, 0., 2.);
    CheckAllClose<int32_t>({2}, {-1, 0}, {1, 2}, 2., 0.);
    CheckAllClose<int32_t>({2}, {-1, 0}, {1, 2}, 1., 1.);
    CheckNotAllClose<int32_t>({2}, {-1, 0}, {1, 2}, 0., 0.);
    CheckNotAllClose<int32_t>({2}, {-1, 0}, {1, 2}, 0., 1.);
    CheckNotAllClose<int32_t>({2}, {-1, 0}, {1, 2}, 1., 0.);

    CheckAllClose<int64_t>({2}, {1, 2}, {1, 2}, 0., 0.);
    CheckAllClose<int64_t>({2}, {-1, 0}, {1, 2}, 0., 2.);
    CheckAllClose<int64_t>({2}, {-1, 0}, {1, 2}, 2., 0.);
    CheckAllClose<int64_t>({2}, {-1, 0}, {1, 2}, 1., 1.);
    CheckNotAllClose<int64_t>({2}, {-1, 0}, {1, 2}, 0., 0.);
    CheckNotAllClose<int64_t>({2}, {-1, 0}, {1, 2}, 0., 1.);
    CheckNotAllClose<int64_t>({2}, {-1, 0}, {1, 2}, 1., 0.);

    CheckAllClose<uint8_t>({2}, {1, 2}, {1, 2}, 0., 0.);
    CheckAllClose<uint8_t>({2}, {1, 2}, {3, 4}, 0., 2.);
    CheckAllClose<uint8_t>({2}, {1, 2}, {3, 4}, 2., 0.);
    CheckAllClose<uint8_t>({2}, {1, 2}, {3, 4}, 1., 1.);
    CheckNotAllClose<uint8_t>({2}, {1, 2}, {3, 4}, 0., 0.);
    CheckNotAllClose<uint8_t>({2}, {1, 2}, {3, 4}, 0., 1.);
    CheckNotAllClose<uint8_t>({2}, {3, 4}, {1, 2}, 1., 0.);

    {
        float eps = 1e-5f;
        CheckAllClose<float>({2}, {1.f, 2.f}, {1.f, 2.f}, 0., 0. + eps);
        CheckAllClose<float>({2}, {1.f, 2.f}, {1.2f, 2.f}, 0., .2 + eps);
        CheckAllClose<float>({2}, {1.f, 2.f}, {1.2f, 2.5f}, .3, 0. + eps);
        CheckAllClose<float>({2}, {1.f, 2.f}, {1.2f, 2.2f}, .2, .2 + eps);
        CheckNotAllClose<float>({2}, {1.f, 2.f}, {1.1f, 2.2f}, 0., 0. + eps);
        CheckNotAllClose<float>({2}, {1.f, 2.f}, {1.1f, 2.2f}, 0., 1e-2 + eps);
        CheckNotAllClose<float>({2}, {1.f, 2.f}, {1.1f, 2.2f}, 1e-2, 0. + eps);
    }

    {
        double eps = 1e-8;
        CheckAllClose<double>({2}, {1., 2.}, {1., 2.}, 0., 0. + eps);
        CheckAllClose<double>({2}, {1., 2.}, {1.2, 2.}, 0., .2 + eps);
        CheckAllClose<double>({2}, {1., 2.}, {1.2, 2.5}, .3, 0. + eps);
        CheckAllClose<double>({2}, {1., 2.}, {1.2, 2.2}, .2, .2 + eps);
        CheckNotAllClose<double>({2}, {1., 2.}, {1.1, 2.2}, 0., 0. + eps);
        CheckNotAllClose<double>({2}, {1., 2.}, {1.1, 2.2}, 0., 1e-2 + eps);
        CheckNotAllClose<double>({2}, {1., 2.}, {1.1, 2.2}, 1e-2, 0. + eps);
    }
}

TEST_F(NumericTest, AllCloseMixed) {
    CheckAllCloseThrow<bool, int8_t>({3}, {true, false, true}, {1, 2, 3}, 2., 1.);
    CheckAllCloseThrow<int16_t, int8_t>({3}, {1, 2, 3}, {4, 5, 6}, 8., 7.);
    CheckAllCloseThrow<int8_t, int32_t>({3}, {1, 2, 3}, {4, 5, 6}, 8., 7.);
    CheckAllCloseThrow<int64_t, int8_t>({3}, {1, 2, 3}, {1, 2, 3}, 2., 1.);
    CheckAllCloseThrow<int32_t, float>({3}, {1, 2, 3}, {1.f, 2.f, 3.f}, 2., 1.);
    CheckAllCloseThrow<double, int32_t>({3}, {1., 2., 3.}, {1, 2, 3}, 2., 1.);
}

}  // namespace testing
}  // namespace xchainer
