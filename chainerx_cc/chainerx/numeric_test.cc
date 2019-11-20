#include "chainerx/numeric.h"

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <absl/types/optional.h>
#include <gtest/gtest.h>

#include "chainerx/array.h"
#include "chainerx/context.h"
#include "chainerx/error.h"
#include "chainerx/scalar.h"
#include "chainerx/shape.h"
#include "chainerx/testing/array.h"
#include "chainerx/testing/device_session.h"

namespace chainerx {

class NumericTest : public ::testing::TestWithParam<std::string> {
protected:
    void SetUp() override { device_session_.emplace(DeviceId{GetParam(), 0}); }

    void TearDown() override { device_session_.reset(); }

public:
    template <typename T>
    void CheckAllClose(
            const Shape& shape,
            const std::vector<T>& adata,
            const std::vector<T>& bdata,
            double rtol,
            double atol,
            bool equal_nan = false) {
        Array a = testing::BuildArray(shape).WithData(adata);
        Array b = testing::BuildArray(shape).WithData(bdata);
        EXPECT_TRUE(AllClose(a, b, rtol, atol, equal_nan));
    }

    template <typename T>
    void CheckNotAllClose(
            const Shape& shape,
            const std::vector<T>& adata,
            const std::vector<T>& bdata,
            double rtol,
            double atol,
            bool equal_nan = false) {
        Array a = testing::BuildArray(shape).WithData(adata);
        Array b = testing::BuildArray(shape).WithData(bdata);
        EXPECT_FALSE(AllClose(a, b, rtol, atol, equal_nan));
    }

    template <typename T, typename U>
    void CheckAllCloseThrow(
            const Shape& shape,
            const std::vector<T>& adata,
            const std::vector<U>& bdata,
            double rtol,
            double atol,
            bool equal_nan = false) {
        Array a = testing::BuildArray(shape).WithData<T>(adata);
        Array b = testing::BuildArray(shape).WithData<U>(bdata);
        EXPECT_THROW(AllClose(a, b, rtol, atol, equal_nan), DtypeError);
    }

private:
    absl::optional<testing::DeviceSession> device_session_;
};

TEST_P(NumericTest, AllClose) {
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

TEST_P(NumericTest, AllCloseMixed) {
    CheckAllCloseThrow<bool, int8_t>({3}, {true, false, true}, {1, 2, 3}, 2., 1.);
    CheckAllCloseThrow<int16_t, int8_t>({3}, {1, 2, 3}, {4, 5, 6}, 8., 7.);
    CheckAllCloseThrow<int8_t, int32_t>({3}, {1, 2, 3}, {4, 5, 6}, 8., 7.);
    CheckAllCloseThrow<int64_t, int8_t>({3}, {1, 2, 3}, {1, 2, 3}, 2., 1.);
    CheckAllCloseThrow<int32_t, float>({3}, {1, 2, 3}, {1.f, 2.f, 3.f}, 2., 1.);
    CheckAllCloseThrow<double, int32_t>({3}, {1., 2., 3.}, {1, 2, 3}, 2., 1.);
}

TEST_P(NumericTest, AllCloseEqualNan) {
    double eps = 1e-5;
    CheckAllClose<double>({3}, {1., std::nan(""), 3.}, {1., std::nan(""), 3.}, 0., 0. + eps, true);
    CheckAllClose<double>({3}, {1., 2., 3.}, {1., 2., 3.}, 0., 0. + eps, true);
    CheckNotAllClose<double>({3}, {1., std::nan(""), 3.}, {1., 2., std::nan("")}, 0., 0. + eps, true);
    CheckNotAllClose<double>({3}, {1., 2., 3.}, {1., std::nan(""), 3.}, 0., 0. + eps, true);
}

TEST_P(NumericTest, AllCloseNotEqualNan) {
    double eps = 1e-5;
    CheckNotAllClose<double>({3}, {1., std::nan(""), 3.}, {1., 2., 3.}, 0., 0. + eps, false);
    CheckNotAllClose<double>({3}, {1., std::nan(""), 2.}, {1., std::nan(""), 2.}, 0., 0. + eps, false);
}

INSTANTIATE_TEST_CASE_P(
        ForEachBackend,
        NumericTest,
        ::testing::Values(
#ifdef CHAINERX_ENABLE_CUDA
                std::string{"cuda"},
#endif  // CHAINERX_ENABLE_CUDA
                std::string{"native"}));

}  // namespace chainerx
