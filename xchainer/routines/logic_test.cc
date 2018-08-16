#include "xchainer/routines/logic.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/device_id.h"
#include "xchainer/dtype.h"
#include "xchainer/testing/array.h"
#include "xchainer/testing/array_check.h"
#include "xchainer/testing/device_session.h"

namespace xchainer {
namespace {

class LogicTest : public ::testing::TestWithParam<std::string> {
protected:
    void SetUp() override {
        const std::string& backend_name = GetParam();
        device_session_.emplace(DeviceId{backend_name, 0});
    }

    void TearDown() override { device_session_.reset(); }

private:
    nonstd::optional<testing::DeviceSession> device_session_;
};

TEST_P(LogicTest, Equal) {
    using T = float;

    struct Param {
        T a;
        T b;
        bool e;
    };

    std::vector<Param> data = {{1.0f, 1.0f, true},
                               {1.0f, -1.0f, false},
                               {2.0f, 3.0f, false},
                               {1.0f, std::nanf(""), false},
                               {std::nanf(""), std::nanf(""), false},
                               {std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity(), true},
                               {0.0f, -0.0f, true}};
    std::vector<T> a_data;
    std::vector<T> b_data;
    std::vector<bool> e_data;
    std::transform(data.begin(), data.end(), std::back_inserter(a_data), [](const auto& param) { return param.a; });
    std::transform(data.begin(), data.end(), std::back_inserter(b_data), [](const auto& param) { return param.b; });
    std::transform(data.begin(), data.end(), std::back_inserter(e_data), [](const auto& param) { return param.e; });
    Shape shape{static_cast<int64_t>(data.size())};
    Array a = testing::BuildArray(shape).WithData<T>(a_data);
    Array b = testing::BuildArray(shape).WithData<T>(b_data);
    Array e = testing::BuildArray(shape).WithData<bool>(e_data);
    Array c = Equal(a, b);

    ASSERT_EQ(c.dtype(), Dtype::kBool);
    EXPECT_TRUE(c.IsContiguous());
    EXPECT_ARRAY_EQ(e, c);
}

TEST_P(LogicTest, EqualBroadcast) {
    using T = int32_t;

    Array a = testing::BuildArray({2, 3}).WithData<T>({1, 2, 3, 4, 3, 2});
    Array b = testing::BuildArray({2, 1}).WithData<T>({3, 2});
    Array e = testing::BuildArray({2, 3}).WithData<bool>({false, false, true, false, false, true});
    Array o = Equal(a, b);
    EXPECT_ARRAY_EQ(e, o);
}

TEST_P(LogicTest, Greater) {
    using T = float;

    struct Param {
        T a;
        T b;
        bool e;
    };

    std::vector<Param> data = {{1.0f, 1.0f, false},
                               {1.0f, -1.0f, true},
                               {2.0f, 3.0f, false},
                               {1.0f, std::nanf(""), false},
                               {std::nanf(""), std::nanf(""), false},
                               {std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity(), false},
                               {std::numeric_limits<T>::infinity(), 100, true},
                               {-std::numeric_limits<T>::infinity(), 100, false},
                               {0.0f, -0.0f, false}};
    std::vector<T> a_data;
    std::vector<T> b_data;
    std::vector<bool> e_data;
    std::transform(data.begin(), data.end(), std::back_inserter(a_data), [](const auto& param) { return param.a; });
    std::transform(data.begin(), data.end(), std::back_inserter(b_data), [](const auto& param) { return param.b; });
    std::transform(data.begin(), data.end(), std::back_inserter(e_data), [](const auto& param) { return param.e; });
    Shape shape{static_cast<int64_t>(data.size())};
    Array a = testing::BuildArray(shape).WithData<T>(a_data);
    Array b = testing::BuildArray(shape).WithData<T>(b_data);
    Array e = testing::BuildArray(shape).WithData<bool>(e_data);
    Array c = Greater(a, b);

    ASSERT_EQ(c.dtype(), Dtype::kBool);
    EXPECT_TRUE(c.IsContiguous());
    EXPECT_ARRAY_EQ(e, c);
}

TEST_P(LogicTest, GreaterBroadcast) {
    using T = int32_t;

    Array a = testing::BuildArray({2, 3}).WithData<T>({1, 2, 3, 4, 3, 2});
    Array b = testing::BuildArray({2, 1}).WithData<T>({2, 2});
    Array e = testing::BuildArray({2, 3}).WithData<bool>({false, false, true, true, true, false});
    Array o = Greater(a, b);
    EXPECT_ARRAY_EQ(e, o);
}

TEST_P(LogicTest, Not) {
    using T = float;

    struct Param {
        T a;
        bool e;
    };

    std::vector<Param> data = {
            {1.0f, false}, {0.0f, true}, {-0.0f, true}, {std::nanf(""), false}, {std::numeric_limits<T>::infinity(), false}};
    std::vector<T> a_data;
    std::vector<bool> e_data;
    std::transform(data.begin(), data.end(), std::back_inserter(a_data), [](const auto& param) { return param.a; });
    std::transform(data.begin(), data.end(), std::back_inserter(e_data), [](const auto& param) { return param.e; });
    Shape shape{static_cast<int64_t>(data.size())};
    Array a = testing::BuildArray(shape).WithData<T>(a_data);
    Array e = testing::BuildArray(shape).WithData<bool>(e_data);
    Array c = Not(a);

    ASSERT_EQ(c.dtype(), Dtype::kBool);
    EXPECT_TRUE(c.IsContiguous());
    EXPECT_ARRAY_EQ(e, c);
}

INSTANTIATE_TEST_CASE_P(
        ForEachBackend,
        LogicTest,
        ::testing::Values(
#ifdef XCHAINER_ENABLE_CUDA
                std::string{"cuda"},
#endif  // XCHAINER_ENABLE_CUDA
                std::string{"native"}));

}  // namespace
}  // namespace xchainer
