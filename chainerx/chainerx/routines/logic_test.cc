#include "chainerx/routines/logic.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/device_id.h"
#include "chainerx/dtype.h"
#include "chainerx/testing/array.h"
#include "chainerx/testing/array_check.h"
#include "chainerx/testing/device_session.h"
#include "chainerx/testing/routines.h"
#include "chainerx/testing/threading.h"

namespace chainerx {
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

TEST_THREAD_SAFE_P(LogicTest, Equal) {
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

    Run([&]() {
        testing::CheckForward(
                [](const std::vector<Array>& xs) {
                    Array y = Equal(xs[0], xs[1]);
                    EXPECT_EQ(y.dtype(), Dtype::kBool);
                    EXPECT_TRUE(y.IsContiguous());
                    return std::vector<Array>{y};
                },
                {a, b},
                {e});
    });
}

TEST_THREAD_SAFE_P(LogicTest, EqualBroadcast) {
    using T = int32_t;

    Array a = testing::BuildArray({2, 3}).WithData<T>({1, 2, 3, 4, 3, 2});
    Array b = testing::BuildArray({2, 1}).WithData<T>({3, 2});
    Array e = testing::BuildArray({2, 3}).WithData<bool>({false, false, true, false, false, true});

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Equal(xs[0], xs[1])}; }, {a, b}, {e});
    });
}

TEST_P(LogicTest, NotEqual) {
    using T = float;

    struct Param {
        T a;
        T b;
        bool e;
    };

    std::vector<Param> data = {{1.0f, 1.0f, false},
                               {1.0f, -1.0f, true},
                               {2.0f, 3.0f, true},
                               {1.0f, std::nanf(""), true},
                               {std::nanf(""), std::nanf(""), true},
                               {std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity(), false},
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
    Array c = NotEqual(a, b);

    ASSERT_EQ(c.dtype(), Dtype::kBool);
    EXPECT_TRUE(c.IsContiguous());
    EXPECT_ARRAY_EQ(e, c);
}

TEST_P(LogicTest, NotEqualBroadcast) {
    using T = int32_t;

    Array a = testing::BuildArray({2, 3}).WithData<T>({1, 2, 3, 4, 3, 2});
    Array b = testing::BuildArray({2, 1}).WithData<T>({3, 2});
    Array e = testing::BuildArray({2, 3}).WithData<bool>({true, true, false, true, true, false});
    Array o = NotEqual(a, b);
    EXPECT_ARRAY_EQ(e, o);
}

TEST_THREAD_SAFE_P(LogicTest, Greater) {
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

    Run([&]() {
        testing::CheckForward(
                [](const std::vector<Array>& xs) {
                    Array y = Greater(xs[0], xs[1]);
                    EXPECT_EQ(y.dtype(), Dtype::kBool);
                    EXPECT_TRUE(y.IsContiguous());
                    return std::vector<Array>{y};
                },
                {a, b},
                {e});
    });
}

TEST_THREAD_SAFE_P(LogicTest, GreaterBroadcast) {
    using T = int32_t;

    Array a = testing::BuildArray({2, 3}).WithData<T>({1, 2, 3, 4, 3, 2});
    Array b = testing::BuildArray({2, 1}).WithData<T>({2, 2});
    Array e = testing::BuildArray({2, 3}).WithData<bool>({false, false, true, true, true, false});

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Greater(xs[0], xs[1])}; }, {a, b}, {e});
    });
}

TEST_P(LogicTest, GreaterEqual) {
    using T = float;

    struct Param {
        T a;
        T b;
        bool e;
    };

    std::vector<Param> data = {{1.0f, 1.0f, true},
                               {1.0f, -1.0f, true},
                               {2.0f, 3.0f, false},
                               {1.0f, std::nanf(""), false},
                               {std::nanf(""), std::nanf(""), false},
                               {std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity(), true},
                               {std::numeric_limits<T>::infinity(), 100, true},
                               {-std::numeric_limits<T>::infinity(), 100, false},
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
    Array c = GreaterEqual(a, b);

    ASSERT_EQ(c.dtype(), Dtype::kBool);
    EXPECT_TRUE(c.IsContiguous());
    EXPECT_ARRAY_EQ(e, c);
}

TEST_P(LogicTest, GreaterEqualBroadcast) {
    using T = int32_t;

    Array a = testing::BuildArray({2, 3}).WithData<T>({1, 2, 3, 4, 3, 2});
    Array b = testing::BuildArray({2, 1}).WithData<T>({2, 2});
    Array e = testing::BuildArray({2, 3}).WithData<bool>({false, true, true, true, true, true});
    Array o = GreaterEqual(a, b);
    EXPECT_ARRAY_EQ(e, o);
}

TEST_P(LogicTest, Less) {
    using T = float;

    struct Param {
        T a;
        T b;
        bool e;
    };

    std::vector<Param> data = {{1.0f, 1.0f, false},
                               {1.0f, -1.0f, false},
                               {2.0f, 3.0f, true},
                               {1.0f, std::nanf(""), false},
                               {std::nanf(""), std::nanf(""), false},
                               {std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity(), false},
                               {std::numeric_limits<T>::infinity(), 100, false},
                               {-std::numeric_limits<T>::infinity(), 100, true},
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
    Array c = Less(a, b);

    ASSERT_EQ(c.dtype(), Dtype::kBool);
    EXPECT_TRUE(c.IsContiguous());
    EXPECT_ARRAY_EQ(e, c);
}

TEST_P(LogicTest, LessBroadcast) {
    using T = int32_t;

    Array a = testing::BuildArray({2, 3}).WithData<T>({1, 2, 3, 4, 3, 2});
    Array b = testing::BuildArray({2, 1}).WithData<T>({2, 2});
    Array e = testing::BuildArray({2, 3}).WithData<bool>({true, false, false, false, false, false});
    Array o = Less(a, b);
    EXPECT_ARRAY_EQ(e, o);
}

TEST_P(LogicTest, LessEqual) {
    using T = float;

    struct Param {
        T a;
        T b;
        bool e;
    };

    std::vector<Param> data = {{1.0f, 1.0f, true},
                               {1.0f, -1.0f, false},
                               {2.0f, 3.0f, true},
                               {1.0f, std::nanf(""), false},
                               {std::nanf(""), std::nanf(""), false},
                               {std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity(), true},
                               {std::numeric_limits<T>::infinity(), 100, false},
                               {-std::numeric_limits<T>::infinity(), 100, true},
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
    Array c = LessEqual(a, b);

    ASSERT_EQ(c.dtype(), Dtype::kBool);
    EXPECT_TRUE(c.IsContiguous());
    EXPECT_ARRAY_EQ(e, c);
}

TEST_P(LogicTest, LessEqualBroadcast) {
    using T = int32_t;

    Array a = testing::BuildArray({2, 3}).WithData<T>({1, 2, 3, 4, 3, 2});
    Array b = testing::BuildArray({2, 1}).WithData<T>({2, 2});
    Array e = testing::BuildArray({2, 3}).WithData<bool>({true, true, false, false, false, true});
    Array o = LessEqual(a, b);
    EXPECT_ARRAY_EQ(e, o);
}

TEST_THREAD_SAFE_P(LogicTest, LogicalNot) {
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

    Run([&]() {
        testing::CheckForward(
                [](const std::vector<Array>& xs) {
                    Array y = LogicalNot(xs[0]);
                    EXPECT_EQ(y.dtype(), Dtype::kBool);
                    EXPECT_TRUE(y.IsContiguous());
                    return std::vector<Array>{y};
                },
                {a},
                {e});
    });
}

INSTANTIATE_TEST_CASE_P(
        ForEachBackend,
        LogicTest,
        ::testing::Values(
#ifdef CHAINERX_ENABLE_CUDA
                std::string{"cuda"},
#endif  // CHAINERX_ENABLE_CUDA
                std::string{"native"}));

}  // namespace
}  // namespace chainerx
