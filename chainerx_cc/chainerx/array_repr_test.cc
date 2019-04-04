#include "chainerx/array_repr.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "chainerx/backprop_scope.h"
#include "chainerx/context.h"
#include "chainerx/device.h"
#include "chainerx/device_id.h"
#include "chainerx/graph.h"
#include "chainerx/native/native_backend.h"
#include "chainerx/native/native_device.h"
#include "chainerx/routines/creation.h"
#include "chainerx/testing/array.h"
#include "chainerx/testing/device_session.h"

namespace chainerx {
namespace {

template <typename T>
void CheckArrayRepr(
        const std::string& expected,
        const std::vector<T>& data_vec,
        const Shape& shape,
        Device& device,
        const std::vector<BackpropId>& backprop_ids = {}) {
    // Copy to a contiguous memory block because std::vector<bool> is not packed as a sequence of bool's.
    std::shared_ptr<T> data_ptr{new T[data_vec.size()], std::default_delete<T[]>{}};
    std::copy(data_vec.begin(), data_vec.end(), data_ptr.get());
    Array array = FromContiguousHostData(shape, TypeToDtype<T>, static_cast<std::shared_ptr<void>>(data_ptr), device);
    for (const BackpropId& backprop_id : backprop_ids) {
        array.RequireGrad(backprop_id);
    }

    // std::string version
    EXPECT_EQ(expected, ArrayRepr(array));

    // std::ostream version
    std::ostringstream os;
    os << array;
    EXPECT_EQ(expected, os.str());
}

TEST(ArrayReprTest, AllDtypesOnNativeBackend) {
    testing::DeviceSession device_session{DeviceId{"native:0"}};
    Device& device = device_session.device();

    // bool
    CheckArrayRepr<bool>("array([False], shape=(1,), dtype=bool, device='native:0')", {false}, Shape({1}), device);
    CheckArrayRepr<bool>("array([ True], shape=(1,), dtype=bool, device='native:0')", {true}, Shape({1}), device);
    CheckArrayRepr<bool>(
            "array([[False,  True,  True],\n"
            "       [ True, False,  True]], shape=(2, 3), dtype=bool, device='native:0')",
            {false, true, true, true, false, true},
            Shape({2, 3}),
            device);
    CheckArrayRepr<bool>("array([[[[ True]]]], shape=(1, 1, 1, 1), dtype=bool, device='native:0')", {true}, Shape({1, 1, 1, 1}), device);

    // int8
    CheckArrayRepr<int8_t>("array([0], shape=(1,), dtype=int8, device='native:0')", {0}, Shape({1}), device);
    CheckArrayRepr<int8_t>("array([-2], shape=(1,), dtype=int8, device='native:0')", {-2}, Shape({1}), device);
    CheckArrayRepr<int8_t>(
            "array([[0, 1, 2],\n"
            "       [3, 4, 5]], shape=(2, 3), dtype=int8, device='native:0')",
            {0, 1, 2, 3, 4, 5},
            Shape({2, 3}),
            device);
    CheckArrayRepr<int8_t>(
            "array([[ 0,  1,  2],\n"
            "       [-3,  4,  5]], shape=(2, 3), dtype=int8, device='native:0')",
            {0, 1, 2, -3, 4, 5},
            Shape({2, 3}),
            device);
    CheckArrayRepr<int8_t>("array([[[[3]]]], shape=(1, 1, 1, 1), dtype=int8, device='native:0')", {3}, Shape({1, 1, 1, 1}), device);

    // int16
    CheckArrayRepr<int16_t>("array([0], shape=(1,), dtype=int16, device='native:0')", {0}, Shape({1}), device);
    CheckArrayRepr<int16_t>("array([-2], shape=(1,), dtype=int16, device='native:0')", {-2}, Shape({1}), device);
    CheckArrayRepr<int16_t>(
            "array([[0, 1, 2],\n"
            "       [3, 4, 5]], shape=(2, 3), dtype=int16, device='native:0')",
            {0, 1, 2, 3, 4, 5},
            Shape({2, 3}),
            device);
    CheckArrayRepr<int16_t>(
            "array([[ 0,  1,  2],\n"
            "       [-3,  4,  5]], shape=(2, 3), dtype=int16, device='native:0')",
            {0, 1, 2, -3, 4, 5},
            Shape({2, 3}),
            device);
    CheckArrayRepr<int16_t>("array([[[[3]]]], shape=(1, 1, 1, 1), dtype=int16, device='native:0')", {3}, Shape({1, 1, 1, 1}), device);

    // int32
    CheckArrayRepr<int32_t>("array([0], shape=(1,), dtype=int32, device='native:0')", {0}, Shape({1}), device);
    CheckArrayRepr<int32_t>("array([-2], shape=(1,), dtype=int32, device='native:0')", {-2}, Shape({1}), device);
    CheckArrayRepr<int32_t>(
            "array([[0, 1, 2],\n"
            "       [3, 4, 5]], shape=(2, 3), dtype=int32, device='native:0')",
            {0, 1, 2, 3, 4, 5},
            Shape({2, 3}),
            device);
    CheckArrayRepr<int32_t>(
            "array([[ 0,  1,  2],\n"
            "       [-3,  4,  5]], shape=(2, 3), dtype=int32, device='native:0')",
            {0, 1, 2, -3, 4, 5},
            Shape({2, 3}),
            device);
    CheckArrayRepr<int32_t>("array([[[[3]]]], shape=(1, 1, 1, 1), dtype=int32, device='native:0')", {3}, Shape({1, 1, 1, 1}), device);

    // int64
    CheckArrayRepr<int64_t>("array([0], shape=(1,), dtype=int64, device='native:0')", {0}, Shape({1}), device);
    CheckArrayRepr<int64_t>("array([-2], shape=(1,), dtype=int64, device='native:0')", {-2}, Shape({1}), device);
    CheckArrayRepr<int64_t>(
            "array([[0, 1, 2],\n"
            "       [3, 4, 5]], shape=(2, 3), dtype=int64, device='native:0')",
            {0, 1, 2, 3, 4, 5},
            Shape({2, 3}),
            device);
    CheckArrayRepr<int64_t>(
            "array([[ 0,  1,  2],\n"
            "       [-3,  4,  5]], shape=(2, 3), dtype=int64, device='native:0')",
            {0, 1, 2, -3, 4, 5},
            Shape({2, 3}),
            device);
    CheckArrayRepr<int64_t>("array([[[[3]]]], shape=(1, 1, 1, 1), dtype=int64, device='native:0')", {3}, Shape({1, 1, 1, 1}), device);

    // uint8
    CheckArrayRepr<uint8_t>("array([0], shape=(1,), dtype=uint8, device='native:0')", {0}, Shape({1}), device);
    CheckArrayRepr<uint8_t>("array([2], shape=(1,), dtype=uint8, device='native:0')", {2}, Shape({1}), device);
    CheckArrayRepr<uint8_t>(
            "array([[0, 1, 2],\n"
            "       [3, 4, 5]], shape=(2, 3), dtype=uint8, device='native:0')",
            {0, 1, 2, 3, 4, 5},
            Shape({2, 3}),
            device);
    CheckArrayRepr<uint8_t>("array([[[[3]]]], shape=(1, 1, 1, 1), dtype=uint8, device='native:0')", {3}, Shape({1, 1, 1, 1}), device);

    // float32
    CheckArrayRepr<float>("array([0.], shape=(1,), dtype=float32, device='native:0')", {0}, Shape({1}), device);
    CheckArrayRepr<float>("array([3.25], shape=(1,), dtype=float32, device='native:0')", {3.25}, Shape({1}), device);
    CheckArrayRepr<float>("array([-3.25], shape=(1,), dtype=float32, device='native:0')", {-3.25}, Shape({1}), device);
    CheckArrayRepr<float>(
            "array([ inf], shape=(1,), dtype=float32, device='native:0')", {std::numeric_limits<float>::infinity()}, Shape({1}), device);
    CheckArrayRepr<float>(
            "array([ -inf], shape=(1,), dtype=float32, device='native:0')", {-std::numeric_limits<float>::infinity()}, Shape({1}), device);
    CheckArrayRepr<float>("array([ nan], shape=(1,), dtype=float32, device='native:0')", {std::nanf("")}, Shape({1}), device);
    CheckArrayRepr<float>(
            "array([[0., 1., 2.],\n"
            "       [3., 4., 5.]], shape=(2, 3), dtype=float32, device='native:0')",
            {0, 1, 2, 3, 4, 5},
            Shape({2, 3}),
            device);
    CheckArrayRepr<float>(
            "array([[0.  , 1.  , 2.  ],\n"
            "       [3.25, 4.  , 5.  ]], shape=(2, 3), dtype=float32, device='native:0')",
            {0, 1, 2, 3.25, 4, 5},
            Shape({2, 3}),
            device);
    CheckArrayRepr<float>(
            "array([[ 0.  ,  1.  ,  2.  ],\n"
            "       [-3.25,  4.  ,  5.  ]], shape=(2, 3), dtype=float32, device='native:0')",
            {0, 1, 2, -3.25, 4, 5},
            Shape({2, 3}),
            device);
    CheckArrayRepr<float>("array([[[[3.25]]]], shape=(1, 1, 1, 1), dtype=float32, device='native:0')", {3.25}, Shape({1, 1, 1, 1}), device);

    // float64
    CheckArrayRepr<double>("array([0.], shape=(1,), dtype=float64, device='native:0')", {0}, Shape({1}), device);
    CheckArrayRepr<double>("array([3.25], shape=(1,), dtype=float64, device='native:0')", {3.25}, Shape({1}), device);
    CheckArrayRepr<double>("array([-3.25], shape=(1,), dtype=float64, device='native:0')", {-3.25}, Shape({1}), device);
    CheckArrayRepr<double>(
            "array([ inf], shape=(1,), dtype=float64, device='native:0')", {std::numeric_limits<double>::infinity()}, Shape({1}), device);
    CheckArrayRepr<double>(
            "array([ -inf], shape=(1,), dtype=float64, device='native:0')", {-std::numeric_limits<double>::infinity()}, Shape({1}), device);
    CheckArrayRepr<double>("array([ nan], shape=(1,), dtype=float64, device='native:0')", {std::nan("")}, Shape({1}), device);
    CheckArrayRepr<double>(
            "array([[0., 1., 2.],\n"
            "       [3., 4., 5.]], shape=(2, 3), dtype=float64, device='native:0')",
            {0, 1, 2, 3, 4, 5},
            Shape({2, 3}),
            device);
    CheckArrayRepr<double>(
            "array([[0.  , 1.  , 2.  ],\n"
            "       [3.25, 4.  , 5.  ]], shape=(2, 3), dtype=float64, device='native:0')",
            {0, 1, 2, 3.25, 4, 5},
            Shape({2, 3}),
            device);
    CheckArrayRepr<double>(
            "array([[ 0.  ,  1.  ,  2.  ],\n"
            "       [-3.25,  4.  ,  5.  ]], shape=(2, 3), dtype=float64, device='native:0')",
            {0, 1, 2, -3.25, 4, 5},
            Shape({2, 3}),
            device);
    CheckArrayRepr<double>(
            "array([[[[3.25]]]], shape=(1, 1, 1, 1), dtype=float64, device='native:0')", {3.25}, Shape({1, 1, 1, 1}), device);

    // 0-sized
    {
        BackpropScope backprop_scope{"bp1"};
        BackpropId backprop_id = backprop_scope.backprop_id();

        CheckArrayRepr<float>(
                "array([], shape=(0, 1, 2), dtype=float32, device='native:0', backprop_ids=['bp1'])",
                {},
                Shape({0, 1, 2}),
                device,
                {backprop_id});
    }

    // Single graph
    {
        BackpropScope backprop_scope{"bp1"};
        BackpropId backprop_id = backprop_scope.backprop_id();

        CheckArrayRepr<float>(
                "array([-2.], shape=(1,), dtype=float32, device='native:0', backprop_ids=['bp1'])",
                {-2},
                Shape({1}),
                device,
                {backprop_id});
    }

    // Two graphs
    {
        BackpropScope backprop_scope1{"bp1"};
        BackpropScope backprop_scope2{"bp2"};
        BackpropId backprop_id1 = backprop_scope1.backprop_id();
        BackpropId backprop_id2 = backprop_scope2.backprop_id();

        CheckArrayRepr<float>(
                "array([1.], shape=(1,), dtype=float32, device='native:0', backprop_ids=['bp1', 'bp2'])",
                {1},
                Shape({1}),
                device,
                {backprop_id1, backprop_id2});
    }

    // Multiple graphs
    {
        BackpropScope backprop_scope1{"bp1"};
        BackpropScope backprop_scope2{"bp2"};
        BackpropScope backprop_scope3{"bp3"};
        BackpropScope backprop_scope4{"bp4"};
        BackpropScope backprop_scope5{"bp5"};
        BackpropId backprop_id1 = backprop_scope1.backprop_id();
        BackpropId backprop_id2 = backprop_scope2.backprop_id();
        BackpropId backprop_id3 = backprop_scope3.backprop_id();
        BackpropId backprop_id4 = backprop_scope4.backprop_id();
        BackpropId backprop_id5 = backprop_scope5.backprop_id();

        CheckArrayRepr<float>(
                "array([-9.], shape=(1,), dtype=float32, device='native:0', backprop_ids=['bp1', 'bp2', 'bp3', 'bp4', "
                "'bp5'])",
                {-9},
                Shape({1}),
                device,
                {backprop_id1, backprop_id2, backprop_id3, backprop_id4, backprop_id5});
    }
}

TEST(ArrayReprTest, ExpiredBackprop) {
    testing::DeviceSession device_session{DeviceId{"native:0"}};

    Array a{};
    {
        BackpropScope backprop_scope{"bp1"};
        BackpropId backprop_id = backprop_scope.backprop_id();
        a = testing::BuildArray({1}).WithData<float>({3.0f});
        a.RequireGrad(backprop_id);
    }

    std::ostringstream os;
    os << a;
    EXPECT_EQ("array([3.], shape=(1,), dtype=float32, device='native:0', backprop_ids=['<expired>'])", os.str());
}

}  // namespace
}  // namespace chainerx
