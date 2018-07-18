#include "xchainer/array_repr.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "xchainer/context.h"
#include "xchainer/device.h"
#include "xchainer/device_id.h"
#include "xchainer/graph.h"
#include "xchainer/graph_scope.h"
#include "xchainer/native/native_backend.h"
#include "xchainer/native/native_device.h"
#include "xchainer/routines/creation.h"
#include "xchainer/testing/device_session.h"

namespace xchainer {
namespace {

template <typename T>
void CheckArrayRepr(
        const std::string& expected,
        const std::vector<T>& data_vec,
        const Shape& shape,
        Device& device,
        const std::vector<GraphId>& graph_ids = {}) {
    // Copy to a contiguous memory block because std::vector<bool> is not packed as a sequence of bool's.
    std::shared_ptr<T> data_ptr{new T[data_vec.size()], std::default_delete<T[]>{}};
    std::copy(data_vec.begin(), data_vec.end(), data_ptr.get());
    Array array = internal::FromContiguousHostData(shape, TypeToDtype<T>, static_cast<std::shared_ptr<void>>(data_ptr), device);
    for (const GraphId& graph_id : graph_ids) {
        array.RequireGrad(graph_id);
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
        GraphScope graph_scope{"graph_1"};
        GraphId graph_id = graph_scope.graph_id();

        CheckArrayRepr<int32_t>(
                "array([], shape=(0, 1, 2), dtype=int32, device='native:0', graph_ids=['1'])", {}, Shape({0, 1, 2}), device, {graph_id});
    }

    // Single graph
    {
        GraphScope graph_scope{"graph_1"};
        GraphId graph_id = graph_scope.graph_id();

        CheckArrayRepr<int32_t>(
                "array([-2], shape=(1,), dtype=int32, device='native:0', graph_ids=['2'])", {-2}, Shape({1}), device, {graph_id});
    }

    // Two graphs
    {
        GraphScope graph_scope1{"graph_1"};
        GraphScope graph_scope2{"graph_2"};
        GraphId graph_id1 = graph_scope1.graph_id();
        GraphId graph_id2 = graph_scope2.graph_id();

        CheckArrayRepr<int32_t>(
                "array([1], shape=(1,), dtype=int32, device='native:0', graph_ids=['3', '4'])",
                {1},
                Shape({1}),
                device,
                {graph_id1, graph_id2});
    }

    // Multiple graphs
    {
        GraphScope graph_scope1{"graph_1"};
        GraphScope graph_scope2{"graph_2"};
        GraphScope graph_scope3{"graph_3"};
        GraphScope graph_scope4{"graph_4"};
        GraphScope graph_scope5{"graph_5"};
        GraphId graph_id1 = graph_scope1.graph_id();
        GraphId graph_id2 = graph_scope2.graph_id();
        GraphId graph_id3 = graph_scope3.graph_id();
        GraphId graph_id4 = graph_scope4.graph_id();
        GraphId graph_id5 = graph_scope5.graph_id();

        CheckArrayRepr<int32_t>(
                "array([-9], shape=(1,), dtype=int32, device='native:0', graph_ids=['5', '6', '7', '8', '9'])",
                {-9},
                Shape({1}),
                device,
                {graph_id1, graph_id2, graph_id3, graph_id4, graph_id5});
    }
}

}  // namespace
}  // namespace xchainer
