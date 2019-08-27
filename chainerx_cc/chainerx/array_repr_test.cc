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

    // float16
    CheckArrayRepr<Float16>("array([0.], shape=(1,), dtype=float16, device='native:0')", {Float16{0}}, Shape({1}), device);
    CheckArrayRepr<Float16>("array([3.25], shape=(1,), dtype=float16, device='native:0')", {Float16{3.25}}, Shape({1}), device);
    CheckArrayRepr<Float16>("array([-3.25], shape=(1,), dtype=float16, device='native:0')", {Float16{-3.25}}, Shape({1}), device);
    CheckArrayRepr<Float16>(
            "array([ inf], shape=(1,), dtype=float16, device='native:0')",
            {Float16{std::numeric_limits<float>::infinity()}},
            Shape({1}),
            device);
    CheckArrayRepr<Float16>(
            "array([ -inf], shape=(1,), dtype=float16, device='native:0')",
            {Float16{-std::numeric_limits<float>::infinity()}},
            Shape({1}),
            device);
    CheckArrayRepr<Float16>("array([ nan], shape=(1,), dtype=float16, device='native:0')", {Float16{std::nanf("")}}, Shape({1}), device);
    CheckArrayRepr<Float16>(
            "array([[0., 1., 2.],\n"
            "       [3., 4., 5.]], shape=(2, 3), dtype=float16, device='native:0')",
            {Float16{0}, Float16{1}, Float16{2}, Float16{3}, Float16{4}, Float16{5}},
            Shape({2, 3}),
            device);
    CheckArrayRepr<Float16>(
            "array([[0.  , 1.  , 2.  ],\n"
            "       [3.25, 4.  , 5.  ]], shape=(2, 3), dtype=float16, device='native:0')",
            {Float16{0}, Float16{1}, Float16{2}, Float16{3.25}, Float16{4}, Float16{5}},
            Shape({2, 3}),
            device);
    CheckArrayRepr<Float16>(
            "array([[ 0.  ,  1.  ,  2.  ],\n"
            "       [-3.25,  4.  ,  5.  ]], shape=(2, 3), dtype=float16, device='native:0')",
            {Float16{0}, Float16{1}, Float16{2}, Float16{-3.25}, Float16{4}, Float16{5}},
            Shape({2, 3}),
            device);
    CheckArrayRepr<Float16>(
            "array([[[[3.25]]]], shape=(1, 1, 1, 1), dtype=float16, device='native:0')", {Float16{3.25}}, Shape({1, 1, 1, 1}), device);

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

        CheckArrayRepr<float>(
                "array([], shape=(2, 1, 0), dtype=float32, device='native:0', backprop_ids=['bp1'])",
                {},
                Shape({2, 1, 0}),
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

TEST(ArrayReprTest, AbbreviateRepr) {
    testing::DeviceSession device_session{DeviceId{"native:0"}};
    Device& device = device_session.device();

    {
        Array x = Arange(10000, Dtype::kFloat64, device);
        std::ostringstream os;
        os << x;
        EXPECT_EQ("array([   0.,    1.,    2., ..., 9997., 9998., 9999.], shape=(10000,), dtype=float64, device='native:0')", os.str());
    }

    {
        Array x = Zeros(Shape{100, 100}, Dtype::kInt16, device);
        std::ostringstream os;
        os << x;
        EXPECT_EQ(
                "array([[0, 0, 0, ..., 0, 0, 0],\n"
                "       [0, 0, 0, ..., 0, 0, 0],\n"
                "       [0, 0, 0, ..., 0, 0, 0],\n"
                "       ...,\n"
                "       [0, 0, 0, ..., 0, 0, 0],\n"
                "       [0, 0, 0, ..., 0, 0, 0],\n"
                "       [0, 0, 0, ..., 0, 0, 0]], shape=(100, 100), dtype=int16, device='native:0')",
                os.str());
    }

    {
        Array x = Arange(10000, Dtype::kInt32, device).Reshape(Shape{100, 100});
        std::ostringstream os;
        os << x;
        EXPECT_EQ(
                "array([[   0,    1,    2, ...,   97,   98,   99],\n"
                "       [ 100,  101,  102, ...,  197,  198,  199],\n"
                "       [ 200,  201,  202, ...,  297,  298,  299],\n"
                "       ...,\n"
                "       [9700, 9701, 9702, ..., 9797, 9798, 9799],\n"
                "       [9800, 9801, 9802, ..., 9897, 9898, 9899],\n"
                "       [9900, 9901, 9902, ..., 9997, 9998, 9999]], shape=(100, 100), dtype=int32, device='native:0')",
                os.str());
    }

    {
        Array x = Arange(27000, Dtype::kFloat32, device).Reshape(Shape{30, 30, 30});
        std::ostringstream os;
        os << x;
        EXPECT_EQ(
                "array([[[    0.,     1.,     2., ...,    27.,    28.,    29.],\n"
                "        [   30.,    31.,    32., ...,    57.,    58.,    59.],\n"
                "        [   60.,    61.,    62., ...,    87.,    88.,    89.],\n"
                "        ...,\n"
                "        [  810.,   811.,   812., ...,   837.,   838.,   839.],\n"
                "        [  840.,   841.,   842., ...,   867.,   868.,   869.],\n"
                "        [  870.,   871.,   872., ...,   897.,   898.,   899.]],\n"
                "\n"
                "       [[  900.,   901.,   902., ...,   927.,   928.,   929.],\n"
                "        [  930.,   931.,   932., ...,   957.,   958.,   959.],\n"
                "        [  960.,   961.,   962., ...,   987.,   988.,   989.],\n"
                "        ...,\n"
                "        [ 1710.,  1711.,  1712., ...,  1737.,  1738.,  1739.],\n"
                "        [ 1740.,  1741.,  1742., ...,  1767.,  1768.,  1769.],\n"
                "        [ 1770.,  1771.,  1772., ...,  1797.,  1798.,  1799.]],\n"
                "\n"
                "       [[ 1800.,  1801.,  1802., ...,  1827.,  1828.,  1829.],\n"
                "        [ 1830.,  1831.,  1832., ...,  1857.,  1858.,  1859.],\n"
                "        [ 1860.,  1861.,  1862., ...,  1887.,  1888.,  1889.],\n"
                "        ...,\n"
                "        [ 2610.,  2611.,  2612., ...,  2637.,  2638.,  2639.],\n"
                "        [ 2640.,  2641.,  2642., ...,  2667.,  2668.,  2669.],\n"
                "        [ 2670.,  2671.,  2672., ...,  2697.,  2698.,  2699.]],\n"
                "\n"
                "       ...,\n"
                "\n"
                "       [[24300., 24301., 24302., ..., 24327., 24328., 24329.],\n"
                "        [24330., 24331., 24332., ..., 24357., 24358., 24359.],\n"
                "        [24360., 24361., 24362., ..., 24387., 24388., 24389.],\n"
                "        ...,\n"
                "        [25110., 25111., 25112., ..., 25137., 25138., 25139.],\n"
                "        [25140., 25141., 25142., ..., 25167., 25168., 25169.],\n"
                "        [25170., 25171., 25172., ..., 25197., 25198., 25199.]],\n"
                "\n"
                "       [[25200., 25201., 25202., ..., 25227., 25228., 25229.],\n"
                "        [25230., 25231., 25232., ..., 25257., 25258., 25259.],\n"
                "        [25260., 25261., 25262., ..., 25287., 25288., 25289.],\n"
                "        ...,\n"
                "        [26010., 26011., 26012., ..., 26037., 26038., 26039.],\n"
                "        [26040., 26041., 26042., ..., 26067., 26068., 26069.],\n"
                "        [26070., 26071., 26072., ..., 26097., 26098., 26099.]],\n"
                "\n"
                "       [[26100., 26101., 26102., ..., 26127., 26128., 26129.],\n"
                "        [26130., 26131., 26132., ..., 26157., 26158., 26159.],\n"
                "        [26160., 26161., 26162., ..., 26187., 26188., 26189.],\n"
                "        ...,\n"
                "        [26910., 26911., 26912., ..., 26937., 26938., 26939.],\n"
                "        [26940., 26941., 26942., ..., 26967., 26968., 26969.],\n"
                "        [26970., 26971., 26972., ..., 26997., 26998., 26999.]]], shape=(30, 30, 30), dtype=float32, device='native:0')",
                os.str());
    }

    {
        Array x = Arange(40000, Dtype::kInt32, device).Reshape(Shape{100, 100, 4});
        std::ostringstream os;
        os << x;
        EXPECT_EQ(
                "array([[[    0,     1,     2,     3],\n"
                "        [    4,     5,     6,     7],\n"
                "        [    8,     9,    10,    11],\n"
                "        ...,\n"
                "        [  388,   389,   390,   391],\n"
                "        [  392,   393,   394,   395],\n"
                "        [  396,   397,   398,   399]],\n"
                "\n"
                "       [[  400,   401,   402,   403],\n"
                "        [  404,   405,   406,   407],\n"
                "        [  408,   409,   410,   411],\n"
                "        ...,\n"
                "        [  788,   789,   790,   791],\n"
                "        [  792,   793,   794,   795],\n"
                "        [  796,   797,   798,   799]],\n"
                "\n"
                "       [[  800,   801,   802,   803],\n"
                "        [  804,   805,   806,   807],\n"
                "        [  808,   809,   810,   811],\n"
                "        ...,\n"
                "        [ 1188,  1189,  1190,  1191],\n"
                "        [ 1192,  1193,  1194,  1195],\n"
                "        [ 1196,  1197,  1198,  1199]],\n"
                "\n"
                "       ...,\n"
                "\n"
                "       [[38800, 38801, 38802, 38803],\n"
                "        [38804, 38805, 38806, 38807],\n"
                "        [38808, 38809, 38810, 38811],\n"
                "        ...,\n"
                "        [39188, 39189, 39190, 39191],\n"
                "        [39192, 39193, 39194, 39195],\n"
                "        [39196, 39197, 39198, 39199]],\n"
                "\n"
                "       [[39200, 39201, 39202, 39203],\n"
                "        [39204, 39205, 39206, 39207],\n"
                "        [39208, 39209, 39210, 39211],\n"
                "        ...,\n"
                "        [39588, 39589, 39590, 39591],\n"
                "        [39592, 39593, 39594, 39595],\n"
                "        [39596, 39597, 39598, 39599]],\n"
                "\n"
                "       [[39600, 39601, 39602, 39603],\n"
                "        [39604, 39605, 39606, 39607],\n"
                "        [39608, 39609, 39610, 39611],\n"
                "        ...,\n"
                "        [39988, 39989, 39990, 39991],\n"
                "        [39992, 39993, 39994, 39995],\n"
                "        [39996, 39997, 39998, 39999]]], shape=(100, 100, 4), dtype=int32, device='native:0')",
                os.str());
    }

    {
        Array x = Arange(40000, Dtype::kInt32, device).Reshape(Shape{100, 4, 100});
        std::ostringstream os;
        os << x;
        EXPECT_EQ(
                "array([[[    0,     1,     2, ...,    97,    98,    99],\n"
                "        [  100,   101,   102, ...,   197,   198,   199],\n"
                "        [  200,   201,   202, ...,   297,   298,   299],\n"
                "        [  300,   301,   302, ...,   397,   398,   399]],\n"
                "\n"
                "       [[  400,   401,   402, ...,   497,   498,   499],\n"
                "        [  500,   501,   502, ...,   597,   598,   599],\n"
                "        [  600,   601,   602, ...,   697,   698,   699],\n"
                "        [  700,   701,   702, ...,   797,   798,   799]],\n"
                "\n"
                "       [[  800,   801,   802, ...,   897,   898,   899],\n"
                "        [  900,   901,   902, ...,   997,   998,   999],\n"
                "        [ 1000,  1001,  1002, ...,  1097,  1098,  1099],\n"
                "        [ 1100,  1101,  1102, ...,  1197,  1198,  1199]],\n"
                "\n"
                "       ...,\n"
                "\n"
                "       [[38800, 38801, 38802, ..., 38897, 38898, 38899],\n"
                "        [38900, 38901, 38902, ..., 38997, 38998, 38999],\n"
                "        [39000, 39001, 39002, ..., 39097, 39098, 39099],\n"
                "        [39100, 39101, 39102, ..., 39197, 39198, 39199]],\n"
                "\n"
                "       [[39200, 39201, 39202, ..., 39297, 39298, 39299],\n"
                "        [39300, 39301, 39302, ..., 39397, 39398, 39399],\n"
                "        [39400, 39401, 39402, ..., 39497, 39498, 39499],\n"
                "        [39500, 39501, 39502, ..., 39597, 39598, 39599]],\n"
                "\n"
                "       [[39600, 39601, 39602, ..., 39697, 39698, 39699],\n"
                "        [39700, 39701, 39702, ..., 39797, 39798, 39799],\n"
                "        [39800, 39801, 39802, ..., 39897, 39898, 39899],\n"
                "        [39900, 39901, 39902, ..., 39997, 39998, 39999]]], shape=(100, 4, 100), dtype=int32, device='native:0')",
                os.str());
    }

    {
        Array x = Arange(40000, Dtype::kInt32, device).Reshape(Shape{4, 100, 100});
        std::ostringstream os;
        os << x;
        EXPECT_EQ(
                "array([[[    0,     1,     2, ...,    97,    98,    99],\n"
                "        [  100,   101,   102, ...,   197,   198,   199],\n"
                "        [  200,   201,   202, ...,   297,   298,   299],\n"
                "        ...,\n"
                "        [ 9700,  9701,  9702, ...,  9797,  9798,  9799],\n"
                "        [ 9800,  9801,  9802, ...,  9897,  9898,  9899],\n"
                "        [ 9900,  9901,  9902, ...,  9997,  9998,  9999]],\n"
                "\n"
                "       [[10000, 10001, 10002, ..., 10097, 10098, 10099],\n"
                "        [10100, 10101, 10102, ..., 10197, 10198, 10199],\n"
                "        [10200, 10201, 10202, ..., 10297, 10298, 10299],\n"
                "        ...,\n"
                "        [19700, 19701, 19702, ..., 19797, 19798, 19799],\n"
                "        [19800, 19801, 19802, ..., 19897, 19898, 19899],\n"
                "        [19900, 19901, 19902, ..., 19997, 19998, 19999]],\n"
                "\n"
                "       [[20000, 20001, 20002, ..., 20097, 20098, 20099],\n"
                "        [20100, 20101, 20102, ..., 20197, 20198, 20199],\n"
                "        [20200, 20201, 20202, ..., 20297, 20298, 20299],\n"
                "        ...,\n"
                "        [29700, 29701, 29702, ..., 29797, 29798, 29799],\n"
                "        [29800, 29801, 29802, ..., 29897, 29898, 29899],\n"
                "        [29900, 29901, 29902, ..., 29997, 29998, 29999]],\n"
                "\n"
                "       [[30000, 30001, 30002, ..., 30097, 30098, 30099],\n"
                "        [30100, 30101, 30102, ..., 30197, 30198, 30199],\n"
                "        [30200, 30201, 30202, ..., 30297, 30298, 30299],\n"
                "        ...,\n"
                "        [39700, 39701, 39702, ..., 39797, 39798, 39799],\n"
                "        [39800, 39801, 39802, ..., 39897, 39898, 39899],\n"
                "        [39900, 39901, 39902, ..., 39997, 39998, 39999]]], shape=(4, 100, 100), dtype=int32, device='native:0')",
                os.str());
    }
}

}  // namespace
}  // namespace chainerx
