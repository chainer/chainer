#include "chainerx/routines/creation.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/check_backward.h"
#include "chainerx/device.h"
#include "chainerx/device_id.h"
#include "chainerx/dtype.h"
#include "chainerx/routines/type_util.h"
#include "chainerx/scalar.h"
#include "chainerx/shape.h"
#include "chainerx/testing/array.h"
#include "chainerx/testing/array_check.h"
#include "chainerx/testing/device_session.h"
#include "chainerx/testing/routines.h"
#include "chainerx/testing/threading.h"

#define EXPECT_ARRAYS_ARE_EQUAL_COPY(orig, copy)             \
    do {                                                     \
        EXPECT_TRUE((copy).IsContiguous());                  \
        EXPECT_EQ((copy).offset(), 0);                       \
        EXPECT_NE((orig).data().get(), (copy).data().get()); \
        EXPECT_ARRAY_EQ((orig), (copy));                     \
    } while (0)

namespace chainerx {
namespace {

class CreationTest : public ::testing::TestWithParam<std::string> {
protected:
    void SetUp() override {
        const std::string& backend_name = GetParam();
        device_session_.emplace(DeviceId{backend_name, 0});
    }

    void TearDown() override { device_session_.reset(); }

public:
    template <typename T>
    void CheckEmpty() {
        testing::RunTestWithThreads([]() {
            Dtype dtype = TypeToDtype<T>;
            Array x = Empty(Shape{3, 2}, dtype);
            EXPECT_NE(x.data(), nullptr);
            EXPECT_EQ(x.shape(), Shape({3, 2}));
            EXPECT_EQ(x.dtype(), dtype);
            EXPECT_TRUE(x.IsContiguous());
            EXPECT_EQ(0, x.offset());
            EXPECT_EQ(&GetDefaultDevice(), &x.device());
        });
    }

    template <typename T>
    void CheckEmptyLike() {
        testing::RunTestWithThreads([]() {
            Dtype dtype = TypeToDtype<T>;
            Array x_orig = Empty(Shape{3, 2}, dtype);
            Array x = EmptyLike(x_orig);
            EXPECT_NE(x.data(), nullptr);
            EXPECT_NE(x.data(), x_orig.data());
            EXPECT_EQ(x.shape(), x_orig.shape());
            EXPECT_EQ(x.dtype(), x_orig.dtype());
            EXPECT_TRUE(x.IsContiguous());
            EXPECT_EQ(0, x.offset());
            EXPECT_EQ(&GetDefaultDevice(), &x.device());
        });
    }

    template <typename T>
    void CheckFullWithGivenDtype(T expected, Scalar scalar) {
        testing::RunTestWithThreads([&expected, &scalar]() {
            Dtype dtype = TypeToDtype<T>;
            Array x = Full(Shape{3, 2}, scalar, dtype);
            EXPECT_NE(x.data(), nullptr);
            EXPECT_EQ(x.shape(), Shape({3, 2}));
            EXPECT_EQ(x.dtype(), dtype);
            EXPECT_TRUE(x.IsContiguous());
            EXPECT_EQ(0, x.offset());
            testing::ExpectDataEqual(expected, x);
            EXPECT_EQ(&GetDefaultDevice(), &x.device());
        });
    }

    template <typename T>
    void CheckFullWithGivenDtype(T value) {
        CheckFullWithGivenDtype(value, value);
    }

    template <typename T>
    void CheckFullWithScalarDtype(T value) {
        testing::RunTestWithThreads([&value]() {
            Scalar scalar = {value};
            Array x = Full(Shape{3, 2}, scalar);
            EXPECT_NE(x.data(), nullptr);
            EXPECT_EQ(x.shape(), Shape({3, 2}));
            EXPECT_EQ(x.dtype(), internal::GetDefaultDtype(scalar.kind()));
            EXPECT_TRUE(x.IsContiguous());
            EXPECT_EQ(0, x.offset());
            testing::ExpectDataEqual(value, x);
            EXPECT_EQ(&GetDefaultDevice(), &x.device());
        });
    }

    template <typename T>
    void CheckFullLike(T expected, Scalar scalar) {
        testing::RunTestWithThreads([&expected, &scalar]() {
            Dtype dtype = TypeToDtype<T>;
            Array x_orig = Empty(Shape{3, 2}, dtype);
            Array x = FullLike(x_orig, scalar);
            EXPECT_NE(x.data(), nullptr);
            EXPECT_NE(x.data(), x_orig.data());
            EXPECT_EQ(x.shape(), x_orig.shape());
            EXPECT_EQ(x.dtype(), x_orig.dtype());
            EXPECT_TRUE(x.IsContiguous());
            EXPECT_EQ(0, x.offset());
            testing::ExpectDataEqual(expected, x);
            EXPECT_EQ(&GetDefaultDevice(), &x.device());
        });
    }

    template <typename T>
    void CheckFullLike(T value) {
        CheckFullLike(value, value);
    }

    template <typename T>
    void CheckZeros() {
        testing::RunTestWithThreads([]() {
            Dtype dtype = TypeToDtype<T>;
            Array x = Zeros(Shape{3, 2}, dtype);
            EXPECT_NE(x.data(), nullptr);
            EXPECT_EQ(x.shape(), Shape({3, 2}));
            EXPECT_EQ(x.dtype(), dtype);
            EXPECT_TRUE(x.IsContiguous());
            EXPECT_EQ(0, x.offset());
            T expected{0};
            testing::ExpectDataEqual(expected, x);
            EXPECT_EQ(&GetDefaultDevice(), &x.device());
        });
    }

    template <typename T>
    void CheckZerosLike() {
        testing::RunTestWithThreads([]() {
            Dtype dtype = TypeToDtype<T>;
            Array x_orig = Empty(Shape{3, 2}, dtype);
            Array x = ZerosLike(x_orig);
            EXPECT_NE(x.data(), nullptr);
            EXPECT_NE(x.data(), x_orig.data());
            EXPECT_EQ(x.shape(), x_orig.shape());
            EXPECT_EQ(x.dtype(), x_orig.dtype());
            EXPECT_TRUE(x.IsContiguous());
            EXPECT_EQ(0, x.offset());
            T expected{0};
            testing::ExpectDataEqual(expected, x);
            EXPECT_EQ(&GetDefaultDevice(), &x.device());
        });
    }

    template <typename T>
    void CheckOnes() {
        testing::RunTestWithThreads([]() {
            Dtype dtype = TypeToDtype<T>;
            Array x = Ones(Shape{3, 2}, dtype);
            EXPECT_NE(x.data(), nullptr);
            EXPECT_EQ(x.shape(), Shape({3, 2}));
            EXPECT_EQ(x.dtype(), dtype);
            EXPECT_TRUE(x.IsContiguous());
            EXPECT_EQ(0, x.offset());
            T expected{1};
            testing::ExpectDataEqual(expected, x);
            EXPECT_EQ(&GetDefaultDevice(), &x.device());
        });
    }

    template <typename T>
    void CheckOnesLike() {
        testing::RunTestWithThreads([]() {
            Dtype dtype = TypeToDtype<T>;
            Array x_orig = Empty(Shape{3, 2}, dtype);
            Array x = OnesLike(x_orig);
            EXPECT_NE(x.data(), nullptr);
            EXPECT_NE(x.data(), x_orig.data());
            EXPECT_EQ(x.shape(), x_orig.shape());
            EXPECT_EQ(x.dtype(), x_orig.dtype());
            EXPECT_TRUE(x.IsContiguous());
            EXPECT_EQ(0, x.offset());
            T expected{1};
            testing::ExpectDataEqual(expected, x);
            EXPECT_EQ(&GetDefaultDevice(), &x.device());
        });
    }

private:
    nonstd::optional<testing::DeviceSession> device_session_;
};

TEST_P(CreationTest, FromContiguousHostData) {
    using T = int32_t;
    Shape shape{3, 2};

    T raw_data[] = {0, 1, 2, 3, 4, 5};
    std::shared_ptr<T> data{raw_data, [](const T*) {}};

    Dtype dtype = TypeToDtype<T>;

    testing::RunTestWithThreads([&shape, &dtype, &data]() {
        Array x = FromContiguousHostData(shape, dtype, data);

        // Basic attributes
        EXPECT_EQ(shape, x.shape());
        EXPECT_EQ(dtype, x.dtype());
        EXPECT_EQ(2, x.ndim());
        EXPECT_EQ(3 * 2, x.GetTotalSize());
        EXPECT_EQ(int64_t{sizeof(T)}, x.GetItemSize());
        EXPECT_EQ(shape.GetTotalSize() * int64_t{sizeof(T)}, x.GetNBytes());
        EXPECT_TRUE(x.IsContiguous());
        EXPECT_EQ(0, x.offset());

        // Array::data
        testing::ExpectDataEqual<T>(data.get(), x);

        Device& device = GetDefaultDevice();
        EXPECT_EQ(&device, &x.device());
        if (device.backend().GetName() == "native") {
            EXPECT_EQ(data.get(), x.data().get());
        } else {
            CHAINERX_ASSERT(device.backend().GetName() == "cuda");
            EXPECT_NE(data.get(), x.data().get());
        }
    });
}

namespace {

template <typename T>
void CheckFromData(
        const Array& x, const Shape& shape, Dtype dtype, const Strides& strides, int64_t offset, const T* raw_data, const void* data_ptr) {
    EXPECT_EQ(shape, x.shape());
    EXPECT_EQ(dtype, x.dtype());
    EXPECT_EQ(strides, x.strides());
    EXPECT_EQ(shape.ndim(), x.ndim());
    EXPECT_EQ(shape.GetTotalSize(), x.GetTotalSize());
    EXPECT_EQ(int64_t{sizeof(T)}, x.GetItemSize());
    EXPECT_EQ(shape.GetTotalSize() * int64_t{sizeof(T)}, x.GetNBytes());
    EXPECT_EQ(offset, x.offset());
    EXPECT_EQ(internal::IsContiguous(shape, strides, GetItemSize(dtype)), x.IsContiguous());
    EXPECT_EQ(&GetDefaultDevice(), &x.device());

    testing::ExpectDataEqual<T>(raw_data, x);
    EXPECT_EQ(data_ptr, x.data().get());
}

}  // namespace

TEST_P(CreationTest, FromData) {
    using T = int32_t;
    Dtype dtype = TypeToDtype<T>;
    Device& device = GetDefaultDevice();

    T raw_data[] = {0, 1, 2, 3, 4, 5};
    std::shared_ptr<void> host_data{raw_data, [](const T*) {}};

    // non-contiguous array like a[:,1]
    T expected_data[] = {1, 4};
    Shape shape{2};
    Strides strides{sizeof(T) * 3};
    int64_t offset = sizeof(T);

    testing::RunTestWithThreads([&device, &host_data, &shape, &dtype, &strides, &offset, &expected_data]() {
        Array x;
        void* data_ptr{};
        {
            // test potential freed memory
            std::shared_ptr<void> data = device.FromHostMemory(host_data, sizeof(raw_data));
            data_ptr = data.get();
            x = FromData(shape, dtype, data, strides, offset);
        }

        CheckFromData<T>(x, shape, dtype, strides, offset, expected_data, data_ptr);
    });
}

TEST_P(CreationTest, FromData_Contiguous) {
    using T = int32_t;
    Dtype dtype = TypeToDtype<T>;
    Device& device = GetDefaultDevice();

    T raw_data[] = {0, 1, 2, 3, 4, 5};
    std::shared_ptr<void> host_data{raw_data, [](const T*) {}};

    // contiguous array like a[1,:]
    T* expected_data = raw_data + 3;
    Shape shape{3};
    Strides strides{sizeof(T)};
    int64_t offset = sizeof(T) * 3;

    testing::RunTestWithThreads([&device, &host_data, &shape, &dtype, &strides, &offset, &expected_data]() {
        Array x;
        void* data_ptr{};
        {
            // test potential freed memory
            std::shared_ptr<void> data = device.FromHostMemory(host_data, sizeof(raw_data));
            data_ptr = data.get();
            // nullopt strides creates an array from a contiguous data
            x = FromData(shape, dtype, data, nonstd::nullopt, offset);
        }

        CheckFromData<T>(x, shape, dtype, strides, offset, expected_data, data_ptr);
    });
}

// TODO(sonots): Checking `MakeDataFromForeignPointer` called is enough as a unit-test here. Use mock library if it becomes available.
#ifdef CHAINERX_ENABLE_CUDA
TEST(CreationTest, FromData_FromAnotherDevice) {
    Context ctx;
    Device& cuda_device = ctx.GetDevice({"cuda", 0});
    Device& native_device = ctx.GetDevice({"native", 0});

    using T = int32_t;
    Dtype dtype = TypeToDtype<T>;
    Shape shape{3};
    Strides strides{shape, dtype};
    int64_t offset = 0;
    std::shared_ptr<void> data = native_device.Allocate(3 * sizeof(T));

    EXPECT_THROW(FromData(shape, dtype, data, strides, offset, cuda_device), ChainerxError);
}
#endif  // CHAINERX_ENABLE_CUDA

TEST_P(CreationTest, FromHostData) {
    using T = int32_t;
    Dtype dtype = TypeToDtype<T>;
    Device& device = GetDefaultDevice();

    // non-contiguous array like a[:,1]
    Shape shape{2};
    Strides strides{sizeof(T) * 3};
    int64_t offset = sizeof(T);

    T raw_data[] = {0, 1, 2, 3, 4, 5};
    std::shared_ptr<void> host_data{raw_data, [](const T*) {}};

    testing::RunTestWithThreads([&shape, &dtype, &host_data, &strides, &offset, &device]() {
        Array x = internal::FromHostData(shape, dtype, host_data, strides, offset, device);

        EXPECT_EQ(shape, x.shape());
        EXPECT_EQ(dtype, x.dtype());
        EXPECT_EQ(strides, x.strides());
        EXPECT_EQ(offset, x.offset());
        EXPECT_EQ(&device, &x.device());
        // std::array<T> is used instead of T[] to avoid a clang-tidy warning
        std::array<T, 2> expected_data = {1, 4};
        testing::ExpectDataEqual<T>(expected_data, x);
    });
}

TEST_P(CreationTest, Empty) {
    CheckEmpty<bool>();
    CheckEmpty<int8_t>();
    CheckEmpty<int16_t>();
    CheckEmpty<int32_t>();
    CheckEmpty<int64_t>();
    CheckEmpty<uint8_t>();
    CheckEmpty<float>();
    CheckEmpty<double>();
}

TEST_P(CreationTest, EmptyWithVariousShapes) {
    testing::RunTestWithThreads([]() {
        {
            Array x = Empty(Shape{}, Dtype::kFloat32);
            EXPECT_EQ(0, x.ndim());
            EXPECT_EQ(1, x.GetTotalSize());
            EXPECT_EQ(int64_t{sizeof(float)}, x.GetNBytes());
            EXPECT_TRUE(x.IsContiguous());
        }
        {
            Array x = Empty(Shape{0}, Dtype::kFloat32);
            EXPECT_EQ(1, x.ndim());
            EXPECT_EQ(0, x.GetTotalSize());
            EXPECT_EQ(0, x.GetNBytes());
            EXPECT_TRUE(x.IsContiguous());
        }
        {
            Array x = Empty(Shape{1}, Dtype::kFloat32);
            EXPECT_EQ(1, x.ndim());
            EXPECT_EQ(1, x.GetTotalSize());
            EXPECT_EQ(int64_t{sizeof(float)}, x.GetNBytes());
            EXPECT_TRUE(x.IsContiguous());
        }
        {
            Array x = Empty(Shape{2, 3}, Dtype::kFloat32);
            EXPECT_EQ(2, x.ndim());
            EXPECT_EQ(6, x.GetTotalSize());
            EXPECT_EQ(6 * int64_t{sizeof(float)}, x.GetNBytes());
            EXPECT_TRUE(x.IsContiguous());
        }
        {
            Array x = Empty(Shape{1, 1, 1}, Dtype::kFloat32);
            EXPECT_EQ(3, x.ndim());
            EXPECT_EQ(1, x.GetTotalSize());
            EXPECT_EQ(int64_t{sizeof(float)}, x.GetNBytes());
            EXPECT_TRUE(x.IsContiguous());
        }
        {
            Array x = Empty(Shape{2, 0, 3}, Dtype::kFloat32);
            EXPECT_EQ(3, x.ndim());
            EXPECT_EQ(0, x.GetTotalSize());
            EXPECT_EQ(0, x.GetNBytes());
            EXPECT_TRUE(x.IsContiguous());
        }
    });
}

TEST_P(CreationTest, EmptyLike) {
    CheckEmptyLike<bool>();
    CheckEmptyLike<int8_t>();
    CheckEmptyLike<int16_t>();
    CheckEmptyLike<int32_t>();
    CheckEmptyLike<int64_t>();
    CheckEmptyLike<uint8_t>();
    CheckEmptyLike<float>();
    CheckEmptyLike<double>();
}

TEST_P(CreationTest, FullWithGivenDtype) {
    CheckFullWithGivenDtype(true);
    CheckFullWithGivenDtype(int8_t{2});
    CheckFullWithGivenDtype(int16_t{2});
    CheckFullWithGivenDtype(int32_t{2});
    CheckFullWithGivenDtype(int64_t{2});
    CheckFullWithGivenDtype(uint8_t{2});
    CheckFullWithGivenDtype(float{2.0f});
    CheckFullWithGivenDtype(double{2.0});

    CheckFullWithGivenDtype(true, Scalar(int32_t{1}));
    CheckFullWithGivenDtype(true, Scalar(int32_t{2}));
    CheckFullWithGivenDtype(true, Scalar(int32_t{-1}));
    CheckFullWithGivenDtype(false, Scalar(int32_t{0}));
}

TEST_P(CreationTest, FullWithScalarDtype) {
    CheckFullWithScalarDtype(true);
    CheckFullWithScalarDtype(int8_t{2});
    CheckFullWithScalarDtype(int16_t{2});
    CheckFullWithScalarDtype(int32_t{2});
    CheckFullWithScalarDtype(int64_t{2});
    CheckFullWithScalarDtype(uint8_t{2});
    CheckFullWithScalarDtype(float{2.0f});
    CheckFullWithScalarDtype(double{2.0});
}

TEST_P(CreationTest, FullLike) {
    CheckFullLike(true);
    CheckFullLike(int8_t{2});
    CheckFullLike(int16_t{2});
    CheckFullLike(int32_t{2});
    CheckFullLike(int64_t{2});
    CheckFullLike(uint8_t{2});
    CheckFullLike(float{2.0f});
    CheckFullLike(double{2.0});

    CheckFullLike(true, Scalar(int32_t{1}));
    CheckFullLike(true, Scalar(int32_t{2}));
    CheckFullLike(true, Scalar(int32_t{-1}));
    CheckFullLike(false, Scalar(int32_t{0}));
}

TEST_P(CreationTest, Zeros) {
    CheckZeros<bool>();
    CheckZeros<int8_t>();
    CheckZeros<int16_t>();
    CheckZeros<int32_t>();
    CheckZeros<int64_t>();
    CheckZeros<uint8_t>();
    CheckZeros<float>();
    CheckZeros<double>();
}

TEST_P(CreationTest, ZerosLike) {
    CheckZerosLike<bool>();
    CheckZerosLike<int8_t>();
    CheckZerosLike<int16_t>();
    CheckZerosLike<int32_t>();
    CheckZerosLike<int64_t>();
    CheckZerosLike<uint8_t>();
    CheckZerosLike<float>();
    CheckZerosLike<double>();
}

TEST_P(CreationTest, Ones) {
    CheckOnes<bool>();
    CheckOnes<int8_t>();
    CheckOnes<int16_t>();
    CheckOnes<int32_t>();
    CheckOnes<int64_t>();
    CheckOnes<uint8_t>();
    CheckOnes<float>();
    CheckOnes<double>();
}

TEST_P(CreationTest, OnesLike) {
    CheckOnesLike<bool>();
    CheckOnesLike<int8_t>();
    CheckOnesLike<int16_t>();
    CheckOnesLike<int32_t>();
    CheckOnesLike<int64_t>();
    CheckOnesLike<uint8_t>();
    CheckOnesLike<float>();
    CheckOnesLike<double>();
}

TEST_P(CreationTest, Arange) {
    testing::RunTestWithThreads([]() {
        Array a = Arange(0, 3, 1);
        Array e = testing::BuildArray({3}).WithData<int32_t>({0, 1, 2});
        EXPECT_ARRAY_EQ(e, a);
    });
}

TEST_P(CreationTest, ArangeStopDtype) {
    testing::RunTestWithThreads([]() {
        Array a = Arange(3, Dtype::kInt32);
        Array e = testing::BuildArray({3}).WithData<int32_t>({0, 1, 2});
        EXPECT_ARRAY_EQ(e, a);
    });
}

TEST_P(CreationTest, ArangeStopDevice) {
    testing::RunTestWithThreads([]() {
        Array a = Arange(Scalar{3}, Dtype::kInt32, GetDefaultDevice());
        Array e = testing::BuildArray({3}).WithData<int32_t>({0, 1, 2});
        EXPECT_ARRAY_EQ(e, a);
    });
}

TEST_P(CreationTest, ArangeStopDtypeDevice) {
    testing::RunTestWithThreads([]() {
        Array a = Arange(3, Dtype::kInt32, GetDefaultDevice());
        Array e = testing::BuildArray({3}).WithData<int32_t>({0, 1, 2});
        EXPECT_ARRAY_EQ(e, a);
    });
}

TEST_P(CreationTest, ArangeStartStopDtype) {
    testing::RunTestWithThreads([]() {
        Array a = Arange(1, 3, Dtype::kInt32);
        Array e = testing::BuildArray({2}).WithData<int32_t>({1, 2});
        EXPECT_ARRAY_EQ(e, a);
    });
}

TEST_P(CreationTest, ArangeStartStopDevice) {
    testing::RunTestWithThreads([]() {
        Array a = Arange(1, 3, GetDefaultDevice());
        Array e = testing::BuildArray({2}).WithData<int32_t>({1, 2});
        EXPECT_ARRAY_EQ(e, a);
    });
}

TEST_P(CreationTest, ArangeStartStopDtypeDevice) {
    testing::RunTestWithThreads([]() {
        Array a = Arange(1, 3, Dtype::kInt32, GetDefaultDevice());
        Array e = testing::BuildArray({2}).WithData<int32_t>({1, 2});
        EXPECT_ARRAY_EQ(e, a);
    });
}

TEST_P(CreationTest, ArangeStartStopStepDtype) {
    testing::RunTestWithThreads([]() {
        Array a = Arange(1, 7, 2, Dtype::kInt32);
        Array e = testing::BuildArray({3}).WithData<int32_t>({1, 3, 5});
        EXPECT_ARRAY_EQ(e, a);
    });
}

TEST_P(CreationTest, ArangeStartStopStepDevice) {
    testing::RunTestWithThreads([]() {
        Array a = Arange(1, 7, 2, GetDefaultDevice());
        Array e = testing::BuildArray({3}).WithData<int32_t>({1, 3, 5});
        EXPECT_ARRAY_EQ(e, a);
    });
}

TEST_P(CreationTest, ArangeStartStopStepDtypeDevice) {
    testing::RunTestWithThreads([]() {
        Array a = Arange(1, 7, 2, Dtype::kInt32, GetDefaultDevice());
        Array e = testing::BuildArray({3}).WithData<int32_t>({1, 3, 5});
        EXPECT_ARRAY_EQ(e, a);
    });
}

TEST_P(CreationTest, ArangeNegativeStep) {
    testing::RunTestWithThreads([]() {
        Array a = Arange(4.f, 0.f, -1.5f, Dtype::kFloat32);
        Array e = testing::BuildArray({3}).WithData<float>({4.f, 2.5f, 1.f});
        EXPECT_ARRAY_EQ(e, a);
    });
}

TEST_P(CreationTest, ArangeLargeStep) {
    testing::RunTestWithThreads([]() {
        Array a = Arange(2, 3, 5, Dtype::kInt32);
        Array e = testing::BuildArray({1}).WithData<int32_t>({2});
        EXPECT_ARRAY_EQ(e, a);
    });
}

TEST_P(CreationTest, ArangeEmpty) {
    testing::RunTestWithThreads([]() {
        Array a = Arange(2, 1, 1, Dtype::kInt32);
        Array e = testing::BuildArray({0}).WithData<int32_t>({});
        EXPECT_ARRAY_EQ(e, a);
    });
}

TEST_P(CreationTest, ArangeScalar) {
    testing::RunTestWithThreads([]() {
        Array a = Arange(Scalar{1}, Scalar{4}, Scalar{1});
        Array e = testing::BuildArray({3}).WithData<int32_t>({1, 2, 3});
        EXPECT_ARRAY_EQ(e, a);
    });
}

TEST_P(CreationTest, InvalidTooLongBooleanArange) { EXPECT_THROW(Arange(0, 3, 1, Dtype::kBool), DtypeError); }

TEST_P(CreationTest, Copy) {
    testing::RunTestWithThreads([]() {
        {
            Array a = testing::BuildArray({4, 1}).WithData<bool>({true, true, false, false});
            Array o = Copy(a);
            EXPECT_ARRAYS_ARE_EQUAL_COPY(a, o);
        }
        {
            Array a = testing::BuildArray({3, 1}).WithData<int8_t>({1, 2, 3});
            Array o = Copy(a);
            EXPECT_ARRAYS_ARE_EQUAL_COPY(a, o);
        }
        {
            Array a = testing::BuildArray({3, 1}).WithData<float>({1.0f, 2.0f, 3.0f});
            Array o = Copy(a);
            EXPECT_ARRAYS_ARE_EQUAL_COPY(a, o);
        }

        // with padding
        {
            Array a = testing::BuildArray({3, 1}).WithData<float>({1.0f, 2.0f, 3.0f}).WithPadding(1);
            Array o = Copy(a);
            EXPECT_ARRAYS_ARE_EQUAL_COPY(a, o);
        }
    });
}

TEST_P(CreationTest, Identity) {
    testing::RunTestWithThreads([]() {
        Array o = Identity(3, Dtype::kFloat32);
        Array e = testing::BuildArray({3, 3}).WithData<float>({1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f});
        EXPECT_ARRAY_EQ(e, o);
    });
}

TEST_P(CreationTest, IdentityInvalidN) { EXPECT_THROW(Identity(-1, Dtype::kFloat32), DimensionError); }

TEST_P(CreationTest, Eye) {
    testing::RunTestWithThreads([]() {
        {
            Array o = Eye(2, 3, 1, Dtype::kFloat32);
            Array e = testing::BuildArray({2, 3}).WithData<float>({0.f, 1.f, 0.f, 0.f, 0.f, 1.f});
            EXPECT_ARRAY_EQ(e, o);
        }
        {
            Array o = Eye(3, 2, -2, Dtype::kFloat32);
            Array e = testing::BuildArray({3, 2}).WithData<float>({0.f, 0.f, 0.f, 0.f, 1.f, 0.f});
            EXPECT_ARRAY_EQ(e, o);
        }
    });
}

TEST_P(CreationTest, EyeInvalidNM) {
    EXPECT_THROW(Eye(-1, 2, 1, Dtype::kFloat32), DimensionError);
    EXPECT_THROW(Eye(1, -2, 1, Dtype::kFloat32), DimensionError);
    EXPECT_THROW(Eye(-1, -2, 1, Dtype::kFloat32), DimensionError);
}

TEST_THREAD_SAFE_P(CreationTest, AsContiguousArray) {
    Array a = testing::BuildArray({2, 3}).WithLinearData<int32_t>().WithPadding(1);
    ASSERT_FALSE(a.IsContiguous());  // test precondition

    Run([&]() {
        testing::CheckForward(
                [](const std::vector<Array>& xs) {
                    Array y = AsContiguousArray(xs[0]);
                    EXPECT_TRUE(y.IsContiguous());
                    return std::vector<Array>{y};
                },
                {a},
                {a});
    });
}

TEST_THREAD_SAFE_P(CreationTest, AsContiguousArrayNoCopy) {
    Array a = testing::BuildArray({2, 3}).WithLinearData<int32_t>();
    ASSERT_TRUE(a.IsContiguous());  // test precondition

    Run([&]() {
        testing::CheckForward(
                [](const std::vector<Array>& xs) {
                    Array y = AsContiguousArray(xs[0]);
                    EXPECT_EQ(internal::GetArrayBody(y), internal::GetArrayBody(xs[0]));
                    return std::vector<Array>{y};
                },
                {a},
                {a});
    });
}

TEST_THREAD_SAFE_P(CreationTest, AsContiguousArrayDtypeMismatch) {
    Array a = testing::BuildArray({2, 3}).WithLinearData<int32_t>();
    ASSERT_TRUE(a.IsContiguous());  // test precondition

    Run([&]() {
        testing::CheckForward(
                [](const std::vector<Array>& xs) {
                    Array y = AsContiguousArray(xs[0], Dtype::kInt64);
                    EXPECT_NE(internal::GetArrayBody(y), internal::GetArrayBody(xs[0]));
                    EXPECT_TRUE(y.IsContiguous());
                    EXPECT_EQ(Dtype::kInt64, y.dtype());
                    EXPECT_ARRAY_EQ(y, xs[0].AsType(Dtype::kInt64));
                    return std::vector<Array>{};
                },
                {a},
                {});
    });
}

TEST_P(CreationTest, AsContiguousArrayBackward) {
    CheckBackward(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                return {AsContiguousArray(xs[0]).MakeView()};  // Make a view to avoid identical output
            },
            {(*testing::BuildArray({2, 3}).WithLinearData<float>().WithPadding(1)).RequireGrad()},
            {testing::BuildArray({2, 3}).WithLinearData<float>(-2.4f, 0.8f)},
            {Full({2, 3}, 1e-1f)});
}

TEST_P(CreationTest, AsContiguousArrayDoubleBackward) {
    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                auto y = AsContiguousArray(xs[0]);
                return {y * y};  // to make it nonlinear
            },
            {(*testing::BuildArray({2, 3}).WithLinearData<float>().WithPadding(1)).RequireGrad()},
            {(*testing::BuildArray({2, 3}).WithLinearData<float>(-2.4f, 0.8f)).RequireGrad()},
            {testing::BuildArray({2, 3}).WithLinearData<float>(5.2f, -0.5f)},
            {Full({2, 3}, 1e-1f), Full({2, 3}, 1e-1f)});
}

TEST_THREAD_SAFE_P(CreationTest, DiagVecToMatDefaultK) {
    Array v = Arange(1, 3, Dtype::kFloat32);
    Array e = testing::BuildArray({2, 2}).WithData<float>({1.f, 0.f, 0.f, 2.f});

    Run([&]() {
        testing::CheckForward(
                [](const std::vector<Array>& xs) {
                    DeviceScope scope{xs[0].device()};
                    return std::vector<Array>{Diag(xs[0])};
                },
                {v},
                {e});
    });
}

TEST_THREAD_SAFE_P(CreationTest, DiagVecToMat) {
    Array v = Arange(1, 4, Dtype::kFloat32);
    Array e = testing::BuildArray({4, 4}).WithData<float>({0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 2.f, 0.f, 0.f, 0.f, 0.f, 3.f, 0.f, 0.f, 0.f, 0.f});

    Run([&]() {
        testing::CheckForward(
                [](const std::vector<Array>& xs) {
                    DeviceScope scope{xs[0].device()};
                    return std::vector<Array>{Diag(xs[0], 1)};
                },
                {v},
                {e});
    });
}

TEST_THREAD_SAFE_P(CreationTest, DiagVecToMatNegativeK) {
    Array v = Arange(1, 3, Dtype::kFloat32);
    Array e = testing::BuildArray({4, 4}).WithData<float>({0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 2.f, 0.f, 0.f});

    Run([&]() {
        testing::CheckForward(
                [](const std::vector<Array>& xs) {
                    DeviceScope scope{xs[0].device()};
                    return std::vector<Array>{Diag(xs[0], -2)};
                },
                {v},
                {e});
    });
}

TEST_THREAD_SAFE_P(CreationTest, DiagMatToVecDefaultK) {
    Array v = Arange(6, Dtype::kFloat32).Reshape({2, 3});
    Array e = testing::BuildArray({2}).WithData<float>({0.f, 4.f});

    Run([&]() {
        testing::CheckForward(
                [](const std::vector<Array>& xs) {
                    DeviceScope scope{xs[0].device()};
                    Array y = Diag(xs[0]);
                    EXPECT_EQ(xs[0].data().get(), y.data().get());
                    return std::vector<Array>{y};
                },
                {v},
                {e});
    });
}

TEST_THREAD_SAFE_P(CreationTest, DiagMatToVec) {
    Array v = Arange(6, Dtype::kFloat32).Reshape({2, 3});
    Array e = testing::BuildArray({2}).WithData<float>({1.f, 5.f});

    Run([&]() {
        testing::CheckForward(
                [](const std::vector<Array>& xs) {
                    DeviceScope scope{xs[0].device()};
                    Array y = Diag(xs[0], 1);
                    EXPECT_EQ(xs[0].data().get(), y.data().get());
                    return std::vector<Array>{y};
                },
                {v},
                {e});
    });
}

TEST_THREAD_SAFE_P(CreationTest, DiagMatToVecNegativeK) {
    Array v = Arange(6, Dtype::kFloat32).Reshape({2, 3});
    Array e = testing::BuildArray({1}).WithData<float>({3.f});

    Run([&]() {
        testing::CheckForward(
                [](const std::vector<Array>& xs) {
                    DeviceScope scope{xs[0].device()};
                    Array y = Diag(xs[0], -1);
                    EXPECT_EQ(xs[0].data().get(), y.data().get());
                    return std::vector<Array>{y};
                },
                {v},
                {e});
    });
}

TEST_P(CreationTest, DiagVecToMatBackward) {
    using T = double;
    Array v = (*testing::BuildArray({3}).WithLinearData<T>(-3).WithPadding(1)).RequireGrad();
    Array go = testing::BuildArray({4, 4}).WithLinearData<T>(-0.1, 0.1).WithPadding(1);
    Array eps = Full({3}, 1e-3, Dtype::kFloat64);

    CheckBackward([](const std::vector<Array>& xs) -> std::vector<Array> { return {Diag(xs[0], -1)}; }, {v}, {go}, {eps});
}

TEST_P(CreationTest, DiagMatToVecBackward) {
    using T = double;
    Array v = (*testing::BuildArray({4, 4}).WithLinearData<T>(-3).WithPadding(1)).RequireGrad();
    Array go = testing::BuildArray({3}).WithLinearData<T>(-0.1, 0.1).WithPadding(1);
    Array eps = Full({4, 4}, 1e-3, Dtype::kFloat64);

    CheckBackward([](const std::vector<Array>& xs) -> std::vector<Array> { return {Diag(xs[0], 1)}; }, {v}, {go}, {eps});
}

TEST_P(CreationTest, DiagVecToMatDoubleBackward) {
    using T = double;
    Array v = (*testing::BuildArray({3}).WithLinearData<T>(-3).WithPadding(1)).RequireGrad();
    Array go = (*testing::BuildArray({4, 4}).WithLinearData<T>(-0.1, 0.1).WithPadding(1)).RequireGrad();
    Array ggv = testing::BuildArray({3}).WithLinearData<T>(-0.1, 0.1).WithPadding(1);
    Array eps_v = Full(Shape{3}, 1e-3, Dtype::kFloat64);
    Array eps_go = Full(Shape{4, 4}, 1e-3, Dtype::kFloat64);

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                auto y = Diag(xs[0], -1);
                return {y * y};  // to make it nonlinear
            },
            {v},
            {go},
            {ggv},
            {eps_v, eps_go});
}

TEST_P(CreationTest, DiagMatToVecDoubleBackward) {
    using T = double;
    Array v = (*testing::BuildArray({4, 4}).WithLinearData<T>(-3).WithPadding(1)).RequireGrad();
    Array go = (*testing::BuildArray({3}).WithLinearData<T>(-0.1, 0.1).WithPadding(1)).RequireGrad();
    Array ggv = testing::BuildArray({4, 4}).WithLinearData<T>(-0.1, 0.1).WithPadding(1);
    Array eps_v = Full(Shape{4, 4}, 1e-3, Dtype::kFloat64);
    Array eps_go = Full(Shape{3}, 1e-3, Dtype::kFloat64);

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                auto y = Diag(xs[0], -1);
                return {y * y};  // to make it nonlinear
            },
            {v},
            {go},
            {ggv},
            {eps_v, eps_go});
}

TEST_THREAD_SAFE_P(CreationTest, Diagflat1) {
    Array v = Arange(1, 3, Dtype::kFloat32);
    Array e = testing::BuildArray({2, 2}).WithData<float>({1.f, 0.f, 0.f, 2.f});

    Run([&]() {
        testing::CheckForward(
                [](const std::vector<Array>& xs) {
                    DeviceScope scope{xs[0].device()};
                    return std::vector<Array>{Diagflat(xs[0])};
                },
                {v},
                {e});
    });
}

TEST_THREAD_SAFE_P(CreationTest, Diagflat2) {
    Array v = Arange(1, 5, Dtype::kFloat32).Reshape({2, 2});
    Array e = testing::BuildArray({5, 5}).WithData<float>(
            {0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 2.f, 0.f, 0.f, 0.f, 0.f, 0.f, 3.f, 0.f, 0.f, 0.f, 0.f, 0.f, 4.f, 0.f, 0.f, 0.f, 0.f, 0.f});

    Run([&]() {
        testing::CheckForward(
                [](const std::vector<Array>& xs) {
                    DeviceScope scope{xs[0].device()};
                    return std::vector<Array>{Diagflat(xs[0], 1)};
                },
                {v},
                {e});
    });
}

TEST_THREAD_SAFE_P(CreationTest, Diagflat3) {
    Array v = Arange(1, 3, Dtype::kFloat32).Reshape({1, 2});
    Array e = testing::BuildArray({3, 3}).WithData<float>({0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 2.f, 0.f});

    Run([&]() {
        testing::CheckForward(
                [](const std::vector<Array>& xs) {
                    DeviceScope scope{xs[0].device()};
                    return std::vector<Array>{Diagflat(xs[0], -1)};
                },
                {v},
                {e});
    });
}

TEST_P(CreationTest, DiagflatBackward) {
    using T = double;
    Array v = (*testing::BuildArray({3}).WithLinearData<T>(-3).WithPadding(1)).RequireGrad();
    Array go = testing::BuildArray({4, 4}).WithLinearData<T>(-0.1, 0.1).WithPadding(1);
    Array eps = Full({3}, 1e-3, Dtype::kFloat64);

    CheckBackward([](const std::vector<Array>& xs) -> std::vector<Array> { return {Diagflat(xs[0], 1)}; }, {v}, {go}, {eps});
}

TEST_P(CreationTest, DiagflatDoubleBackward) {
    using T = double;
    Array v = (*testing::BuildArray({3}).WithLinearData<T>(-3).WithPadding(1)).RequireGrad();
    Array go = (*testing::BuildArray({4, 4}).WithLinearData<T>(-0.1, 0.1).WithPadding(1)).RequireGrad();
    Array ggv = testing::BuildArray({3}).WithLinearData<T>(-0.1, 0.1).WithPadding(1);
    Array eps_v = Full(Shape{3}, 1e-3, Dtype::kFloat64);
    Array eps_go = Full(Shape{4, 4}, 1e-3, Dtype::kFloat64);

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                auto y = Diagflat(xs[0], -1);
                return {y * y};  // to make it nonlinear
            },
            {v},
            {go},
            {ggv},
            {eps_v, eps_go});
}

TEST_P(CreationTest, Linspace) {
    testing::RunTestWithThreads([]() {
        Array o = Linspace(3.0, 10.0, 4, true, Dtype::kInt32);
        Array e = testing::BuildArray({4}).WithData<int32_t>({3, 5, 7, 10});
        EXPECT_ARRAY_EQ(e, o);
    });
}

TEST_P(CreationTest, LinspaceEndPointFalse) {
    testing::RunTestWithThreads([]() {
        Array o = Linspace(3.0, 10.0, 4, false, Dtype::kInt32);
        Array e = testing::BuildArray({4}).WithData<int32_t>({3, 4, 6, 8});
        EXPECT_ARRAY_EQ(e, o);
    });
}

INSTANTIATE_TEST_CASE_P(
        ForEachBackend,
        CreationTest,
        ::testing::Values(
#ifdef CHAINERX_ENABLE_CUDA
                std::string{"cuda"},
#endif  // CHAINERX_ENABLE_CUDA
                std::string{"native"}));

}  // namespace
}  // namespace chainerx
