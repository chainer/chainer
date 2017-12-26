#include "xchainer/array.h"

#include <array>
#include <cstddef>
#include <initializer_list>
#include <type_traits>

#ifdef XCHAINER_ENABLE_CUDA
#include <cuda_runtime.h>
#endif  // XCHAINER_ENABLE_CUDA
#include <gtest/gtest.h>

#ifdef XCHAINER_ENABLE_CUDA
#include "xchainer/cuda/cuda_runtime.h"
#endif  // XCHAINER_ENABLE_CUDA
#include "xchainer/device.h"

namespace xchainer {
namespace {

class ArrayTest : public ::testing::TestWithParam<::testing::tuple<std::string>> {
protected:
    virtual void SetUp() {
        std::string device_name = ::testing::get<0>(GetParam());
        device_scope_ = std::make_unique<DeviceScope>(device_name);
    }

    virtual void TearDown() { device_scope_.reset(); }

public:
    template <typename T>
    Array MakeArray(std::initializer_list<int64_t> shape, std::shared_ptr<void> data) {
        return {shape, TypeToDtype<T>, data};
    }

    template <typename T>
    Array MakeArray(std::initializer_list<int64_t> shape, std::initializer_list<T> data) {
        auto a = std::make_unique<T[]>(data.size());
        std::copy(data.begin(), data.end(), a.get());
        return {shape, TypeToDtype<T>, std::move(a)};
    }

    template <typename T>
    void AssertEqual(const Array& lhs, const Array& rhs) {
#ifdef XCHAINER_ENABLE_CUDA
        std::string device_name = ::testing::get<0>(GetParam());
        if (device_name == "cuda") {
            cuda::CheckError(cudaDeviceSynchronize());
        }
#endif  // XCHAINER_ENABLE_CUDA
        ASSERT_NO_THROW(CheckEqual(lhs.dtype(), rhs.dtype()));
        ASSERT_NO_THROW(CheckEqual(lhs.shape(), rhs.shape()));
        auto total_size = lhs.shape().total_size();
        const T* ldata = static_cast<const T*>(lhs.data().get());
        const T* rdata = static_cast<const T*>(rhs.data().get());
        for (decltype(total_size) i = 0; i < total_size; i++) {
            ASSERT_EQ(ldata[i], rdata[i]);
        }
    }

    bool IsPointerCudaManaged(const void* ptr) {
#ifdef XCHAINER_ENABLE_CUDA
        cudaPointerAttributes attr = {};
        cuda::CheckError(cudaPointerGetAttributes(&attr, ptr));
        return attr.isManaged != 0;
#else
        (void)ptr;
        return false;
#endif  // XCHAINER_ENABLE_CUDA
    }

    template <bool is_const>
    void CheckArray() {
        using TargetArray = std::conditional_t<is_const, const Array, Array>;

        std::shared_ptr<void> data = std::make_unique<float[]>(2 * 3 * 4);
        TargetArray x = MakeArray<float>({2, 3, 4}, data);

        // Basic attributes
        ASSERT_EQ(TypeToDtype<float>, x.dtype());
        ASSERT_EQ(3, x.ndim());
        ASSERT_EQ(2 * 3 * 4, x.total_size());
        ASSERT_EQ(4, x.element_bytes());
        ASSERT_EQ(2 * 3 * 4 * 4, x.total_bytes());
        ASSERT_EQ(0, x.offset());
        ASSERT_TRUE(x.is_contiguous());

        // Array::data
        std::shared_ptr<const void> x_data = x.data();
        if (GetCurrentDevice() == MakeDevice("cpu")) {
            ASSERT_EQ(data, x_data);
        } else if (GetCurrentDevice() == MakeDevice("cuda")) {
            ASSERT_NE(data, x_data);
            ASSERT_TRUE(IsPointerCudaManaged(x_data.get()));
        } else {
            FAIL() << "invalid device";
        }
    }

private:
    std::unique_ptr<DeviceScope> device_scope_;
};

TEST_P(ArrayTest, ArrayCtor) { CheckArray<false>(); }

TEST_P(ArrayTest, ConstArrayCtor) { CheckArray<true>(); }

TEST_P(ArrayTest, IAdd) {
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

TEST_P(ArrayTest, IMul) {
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

TEST_P(ArrayTest, Add) {
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

TEST_P(ArrayTest, Mul) {
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

TEST_P(ArrayTest, ChainedMath) {
    {
        Array a = MakeArray<bool>({4, 1}, {true, true, false, false});
        Array b = MakeArray<bool>({4, 1}, {true, false, true, false});
        Array e = MakeArray<bool>({4, 1}, {true, true, false, false});
        Array c = a.Mul(b);
        Array o = a.Add(c);
        AssertEqual<bool>(e, o);
    }
    {
        Array a = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array b = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array e = MakeArray<int8_t>({3, 1}, {2, 6, 12});
        Array c = a.Mul(b);
        Array o = a.Add(c);
        AssertEqual<int8_t>(e, o);
    }
    {
        Array a = MakeArray<float>({3, 1}, {1, 2, 3});
        Array b = MakeArray<float>({3, 1}, {1, 2, 3});
        Array e = MakeArray<float>({3, 1}, {2, 6, 12});
        Array c = a.Mul(b);
        Array o = a.Add(c);
        AssertEqual<float>(e, o);
    }
}

TEST_P(ArrayTest, ChainedInplaceMath) {
    {
        Array a = MakeArray<bool>({4, 1}, {true, true, false, false});
        Array b = MakeArray<bool>({4, 1}, {true, false, true, false});
        Array e = MakeArray<bool>({4, 1}, {true, true, false, false});
        b.IMul(a);
        a.IAdd(b);
        AssertEqual<bool>(e, a);
    }
    {
        Array a = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array b = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array e = MakeArray<int8_t>({3, 1}, {2, 6, 12});
        b.IMul(a);
        a.IAdd(b);
        AssertEqual<int8_t>(e, a);
    }
    {
        Array a = MakeArray<float>({3, 1}, {1, 2, 3});
        Array b = MakeArray<float>({3, 1}, {1, 2, 3});
        Array e = MakeArray<float>({3, 1}, {2, 6, 12});
        b.IMul(a);
        a.IAdd(b);
        AssertEqual<float>(e, a);
    }
}

INSTANTIATE_TEST_CASE_P(ForEachDevice, ArrayTest, ::testing::Values(
#ifdef XCHAINER_ENABLE_CUDA
                                                      std::string{"cuda"},
#endif  // XCHAINER_ENABLE_CUDA
                                                      std::string{"cpu"}));

}  // namespace
}  // namespace xchainer
