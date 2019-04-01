#include "chainerx/routines/creation.h"

#include <arpa/inet.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
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
#include "chainerx/testing/util.h"

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

    static constexpr char kTestDelimiter = ',';

    static constexpr size_t kRandomBoolBinLength = 100;
    static constexpr size_t kRandomInt32BinLength = 400;
    static constexpr size_t kRandomFloat16BinLength = 200;
    static constexpr size_t kRandomFloat32BinLength = 400;
    static constexpr size_t kSpecialFloat32BinLength = 120;

    static constexpr char kRandomBoolText[] =
            "False,True,True,False,False,False,False,True,False,False,False,"
            "True,True,False,True,False,True,False,False,True,False,False,True,"
            "False,False,True,False,True,True,False,False,False,False,False,False,"
            "True,True,False,True,False,True,False,False,True,False,False,True,"
            "True,True,True,True,False,False,True,False,True,False,True,False,"
            "False,True,True,True,False,True,False,False,False,False,False,True,"
            "True,False,True,False,True,True,False,False,False,True,True,False,"
            "False,True,False,False,True,False,True,True,True,False,True,True,"
            "True,True,False,False,True";

    static constexpr char kRandomInt32Text[] =
            "4,-4,-1,4,0,3,2,1,0,-3,4,3,1,-5,4,1,-2,-2,-3,-3,-5,-2,3,4,0,3,-5,"
            "3,0,-4,-4,-3,1,-5,-1,-5,-2,0,4,1,2,-5,4,3,2,4,1,-2,3,-2,1,0,-2,3,"
            "-1,-5,0,-4,1,3,2,0,-4,3,-2,4,2,3,0,1,-5,-5,3,1,1,1,0,0,4,-5,1,2,"
            "-1,3,-2,-5,-5,-5,-4,4,1,-5,-3,-4,4,-2,-4,-3,-3,-2";

    static constexpr char kRandomFloat16Text[] =
            "-489.5,-153.5,67.9,85.25,-330.2,-193.9,263.2,-244.6,59.94,68.9,-167.2,"
            "210.4,326.8,286.8,-318.8,-287.8,-276.0,-41.2,242.8,256.0,-249.6,477.2,"
            "-148.6,-119.56,-9.516,193.5,239.6,223.1,18.62,-70.44,406.5,478.5,-255.9,"
            "-77.3,-221.1,104.5,-272.5,79.5,493.5,224.5,234.8,-32.88,32.9,-107.3,-443.2,"
            "481.8,-192.6,162.5,20.89,-343.2,-33.56,-481.2,373.8,157.9,339.2,-465.0,"
            "-434.8,-68.25,-57.38,104.6,-488.0,-221.5,489.8,-146.2,-287.0,-59.12,144.1,"
            "429.5,-208.6,-256.2,-442.8,433.8,398.2,-163.1,112.3,112.3,270.2,-458.5,"
            "278.5,-112.5,130.0,188.5,-365.5,-371.5,137.1,407.2,369.0,258.0,251.8,"
            "-490.2,30.23,45.38,-346.2,-499.8,-466.0,250.9,480.8,248.5,462.0,-330.8";

    static constexpr char kRandomFloat32Text[] =
            "-489.52817,-153.45093,67.86676,85.27582,-330.17505,-193.81528,"
            "263.23346,-244.68239,59.939575,68.89496,-167.29025,210.34674,"
            "326.85004,286.6725,-318.81125,-287.70892,-276.01904,-41.196655,"
            "242.77338,255.94086,-249.60991,477.24445,-148.66815,-119.55133,"
            "-9.512756,193.52155,239.63647,223.14655,18.628479,-70.459625,"
            "406.3797,478.56787,-255.81458,-77.322754,-221.09692,104.481995,"
            "-272.51776,79.51709,493.5265,224.5122,234.76495,-32.880127,32.899902,"
            "-107.33298,-443.27704,481.87183,-192.63309,162.48883,20.88617,"
            "-343.16333,-33.555756,-481.25684,373.68207,157.91406,339.28485,"
            "-464.95462,-434.7118,-68.27731,-57.375427,104.630066,-487.9083,"
            "-221.55606,489.63654,-146.25946,-287.0644,-59.134033,144.11603,"
            "429.3761,-208.68625,-256.1972,-442.64804,433.63,398.1725,-163.08743,"
            "112.322754,112.289856,270.16742,-458.46924,278.55164,-112.471924,"
            "129.9386,188.46356,-365.4979,-371.4936,137.07263,407.36652,369.00543,"
            "258.0064,251.78638,-490.22543,30.232178,45.373047,-346.2873,"
            "-499.76062,-466.06503,250.8642,480.76123,248.55377,461.9134,-330.63153";

    static constexpr char kSpecialFloat32Text[] =
            "nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,inf,inf,inf,inf,inf,inf,inf,inf,"
            "inf,inf,-inf,-inf,-inf,-inf,-inf,-inf,-inf,-inf,-inf,-inf";

    static const std::vector<bool> kRandomBoolVector;
    static const std::vector<int32_t> kRandomInt32Vector;
    static const std::vector<float> kRandomFloat16Vector;
    static const std::vector<float> kRandomFloat32Vector;
    static const std::vector<float> kSpecialFloat32Vector;

    static constexpr unsigned char kRandomBoolBin[] = {
            0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x01,
            0x00, 0x00, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x01, 0x00,
            0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00,
            0x01, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00,
            0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x00, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01};

    static constexpr unsigned char kRandomInt32LittleEndianBin[] = {
            0x04, 0x00, 0x00, 0x00, 0xfc, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x03, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xfd, 0xff, 0xff, 0xff,
            0x04, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xfb, 0xff, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00,
            0x01, 0x00, 0x00, 0x00, 0xfe, 0xff, 0xff, 0xff, 0xfe, 0xff, 0xff, 0xff, 0xfd, 0xff, 0xff, 0xff, 0xfd, 0xff, 0xff, 0xff,
            0xfb, 0xff, 0xff, 0xff, 0xfe, 0xff, 0xff, 0xff, 0x03, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x03, 0x00, 0x00, 0x00, 0xfb, 0xff, 0xff, 0xff, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xfc, 0xff, 0xff, 0xff,
            0xfc, 0xff, 0xff, 0xff, 0xfd, 0xff, 0xff, 0xff, 0x01, 0x00, 0x00, 0x00, 0xfb, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xfb, 0xff, 0xff, 0xff, 0xfe, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
            0x02, 0x00, 0x00, 0x00, 0xfb, 0xff, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
            0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xfe, 0xff, 0xff, 0xff, 0x03, 0x00, 0x00, 0x00, 0xfe, 0xff, 0xff, 0xff,
            0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xfe, 0xff, 0xff, 0xff, 0x03, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff,
            0xfb, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0xfc, 0xff, 0xff, 0xff, 0x01, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
            0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xfc, 0xff, 0xff, 0xff, 0x03, 0x00, 0x00, 0x00, 0xfe, 0xff, 0xff, 0xff,
            0x04, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
            0xfb, 0xff, 0xff, 0xff, 0xfb, 0xff, 0xff, 0xff, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
            0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0xfb, 0xff, 0xff, 0xff,
            0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0x03, 0x00, 0x00, 0x00, 0xfe, 0xff, 0xff, 0xff,
            0xfb, 0xff, 0xff, 0xff, 0xfb, 0xff, 0xff, 0xff, 0xfb, 0xff, 0xff, 0xff, 0xfc, 0xff, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00,
            0x01, 0x00, 0x00, 0x00, 0xfb, 0xff, 0xff, 0xff, 0xfd, 0xff, 0xff, 0xff, 0xfc, 0xff, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00,
            0xfe, 0xff, 0xff, 0xff, 0xfc, 0xff, 0xff, 0xff, 0xfd, 0xff, 0xff, 0xff, 0xfd, 0xff, 0xff, 0xff, 0xfe, 0xff, 0xff, 0xff};

    static constexpr unsigned char kRandomInt32BigEndianBin[] = {
            0x00, 0x00, 0x00, 0x04, 0xff, 0xff, 0xff, 0xfc, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xfd,
            0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0xff, 0xff, 0xff, 0xfb, 0x00, 0x00, 0x00, 0x04,
            0x00, 0x00, 0x00, 0x01, 0xff, 0xff, 0xff, 0xfe, 0xff, 0xff, 0xff, 0xfe, 0xff, 0xff, 0xff, 0xfd, 0xff, 0xff, 0xff, 0xfd,
            0xff, 0xff, 0xff, 0xfb, 0xff, 0xff, 0xff, 0xfe, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x03, 0xff, 0xff, 0xff, 0xfb, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xfc,
            0xff, 0xff, 0xff, 0xfc, 0xff, 0xff, 0xff, 0xfd, 0x00, 0x00, 0x00, 0x01, 0xff, 0xff, 0xff, 0xfb, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xfb, 0xff, 0xff, 0xff, 0xfe, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x01,
            0x00, 0x00, 0x00, 0x02, 0xff, 0xff, 0xff, 0xfb, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x02,
            0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x01, 0xff, 0xff, 0xff, 0xfe, 0x00, 0x00, 0x00, 0x03, 0xff, 0xff, 0xff, 0xfe,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xfe, 0x00, 0x00, 0x00, 0x03, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xfb, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xfc, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x03,
            0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xfc, 0x00, 0x00, 0x00, 0x03, 0xff, 0xff, 0xff, 0xfe,
            0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
            0xff, 0xff, 0xff, 0xfb, 0xff, 0xff, 0xff, 0xfb, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0xff, 0xff, 0xff, 0xfb,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x03, 0xff, 0xff, 0xff, 0xfe,
            0xff, 0xff, 0xff, 0xfb, 0xff, 0xff, 0xff, 0xfb, 0xff, 0xff, 0xff, 0xfb, 0xff, 0xff, 0xff, 0xfc, 0x00, 0x00, 0x00, 0x04,
            0x00, 0x00, 0x00, 0x01, 0xff, 0xff, 0xff, 0xfb, 0xff, 0xff, 0xff, 0xfd, 0xff, 0xff, 0xff, 0xfc, 0x00, 0x00, 0x00, 0x04,
            0xff, 0xff, 0xff, 0xfe, 0xff, 0xff, 0xff, 0xfc, 0xff, 0xff, 0xff, 0xfd, 0xff, 0xff, 0xff, 0xfd, 0xff, 0xff, 0xff, 0xfe};

    static constexpr unsigned char kRandomFloat16LittleEndianBin[] = {
            0xa6, 0xdf, 0xcc, 0xd8, 0x3e, 0x54, 0x54, 0x55, 0x29, 0xdd, 0x0f, 0xda, 0x1d, 0x5c, 0xa5, 0xdb, 0x7e, 0x53, 0x4e, 0x54,
            0x3a, 0xd9, 0x93, 0x5a, 0x1b, 0x5d, 0x7b, 0x5c, 0xfb, 0xdc, 0x7f, 0xdc, 0x50, 0xdc, 0x26, 0xd1, 0x96, 0x5b, 0x00, 0x5c,
            0xcd, 0xdb, 0x75, 0x5f, 0xa5, 0xd8, 0x79, 0xd7, 0xc2, 0xc8, 0x0c, 0x5a, 0x7d, 0x5b, 0xf9, 0x5a, 0xa8, 0x4c, 0x67, 0xd4,
            0x5a, 0x5e, 0x7a, 0x5f, 0xff, 0xdb, 0xd5, 0xd4, 0xe9, 0xda, 0x88, 0x56, 0x42, 0xdc, 0xf8, 0x54, 0xb6, 0x5f, 0x04, 0x5b,
            0x56, 0x5b, 0x1c, 0xd0, 0x1d, 0x50, 0xb5, 0xd6, 0xed, 0xde, 0x87, 0x5f, 0x05, 0xda, 0x14, 0x59, 0x39, 0x4d, 0x5d, 0xdd,
            0x32, 0xd0, 0x85, 0xdf, 0xd7, 0x5d, 0xef, 0x58, 0x4d, 0x5d, 0x44, 0xdf, 0xcb, 0xde, 0x44, 0xd4, 0x2c, 0xd3, 0x8a, 0x56,
            0xa0, 0xdf, 0xec, 0xda, 0xa7, 0x5f, 0x92, 0xd8, 0x7c, 0xdc, 0x64, 0xd3, 0x81, 0x58, 0xb6, 0x5e, 0x85, 0xda, 0x01, 0xdc,
            0xeb, 0xde, 0xc7, 0x5e, 0x39, 0x5e, 0x19, 0xd9, 0x05, 0x57, 0x05, 0x57, 0x39, 0x5c, 0x2a, 0xdf, 0x5a, 0x5c, 0x08, 0xd7,
            0x10, 0x58, 0xe4, 0x59, 0xb6, 0xdd, 0xce, 0xdd, 0x49, 0x58, 0x5d, 0x5e, 0xc4, 0x5d, 0x08, 0x5c, 0xde, 0x5b, 0xa9, 0xdf,
            0x8f, 0x4f, 0xac, 0x51, 0x69, 0xdd, 0xcf, 0xdf, 0x48, 0xdf, 0xd7, 0x5b, 0x83, 0x5f, 0xc4, 0x5b, 0x38, 0x5f, 0x2b, 0xdd};

    static constexpr unsigned char kRandomFloat16BigEndianBin[] = {
            0xdf, 0xa6, 0xd8, 0xcc, 0x54, 0x3e, 0x55, 0x54, 0xdd, 0x29, 0xda, 0x0f, 0x5c, 0x1d, 0xdb, 0xa5, 0x53, 0x7e, 0x54, 0x4e,
            0xd9, 0x3a, 0x5a, 0x93, 0x5d, 0x1b, 0x5c, 0x7b, 0xdc, 0xfb, 0xdc, 0x7f, 0xdc, 0x50, 0xd1, 0x26, 0x5b, 0x96, 0x5c, 0x00,
            0xdb, 0xcd, 0x5f, 0x75, 0xd8, 0xa5, 0xd7, 0x79, 0xc8, 0xc2, 0x5a, 0x0c, 0x5b, 0x7d, 0x5a, 0xf9, 0x4c, 0xa8, 0xd4, 0x67,
            0x5e, 0x5a, 0x5f, 0x7a, 0xdb, 0xff, 0xd4, 0xd5, 0xda, 0xe9, 0x56, 0x88, 0xdc, 0x42, 0x54, 0xf8, 0x5f, 0xb6, 0x5b, 0x04,
            0x5b, 0x56, 0xd0, 0x1c, 0x50, 0x1d, 0xd6, 0xb5, 0xde, 0xed, 0x5f, 0x87, 0xda, 0x05, 0x59, 0x14, 0x4d, 0x39, 0xdd, 0x5d,
            0xd0, 0x32, 0xdf, 0x85, 0x5d, 0xd7, 0x58, 0xef, 0x5d, 0x4d, 0xdf, 0x44, 0xde, 0xcb, 0xd4, 0x44, 0xd3, 0x2c, 0x56, 0x8a,
            0xdf, 0xa0, 0xda, 0xec, 0x5f, 0xa7, 0xd8, 0x92, 0xdc, 0x7c, 0xd3, 0x64, 0x58, 0x81, 0x5e, 0xb6, 0xda, 0x85, 0xdc, 0x01,
            0xde, 0xeb, 0x5e, 0xc7, 0x5e, 0x39, 0xd9, 0x19, 0x57, 0x05, 0x57, 0x05, 0x5c, 0x39, 0xdf, 0x2a, 0x5c, 0x5a, 0xd7, 0x08,
            0x58, 0x10, 0x59, 0xe4, 0xdd, 0xb6, 0xdd, 0xce, 0x58, 0x49, 0x5e, 0x5d, 0x5d, 0xc4, 0x5c, 0x08, 0x5b, 0xde, 0xdf, 0xa9,
            0x4f, 0x8f, 0x51, 0xac, 0xdd, 0x69, 0xdf, 0xcf, 0xdf, 0x48, 0x5b, 0xd7, 0x5f, 0x83, 0x5b, 0xc4, 0x5f, 0x38, 0xdd, 0x2b};

    static constexpr unsigned char kRandomFloat32BigEndianBin[] = {
            0xc3, 0xf4, 0xc3, 0x9b, 0xc3, 0x19, 0x73, 0x70, 0x42, 0x87, 0xbb, 0xc8, 0x42, 0xaa, 0x8d, 0x38, 0xc3, 0xa5, 0x16, 0x68,
            0xc3, 0x41, 0xd0, 0xb6, 0x43, 0x83, 0x9d, 0xe2, 0xc3, 0x74, 0xae, 0xb1, 0x42, 0x6f, 0xc2, 0x20, 0x42, 0x89, 0xca, 0x38,
            0xc3, 0x27, 0x4a, 0x4e, 0x43, 0x52, 0x58, 0xc4, 0x43, 0xa3, 0x6c, 0xce, 0x43, 0x8f, 0x56, 0x14, 0xc3, 0x9f, 0x67, 0xd7,
            0xc3, 0x8f, 0xda, 0xbe, 0xc3, 0x8a, 0x02, 0x70, 0xc2, 0x24, 0xc9, 0x60, 0x43, 0x72, 0xc5, 0xfc, 0x43, 0x7f, 0xf0, 0xdc,
            0xc3, 0x79, 0x9c, 0x23, 0x43, 0xee, 0x9f, 0x4a, 0xc3, 0x14, 0xab, 0x0c, 0xc2, 0xef, 0x1a, 0x48, 0xc1, 0x18, 0x34, 0x40,
            0x43, 0x41, 0x85, 0x84, 0x43, 0x6f, 0xa2, 0xf0, 0x43, 0x5f, 0x25, 0x84, 0x41, 0x95, 0x07, 0x20, 0xc2, 0x8c, 0xeb, 0x54,
            0x43, 0xcb, 0x30, 0x9a, 0x43, 0xef, 0x48, 0xb0, 0xc3, 0x7f, 0xd0, 0x88, 0xc2, 0x9a, 0xa5, 0x40, 0xc3, 0x5d, 0x18, 0xd0,
            0x42, 0xd0, 0xf6, 0xc8, 0xc3, 0x88, 0x42, 0x46, 0x42, 0x9f, 0x08, 0xc0, 0x43, 0xf6, 0xc3, 0x64, 0x43, 0x60, 0x83, 0x20,
            0x43, 0x6a, 0xc3, 0xd4, 0xc2, 0x03, 0x85, 0x40, 0x42, 0x03, 0x99, 0x80, 0xc2, 0xd6, 0xaa, 0x7c, 0xc3, 0xdd, 0xa3, 0x76,
            0x43, 0xf0, 0xef, 0x98, 0xc3, 0x40, 0xa2, 0x12, 0x43, 0x22, 0x7d, 0x24, 0x41, 0xa7, 0x16, 0xe0, 0xc3, 0xab, 0x94, 0xe8,
            0xc2, 0x06, 0x39, 0x18, 0xc3, 0xf0, 0xa0, 0xe0, 0x43, 0xba, 0xd7, 0x4e, 0x43, 0x1d, 0xea, 0x00, 0x43, 0xa9, 0xa4, 0x76,
            0xc3, 0xe8, 0x7a, 0x31, 0xc3, 0xd9, 0x5b, 0x1c, 0xc2, 0x88, 0x8d, 0xfc, 0xc2, 0x65, 0x80, 0x70, 0x42, 0xd1, 0x42, 0x98,
            0xc3, 0xf3, 0xf4, 0x43, 0xc3, 0x5d, 0x8e, 0x5a, 0x43, 0xf4, 0xd1, 0x7a, 0xc3, 0x12, 0x42, 0x6c, 0xc3, 0x8f, 0x88, 0x3e,
            0xc2, 0x6c, 0x89, 0x40, 0x43, 0x10, 0x1d, 0xb4, 0x43, 0xd6, 0xb0, 0x24, 0xc3, 0x50, 0xaf, 0xae, 0xc3, 0x80, 0x19, 0x3e,
            0xc3, 0xdd, 0x52, 0xf3, 0x43, 0xd8, 0xd0, 0xa4, 0x43, 0xc7, 0x16, 0x14, 0xc3, 0x23, 0x16, 0x62, 0x42, 0xe0, 0xa5, 0x40,
            0x42, 0xe0, 0x94, 0x68, 0x43, 0x87, 0x15, 0x6e, 0xc3, 0xe5, 0x3c, 0x10, 0x43, 0x8b, 0x46, 0x9c, 0xc2, 0xe0, 0xf1, 0xa0,
            0x43, 0x01, 0xf0, 0x48, 0x43, 0x3c, 0x76, 0xac, 0xc3, 0xb6, 0xbf, 0xbb, 0xc3, 0xb9, 0xbf, 0x2e, 0x43, 0x09, 0x12, 0x98,
            0x43, 0xcb, 0xae, 0xea, 0x43, 0xb8, 0x80, 0xb2, 0x43, 0x81, 0x00, 0xd2, 0x43, 0x7b, 0xc9, 0x50, 0xc3, 0xf5, 0x1c, 0xdb,
            0x41, 0xf1, 0xdb, 0x80, 0x42, 0x35, 0x7e, 0x00, 0xc3, 0xad, 0x24, 0xc6, 0xc3, 0xf9, 0xe1, 0x5c, 0xc3, 0xe9, 0x08, 0x53,
            0x43, 0x7a, 0xdd, 0x3c, 0x43, 0xf0, 0x61, 0x70, 0x43, 0x78, 0x8d, 0xc4, 0x43, 0xe6, 0xf4, 0xea, 0xc3, 0xa5, 0x50, 0xd6};

    static constexpr unsigned char kRandomFloat32LittleEndianBin[] = {
            0x9b, 0xc3, 0xf4, 0xc3, 0x70, 0x73, 0x19, 0xc3, 0xc8, 0xbb, 0x87, 0x42, 0x38, 0x8d, 0xaa, 0x42, 0x68, 0x16, 0xa5, 0xc3,
            0xb6, 0xd0, 0x41, 0xc3, 0xe2, 0x9d, 0x83, 0x43, 0xb1, 0xae, 0x74, 0xc3, 0x20, 0xc2, 0x6f, 0x42, 0x38, 0xca, 0x89, 0x42,
            0x4e, 0x4a, 0x27, 0xc3, 0xc4, 0x58, 0x52, 0x43, 0xce, 0x6c, 0xa3, 0x43, 0x14, 0x56, 0x8f, 0x43, 0xd7, 0x67, 0x9f, 0xc3,
            0xbe, 0xda, 0x8f, 0xc3, 0x70, 0x02, 0x8a, 0xc3, 0x60, 0xc9, 0x24, 0xc2, 0xfc, 0xc5, 0x72, 0x43, 0xdc, 0xf0, 0x7f, 0x43,
            0x23, 0x9c, 0x79, 0xc3, 0x4a, 0x9f, 0xee, 0x43, 0x0c, 0xab, 0x14, 0xc3, 0x48, 0x1a, 0xef, 0xc2, 0x40, 0x34, 0x18, 0xc1,
            0x84, 0x85, 0x41, 0x43, 0xf0, 0xa2, 0x6f, 0x43, 0x84, 0x25, 0x5f, 0x43, 0x20, 0x07, 0x95, 0x41, 0x54, 0xeb, 0x8c, 0xc2,
            0x9a, 0x30, 0xcb, 0x43, 0xb0, 0x48, 0xef, 0x43, 0x88, 0xd0, 0x7f, 0xc3, 0x40, 0xa5, 0x9a, 0xc2, 0xd0, 0x18, 0x5d, 0xc3,
            0xc8, 0xf6, 0xd0, 0x42, 0x46, 0x42, 0x88, 0xc3, 0xc0, 0x08, 0x9f, 0x42, 0x64, 0xc3, 0xf6, 0x43, 0x20, 0x83, 0x60, 0x43,
            0xd4, 0xc3, 0x6a, 0x43, 0x40, 0x85, 0x03, 0xc2, 0x80, 0x99, 0x03, 0x42, 0x7c, 0xaa, 0xd6, 0xc2, 0x76, 0xa3, 0xdd, 0xc3,
            0x98, 0xef, 0xf0, 0x43, 0x12, 0xa2, 0x40, 0xc3, 0x24, 0x7d, 0x22, 0x43, 0xe0, 0x16, 0xa7, 0x41, 0xe8, 0x94, 0xab, 0xc3,
            0x18, 0x39, 0x06, 0xc2, 0xe0, 0xa0, 0xf0, 0xc3, 0x4e, 0xd7, 0xba, 0x43, 0x00, 0xea, 0x1d, 0x43, 0x76, 0xa4, 0xa9, 0x43,
            0x31, 0x7a, 0xe8, 0xc3, 0x1c, 0x5b, 0xd9, 0xc3, 0xfc, 0x8d, 0x88, 0xc2, 0x70, 0x80, 0x65, 0xc2, 0x98, 0x42, 0xd1, 0x42,
            0x43, 0xf4, 0xf3, 0xc3, 0x5a, 0x8e, 0x5d, 0xc3, 0x7a, 0xd1, 0xf4, 0x43, 0x6c, 0x42, 0x12, 0xc3, 0x3e, 0x88, 0x8f, 0xc3,
            0x40, 0x89, 0x6c, 0xc2, 0xb4, 0x1d, 0x10, 0x43, 0x24, 0xb0, 0xd6, 0x43, 0xae, 0xaf, 0x50, 0xc3, 0x3e, 0x19, 0x80, 0xc3,
            0xf3, 0x52, 0xdd, 0xc3, 0xa4, 0xd0, 0xd8, 0x43, 0x14, 0x16, 0xc7, 0x43, 0x62, 0x16, 0x23, 0xc3, 0x40, 0xa5, 0xe0, 0x42,
            0x68, 0x94, 0xe0, 0x42, 0x6e, 0x15, 0x87, 0x43, 0x10, 0x3c, 0xe5, 0xc3, 0x9c, 0x46, 0x8b, 0x43, 0xa0, 0xf1, 0xe0, 0xc2,
            0x48, 0xf0, 0x01, 0x43, 0xac, 0x76, 0x3c, 0x43, 0xbb, 0xbf, 0xb6, 0xc3, 0x2e, 0xbf, 0xb9, 0xc3, 0x98, 0x12, 0x09, 0x43,
            0xea, 0xae, 0xcb, 0x43, 0xb2, 0x80, 0xb8, 0x43, 0xd2, 0x00, 0x81, 0x43, 0x50, 0xc9, 0x7b, 0x43, 0xdb, 0x1c, 0xf5, 0xc3,
            0x80, 0xdb, 0xf1, 0x41, 0x00, 0x7e, 0x35, 0x42, 0xc6, 0x24, 0xad, 0xc3, 0x5c, 0xe1, 0xf9, 0xc3, 0x53, 0x08, 0xe9, 0xc3,
            0x3c, 0xdd, 0x7a, 0x43, 0x70, 0x61, 0xf0, 0x43, 0xc4, 0x8d, 0x78, 0x43, 0xea, 0xf4, 0xe6, 0x43, 0xd6, 0x50, 0xa5, 0xc3};

    static constexpr unsigned char kSpecialFloat32LittleEndianBin[] = {
            0x00, 0x00, 0xc0, 0x7f, 0x00, 0x00, 0xc0, 0x7f, 0x00, 0x00, 0xc0, 0x7f, 0x00, 0x00, 0xc0, 0x7f, 0x00, 0x00, 0xc0, 0x7f,
            0x00, 0x00, 0xc0, 0x7f, 0x00, 0x00, 0xc0, 0x7f, 0x00, 0x00, 0xc0, 0x7f, 0x00, 0x00, 0xc0, 0x7f, 0x00, 0x00, 0xc0, 0x7f,
            0x00, 0x00, 0x80, 0x7f, 0x00, 0x00, 0x80, 0x7f, 0x00, 0x00, 0x80, 0x7f, 0x00, 0x00, 0x80, 0x7f, 0x00, 0x00, 0x80, 0x7f,
            0x00, 0x00, 0x80, 0x7f, 0x00, 0x00, 0x80, 0x7f, 0x00, 0x00, 0x80, 0x7f, 0x00, 0x00, 0x80, 0x7f, 0x00, 0x00, 0x80, 0x7f,
            0x00, 0x00, 0x80, 0xff, 0x00, 0x00, 0x80, 0xff, 0x00, 0x00, 0x80, 0xff, 0x00, 0x00, 0x80, 0xff, 0x00, 0x00, 0x80, 0xff,
            0x00, 0x00, 0x80, 0xff, 0x00, 0x00, 0x80, 0xff, 0x00, 0x00, 0x80, 0xff, 0x00, 0x00, 0x80, 0xff, 0x00, 0x00, 0x80, 0xff};

    static constexpr unsigned char kSpecialFloat32BigEndianBin[] = {
            0x7f, 0xc0, 0x00, 0x00, 0x7f, 0xc0, 0x00, 0x00, 0x7f, 0xc0, 0x00, 0x00, 0x7f, 0xc0, 0x00, 0x00, 0x7f, 0xc0, 0x00, 0x00,
            0x7f, 0xc0, 0x00, 0x00, 0x7f, 0xc0, 0x00, 0x00, 0x7f, 0xc0, 0x00, 0x00, 0x7f, 0xc0, 0x00, 0x00, 0x7f, 0xc0, 0x00, 0x00,
            0x7f, 0x80, 0x00, 0x00, 0x7f, 0x80, 0x00, 0x00, 0x7f, 0x80, 0x00, 0x00, 0x7f, 0x80, 0x00, 0x00, 0x7f, 0x80, 0x00, 0x00,
            0x7f, 0x80, 0x00, 0x00, 0x7f, 0x80, 0x00, 0x00, 0x7f, 0x80, 0x00, 0x00, 0x7f, 0x80, 0x00, 0x00, 0x7f, 0x80, 0x00, 0x00,
            0xff, 0x80, 0x00, 0x00, 0xff, 0x80, 0x00, 0x00, 0xff, 0x80, 0x00, 0x00, 0xff, 0x80, 0x00, 0x00, 0xff, 0x80, 0x00, 0x00,
            0xff, 0x80, 0x00, 0x00, 0xff, 0x80, 0x00, 0x00, 0xff, 0x80, 0x00, 0x00, 0xff, 0x80, 0x00, 0x00, 0xff, 0x80, 0x00, 0x00};

private:
    nonstd::optional<testing::DeviceSession> device_session_;
};

constexpr char CreationTest::kTestDelimiter;

constexpr size_t CreationTest::kRandomBoolBinLength;
constexpr size_t CreationTest::kRandomInt32BinLength;
constexpr size_t CreationTest::kRandomFloat16BinLength;
constexpr size_t CreationTest::kRandomFloat32BinLength;
constexpr size_t CreationTest::kSpecialFloat32BinLength;

constexpr char CreationTest::kRandomBoolText[];
constexpr char CreationTest::kRandomInt32Text[];
constexpr char CreationTest::kRandomFloat16Text[];
constexpr char CreationTest::kRandomFloat32Text[];
constexpr char CreationTest::kSpecialFloat32Text[];

constexpr unsigned char CreationTest::kRandomBoolBin[];
constexpr unsigned char CreationTest::kRandomInt32LittleEndianBin[];
constexpr unsigned char CreationTest::kRandomInt32BigEndianBin[];
constexpr unsigned char CreationTest::kRandomFloat16LittleEndianBin[];
constexpr unsigned char CreationTest::kRandomFloat16BigEndianBin[];
constexpr unsigned char CreationTest::kRandomFloat32LittleEndianBin[];
constexpr unsigned char CreationTest::kRandomFloat32BigEndianBin[];
constexpr unsigned char CreationTest::kSpecialFloat32LittleEndianBin[];
constexpr unsigned char CreationTest::kSpecialFloat32BigEndianBin[];

const std::vector<bool> CreationTest::kRandomBoolVector = {
        false, true,  true, false, false, false, false, true,  false, false, false, true,  true,  false, true,  false, true,
        false, false, true, false, false, true,  false, false, true,  false, true,  true,  false, false, false, false, false,
        false, true,  true, false, true,  false, true,  false, false, true,  false, false, true,  true,  true,  true,  true,
        false, false, true, false, true,  false, true,  false, false, true,  true,  true,  false, true,  false, false, false,
        false, false, true, true,  false, true,  false, true,  true,  false, false, false, true,  true,  false, false, true,
        false, false, true, false, true,  true,  true,  false, true,  true,  true,  true,  false, false, true};

const std::vector<int32_t> CreationTest::kRandomInt32Vector = {
        4,  -4, -1, 4,  0, 3, 2, 1,  0, -3, 4, 3,  1, -5, 4,  1,  -2, -2, -3, -3, -5, -2, 3, 4,  0,  3,  -5, 3,  0,  -4, -4, -3, 1, -5,
        -1, -5, -2, 0,  4, 1, 2, -5, 4, 3,  2, 4,  1, -2, 3,  -2, 1,  0,  -2, 3,  -1, -5, 0, -4, 1,  3,  2,  0,  -4, 3,  -2, 4,  2, 3,
        0,  1,  -5, -5, 3, 1, 1, 1,  0, 0,  4, -5, 1, 2,  -1, 3,  -2, -5, -5, -5, -4, 4,  1, -5, -3, -4, 4,  -2, -4, -3, -3, -2};

const std::vector<float> CreationTest::kRandomFloat16Vector = {
        -489.5, -153.5, 67.9,   85.25,  -330.2, -193.9, 263.2,  -244.6, 59.94,   68.9,   -167.2, 210.4,  326.8,  286.8,  -318.8,
        -287.8, -276.0, -41.2,  242.8,  256.0,  -249.6, 477.2,  -148.6, -119.56, -9.516, 193.5,  239.6,  223.1,  18.62,  -70.44,
        406.5,  478.5,  -255.9, -77.3,  -221.1, 104.5,  -272.5, 79.5,   493.5,   224.5,  234.8,  -32.88, 32.9,   -107.3, -443.2,
        481.8,  -192.6, 162.5,  20.89,  -343.2, -33.56, -481.2, 373.8,  157.9,   339.2,  -465.0, -434.8, -68.25, -57.38, 104.6,
        -488.0, -221.5, 489.8,  -146.2, -287.0, -59.12, 144.1,  429.5,  -208.6,  -256.2, -442.8, 433.8,  398.2,  -163.1, 112.3,
        112.3,  270.2,  -458.5, 278.5,  -112.5, 130.0,  188.5,  -365.5, -371.5,  137.1,  407.2,  369.0,  258.0,  251.8,  -490.2,
        30.23,  45.38,  -346.2, -499.8, -466.0, 250.9,  480.8,  248.5,  462.0,   -330.8};

const std::vector<float> CreationTest::kRandomFloat32Vector = {
        -489.52817, -153.45093, 67.86676,   85.27582,   -330.17505, -193.81528, 263.23346,  -244.68239, 59.939575,  68.89496,
        -167.29025, 210.34674,  326.85004,  286.6725,   -318.81125, -287.70892, -276.01904, -41.196655, 242.77338,  255.94086,
        -249.60991, 477.24445,  -148.66815, -119.55133, -9.512756,  193.52155,  239.63647,  223.14655,  18.628479,  -70.459625,
        406.3797,   478.56787,  -255.81458, -77.322754, -221.09692, 104.481995, -272.51776, 79.51709,   493.5265,   224.5122,
        234.76495,  -32.880127, 32.899902,  -107.33298, -443.27704, 481.87183,  -192.63309, 162.48883,  20.88617,   -343.16333,
        -33.555756, -481.25684, 373.68207,  157.91406,  339.28485,  -464.95462, -434.7118,  -68.27731,  -57.375427, 104.630066,
        -487.9083,  -221.55606, 489.63654,  -146.25946, -287.0644,  -59.134033, 144.11603,  429.3761,   -208.68625, -256.1972,
        -442.64804, 433.63,     398.1725,   -163.08743, 112.322754, 112.289856, 270.16742,  -458.46924, 278.55164,  -112.471924,
        129.9386,   188.46356,  -365.4979,  -371.4936,  137.07263,  407.36652,  369.00543,  258.0064,   251.78638,  -490.22543,
        30.232178,  45.373047,  -346.2873,  -499.76062, -466.06503, 250.8642,   480.76123,  248.55377,  461.9134,   -330.63153};

const std::vector<float> CreationTest::kSpecialFloat32Vector = {
        NAN,       NAN,       NAN,       NAN,       NAN,       NAN,       NAN,       NAN,       NAN,       NAN,
        INFINITY,  INFINITY,  INFINITY,  INFINITY,  INFINITY,  INFINITY,  INFINITY,  INFINITY,  INFINITY,  INFINITY,
        -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY};

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

TEST_P(CreationTest, FromString) {
    static const std::vector<Dtype> kDtypes = {
            Dtype::kBool,
            Dtype::kInt32,
            Dtype::kFloat16,
            Dtype::kFloat32,
            Dtype::kFloat32,
    };

    static const char* kText[] = {
            kRandomBoolText,
            kRandomInt32Text,
            kRandomFloat16Text,
            kRandomFloat32Text,
            kSpecialFloat32Text,
    };

    static const size_t kTextLengths[] = {
            sizeof(kRandomBoolText),
            sizeof(kRandomInt32Text),
            sizeof(kRandomFloat16Text),
            sizeof(kRandomFloat32Text),
            sizeof(kSpecialFloat32Text),
    };

    static const size_t kDataLengths[] = {
            kRandomBoolVector.size(),
            kRandomInt32Vector.size(),
            kRandomFloat16Vector.size(),
            kRandomFloat32Vector.size(),
            kSpecialFloat32Vector.size(),
    };

    static const unsigned char* kLittleEndianBins[] = {
            kRandomBoolBin,
            kRandomInt32LittleEndianBin,
            kRandomFloat16LittleEndianBin,
            kRandomFloat32LittleEndianBin,
            kSpecialFloat32LittleEndianBin,
    };

    static const unsigned char* kBigEndianBins[] = {
            kRandomBoolBin,
            kRandomInt32BigEndianBin,
            kRandomFloat16BigEndianBin,
            kRandomFloat32BigEndianBin,
            kSpecialFloat32BigEndianBin,
    };

    static const size_t kBinLengths[] = {
            kRandomBoolBinLength,
            kRandomInt32BinLength,
            kRandomFloat16BinLength,
            kRandomFloat32BinLength,
            kSpecialFloat32BinLength,
    };

    static const std::function<void(Array)> kExpectDataEqualChecks[] = {
            [](const auto& x) { return testing::ExpectDataEqual<bool>(kRandomBoolVector, x); },
            [](const auto& x) { return testing::ExpectDataEqual<int32_t>(kRandomInt32Vector, x); },
            [](const auto& x) { return testing::ExpectDataClose<Float16>(kRandomFloat16Vector, x, 1e-1); },
            [](const auto& x) { return testing::ExpectDataClose<float>(kRandomFloat32Vector, x, 1e-8); },
            [](const auto& x) { return testing::ExpectDataClose<float>(kSpecialFloat32Vector, x, 1e-8); },
    };

    Device& device = GetDefaultDevice();

    testing::RunTestWithThreads([&device]() {
        for (size_t j = 0; j < kDtypes.size(); j++) {
            std::cerr << "Testing dtype " << kDtypes[j] << std::endl;
            int64_t counts[2] = {int64_t(kDataLengths[j]), -1};
            for (size_t i = 0; i < 2; i++) {
                {
                    std::cerr << "Running with text" << std::endl;
                    Array x = FromString(std::string(kText[j], kTextLengths[j]), kDtypes[j], counts[i], kTestDelimiter, device);

                    Shape expected_shape{int64_t(kDataLengths[j])};
                    Strides expected_strides{expected_shape, kDtypes[j]};

                    EXPECT_EQ(expected_shape, x.shape());
                    EXPECT_EQ(kDtypes[j], x.dtype());
                    EXPECT_EQ(expected_strides, x.strides());
                    EXPECT_EQ(0, x.offset());
                    EXPECT_EQ(&device, &x.device());

                    kExpectDataEqualChecks[j](x);
                }

                {
                    std::cerr << "Running with binary" << std::endl;
                    std::string data = (testing::testing_internal::IsLittleEndian())
                                               ? std::string(reinterpret_cast<const char*>(kLittleEndianBins[j]), kBinLengths[j])
                                               : std::string(reinterpret_cast<const char*>(kBigEndianBins[j]), kBinLengths[j]);
                    Array x = FromString(data, kDtypes[j], counts[i], nonstd::nullopt, device);

                    Shape expected_shape{int64_t(kDataLengths[j])};
                    Strides expected_strides{expected_shape, kDtypes[j]};

                    EXPECT_EQ(expected_shape, x.shape());
                    EXPECT_EQ(kDtypes[j], x.dtype());
                    EXPECT_EQ(expected_strides, x.strides());
                    EXPECT_EQ(0, x.offset());
                    EXPECT_EQ(&device, &x.device());

                    kExpectDataEqualChecks[j](x);
                }
            }
        }
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
