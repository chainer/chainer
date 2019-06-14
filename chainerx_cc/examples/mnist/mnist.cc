#include "mnist.h"

#include <cassert>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>

#include "chainerx/array.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/routines/creation.h"
#include "chainerx/shape.h"

namespace chx = chainerx;

namespace {

chx::Array ReadArray(std::ifstream& ifs, const chx::Shape& shape) {
    int64_t n = shape.GetTotalSize();

    std::shared_ptr<uint8_t> data{new uint8_t[n], std::default_delete<uint8_t[]>{}};
    ifs.read(reinterpret_cast<char*>(data.get()), n);

    return chx::FromContiguousHostData(shape, chx::Dtype::kUInt8, static_cast<std::shared_ptr<void>>(data), chx::GetDefaultDevice());
}

int32_t ReadInt32(std::ifstream& ifs) {
    uint32_t result = 0;
    for (int i = 0; i < 4; ++i) {
        char byte{};
        ifs.read(&byte, sizeof(byte));
        result = (result << 8) | static_cast<uint32_t>(static_cast<uint8_t>(byte));
    }
    return static_cast<int32_t>(result);
}

}  // namespace

chx::Array ReadMnistImages(const std::string& filename) {
    std::ifstream ifs{filename.c_str(), std::ios::in | std::ios::binary};
    if (!ifs.is_open()) {
        throw std::runtime_error("Could not open MNIST images: " + filename);
    }

    if (ReadInt32(ifs) != 0x803) {
        throw std::runtime_error("Bad MNIST images file: " + filename);
    }

    int32_t n = ReadInt32(ifs);
    int32_t height = ReadInt32(ifs);
    int32_t width = ReadInt32(ifs);

    assert(n == 60000 || n == 10000);
    assert(height == 28);
    assert(width == 28);

    return ReadArray(ifs, {n, height * width});
}

chx::Array ReadMnistLabels(const std::string& filename) {
    std::ifstream ifs{filename.c_str(), std::ios::in | std::ios::binary};
    if (!ifs.is_open()) {
        throw std::runtime_error("Could not open MNIST labels: " + filename);
    }

    if (ReadInt32(ifs) != 0x801) {
        throw std::runtime_error("Bad MNIST labels file: " + filename);
    }

    int32_t n = ReadInt32(ifs);

    assert(n == 60000 || n == 10000);

    return ReadArray(ifs, {n});
}
