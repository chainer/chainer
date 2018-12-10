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

int32_t ReadHeader(std::ifstream& ifs) {
    int32_t header{0};
    ifs.read(reinterpret_cast<char*>(&header), sizeof(header));

    // Swap byte order.
    uint32_t b1 = header & 255;
    uint32_t b2 = (header >> 8) & 255;
    uint32_t b3 = (header >> 16) & 255;
    uint32_t b4 = (header >> 24) & 255;

    return (b1 << 24) | (b2 << 16) | (b3 << 8) | b4;
}

}  // namespace

chx::Array ReadMnistImages(const std::string& filename) {
    std::ifstream ifs{filename.c_str(), std::ios::in | std::ios::binary};
    if (!ifs.is_open()) {
        throw std::runtime_error("Could not open MNIST images: " + filename);
    }

    if (ReadHeader(ifs) != 0x803) {
        throw std::runtime_error("Bad MNIST images file: " + filename);
    }

    int32_t n = ReadHeader(ifs);
    int32_t height = ReadHeader(ifs);
    int32_t width = ReadHeader(ifs);

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

    if (ReadHeader(ifs) != 0x801) {
        throw std::runtime_error("Bad MNIST labels file: " + filename);
    }

    int32_t n = ReadHeader(ifs);

    assert(n == 60000 || n == 10000);

    return ReadArray(ifs, {n});
}
