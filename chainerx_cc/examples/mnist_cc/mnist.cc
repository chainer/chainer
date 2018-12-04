#include "mnist.h"

#include <algorithm>
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

namespace {

std::unique_ptr<char[]> ReadFile(const std::string& filename) {
    std::ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) {
        throw std::runtime_error("Could not open " + filename);
    }

    size_t size = ifs.tellg();

    std::unique_ptr<char[]> buffer{new char[size]};

    ifs.seekg(0, std::ios::beg);
    ifs.read(buffer.get(), size);
    ifs.close();

    return buffer;
}

int32_t ReadHeader(const std::unique_ptr<char[]>& buffer, size_t offset) {
    uint32_t header = *reinterpret_cast<uint32_t*>(buffer.get() + offset);

    // Swap byte order.
    uint32_t b1 = header & 255;
    uint32_t b2 = (header >> 8) & 255;
    uint32_t b3 = (header >> 16) & 255;
    uint32_t b4 = (header >> 24) & 255;

    return (b1 << 24) | (b2 << 16) | (b3 << 8) | b4;
}

chainerx::Array ReadArray(const std::unique_ptr<char[]>& buffer, const chainerx::Shape& shape, size_t offset) {
    int64_t n = shape.GetTotalSize();

    std::shared_ptr<uint8_t> data{new uint8_t[n], std::default_delete<uint8_t[]>{}};
    std::copy_n(reinterpret_cast<uint8_t*>(buffer.get() + offset), n, data.get());

    return chainerx::FromContiguousHostData(
            shape, chainerx::Dtype::kUInt8, static_cast<std::shared_ptr<void>>(data), chainerx::GetDefaultDevice());
}

}  // namespace

chainerx::Array ReadMnistImages(const std::string& filename) {
    std::unique_ptr<char[]> buffer = ReadFile(filename);

    int32_t n = ReadHeader(buffer, 4);
    int32_t height = ReadHeader(buffer, 8);
    int32_t width = ReadHeader(buffer, 12);

    assert(n == 60000 || n == 10000);
    assert(height == 28);
    assert(width == 28);

    return ReadArray(buffer, {n, height * width}, 16);
}

chainerx::Array ReadMnistLabels(const std::string& filename) {
    std::unique_ptr<char[]> buffer = ReadFile(filename);

    int32_t n = ReadHeader(buffer, 4);

    assert(n == 60000 || n == 10000);

    return ReadArray(buffer, {n}, 8);
}
