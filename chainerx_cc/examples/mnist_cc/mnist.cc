#include "mnist.h"

#include <algorithm>
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

int32_t ReverseInt(int32_t i) {
    uint8_t c1 = i & 255;
    uint8_t c2 = (i >> 8) & 255;
    uint8_t c3 = (i >> 16) & 255;
    uint8_t c4 = (i >> 24) & 255;

    return (static_cast<int32_t>(c1) << 24) + (static_cast<int32_t>(c2) << 16) + (static_cast<int32_t>(c3) << 8) + c4;
}

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
    return ReverseInt(*(reinterpret_cast<int32_t*>(buffer.get() + offset)));
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

    return ReadArray(buffer, {n, height * width}, 16);
}

chainerx::Array ReadMnistLabels(const std::string& filename) {
    std::unique_ptr<char[]> buffer = ReadFile(filename);

    int32_t n = ReadHeader(buffer, 4);

    return ReadArray(buffer, {n}, 8);
}
