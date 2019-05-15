#pragma once

namespace chainerx {

class Kernel {
public:
    Kernel() = default;

    virtual ~Kernel() = default;

    Kernel(const Kernel&) = delete;
    Kernel(Kernel&&) = delete;
    Kernel& operator=(const Kernel&) = delete;
    Kernel& operator=(Kernel&&) = delete;
};

}  // namespace chainerx
