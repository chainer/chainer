#pragma once

#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/kernel.h"
#include "chainerx/scalar.h"

namespace chainerx {

class BitwiseAndKernel : public Kernel {
public:
    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class BitwiseAndASKernel : public Kernel {
public:
    virtual void Call(const Array& x1, Scalar x2, const Array& out) = 0;
};

class BitwiseOrKernel : public Kernel {
public:
    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class BitwiseOrASKernel : public Kernel {
public:
    virtual void Call(const Array& x1, Scalar x2, const Array& out) = 0;
};

class BitwiseXorKernel : public Kernel {
public:
    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class BitwiseXorASKernel : public Kernel {
public:
    virtual void Call(const Array& x1, Scalar x2, const Array& out) = 0;
};

class LeftShiftAAKernel : public Kernel {
public:
    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class LeftShiftASKernel : public Kernel {
public:
    virtual void Call(const Array& x1, Scalar x2, const Array& out) = 0;
};

class LeftShiftSAKernel : public Kernel {
public:
    virtual void Call(Scalar x1, const Array& x2, const Array& out) = 0;
};

class RightShiftAAKernel : public Kernel {
public:
    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class RightShiftASKernel : public Kernel {
public:
    virtual void Call(const Array& x1, Scalar x2, const Array& out) = 0;
};

class RightShiftSAKernel : public Kernel {
public:
    virtual void Call(Scalar x1, const Array& x2, const Array& out) = 0;
};

}  // namespace chainerx
