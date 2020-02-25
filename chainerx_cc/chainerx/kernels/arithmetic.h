#pragma once

#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/kernel.h"
#include "chainerx/scalar.h"

namespace chainerx {

class AddKernel : public Kernel {
public:
    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class AddASKernel : public Kernel {
public:
    virtual void Call(const Array& x1, Scalar x2, const Array& out) = 0;
};

class SubtractKernel : public Kernel {
public:
    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class SubtractASKernel : public Kernel {
public:
    virtual void Call(const Array& x1, Scalar x2, const Array& out) = 0;
};

class MultiplyKernel : public Kernel {
public:
    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class MultiplyASKernel : public Kernel {
public:
    virtual void Call(const Array& x1, Scalar x2, const Array& out) = 0;
};

class FloorDivideKernel : public Kernel {
public:
    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class FloorDivideASKernel : public Kernel {
public:
    virtual void Call(const Array& x1, Scalar x2, const Array& out) = 0;
};

class FloorDivideSAKernel : public Kernel {
public:
    virtual void Call(Scalar x1, const Array& x2, const Array& out) = 0;
};

class DivideKernel : public Kernel {
public:
    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class DivideASKernel : public Kernel {
public:
    virtual void Call(const Array& x1, Scalar x2, const Array& out) = 0;
};

class DivideSAKernel : public Kernel {
public:
    virtual void Call(Scalar x1, const Array& x2, const Array& out) = 0;
};

class PowerKernel : public Kernel {
public:
    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class PowerASKernel : public Kernel {
public:
    virtual void Call(const Array& x1, Scalar x2, const Array& out) = 0;
};

class PowerSAKernel : public Kernel {
public:
    virtual void Call(Scalar x1, const Array& x2, const Array& out) = 0;
};

class ModAAKernel : public Kernel {
public:
    static const char* name() { return "ModAA"; }

    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class ModASKernel : public Kernel {
public:
    static const char* name() { return "ModAS"; }

    virtual void Call(const Array& x1, Scalar x2, const Array& out) = 0;
};

class ModSAKernel : public Kernel {
public:
    static const char* name() { return "ModSA"; }

    virtual void Call(Scalar x1, const Array& x2, const Array& out) = 0;
};

class FmodKernel : public Kernel {
public:
    static const char* name() { return "Fmod"; }

    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

}  // namespace chainerx
