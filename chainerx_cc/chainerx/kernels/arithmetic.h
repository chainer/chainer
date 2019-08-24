#pragma once

#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/kernel.h"
#include "chainerx/scalar.h"

namespace chainerx {

class AddKernel : public Kernel {
public:
    static const char* name() { return "Add"; }

    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class AddASKernel : public Kernel {
public:
    static const char* name() { return "AddAS"; }

    virtual void Call(const Array& x1, Scalar x2, const Array& out) = 0;
};

class SubtractKernel : public Kernel {
public:
    static const char* name() { return "Subtract"; }

    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class SubtractASKernel : public Kernel {
public:
    static const char* name() { return "SubtractAS"; }

    virtual void Call(const Array& x1, Scalar x2, const Array& out) = 0;
};

class MultiplyKernel : public Kernel {
public:
    static const char* name() { return "Multiply"; }

    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class MultiplyASKernel : public Kernel {
public:
    static const char* name() { return "MultiplyAS"; }

    virtual void Call(const Array& x1, Scalar x2, const Array& out) = 0;
};

class FloorDivideKernel : public Kernel {
public:
    static const char* name() { return "FloorDivide"; }

    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class FloorDivideASKernel : public Kernel {
public:
    static const char* name() { return "FloorDivideAS"; }

    virtual void Call(const Array& x1, Scalar x2, const Array& out) = 0;
};

class FloorDivideSAKernel : public Kernel {
public:
    static const char* name() { return "FloorDivideSA"; }

    virtual void Call(Scalar x1, const Array& x2, const Array& out) = 0;
};

class DivideKernel : public Kernel {
public:
    static const char* name() { return "Divide"; }

    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class DivideASKernel : public Kernel {
public:
    static const char* name() { return "DivideAS"; }

    virtual void Call(const Array& x1, Scalar x2, const Array& out) = 0;
};

class DivideSAKernel : public Kernel {
public:
    static const char* name() { return "DivideSA"; }

    virtual void Call(Scalar x1, const Array& x2, const Array& out) = 0;
};

class PowerKernel : public Kernel {
public:
    static const char* name() { return "Power"; }

    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class PowerASKernel : public Kernel {
public:
    static const char* name() { return "PowerAS"; }

    virtual void Call(const Array& x1, Scalar x2, const Array& out) = 0;
};

class PowerSAKernel : public Kernel {
public:
    static const char* name() { return "PowerSA"; }

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

}  // namespace chainerx
