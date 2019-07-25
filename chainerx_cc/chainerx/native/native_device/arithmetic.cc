#include "chainerx/native/native_device.h"

#include <cstdint>

#include "chainerx/arithmetic_ops.h"
#include "chainerx/array.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/float16.h"
#include "chainerx/kernels/arithmetic.h"
#include "chainerx/native/elementwise.h"
#include "chainerx/native/kernel_regist.h"
#include "chainerx/numeric.h"
#include "chainerx/routines/arithmetic.h"
#include "chainerx/scalar.h"

namespace chainerx {
namespace native {
namespace {

CHAINERX_NATIVE_REGISTER_ELTWISE_BINARY_KERNEL(AddKernel, { out = ArithmeticOps<T>::Add(x1, x2); });

class NativeAddASKernel : public AddASKernel {
public:
    void Call(const Array& x1, Scalar x2, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, out);
        const Array& x1_cast = x1.dtype() == out.dtype() ? x1 : x1.AsType(out.dtype());
        VisitDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                void operator()(int64_t /*i*/, T x1, T& out) { out = ArithmeticOps<T>::Add(x1, x2); }
                T x2;
            };
            Elementwise<const T, T>(Impl{static_cast<T>(x2)}, x1_cast, out);
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(AddASKernel, NativeAddASKernel);

CHAINERX_NATIVE_REGISTER_ELTWISE_DTYPE_BINARY_KERNEL(SubtractKernel, { out = ArithmeticOps<T>::Subtract(x1, x2); }, VisitNumericDtype);

class NativeSubtractASKernel : public SubtractASKernel {
public:
    void Call(const Array& x1, Scalar x2, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, out);
        const Array& x1_cast = x1.dtype() == out.dtype() ? x1 : x1.AsType(out.dtype());
        VisitNumericDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                void operator()(int64_t /*i*/, T x1, T& out) { out = ArithmeticOps<T>::Subtract(x1, x2); }
                T x2;
            };
            Elementwise<const T, T>(Impl{static_cast<T>(x2)}, x1_cast, out);
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(SubtractASKernel, NativeSubtractASKernel);

CHAINERX_NATIVE_REGISTER_ELTWISE_BINARY_KERNEL(MultiplyKernel, { out = ArithmeticOps<T>::Multiply(x1, x2); });

class NativeMultiplyASKernel : public MultiplyASKernel {
public:
    void Call(const Array& x1, Scalar x2, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, out);
        const Array& x1_cast = x1.dtype() == out.dtype() ? x1 : x1.AsType(out.dtype());
        VisitDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                void operator()(int64_t /*i*/, T x1, T& out) { out = ArithmeticOps<T>::Multiply(x1, x2); }
                T x2;
            };
            Elementwise<const T, T>(Impl{static_cast<T>(x2)}, x1_cast, out);
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(MultiplyASKernel, NativeMultiplyASKernel);

int32_t FloorDivide(int32_t x, int32_t y) {
    if (y == 0) {
        return 0;
    }
    auto div = std::div(x, y);
    return div.quot - ((y >= 0 ? div.rem : -div.rem) < 0 ? 1 : 0);
}
int64_t FloorDivide(int64_t x, int64_t y) {
    if (y == 0) {
        return 0;
    }
    auto div = std::div(x, y);
    return div.quot - ((y >= 0 ? div.rem : -div.rem) < 0 ? 1 : 0);
}
int8_t FloorDivide(int8_t x, int8_t y) { return static_cast<int8_t>(FloorDivide(static_cast<int32_t>(x), static_cast<int32_t>(y))); }
int16_t FloorDivide(int16_t x, int16_t y) { return static_cast<int16_t>(FloorDivide(static_cast<int32_t>(x), static_cast<int32_t>(y))); }
uint8_t FloorDivide(uint8_t x, uint8_t y) {
    if (y == 0) {
        return 0;
    }
    return x / y;
}
float FloorDivide(float x, float y) {
    float rem = std::fmod(x, y);
    return (x - rem) / y - ((rem < 0 && y > 0) || (rem > 0 && y < 0) ? 1 : 0);
}
double FloorDivide(double x, double y) {
    double rem = std::fmod(x, y);
    return (x - rem) / y - ((rem < 0 && y > 0) || (rem > 0 && y < 0) ? 1 : 0);
}
chainerx::Float16 FloorDivide(chainerx::Float16 x, chainerx::Float16 y) {
    return chainerx::Float16{FloorDivide(static_cast<float>(x), static_cast<float>(y))};
}

CHAINERX_NATIVE_REGISTER_ELTWISE_DTYPE_BINARY_KERNEL(FloorDivideKernel, { out = native::FloorDivide(x1, x2); }, VisitNumericDtype);

class NativeFloorDivideASKernel : public FloorDivideASKernel {
public:
    void Call(const Array& x1, Scalar x2, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, out);
        const Array& x1_cast = x1.dtype() == out.dtype() ? x1 : x1.AsType(out.dtype());
        VisitNumericDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                void operator()(int64_t /*i*/, T x1, T& out) { out = native::FloorDivide(x1, x2); }
                T x2;
            };
            Elementwise<const T, T>(Impl{static_cast<T>(x2)}, x1_cast, out);
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(FloorDivideASKernel, NativeFloorDivideASKernel);

class NativeFloorDivideSAKernel : public FloorDivideSAKernel {
public:
    void Call(Scalar x1, const Array& x2, const Array& out) override {
        Device& device = x2.device();
        device.CheckDevicesCompatible(x2, out);
        const Array& x2_cast = x2.dtype() == out.dtype() ? x2 : x2.AsType(out.dtype());
        VisitNumericDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                void operator()(int64_t /*i*/, T x2, T& out) { out = native::FloorDivide(x1, x2); }
                T x1;
            };
            Elementwise<const T, T>(Impl{static_cast<T>(x1)}, x2_cast, out);
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(FloorDivideSAKernel, NativeFloorDivideSAKernel);

CHAINERX_NATIVE_REGISTER_ELTWISE_BINARY_KERNEL(DivideKernel, { out = ArithmeticOps<T>::Divide(x1, x2); });

class NativeDivideASKernel : public DivideASKernel {
public:
    void Call(const Array& x1, Scalar x2, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, out);
        const Array& x1_cast = x1.dtype() == out.dtype() ? x1 : x1.AsType(out.dtype());
        VisitDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                void operator()(int64_t /*i*/, T x1, T& out) { out = ArithmeticOps<T>::Divide(x1, x2); }
                T x2;
            };
            Elementwise<const T, T>(Impl{static_cast<T>(x2)}, x1_cast, out);
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(DivideASKernel, NativeDivideASKernel);

class NativeDivideSAKernel : public DivideSAKernel {
public:
    void Call(Scalar x1, const Array& x2, const Array& out) override {
        Device& device = x2.device();
        device.CheckDevicesCompatible(x2, out);
        const Array& x2_cast = x2.dtype() == out.dtype() ? x2 : x2.AsType(out.dtype());
        VisitDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                void operator()(int64_t /*i*/, T x2, T& out) { out = ArithmeticOps<T>::Divide(x1, x2); }
                T x1;
            };
            Elementwise<const T, T>(Impl{static_cast<T>(x1)}, x2_cast, out);
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(DivideSAKernel, NativeDivideSAKernel);

CHAINERX_NATIVE_REGISTER_ELTWISE_DTYPE_BINARY_KERNEL(PowerKernel, { out = chainerx::Power(x1, x2); }, VisitNumericDtype);

class NativePowerASKernel : public PowerASKernel {
public:
    void Call(const Array& x1, Scalar x2, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, out);
        const Array& x1_cast = x1.dtype() == out.dtype() ? x1 : x1.AsType(out.dtype());
        VisitNumericDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                void operator()(int64_t /*i*/, T x1, T& out) { out = chainerx::Power(x1, x2); }
                T x2;
            };
            Elementwise<const T, T>(Impl{static_cast<T>(x2)}, x1_cast, out);
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(PowerASKernel, NativePowerASKernel);

class NativePowerSAKernel : public PowerSAKernel {
public:
    void Call(Scalar x1, const Array& x2, const Array& out) override {
        Device& device = x2.device();
        device.CheckDevicesCompatible(x2, out);
        const Array& x2_cast = x2.dtype() == out.dtype() ? x2 : x2.AsType(out.dtype());
        VisitNumericDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                void operator()(int64_t /*i*/, T x2, T& out) { out = chainerx::Power(x1, x2); }
                T x1;
            };
            Elementwise<const T, T>(Impl{static_cast<T>(x1)}, x2_cast, out);
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(PowerSAKernel, NativePowerSAKernel);

int64_t Mod(int64_t x, int64_t y) {
    if (x == 0 || y == 0) {
        return 0;
    }
    if (x < 0) {
        if (y > 0) {
            return (y - (-x) % y) % y;
        }
        return -(-x % (-y));
    }
    if (y < 0) {
        return (y + x % (-y)) % y;
    }
    return x % y;
}
int8_t Mod(int8_t x, int8_t y) { return static_cast<int8_t>(Mod(static_cast<int64_t>(x), static_cast<int64_t>(y))); }
int16_t Mod(int16_t x, int16_t y) { return static_cast<int16_t>(Mod(static_cast<int64_t>(x), static_cast<int64_t>(y))); }
int32_t Mod(int32_t x, int32_t y) { return static_cast<int32_t>(Mod(static_cast<int64_t>(x), static_cast<int64_t>(y))); }
uint8_t Mod(uint8_t x, uint8_t y) {
    if (x == 0 || y == 0) {
        return 0;
    }
    return x % y;
}
float Mod(float x, float y) {
    if (x == 0 || y == 0) {
        return 0;
    }
    if (x < 0) {
        if (y > 0) {
            return std::fmod(y - std::fmod(-x, y), y);
        }
        return -std::fmod(-x, -y);
    }
    if (y < 0) {
        return std::fmod(y + std::fmod(x, -y), y);
    }
    return std::fmod(x, y);
}
double Mod(double x, double y) {
    if (x == 0 || y == 0) {
        return 0;
    }
    if (x < 0) {
        if (y > 0) {
            return std::fmod(y - std::fmod(-x, y), y);
        }
        return -std::fmod(-x, -y);
    }
    if (y < 0) {
        return std::fmod(y + std::fmod(x, -y), y);
    }
    return std::fmod(x, y);
}
chainerx::Float16 Mod(chainerx::Float16 x, chainerx::Float16 y) {
    return chainerx::Float16{Mod(static_cast<float>(x), static_cast<float>(y))};
}

CHAINERX_NATIVE_REGISTER_ELTWISE_DTYPE_BINARY_KERNEL(ModAAKernel, { out = Mod(x1, x2); }, VisitNumericDtype);

class NativeModASKernel : public ModASKernel {
public:
    void Call(const Array& x1, Scalar x2, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, out);
        const Array& x1_cast = x1.dtype() == out.dtype() ? x1 : x1.AsType(out.dtype());
        VisitNumericDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                void operator()(int64_t /*i*/, T x1, T& out) { out = Mod(x1, x2); }
                T x2;
            };
            Elementwise<const T, T>(Impl{static_cast<T>(x2)}, x1_cast, out);
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(ModASKernel, NativeModASKernel);

class NativeModSAKernel : public ModSAKernel {
public:
    void Call(Scalar x1, const Array& x2, const Array& out) override {
        Device& device = x2.device();
        device.CheckDevicesCompatible(x2, out);
        const Array& x2_cast = x2.dtype() == out.dtype() ? x2 : x2.AsType(out.dtype());
        VisitNumericDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                void operator()(int64_t /*i*/, T x2, T& out) { out = Mod(x1, x2); }
                T x1;
            };
            Elementwise<const T, T>(Impl{static_cast<T>(x1)}, x2_cast, out);
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(ModSAKernel, NativeModSAKernel);

}  // namespace
}  // namespace native
}  // namespace chainerx
