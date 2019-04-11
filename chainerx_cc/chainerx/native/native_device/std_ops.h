#define CHAINERX_NATIVE_REGISTER_DTYPE_ELTWISE_UNARY_OP(func, func_def, visit_dtype) \
    class Native##func##Op : public func##Op {                                       \
    public:                                                                          \
        void Call(const Array& x, const Array& out) override {                       \
            Device& device = x.device();                                             \
            device.CheckDevicesCompatible(x, out);                                   \
            visit_dtype(out.dtype(), [&](auto pt) {                                  \
                using T = typename decltype(pt)::type;                               \
                struct Impl {                                                        \
                    void operator()(int64_t i, T x, T& out) { out = x; }             \
                };                                                                   \
                Elementwise<const T, T>(Impl{}, x, out);                             \
            });                                                                      \
        }                                                                            \
    };                                                                               \
                                                                                     \
    CHAINERX_REGISTER_OP_NATIVE(func##Op, Native##func##Op);

#define CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_OP(func, func_def) \
    CHAINERX_NATIVE_REGISTER_DTYPE_ELTWISE_UNARY_OP(func, func_def, VisitFloatingPointDtype)

#define CHAINERX_NATIVE_REGISTER_ELTWISE_UNARY_OP(func, func_def) CHAINERX_NATIVE_REGISTER_DTYPE_ELTWISE_UNARY_OP(func, func_def, VisitDtype)