#define CHAINERX_CUDA_REGISTER_DTYPE_ELTWISE_UNARY_OP(func, func_def, visit_dtype)      \
                                                                                        \
    template <typename T>                                                               \
    struct func##Impl {                                                                 \
        using CudaType = cuda_internal::DataType<T>;                                    \
        __device__ void operator()(int64_t i, CudaType x, CudaType& out) func_def       \
    };                                                                                  \
                                                                                        \
    class Cuda##func##Op : public func##Op {                                            \
    public:                                                                             \
        void Call(const Array& x, const Array& out) override {                          \
            Device& device = x.device();                                                \
            device.CheckDevicesCompatible(x, out);                                      \
            CudaSetDeviceScope scope{device.index()};                                   \
            const Array& x_cast = x.dtype() == out.dtype() ? x : x.AsType(out.dtype()); \
            visit_dtype(out.dtype(), [&](auto pt) {                                     \
                using T = typename decltype(pt)::type;                                  \
                Elementwise<const T, T>(func##Impl<T>{}, x_cast, out);                  \
            });                                                                         \
        }                                                                               \
    };                                                                                  \
                                                                                        \
    CHAINERX_REGISTER_OP_CUDA(func##Op, Cuda##func##Op)

#define CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_UNARY_OP(func, func_def) \
    CHAINERX_CUDA_REGISTER_DTYPE_ELTWISE_UNARY_OP(func, func_def, VisitFloatingPointDtype)

#define CHAINERX_CUDA_REGISTER_ELTWISE_UNARY_OP(func, func_def) CHAINERX_CUDA_REGISTER_DTYPE_ELTWISE_UNARY_OP(func, func_def, VisitDtype)
