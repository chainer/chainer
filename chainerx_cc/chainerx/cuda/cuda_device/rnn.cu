#include "chainerx/cuda/cuda_device.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>
#include <iostream>
#include <memory>
#include <tuple>

#include <cuda.h>
#include <cudnn.h>
#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/constant.h"
#include "chainerx/cuda/cuda.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/backend_util.h"
#include "chainerx/cuda/cuda_backend.h"
#include "chainerx/cuda/cuda_device.h"
#include "chainerx/cuda/cuda_set_device_scope.h"
#include "chainerx/cuda/cudnn.h"
#include "chainerx/device.h"
#include "chainerx/cuda/kernel_regist.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/manipulation.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/kernels/rnn.h"
#include "chainerx/kernels/creation.h"
#include "chainerx/kernels/misc.h"
#include "chainerx/native/kernel_regist.h"
#include "chainerx/kernels/connection.h"
#include "chainerx/macro.h"
#include "chainerx/routines/connection.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/manipulation.h"
#include "chainerx/shape.h"
#include "chainerx/stack_vector.h"
#include "chainerx/shape.h"
#include "chainerx/stack_vector.h"
#include "chainerx/kernels/misc.h"


namespace chainerx {
namespace cuda {
namespace {

__global__ void initGPUData_ker(float *data, int numElements, float* value) {
   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   if (tid < numElements) {
      data[tid] = value[tid];
      
   }
}

void initGPUData(float *data, int numElements, float* value) {
   dim3 gridDim;
   dim3 blockDim;
   
   blockDim.x = 1024;
   gridDim.x = (numElements + blockDim.x - 1) / blockDim.x;
   
   initGPUData_ker <<< gridDim, blockDim >>> (data, numElements, value);
}

Shape GetInferredShape(const Shape& shape, int64_t total_size) {
    Shape inferred_shape = shape;

    auto it = std::find_if(inferred_shape.begin(), inferred_shape.end(), [](int64_t dim) { return dim < 0; });
    if (it != inferred_shape.end()) {
        if (std::find_if(std::next(it), inferred_shape.end(), [](int64_t dim) { return dim < 0; }) != inferred_shape.end()) {
            throw DimensionError{"Can only specify one unknown dimension"};
        }
        int64_t rest_size = std::accumulate(inferred_shape.begin(), it, int64_t{1}, std::multiplies<>()) *
                            std::accumulate(std::next(it), inferred_shape.end(), int64_t{1}, std::multiplies<>());
        *it = total_size / rest_size;
    }

    if (total_size != inferred_shape.GetTotalSize()) {
        throw DimensionError{"Cannot reshape array of size ", total_size, " into shape ", shape};
    }
    return inferred_shape;
}


Array reshape(const Array& a, const Shape& newshape) {
    const Shape& in_shape = a.shape();
    const Strides& in_strides = a.strides();

    // If the shape is unchanged, just return a view.
    if (in_shape == newshape) {
        return a.MakeView();
    }

    // Check for invalid shape.
    int64_t total_size = in_shape.GetTotalSize();
    Shape out_shape = GetInferredShape(newshape, total_size);
    int64_t item_size = a.GetItemSize();
    Strides strides{};
    if (total_size == 0) {
        // Calculate the strides for 0-sized array.
        strides.resize(out_shape.ndim());
        strides.back() = item_size;
        for (int8_t i = out_shape.ndim() - 1; i >= 1; --i) {
            strides[i - 1] = strides[i] * std::max(int64_t{1}, out_shape[i]);
        }
    } else {
        // Calculate the strides for non-0-sized array.

        // reduced_shape and reduced_strides are the shortest shape and strides which can be convertible from input shape and strides
        // without copy.
        Shape reduced_shape{};
        Strides reduced_strides{};
        if (total_size == 1) {
            reduced_shape.emplace_back(int64_t{1});
            reduced_strides.emplace_back(item_size);
        } else {
            int8_t i = 0;
            // Ignore preceding 1-length dimensions
            while (i < in_shape.ndim() && in_shape[i] == 1) {
                ++i;
            }
            // Add the first pair
            reduced_shape.emplace_back(in_shape[i]);
            reduced_strides.emplace_back(in_strides[i]);
            ++i;
            // Reduce the remaining
            for (; i < in_shape.ndim(); ++i) {
                int64_t dim = in_shape[i];
                int64_t st = in_strides[i];
                CHAINERX_ASSERT(dim > 0);
                if (dim == 1) {
                    // If the axis has unit-length, skip this dimension.
                } else if (dim * st == reduced_strides.back()) {
                    // If the pair is compatible with the previous stride, reduce the pair to it.
                    reduced_shape.back() *= dim;
                    reduced_strides.back() = st;
                } else {
                    // Otherwise, add a new shape and stride.
                    reduced_shape.emplace_back(dim);
                    reduced_strides.emplace_back(st);
                }
            }
        }
        CHAINERX_ASSERT(reduced_shape.size() == reduced_strides.size());
        CHAINERX_ASSERT(!reduced_shape.empty());

        // Construct the strides for no-copy reshape.
        // If it's not possible, can_reshape_without_copy will be false.
        bool can_reshape_without_copy = true;
        if (out_shape.ndim() > 0) {
            int64_t last_stride = reduced_shape[0] * reduced_strides[0];
            size_t i_dim = 0;
            for (int64_t dim : out_shape) {
                if (dim <= 1) {
                    strides.emplace_back(last_stride);
                    continue;
                }
                if (i_dim >= reduced_shape.size() || reduced_shape[i_dim] % dim != 0) {
                    strides.clear();
                    can_reshape_without_copy = false;
                    break;
                }
                reduced_shape[i_dim] /= dim;
                last_stride = reduced_shape[i_dim] * reduced_strides[i_dim];
                strides.emplace_back(last_stride);
                if (reduced_shape[i_dim] == 1) {
                    ++i_dim;
                }
            }
        }

        if (!can_reshape_without_copy) {
            // Copy is required.
            return a.Copy().Reshape(out_shape);
        }
        CHAINERX_ASSERT(strides.size() == out_shape.size());
    }

    Array out = internal::MakeArray(out_shape, strides, a.dtype(), a.device(), a.data(), a.offset());
    return out;
}
Array concatenate(const std::vector<Array>& arrays, int8_t axis) {
        if (arrays.empty()) {
        throw DimensionError{"Need at least one array to concatenate"};
    }

    Shape shape = arrays.front().shape();
    Dtype out_dtype = arrays[0].dtype();
    Device& device = arrays.front().device();
    int8_t ndim = arrays.front().ndim();
    axis = internal::NormalizeAxis(axis, ndim);
    shape[axis] = 0;
    std::vector<int64_t> indices;
    indices.reserve(arrays.size() - 1);

    for (const Array& array : arrays) {
        const Shape& s = array.shape();
        if (ndim != array.ndim()) {
            throw DimensionError{"All the input arrays must have same number of dimensions"};
        }
        for (int8_t i = 0; i < ndim; ++i) {
            if (axis == i) {
                shape[i] += s[i];
            } else if (shape[i] != s[i]) {
                throw DimensionError{"All the input array dimensions except for the concatenation axis must match exactly"};
            }
        }
        if (indices.size() < arrays.size() - 1) {
            indices.emplace_back(shape[axis]);
        }
    }

    Strides strides{shape, out_dtype};

    // Aligning with NumPy strides behavior
    auto last_zero_it = std::find(shape.rbegin(), shape.rend(), int64_t{0});
    if (last_zero_it != shape.rend()) {
        std::fill(strides.rbegin() + (last_zero_it - shape.rbegin() + 1), strides.rend(), int64_t{0});
    }

    Array out = internal::Empty(shape, out_dtype, strides, device);

    size_t in_size = arrays.size();

    // If input dtypes are mixed, elements in the input arrays are casted to the resulting dtype.
    // Their original dtypes must therefore be remembered in order to cast the computed gradients back in the backward pass.
    std::vector<Dtype> in_dtypes;
    in_dtypes.reserve(in_size);

    std::vector<ConstArrayRef> array_refs;
    array_refs.reserve(in_size);

    {
        int64_t out_offset = 0;
        for (const Array& array : arrays) {
            const Shape& shape = array.shape();
            Array sliced_out = internal::MakeArray(shape, strides, out_dtype, device, out.data(), out_offset);
            Dtype in_dtype = array.dtype();
            in_dtypes.emplace_back(in_dtype);
            // Note: In CopyKernel, Input Array Elements are casted to the type of Output Array.
            device.backend().CallKernel<CopyKernel>(array, sliced_out);
            array_refs.emplace_back(ConstArrayRef{array});
            out_offset += strides[axis] * shape[axis];
        }
    }
    return out;
}

std::vector<Array> split(const Array& ary, std::vector<int64_t> indices, int8_t axis) {
    const Shape& in_shape = ary.shape();
    int8_t axis_norm = internal::NormalizeAxis(axis, ary.ndim());
    int64_t in_dim = in_shape[axis_norm];

    // Wrap negative indices.
    std::transform(
            indices.begin(), indices.end(), indices.begin(), [in_dim](int64_t index) { return index >= 0 ? index : index + in_dim; });
    indices.emplace_back(in_dim);

    Shape out_shape = in_shape;
    int64_t out_stride = ary.strides()[axis_norm];
    int64_t out_offset = ary.offset();
    int64_t slice_start = 0;
    bool is_empty = ary.GetTotalSize() == 0;

    std::vector<Array> out{};
    out.reserve(indices.size());

    for (int64_t index : indices) {
        int64_t slice_stop = std::min(in_dim, std::max(int64_t{0}, index));
        int64_t slice_step = slice_stop - slice_start;

        // Update the dimension of interest in the output shape.
        out_shape[axis_norm] = std::max(int64_t{0}, slice_step);

        out.emplace_back(internal::MakeArray(out_shape, ary.strides(), ary.dtype(), ary.device(), ary.data(), out_offset));

        // Empty arrays should all have offsets of 0 to e.g. avoid out-of-memory errors.
        if (!is_empty) {
            out_offset += out_stride * slice_step;
        }

        slice_start = slice_stop;
    }
    return out;
}

void weights_forward(
    cuda_internal::DeviceInternals& device_internals,
    cudnnRNNDescriptor_t rnn_desc,
    std::vector<std::vector<Array>> ws,
    std::vector<std::vector<Array>> bs,
    int n_layers,
    int num_directions,
    cudnnTensorDescriptor_t x_desc,
    cudnnFilterDescriptor_t w_desc,
    Array& w
    ) {
    for(int layer = 0 ; layer < n_layers; layer++) {
        for(int8_t di = 0; di < num_directions; di++) {
            for(uint lin_layer_id = 0; lin_layer_id < ws[0].size(); lin_layer_id++) {
                int64_t index = num_directions * layer + di;
                
                cudnnFilterDescriptor_t linlayermatdesc;
                cudnnCreateFilterDescriptor(&linlayermatdesc);
                float* m_offset;
                device_internals.cudnn_handle().Call(
                    cudnnGetRNNLinLayerMatrixParams,
                    rnn_desc,
                    index,
                    x_desc,
                    w_desc,
                    internal::GetRawOffsetData(w),
                    lin_layer_id,
                    linlayermatdesc,
                    (void**)&m_offset         
                );
                cudnnDataType_t dataType;
                cudnnTensorFormat_t format;
                int nbDims;
                int filterDimA[3];


                cudnnGetFilterNdDescriptor(
                linlayermatdesc,
                3,
                &dataType,
                &format,
                &nbDims,
                filterDimA
                );
                ws[index][lin_layer_id] = AsContiguous(ws[index][lin_layer_id].AsType(Dtype::kFloat32));
                initGPUData(m_offset, filterDimA[0] * filterDimA[1] * filterDimA[2], (float*)internal::GetRawOffsetData(ws[index][lin_layer_id]));
                cudnnDestroyFilterDescriptor(linlayermatdesc);
                

                cudnnFilterDescriptor_t linlayerbiasdesc;
                cudnnCreateFilterDescriptor(&linlayerbiasdesc);
                float* b_offset;
                device_internals.cudnn_handle().Call(
                    cudnnGetRNNLinLayerBiasParams,
                    rnn_desc,
                    index,
                    x_desc,
                    w_desc,
                    internal::GetRawOffsetData(w),
                    lin_layer_id,
                    linlayerbiasdesc,
                    (void**)&b_offset
                );
                cudnnGetFilterNdDescriptor(
                                        linlayerbiasdesc,
                                        3,
                                        &dataType,
                                        &format,
                                        &nbDims,
                                        filterDimA
                                        );
                bs[index][lin_layer_id] = AsContiguous(bs[index][lin_layer_id].AsType(Dtype::kFloat32));
                initGPUData(b_offset, filterDimA[0] * filterDimA[1] * filterDimA[2], (float*)internal::GetRawOffsetData(bs[index][lin_layer_id]));
                cudnnDestroyFilterDescriptor(linlayerbiasdesc);
            }
        }
    }
}

std::vector<Array> weights_backward(
    CudaDevice& device,
    cudnnRNNDescriptor_t& rnn_desc,
    cudnnTensorDescriptor_t dummy_x_desc,
    cudnnFilterDescriptor_t w_desc,
    Array w,
    std::vector<std::vector<Array>> ws,
    std::vector<std::vector<Array>> bs,
    int64_t n_layers,
    int64_t num_directions,
    Dtype type
    ) {
    cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(device);
    std::vector<Array> ret;
    
    for(int64_t layer = 0 ; layer < n_layers; layer++) {
        for(int8_t di = 0; di < num_directions; di++) {
            for(uint lin_layer_id = 0; lin_layer_id < ws[0].size(); lin_layer_id++) {
                int64_t index = num_directions * layer + di;
                
                
                cudnnFilterDescriptor_t linlayermatdesc;
                cudnnCreateFilterDescriptor(&linlayermatdesc);
                float* m_offset;
                device_internals.cudnn_handle().Call(
                    cudnnGetRNNLinLayerMatrixParams,
                    rnn_desc,
                    index,
                    dummy_x_desc,
                    w_desc,
                    internal::GetRawOffsetData(w),
                    lin_layer_id,
                    linlayermatdesc,
                    (void**)&m_offset
                );
                
                Array m = AsContiguous(Zeros(ws[index][lin_layer_id].shape(), type, ws[index][lin_layer_id].device()));
                cudnnDataType_t dataType;
                cudnnTensorFormat_t format;
                int nbDims;
                int filterDimA[3];


                cudnnGetFilterNdDescriptor(
                linlayermatdesc,
                3,
                &dataType,
                &format,
                &nbDims,
                filterDimA
                );
                initGPUData((float*)internal::GetRawOffsetData(m), filterDimA[0] * filterDimA[1] * filterDimA[2], m_offset);
                cudnnDestroyFilterDescriptor(linlayermatdesc);
                ret.push_back(m);
                
                cudnnFilterDescriptor_t linlayerbiasdesc;
                cudnnCreateFilterDescriptor(&linlayerbiasdesc);
                float* b_offset;
                device_internals.cudnn_handle().Call(
                    cudnnGetRNNLinLayerBiasParams,
                    rnn_desc,
                    index,
                    dummy_x_desc,
                    w_desc,
                    internal::GetRawOffsetData(w),
                    lin_layer_id,
                    linlayerbiasdesc,
                    (void**)&b_offset
                );
                
                Array b = AsContiguous(Zeros(bs[index][lin_layer_id].shape(), type, bs[index][lin_layer_id].device()));
                cudnnGetFilterNdDescriptor(
                                        linlayerbiasdesc,
                                        3,
                                        &dataType,
                                        &format,
                                        &nbDims,
                                        filterDimA
                                        );
                initGPUData((float*)internal::GetRawOffsetData(b), filterDimA[0] * filterDimA[1] * filterDimA[2], b_offset);
                cudnnDestroyFilterDescriptor(linlayerbiasdesc);
                
                ret.push_back(b);
            }
        }
    }
    return ret;
}


class CudaRnnKernel : public RnnKernel {
public:
    std::tuple<std::vector<std::vector<Array>>, std::unique_ptr<chainerx::RnnGradState>> Call(
        int64_t n_layers,
        Array hx,
        nonstd::optional<Array> cx,
        const std::vector<std::vector<Array>>& ws,
        const std::vector<std::vector<Array>>& bs,
        const std::vector<Array>& xs,
        const int8_t bidirectional,
        const int8_t mode) override {

        CudaDevice& device = dynamic_cast<CudaDevice&>(hx.device());
        CudaSetDeviceScope scope{device.index()};
        auto& backend = static_cast<CudaBackend&>(device.backend());  // NOLINT
        Dtype type = hx.dtype();
        cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(device);


        const auto input_dim = xs[0].shape()[1];
        const auto hidden_dim = hx.shape()[2];
        const auto num_directions = bidirectional == 1 ? 2 : 1;
        const auto num_layers = n_layers;
        const auto rnn_direction = bidirectional == 1 ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL;
        const auto rnn_mode = mode == 1? CUDNN_LSTM : CUDNN_GRU ;
        const auto rnn_input = CUDNN_LINEAR_INPUT;
        cudnnRNNDescriptor_t rnn_desc;
        unsigned long long seed = 1337ull; 

        cudnnDropoutDescriptor_t dropoutDesc;
        cudnnCreateDropoutDescriptor(&dropoutDesc);

        size_t stateSize;
        void *states;
        device_internals.cudnn_handle().Call(cudnnDropoutGetStatesSize, &stateSize);

        cudaMalloc(&states, stateSize);

        cudnnSetDropoutDescriptor(dropoutDesc,
                                 device_internals.cudnn_handle().handle(),
                                 0,
                                 states,
                                 stateSize,
                                 seed);
        

        cudnnCreateRNNDescriptor(&rnn_desc);
        device_internals.cudnn_handle().Call(
            cudnnSetRNNDescriptor,
            rnn_desc,
            hidden_dim,
            num_layers,
            dropoutDesc,
            rnn_input,
            rnn_direction,
            rnn_mode,
            CUDNN_RNN_ALGO_STANDARD,
            CUDNN_DATA_FLOAT
        );
        
        cudnnTensorDescriptor_t *x_desc, *y_desc;
        x_desc = (cudnnTensorDescriptor_t*)malloc(xs.size() * sizeof(cudnnTensorDescriptor_t));
        y_desc = (cudnnTensorDescriptor_t*)malloc(xs.size() * sizeof(cudnnTensorDescriptor_t));
        std::vector<cuda_internal::CudnnTensorDescriptor> xs_desc;
        std::vector<cuda_internal::CudnnTensorDescriptor> ys_desc;
        std::vector<Array> ys;
        std::vector<Array> xs_cont;
        for(uint i = 0; i < xs.size(); i++) {
            Shape xs_shape{xs[i].shape()[0], xs[i].shape()[1], 1};
            Shape ys_shape{xs[i].shape()[0], num_directions * hidden_dim, 1};
            xs_cont.push_back(AsContiguous(xs[i].AsType(Dtype::kFloat32)));
            ys.push_back(AsContiguous(Zeros({xs_cont[i].shape()[0], num_directions * hidden_dim}, xs_cont[i].dtype(), xs_cont[i].device())));
            xs_desc.push_back(cuda_internal::CudnnTensorDescriptor(reshape(xs_cont[i], xs_shape)));
            ys_desc.push_back(cuda_internal::CudnnTensorDescriptor(reshape(ys[i], ys_shape)));
            x_desc[i] = *xs_desc[i];
            y_desc[i] = *ys_desc[i];
        }

        
        Array x = concatenate(xs_cont, 0);
        Array y = concatenate(ys, 0);


        size_t weight_size;
        device_internals.cudnn_handle().Call(
            cudnnGetRNNParamsSize,
            rnn_desc,
            x_desc[0],
            &weight_size,
            CUDNN_DATA_FLOAT
        );



        Array w = AsContiguous(Zeros({(int)weight_size / 4, 1, 1}, x.dtype(), x.device()));
        cuda_internal::CudnnFilterDescriptor wDesc{w};

        weights_forward(device_internals, rnn_desc, ws, bs, n_layers, num_directions, x_desc[0], *wDesc, w);

        
        
        size_t workSize;
        
        size_t reserve_size;
        device_internals.cudnn_handle().Call(cudnnGetRNNWorkspaceSize, rnn_desc, xs.size(), x_desc, &workSize);
        Array workspace = AsContiguous(Zeros({(long)workSize}, hx.dtype(), hx.device()));

        device_internals.cudnn_handle().Call(cudnnGetRNNTrainingReserveSize, rnn_desc, xs.size(), x_desc, &reserve_size);
        Array reserve = AsContiguous(Zeros({(long)reserve_size}, hx.dtype(), hx.device()));
        hx = AsContiguous(hx.AsType(Dtype::kFloat32));
        
        Array hy = AsContiguous(Zeros(hx.shape(), hx.dtype(), hx.device()));
        Array cy = AsContiguous(Zeros(hx.shape(), hx.dtype(), hx.device()));
        cuda_internal::CudnnTensorDescriptor hxDesc{hx};
        
        if(cx.has_value())
        {
            *cx = AsContiguous((*cx).AsType(Dtype::kFloat32));
        } else {
            *cx = Zeros(hx.shape(), hx.dtype(), hx.device());
        }
        cuda_internal::CudnnTensorDescriptor cxDesc{*cx};
        cuda_internal::CudnnTensorDescriptor hyDesc{hy};
        cuda_internal::CudnnTensorDescriptor cyDesc{cy};
        device_internals.cudnn_handle().Call(
            cudnnRNNForwardTraining,
            rnn_desc,
            xs.size(),
            x_desc,
            internal::GetRawOffsetData(x),
            *hxDesc,
            internal::GetRawOffsetData(hx),
            *cxDesc,
            internal::GetRawOffsetData(*cx),
            *wDesc,
            internal::GetRawOffsetData(w),
            y_desc,
            internal::GetRawOffsetData(y),
            *hyDesc,
            internal::GetRawOffsetData(hy),
            *cyDesc,
            internal::GetRawOffsetData(cy),
            internal::GetRawOffsetData(workspace),
            workSize,
            internal::GetRawOffsetData(reserve),
            reserve_size
        );

        std::vector<int64_t> split_indices;
        for(uint i = 0; i < xs.size() - 1; i++){
            if ( i != 0 ) {
                split_indices.push_back(split_indices[i-1] + xs[i].shape()[0]);
            } else {
                split_indices.push_back(xs[i].shape()[0]);
            }
        }

        std::unique_ptr<RnnGradState> state = std::make_unique<GenericRnnGradState>(rnn_desc, *wDesc, w, reserve, workspace);
        y = y.AsType(type);
        ys = split(y, split_indices, 0);

        std::vector<Array> out_states;
        out_states.push_back(hy.AsType(type));
        if(cx.has_value())
        {
            out_states.push_back(cy.AsType(type));    
        }
        std::vector<std::vector<Array>> ret;
        ret.push_back(out_states);
        ret.push_back(ys);
        return std::make_tuple(std::move(ret), std::move(state));        
        }
};

CHAINERX_CUDA_REGISTER_KERNEL(RnnKernel, CudaRnnKernel);

class CudaRnnBackwardKernel : public RnnBackwardKernel {
public:
    std::vector<std::vector<Array>> Call(
        int64_t n_layers,
        Array hx,
        nonstd::optional<Array> cx,
        const std::vector<std::vector<Array>>& ws,
        const std::vector<std::vector<Array>>& bs,
        const std::vector<Array>& xs,
        Array dhy,
        nonstd::optional<Array> dcy,
        std::vector<Array> ys,
        std::vector<Array> dys,
        const int8_t bidirectional,
        const std::shared_ptr<chainerx::RnnGradState>& state
        ) override {
        CudaDevice& device = dynamic_cast<CudaDevice&>(hx.device());
        CudaSetDeviceScope scope{device.index()};

        cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(device);
        auto cuda_state = dynamic_cast<GenericRnnGradState&>(*state);
        Dtype type = hx.dtype();
        const auto input_dim = xs[0].shape()[1];
        const auto hidden_dim = hx.shape()[2];
        const auto num_directions = bidirectional == 1 ? 2 : 1;
        

        cudnnRNNDescriptor_t rnn_desc = cuda_state.rnn_desc();
       
        std::vector<Array> dxs;
        std::vector<cuda_internal::CudnnTensorDescriptor> xsDesc, dxsDesc, ysDesc, dysDesc; 
        cudnnTensorDescriptor_t *xs_desc = (cudnnTensorDescriptor_t*)malloc(xs.size() * sizeof(cudnnTensorDescriptor_t));
        cudnnTensorDescriptor_t *dxs_desc = (cudnnTensorDescriptor_t*)malloc(xs.size() * sizeof(cudnnTensorDescriptor_t));
        cudnnTensorDescriptor_t *ys_desc = (cudnnTensorDescriptor_t*)malloc(xs.size() * sizeof(cudnnTensorDescriptor_t));
        cudnnTensorDescriptor_t *dys_desc = (cudnnTensorDescriptor_t*)malloc(xs.size() * sizeof(cudnnTensorDescriptor_t)); 
        std::vector<Array> xs_cont;
        for(uint i = 0; i < xs.size(); i++) {
            Shape xs_shape{xs[i].shape()[0], xs[i].shape()[1], 1};
            Shape ys_shape{ys[i].shape()[0], ys[i].shape()[1], 1};

            xs_cont.push_back(AsContiguous(xs[i].AsType(Dtype::kFloat32)));
            ys[i] = AsContiguous(ys[i].AsType(Dtype::kFloat32));
            dys[i] = AsContiguous(dys[i].AsType(Dtype::kFloat32));
            
            xsDesc.push_back(cuda_internal::CudnnTensorDescriptor(reshape(xs_cont[i], xs_shape)));
            xs_desc[i] = *xsDesc[i];
            
            dxs.push_back(AsContiguous(Zeros(xs_cont[i].shape(), xs_cont[i].dtype(), xs_cont[i].device())));
            
            dxsDesc.push_back(cuda_internal::CudnnTensorDescriptor(reshape(dxs[i], xs_shape)));
            
            dxs_desc[i] = *dxsDesc[i];
            ysDesc.push_back(cuda_internal::CudnnTensorDescriptor(reshape(ys[i], ys_shape)));
            ys_desc[i] = *ysDesc[i];
            dysDesc.push_back(cuda_internal::CudnnTensorDescriptor(reshape(dys[i], ys_shape)));
            dys_desc[i] = *dysDesc[i];
        }
        Array dx = AsContiguous(concatenate(dxs, 0));
        Array x = AsContiguous(concatenate(xs_cont, 0));
        Array y = AsContiguous(concatenate(ys, 0));
        Array dy = AsContiguous(concatenate(dys, 0));
        cudnnFilterDescriptor_t wDesc = cuda_state.wDesc();
        Array w = AsContiguous(cuda_state.w());
        Array reserve = AsContiguous(cuda_state.reserve());
        Array workspace = AsContiguous(cuda_state.workspace());
        size_t reserve_size = reserve.shape()[0];
        size_t workSize = workspace.shape()[0];
        hx = AsContiguous(hx.AsType(Dtype::kFloat32));
        
        dhy = AsContiguous(dhy.AsType(Dtype::kFloat32));
        

        cuda_internal::CudnnTensorDescriptor hx_desc{hx};
        
        if(cx.has_value()) {
            *cx = AsContiguous((*cx).AsType(Dtype::kFloat32));
            *dcy = AsContiguous((*dcy).AsType(Dtype::kFloat32));
        } else {
            *cx = Zeros(hx.shape(), hx.dtype(), hx.device());
            *dcy = Zeros(hx.shape(), hx.dtype(), hx.device());
        }
        cuda_internal::CudnnTensorDescriptor cx_desc{*cx};
        cuda_internal::CudnnTensorDescriptor dcy_desc{*dcy};
        cuda_internal::CudnnTensorDescriptor dhy_desc{dhy};
        Array dhx = AsContiguous(Zeros(hx.shape(), hx.dtype(), hx.device()));
        cuda_internal::CudnnTensorDescriptor dhx_desc{dhx};
        Array dcx = AsContiguous(Zeros(hx.shape(), hx.dtype(), hx.device()));
        cuda_internal::CudnnTensorDescriptor dcx_desc{dcx};
        reserve = AsContiguous(reserve);
        

        device_internals.cudnn_handle().Call(
            cudnnRNNBackwardData,
            rnn_desc,
            xs.size(),
            ys_desc,
            internal::GetRawOffsetData(y),
            dys_desc,
            internal::GetRawOffsetData(dy),
            *dhy_desc,
            internal::GetRawOffsetData(dhy),
            *dcy_desc,
            internal::GetRawOffsetData(*dcy),
            wDesc,
            internal::GetRawOffsetData(w),
            *hx_desc,
            internal::GetRawOffsetData(hx),
            *cx_desc,
            internal::GetRawOffsetData(*cx),
            dxs_desc,
            internal::GetRawOffsetData(dx),
            *dhx_desc,
            internal::GetRawOffsetData(dhx),
            *dcx_desc,
            internal::GetRawOffsetData(dcx),
            internal::GetRawOffsetData(workspace),
            workSize,
            internal::GetRawOffsetData(reserve),
            reserve_size
        );

        Array dw = AsContiguous(Zeros(w.shape(), hx.dtype(), hx.device()));
        cuda_internal::CudnnFilterDescriptor dwDesc{dw};
        
        device_internals.cudnn_handle().Call(
            cudnnRNNBackwardWeights,
            rnn_desc,
            xs.size(),
            xs_desc,
            internal::GetRawOffsetData(x),
            *hx_desc,
            internal::GetRawOffsetData(hx),
            ys_desc,
            internal::GetRawOffsetData(y),
            internal::GetRawOffsetData(workspace),
            workSize,
            *dwDesc,
            internal::GetRawOffsetData(dw),
            internal::GetRawOffsetData(reserve),
            reserve_size
        );

        std::vector<int64_t> split_indices;
        for(uint i = 0; i < xs.size() - 1; i++){
            if ( i != 0 ) {
                split_indices.push_back(split_indices[i - 1] + xs[i].shape()[0]);
            } else {
                split_indices.push_back(xs[i].shape()[0]);
            }
        }
        dx = dx.AsType(type);
        dxs = split(dx, split_indices, 0);
        std::vector<Array> dstate;
        dstate.push_back(dhx.AsType(type));


        if(cx.has_value())
        {
            dstate.push_back(dcx.AsType(type));    
        }
        
        std::vector<std::vector<Array>> ret;
        ret.push_back(dstate);
        ret.push_back(weights_backward(device, rnn_desc, dxs_desc[0], *dwDesc, dw, ws, bs, n_layers, num_directions, type));
        ret.push_back(dxs);
        return ret;
        
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(RnnBackwardKernel, CudaRnnBackwardKernel);

}  // namespace
}  // namespace cuda
}  // namespace chainerx
