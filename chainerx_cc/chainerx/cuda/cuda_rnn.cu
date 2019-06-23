#include "chainerx/cuda/cuda_rnn.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>
#include <iostream>

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/cuda/copy_data.cuh"
#include "chainerx/cuda/cuda.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/backend_util.h"
#include "chainerx/cuda/cuda_backend.h"
#include "chainerx/cuda/cuda_device.h"
#include "chainerx/cuda/cuda_set_device_scope.h"
#include "chainerx/cuda/cudnn.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/hash_combine.h"
#include "chainerx/kernels/connection.h"
#include "chainerx/macro.h"
#include "chainerx/routines/connection.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/manipulation.h"
#include "chainerx/shape.h"
#include "chainerx/stack_vector.h"

namespace chainerx {
namespace cuda {
namespace cuda_internal {


float *reserve;

size_t reserve_size;


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
    CudnnHandle& handle,
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
                handle.Call(
                    cudnnGetRNNLinLayerMatrixParams,
                    rnn_desc,
                    layer,
                    x_desc,
                    w_desc,
                    internal::GetRawOffsetData(AsContiguous(w)),
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
                ws[index][lin_layer_id] = ws[index][lin_layer_id].AsType(Dtype::kFloat32);
                initGPUData(m_offset, filterDimA[0] * filterDimA[1] * filterDimA[2], (float*)internal::GetRawOffsetData(AsContiguous(ws[index][lin_layer_id])));
                cudnnDestroyFilterDescriptor(linlayermatdesc);
                

                cudnnFilterDescriptor_t linlayerbiasdesc;
                cudnnCreateFilterDescriptor(&linlayerbiasdesc);
                float* b_offset;
                handle.Call(
                    cudnnGetRNNLinLayerBiasParams,
                    rnn_desc,
                    layer,
                    x_desc,
                    w_desc,
                    internal::GetRawOffsetData(AsContiguous(w)),
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
                bs[index][lin_layer_id] = bs[index][lin_layer_id].AsType(Dtype::kFloat32);
                initGPUData(b_offset, filterDimA[0] * filterDimA[1] * filterDimA[2], (float*)internal::GetRawOffsetData(AsContiguous(bs[index][lin_layer_id])));
                cudnnDestroyFilterDescriptor(linlayerbiasdesc);
            }
        }
    }
}

std::vector<std::vector<Array>> CudaRnn::n_step_rnn(
        CudaDevice& device,
        int64_t n_layers,
        Array hx,
        Array cx,
        const std::vector<std::vector<Array>>& ws,
        const std::vector<std::vector<Array>>& bs,
        std::vector<Array>& xs,
        const int8_t bidirectional,
        const int8_t mode) {

    CudaSetDeviceScope scope{device.index()};
    auto& backend = static_cast<CudaBackend&>(device.backend());  // NOLINT
    Dtype type = hx.dtype();
    cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(device);

    CudnnHandle& handle = device_internals.cudnn_handle();

    const auto input_dim = xs[0].shape()[1];
    const auto hidden_dim = hx.shape()[2];
    const auto num_directions = bidirectional == 1 ? 2 : 1;
    const auto num_layers = n_layers;
    const auto rnn_direction = bidirectional == 1 ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL;
    const auto rnn_mode = mode == 1? CUDNN_LSTM : CUDNN_GRU ;
    const auto rnn_input = CUDNN_LINEAR_INPUT;
    cudnnDropoutDescriptor_t dropoutDesc;
    cudnnRNNDescriptor_t rnn_desc;

    unsigned long long seed = 1337ull; 

   
   cudnnCreateDropoutDescriptor(&dropoutDesc);

   
   size_t stateSize;
   void *states;
   handle.Call(cudnnDropoutGetStatesSize, &stateSize);

   cudaMalloc(&states, stateSize);

   cudnnSetDropoutDescriptor(dropoutDesc,
                             handle.handle(),
                             0,
                             states,
                             stateSize,
                             seed);

    cudnnCreateRNNDescriptor(&rnn_desc);
    handle.Call(
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
    std::vector<CudnnTensorDescriptor> xs_desc;
    std::vector<CudnnTensorDescriptor> ys_desc;
    std::vector<Array> ys;
    for(uint i = 0; i < xs.size(); i++) {
        Shape xs_shape{xs[i].shape()[0], xs[i].shape()[1], 1};
        Shape ys_shape{xs[i].shape()[0], num_directions * hidden_dim, 1};
        xs[i] = xs[i].AsType(Dtype::kFloat32);
        ys.push_back(Empty({xs[i].shape()[0], num_directions * hidden_dim}, xs[i].dtype(), xs[i].device()));
        xs_desc.push_back(CudnnTensorDescriptor(AsContiguous(xs[i]).Reshape(xs_shape)));
        ys_desc.push_back(CudnnTensorDescriptor(AsContiguous(ys[i]).Reshape( ys_shape)));
        x_desc[i] = *xs_desc[i];
        y_desc[i] = *ys_desc[i];
    }

    
    Array x = Concatenate(xs, 0);
    Array y = Concatenate(ys, 0);


    size_t weight_size;
    handle.Call(
        cudnnGetRNNParamsSize,
        rnn_desc,
        x_desc[0],
        &weight_size,
        CUDNN_DATA_FLOAT
    );



    Array w = Empty({(int)weight_size / 4, 1, 1}, x.dtype(), x.device());
    CudnnFilterDescriptor wDesc{w};

    weights_forward(handle, rnn_desc, ws, bs, n_layers, num_directions, x_desc[0], *wDesc, w);

    
    void *workspace;
    size_t workSize;
    handle.Call(cudnnGetRNNWorkspaceSize, rnn_desc, xs.size(), x_desc, &workSize);
    cudaMalloc((void**)&workspace, workSize);

    handle.Call(cudnnGetRNNTrainingReserveSize, rnn_desc, xs.size(), x_desc, &reserve_size);
    cudaMallocManaged((void**)&reserve, reserve_size);
    hx = hx.AsType(Dtype::kFloat32);
    cx = cx.AsType(Dtype::kFloat32);
    Array hy = Empty(hx.shape(), hx.dtype(), hx.device());
    Array cy = Empty(cx.shape(), cx.dtype(), cx.device());
    CudnnTensorDescriptor hxDesc{AsContiguous(hx)};
    CudnnTensorDescriptor cxDesc{AsContiguous(cx)};

    CudnnTensorDescriptor hyDesc{hy};
    CudnnTensorDescriptor cyDesc{cy};

    handle.Call(
        cudnnRNNForwardTraining,
        rnn_desc,
        xs.size(),
        x_desc,
        internal::GetRawOffsetData(AsContiguous(x)),
        *hxDesc,
        internal::GetRawOffsetData(AsContiguous(hx)),
        *cxDesc,
        internal::GetRawOffsetData(AsContiguous(cx)),
        *wDesc,
        internal::GetRawOffsetData(w),
        y_desc,
        internal::GetRawOffsetData(y),
        *hyDesc,
        internal::GetRawOffsetData(hy),
        *cyDesc,
        internal::GetRawOffsetData(cy),
        workspace,
        workSize,
        reserve,
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
    y = y.AsType(type);
    ys = split(y, split_indices, 0);

    std::vector<Array> out_states;
    out_states.push_back(hy.AsType(type));
    out_states.push_back(cy.AsType(type));
    std::vector<std::vector<Array>> ret;
    ret.push_back(out_states);
    ret.push_back(ys);
    return ret;
}

std::vector<Array> weights_backward(
    CudaDevice &device,
    cudnnRNNDescriptor_t& rnn_desc,
    cudnnTensorDescriptor_t dummy_x_desc,
    cudnnFilterDescriptor_t w_desc,
    Array w,
    const std::vector<std::vector<Array>> ws,
    const std::vector<std::vector<Array>> bs,
    int64_t n_layers,
    int64_t num_directions,
    Dtype type
    ) {
    cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(device);
    std::vector<Array> ret;
    CudnnHandle& handle = device_internals.cudnn_handle();
    
    for(int64_t layer = 0 ; layer < n_layers; layer++) {
        for(int8_t di = 0; di < num_directions; di++) {
            for(uint lin_layer_id = 0; lin_layer_id < ws[0].size(); lin_layer_id++) {
                int64_t index = num_directions * layer + di;
                
                
                cudnnFilterDescriptor_t linlayermatdesc;
                cudnnCreateFilterDescriptor(&linlayermatdesc);
                float* m_offset;
                handle.Call(
                    cudnnGetRNNLinLayerMatrixParams,
                    rnn_desc,
                    layer,
                    dummy_x_desc,
                    w_desc,
                    internal::GetRawOffsetData(w),
                    lin_layer_id,
                    linlayermatdesc,
                    (void**)&m_offset
                );
                
                Array m = Empty(ws[index][lin_layer_id].shape(), ws[index][lin_layer_id].dtype(), ws[index][lin_layer_id].device());
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
                ret.push_back(m.AsType(type));
                
                cudnnFilterDescriptor_t linlayerbiasdesc;
                cudnnCreateFilterDescriptor(&linlayerbiasdesc);
                float* b_offset;
                handle.Call(
                    cudnnGetRNNLinLayerBiasParams,
                    rnn_desc,
                    layer,
                    dummy_x_desc,
                    w_desc,
                    internal::GetRawOffsetData(w),
                    lin_layer_id,
                    linlayerbiasdesc,
                    (void**)&b_offset
                );
                
                Array b = Empty(bs[index][lin_layer_id].shape(), bs[index][lin_layer_id].dtype(), bs[index][lin_layer_id].device());
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
                
                ret.push_back(b.AsType(type));
            }
        }
    }
    return ret;
}

std::vector<std::vector<Array>> CudaRnn::n_step_rnn_backward(
        CudaDevice& device,
        int64_t n_layers,
        Array hx,
        Array cx,
        const std::vector<std::vector<Array>>& ws,
        const std::vector<std::vector<Array>>& bs,
        std::vector<Array>& xs,
        Array dhy,
        Array dcy,
        std::vector<Array> ys,
        std::vector<Array> dys,
        const int8_t bidirectional,
        const int8_t mode) {
    CudaSetDeviceScope scope{device.index()};

    cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(device);

    CudnnHandle& handle = device_internals.cudnn_handle();
    Dtype type = hx.dtype();
    const auto input_dim = xs[0].shape()[1];
    const auto hidden_dim = hx.shape()[2];
    const auto num_directions = bidirectional == 1 ? 2 : 1;
    const auto num_layers = n_layers;
    const auto rnn_direction = bidirectional == 1 ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL;
    const auto rnn_mode = mode == 1 ? CUDNN_LSTM : CUDNN_GRU;
    const auto rnn_input = CUDNN_LINEAR_INPUT;
    unsigned long long seed = 1337ull; 

   cudnnDropoutDescriptor_t dropoutDesc;
   cudnnCreateDropoutDescriptor(&dropoutDesc);

   
   size_t stateSize;
   void *states;
   handle.Call(cudnnDropoutGetStatesSize, &stateSize);

   cudaMalloc(&states, stateSize);

   cudnnSetDropoutDescriptor(dropoutDesc,
                             handle.handle(),
                             0,
                             states,
                             stateSize,
                             seed);


    cudnnRNNDescriptor_t rnn_desc;
    cudnnCreateRNNDescriptor(&rnn_desc);
    handle.Call(
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
    std::vector<Array> dxs;
    std::vector<CudnnTensorDescriptor> xsDesc, dxsDesc, ysDesc, dysDesc; 
    cudnnTensorDescriptor_t *xs_desc = (cudnnTensorDescriptor_t*)malloc(xs.size() * sizeof(cudnnTensorDescriptor_t));
    cudnnTensorDescriptor_t *dxs_desc = (cudnnTensorDescriptor_t*)malloc(xs.size() * sizeof(cudnnTensorDescriptor_t));
    cudnnTensorDescriptor_t *ys_desc = (cudnnTensorDescriptor_t*)malloc(xs.size() * sizeof(cudnnTensorDescriptor_t));
    cudnnTensorDescriptor_t *dys_desc = (cudnnTensorDescriptor_t*)malloc(xs.size() * sizeof(cudnnTensorDescriptor_t)); 

    for(uint i = 0; i < xs.size(); i++) {
        Shape xs_shape{xs[i].shape()[0], xs[i].shape()[1], 1};
        Shape ys_shape{ys[i].shape()[0], ys[i].shape()[1], 1};
        xs[i] = xs[i].AsType(Dtype::kFloat32);
        ys[i] = ys[i].AsType(Dtype::kFloat32);
        dys[i] = dys[i].AsType(Dtype::kFloat32);
        xsDesc.push_back(CudnnTensorDescriptor(AsContiguous(xs[i]).Reshape(xs_shape)));
        xs_desc[i] = *xsDesc[i];
        dxs.push_back(Empty(xs[i].shape(), xs[i].dtype(), xs[i].device()));
        dxsDesc.push_back(CudnnTensorDescriptor(AsContiguous(dxs[i]).Reshape(xs_shape)));
        dxs_desc[i] = *dxsDesc[i];
        ysDesc.push_back(CudnnTensorDescriptor(AsContiguous(ys[i]).Reshape(ys_shape)));
        ys_desc[i] = *ysDesc[i];
        dysDesc.push_back(CudnnTensorDescriptor(AsContiguous(dys[i]).Reshape(ys_shape)));
        dys_desc[i] = *dysDesc[i];
    }
    Array dx = Concatenate(dxs, 0);
    Array x = Concatenate(xs, 0);
    Array y = Concatenate(ys, 0);
    Array dy = Concatenate(dys, 0);
    size_t weight_size;
    handle.Call(
        cudnnGetRNNParamsSize,
        rnn_desc,
        xs_desc[0],
        &weight_size,
        CUDNN_DATA_FLOAT
    );
    Array w = Empty({(int)weight_size / 4, 1, 1}, x.dtype(), x.device());
    CudnnFilterDescriptor wDesc{w};
    
    
    weights_forward(handle, rnn_desc, ws, bs, n_layers, num_directions, xs_desc[0], *wDesc, w);

    void *workspace;
    size_t workSize;
    handle.Call(cudnnGetRNNWorkspaceSize, rnn_desc, xs.size(), xs_desc, &workSize);
    cudaMalloc((void**)&workspace, workSize);

    hx = hx.AsType(Dtype::kFloat32);
    cx = cx.AsType(Dtype::kFloat32);
    dhy = dhy.AsType(Dtype::kFloat32);
    dcy = dcy.AsType(Dtype::kFloat32);

    CudnnTensorDescriptor hx_desc{AsContiguous(hx)};
    CudnnTensorDescriptor cx_desc{AsContiguous(cx)};
    CudnnTensorDescriptor dhy_desc{AsContiguous(dhy)};
    CudnnTensorDescriptor dcy_desc{AsContiguous(dcy)};

    Array dhx = Empty(hx.shape(), hx.dtype(), hx.device());
    CudnnTensorDescriptor dhx_desc{dhx};
    Array dcx = Empty(cx.shape(), cx.dtype(), cx.device());
    CudnnTensorDescriptor dcx_desc{dcx};
    
    handle.Call(
        cudnnRNNBackwardData,
        rnn_desc,
        xs.size(),
        ys_desc,
        internal::GetRawOffsetData(AsContiguous(y)),
        dys_desc,
        internal::GetRawOffsetData(AsContiguous(dy)),
        *dhy_desc,
        internal::GetRawOffsetData(AsContiguous(dhy)),
        *dcy_desc,
        internal::GetRawOffsetData(AsContiguous(dcy)),
        *wDesc,
        internal::GetRawOffsetData(w),
        *hx_desc,
        internal::GetRawOffsetData(AsContiguous(hx)),
        *cx_desc,
        internal::GetRawOffsetData(AsContiguous(cx)),
        dxs_desc,
        internal::GetRawOffsetData(AsContiguous(dx)),
        *dhx_desc,
        internal::GetRawOffsetData(dcx),
        *dcx_desc,
        internal::GetRawOffsetData(dhx),
        workspace,
        workSize,
        reserve,
        reserve_size
    );

    Array dw = Empty({(int)weight_size / 4, 1, 1}, hx.dtype(), hx.device());
    CudnnFilterDescriptor dwDesc{AsContiguous(dw)};
    
    handle.Call(
        cudnnRNNBackwardWeights,
        rnn_desc,
        xs.size(),
        xs_desc,
        internal::GetRawOffsetData(AsContiguous(x)),
        *hx_desc,
        internal::GetRawOffsetData(AsContiguous(hx)),
        ys_desc,
        internal::GetRawOffsetData(AsContiguous(y)),
        workspace,
        workSize,
        *dwDesc,
        internal::GetRawOffsetData(dw),
        reserve,
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
    std::vector<Array> state;
    state.push_back(dhx.AsType(type));
    state.push_back(dcx.AsType(type));
    std::vector<std::vector<Array>> ret;
    ret.push_back(state);
    ret.push_back(weights_backward(device, rnn_desc, dxs_desc[0], *dwDesc, dw, ws, bs, n_layers, num_directions, type));
    ret.push_back(dxs);
    return ret;

}
}  // namespace cuda_internal
}  // namespace cuda
}  // namespace chainerx
