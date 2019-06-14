#include "chainerx/cuda/cuda_rnn.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
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


void *reserve;

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

    size_t max_workspace_size = backend.GetCudnnMaxWorkspaceSize();

    cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(device);

    CudnnHandle& handle = device_internals.cudnn_handle();

    const auto input_dim = xs[0].shape()[1];
    const auto hidden_dim = hx.shape()[2];
    const auto num_directions = bidirectional == 1 ? 2 : 1;
    const auto num_layers = n_layers;
    const auto rnn_direction = bidirectional == 1 ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL;
    const auto rnn_mode = mode == 1 ? CUDNN_LSTM : CUDNN_GRU;
    const auto rnn_input = CUDNN_LINEAR_INPUT;
    cudnnRNNDescriptor_t rnn_desc;
    cudnnCreateRNNDescriptor(&rnn_desc);
    handle.Call(
        cudnnSetRNNDescriptor,
        rnn_desc,
        hidden_dim,
        num_layers,
        (cudnnDropoutDescriptor_t)NULL,
        rnn_input,
        rnn_direction,
        rnn_mode,
        CUDNN_RNN_ALGO_STANDARD,
        CUDNN_DATA_FLOAT
    );
    std::vector<CudnnTensorDescriptor> xs_desc;
    std::vector<cudnnTensorDescriptor_t> xs_desc_arr;
    Array x = Concatenate(xs, 0);
    for(uint i = 0; i < xs.size(); i++) {
        Shape xs_shape{xs[i].shape()[0], xs[i].shape()[1], 1};
        xs_desc.push_back(CudnnTensorDescriptor(AsContiguous(Reshape(xs[i], xs_shape))));
        xs_desc_arr.push_back(*xs_desc[i]);
    }
    size_t weight_size;
    handle.Call(
        cudnnGetRNNParamsSize,
        rnn_desc,
        *xs_desc[0],
        &weight_size,
        CUDNN_DATA_FLOAT
    );

    Shape dummy_x_shape{1, input_dim, 1};
    Array dummy_x = Empty(dummy_x_shape, xs[0].dtype(), xs[0].device());
    CudnnTensorDescriptor dummy_x_desc{dummy_x};
    Shape w_shape{(int64_t)weight_size / 4, 1, 1};
    Array w = Empty(w_shape, xs[0].dtype(), xs[0].device());
    CudnnFilterDescriptor w_desc{AsContiguous(w)};
    for(uint layer = 0 ; layer < n_layers; layer++) {
        for(int8_t di = 0; di < num_directions; di++) {
            for(uint lin_layer_id = 0; lin_layer_id < ws[0].size(); lin_layer_id++) {
                int64_t index = num_directions * layer + di;
                
                cudnnFilterDescriptor_t linlayermatdesc;
                cudnnCreateFilterDescriptor(&linlayermatdesc);
                void* m_offset;
                handle.Call(
                    cudnnGetRNNLinLayerMatrixParams,
                    rnn_desc,
                    layer,
                    *dummy_x_desc,
                    *w_desc,
                    internal::GetRawOffsetData(w),
                    lin_layer_id,
                    linlayermatdesc,
                    &m_offset         
                );
                
                m_offset = internal::GetRawOffsetData(Reshape(ws[index][lin_layer_id], 
                    {1, ws[index][lin_layer_id].shape()[0], ws[index][lin_layer_id].shape()[1]}));
               
                cudnnFilterDescriptor_t linlayerbiasdesc;
                cudnnCreateFilterDescriptor(&linlayerbiasdesc);
                void* b_offset;
                handle.Call(
                    cudnnGetRNNLinLayerBiasParams,
                    rnn_desc,
                    layer,
                    *dummy_x_desc,
                    *w_desc,
                    internal::GetRawOffsetData(w),
                    lin_layer_id,
                    linlayerbiasdesc,
                    &b_offset
                );
                b_offset = internal::GetRawOffsetData(Reshape(bs[index][lin_layer_id], {1, bs[index][lin_layer_id].shape()[0]}));
            }
        }
    }

    size_t reserve_size;
    handle.Call(
        cudnnGetRNNTrainingReserveSize,
        rnn_desc,
        xs.size(),
        xs_desc_arr.data(),
        &reserve_size
    );
    
    cudaMallocManaged(&reserve, reserve_size);

    std::vector<CudnnTensorDescriptor> ys_desc;
    std::vector<cudnnTensorDescriptor_t> ys_desc_arr;
    std::vector<Array> ys;
    std::vector<void*> ys_void;
    for(uint i = 0; i < xs.size(); i++) {
        Shape ys_shape{xs[i].shape()[0], num_directions * hidden_dim, 1};
        ys.push_back(AsContiguous(Empty(ys_shape, xs[i].dtype(), xs[i].device())));
        ys_void.push_back(internal::GetRawOffsetData(ys[i]));
        ys_desc.push_back(CudnnTensorDescriptor(AsContiguous(ys[i])));

        ys_desc_arr.push_back(*ys_desc[i]);
    }

    Array y = AsContiguous(Concatenate(ys, 0)) ;
    y = Reshape(y, {y.shape()[0], y.shape()[1]});
    CudnnTensorDescriptor hx_desc{hx};
    CudnnTensorDescriptor cx_desc{cx};


    Array hy = Empty(hx.shape(), hx.dtype(), hx.device());
    Array cy = Empty(cx.shape(), cx.dtype(), cx.device());


    CudnnTensorDescriptor hy_desc{AsContiguous(hy)};
    CudnnTensorDescriptor cy_desc{AsContiguous(cy)};
    std::shared_ptr<void> workspace = device.Allocate(max_workspace_size);
    handle.Call(
        cudnnRNNForwardTraining,
        rnn_desc,
        xs.size(),
        xs_desc_arr.data(),
        internal::GetRawOffsetData(x),
        *hx_desc,
        internal::GetRawOffsetData(hx),
        *cx_desc,
        internal::GetRawOffsetData(cx),
        *w_desc,
        internal::GetRawOffsetData(w),
        ys_desc_arr.data(),
        internal::GetRawOffsetData(y),
        *hy_desc,
        internal::GetRawOffsetData(hy),
        *cy_desc,
        internal::GetRawOffsetData(cy),
        workspace.get(),
        max_workspace_size,
        reserve,
        reserve_size
    );

    std::vector<Array> states;
    states.push_back(hy);
    states.push_back(cy);
    std::vector<std::vector<Array>> ret;
    ret.push_back(states);
    ret.push_back(ys);
    return ret;
}

std::vector<Array> weights_backward(
    CudaDevice &device,
    cudnnRNNDescriptor_t& rnn_desc,
    CudnnTensorDescriptor& dummy_x_desc,
    Array w,
    const std::vector<std::vector<Array>> ws,
    const std::vector<std::vector<Array>> bs,
    int64_t n_layers,
    int64_t num_directions
    ) {
    cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(device);
    std::vector<Array> ret;
    CudnnHandle& handle = device_internals.cudnn_handle();
    CudnnFilterDescriptor w_desc{w};
    for(int64_t layer = 0 ; layer < n_layers; layer++) {
        for(int8_t di = 0; di < num_directions; di++) {
            for(uint lin_layer_id = 0; lin_layer_id < ws[0].size(); lin_layer_id++) {
                int64_t index = num_directions * layer + di;
                Shape m_shape{1, ws[layer][lin_layer_id].shape()[0], ws[index][lin_layer_id].shape()[1]};
                
                cudnnFilterDescriptor_t linlayermatdesc;
                cudnnCreateFilterDescriptor(&linlayermatdesc);
                std::shared_ptr<void*> m_offset;
                handle.Call(
                    cudnnGetRNNLinLayerMatrixParams ,
                    rnn_desc,
                    layer,
                    *dummy_x_desc,
                    *w_desc,
                    internal::GetRawOffsetData(w),
                    lin_layer_id,
                    linlayermatdesc,
                    m_offset.get()
                );
                Array m = internal::MakeArray(
                    ws[index][lin_layer_id].shape(),
                    ws[index][lin_layer_id].strides(),
                    ws[index][lin_layer_id].dtype(),
                    ws[index][lin_layer_id].device(),
                    m_offset,
                    ws[index][lin_layer_id].offset());

                ret.push_back(m);
                
                
                Shape b_shape{1, bs[index][lin_layer_id].shape()[0]};
                cudnnFilterDescriptor_t linlayerbiasdesc;
                cudnnCreateFilterDescriptor(&linlayerbiasdesc);
                std::shared_ptr<void*> b_offset;
                handle.Call(
                    cudnnGetRNNLinLayerBiasParams,
                    rnn_desc,
                    layer,
                    *dummy_x_desc,
                    *w_desc,
                    internal::GetRawOffsetData(w),
                    lin_layer_id,
                    linlayerbiasdesc,
                    b_offset.get()
                );
                
                Array b = internal::MakeArray(
                    bs[index][lin_layer_id].shape(),
                    bs[index][lin_layer_id].strides(),
                    bs[index][lin_layer_id].dtype(),
                    bs[index][lin_layer_id].device(),
                    b_offset,
                    bs[index][lin_layer_id].offset());
                ret.push_back(b);
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
    auto& backend = static_cast<CudaBackend&>(device.backend());  // NOLINT

    size_t max_workspace_size = backend.GetCudnnMaxWorkspaceSize();

    cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(device);

    CudnnHandle& handle = device_internals.cudnn_handle();

    const auto input_dim = xs[0].shape()[1];
    const auto hidden_dim = hx.shape()[0];
    const auto num_directions = bidirectional == 1 ? 2 : 1;
    const auto num_layers = n_layers;
    const auto rnn_direction = bidirectional == 1 ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL;
    const auto rnn_mode = mode == 1 ? CUDNN_LSTM : CUDNN_GRU;
    const auto rnn_input = CUDNN_LINEAR_INPUT;
    cudnnRNNDescriptor_t rnn_desc;
    cudnnCreateRNNDescriptor(&rnn_desc);
    handle.Call(
        cudnnSetRNNDescriptor,
        rnn_desc,
        hidden_dim,
        num_layers,
        (cudnnDropoutDescriptor_t)NULL,
        rnn_input,
        rnn_direction,
        rnn_mode,
        CUDNN_RNN_ALGO_STANDARD,
        CUDNN_DATA_FLOAT
    );
    std::vector<Array> dxs;
    std::vector<cudnnTensorDescriptor_t> xs_desc;
    std::vector<cudnnTensorDescriptor_t> dxs_desc;
    std::vector<cudnnTensorDescriptor_t> ys_desc;
    std::vector<cudnnTensorDescriptor_t> dys_desc; 

    for(uint i = 0; i < xs.size(); i++) {
        CudnnTensorDescriptor t1 = CudnnTensorDescriptor(AsContiguous(Reshape(xs[i], {xs[i].shape()[0], xs[i].shape()[1], 1})));
        xs_desc.push_back(*t1);
        dxs.push_back(Empty({xs[i].shape()[0], xs[i].shape()[1], 1}, xs[i].dtype(), xs[i].device()));
        CudnnTensorDescriptor t2 = CudnnTensorDescriptor(AsContiguous(dxs[i]));
        dxs_desc.push_back(*t2);
        CudnnTensorDescriptor t3 = CudnnTensorDescriptor(AsContiguous(Reshape(ys[i], {ys[i].shape()[0], ys[i].shape()[1], 1})));
        ys_desc.push_back(*t3);
        CudnnTensorDescriptor t4 = CudnnTensorDescriptor(AsContiguous(Reshape(dys[i], {dys[i].shape()[0], dys[i].shape()[1], 1})));
        dys_desc.push_back(*t4);
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
    Shape dummy_x_shape{1, input_dim, 1};
    Array dummy_x = Empty(dummy_x_shape, xs[0].dtype(), xs[0].device());
    CudnnTensorDescriptor dummy_x_desc{dummy_x};
    Shape w_shape{(int64_t)weight_size / 4, 1, 1};
    Array w = Empty(w_shape, xs[0].dtype(), xs[0].device());
    Array dw = Empty(w_shape, xs[0].dtype(), xs[1].device());
    CudnnFilterDescriptor w_desc{AsContiguous(w)};
    CudnnFilterDescriptor dw_desc{AsContiguous(dw)};
    for(int64_t layer = 0 ; layer < n_layers; layer++) {
        for(int8_t di = 0; di < num_directions; di++) {
            for(uint lin_layer_id = 0; lin_layer_id < ws[0].size(); lin_layer_id++) {
                int64_t index = num_directions * layer + di;
                Shape m_shape{1, ws[layer][lin_layer_id].shape()[0], ws[index][lin_layer_id].shape()[1]};
                
                cudnnFilterDescriptor_t linlayermatdesc;
                cudnnCreateFilterDescriptor(&linlayermatdesc);
                void* m_offset;
                handle.Call(
                    cudnnGetRNNLinLayerMatrixParams,
                    rnn_desc,
                    layer,
                    *dummy_x_desc,
                    *w_desc,
                    internal::GetRawOffsetData(w)   ,
                    lin_layer_id,
                    linlayermatdesc,
                    &m_offset
                );
                m_offset = internal::GetRawOffsetData(Reshape(ws[index][lin_layer_id], m_shape));

                Shape b_shape{1, bs[index][lin_layer_id].shape()[0]};
                
                cudnnFilterDescriptor_t linlayerbiasdesc;
                cudnnCreateFilterDescriptor(&linlayerbiasdesc);
                void* b_offset;
                handle.Call(
                    cudnnGetRNNLinLayerBiasParams,
                    rnn_desc,
                    layer,
                    *dummy_x_desc,
                    *w_desc,
                    internal::GetRawOffsetData(w),
                    lin_layer_id,
                    linlayerbiasdesc,
                    &b_offset
                );
                b_offset = internal::GetRawOffsetData(Reshape(bs[layer][lin_layer_id], b_shape));
            }
        }
    }
    CudnnTensorDescriptor hx_desc{hx};
    CudnnTensorDescriptor cx_desc{cx};
    CudnnTensorDescriptor dhy_desc{dhy};
    CudnnTensorDescriptor dcy_desc{dcy};

    Array dhx = Empty(hx.shape(), hx.dtype(), hx.device());
    CudnnTensorDescriptor dhx_desc{dhx};
    Array dcx = Empty(cx.shape(), cx.dtype(), cx.device());
    CudnnTensorDescriptor dcx_desc{dcx};
    size_t reserve_size;
    handle.Call(
        cudnnGetRNNTrainingReserveSize,
        rnn_desc,
        xs.size(),
        xs_desc.data(),
        &reserve_size
    );
    std::shared_ptr<void> workspace = hx.device().Allocate(max_workspace_size);
    handle.Call(
        cudnnRNNBackwardData,
        rnn_desc,
        xs.size(),
        ys_desc.data(),
        internal::GetRawOffsetData(y),
        dys_desc.data(),
        internal::GetRawOffsetData(dy),
        *dhy_desc,
        internal::GetRawOffsetData(dhy),
        *dcy_desc,
        internal::GetRawOffsetData(dcy),
        *w_desc,
        internal::GetRawOffsetData(w),
        *hx_desc,
        internal::GetRawOffsetData(hx),
        *cx_desc,
        internal::GetRawOffsetData(cx),
        dxs_desc.data(),
        internal::GetRawOffsetData(dx),
        *dhx_desc,
        internal::GetRawOffsetData(hx),
        *dcx_desc,
        internal::GetRawOffsetData(dcx),
        workspace.get(),
        max_workspace_size,
        reserve,
        reserve_size
    );
    handle.Call(
        cudnnRNNBackwardWeights,
        rnn_desc,
        xs.size(),
        xs_desc.data(),
        internal::GetRawOffsetData(x),
        *hx_desc,
        internal::GetRawOffsetData(hx),
        ys_desc.data(),
        internal::GetRawOffsetData(y),
        workspace.get(),
        max_workspace_size,
        *dw_desc,
        internal::GetRawOffsetData(dw),
        reserve,
        reserve_size
        );
    std::vector<Array> state_and_input;
    state_and_input.push_back(dhx);
    state_and_input.push_back(dcx);
    std::vector<std::vector<Array>> ret;
    ret.push_back(state_and_input);
    ret.push_back(dxs);
    ret.push_back(weights_backward(device, rnn_desc, dummy_x_desc, dw, ws, bs, n_layers, num_directions));
    return ret;

}
}  // namespace cuda_internal
}  // namespace cuda
}  // namespace chainerx
