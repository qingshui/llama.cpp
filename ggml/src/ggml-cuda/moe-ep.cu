#include "moe-ep.cuh"
#include "ggml-cuda.h"
#include "ggml-impl.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <vector>
#include <map>
#include <mutex>

#ifdef GGML_CUDA_EP_USE_NCCL
#include <nccl.h>

// NCCL error checking macro
#define NCCL_CHECK(cmd) do {                                    \
    ncclResult_t result = (cmd);                                \
    if (result != ncclSuccess) {                                \
        GGML_LOG_ERROR("NCCL error %s:%d: %s\n",                \
                       __FILE__, __LINE__, ncclGetErrorString(result)); \
        GGML_ABORT("NCCL error");                               \
    }                                                           \
} while(0)
#endif

#ifdef GGML_CUDA_EP_USE_NCCL
// Global NCCL communicator management for EP mode
static std::map<int, ncclComm_t> g_ep_nccl_comms;
static std::mutex g_ep_nccl_mutex;

static ncclUniqueId get_or_create_nccl_id(int group_id) {
    static std::map<int, ncclUniqueId> g_nccl_ids;
    static bool initialized = false;

    if (!initialized) {
        ncclGetUniqueId(&g_nccl_ids[0]);
        initialized = true;
    }

    if (g_nccl_ids.find(group_id) == g_nccl_ids.end()) {
        ncclGetUniqueId(&g_nccl_ids[group_id]);
    }
    return g_nccl_ids[group_id];
}
#endif

// Initialize Expert Parallelism context
void ggml_cuda_ep_init(
    ggml_cuda_ep_context& ctx,
    int64_t n_experts,
    const float* tensor_split,
    int n_devices_requested)
{
    ctx.n_experts_total = n_experts;
    ctx.peer_access_enabled = false;

    // Get available CUDA devices
    int cuda_device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&cuda_device_count));

    ctx.n_devices = std::min(n_devices_requested, cuda_device_count);
    ctx.device_ids.resize(ctx.n_devices);

    for (int i = 0; i < ctx.n_devices; ++i) {
        ctx.device_ids[i] = i;
    }

    // Calculate expert distribution
    ggml_cuda_ep_calculate_distribution(
        n_experts,
        ctx.n_devices,
        tensor_split,
        ctx.expert_offsets,
        ctx.expert_offsets);  // Reuse for experts_per_device

    ctx.n_experts_per_device = (n_experts + ctx.n_devices - 1) / ctx.n_devices;

#ifdef GGML_CUDA_EP_USE_NCCL
    // Initialize NCCL communicator for AllReduce
    int original_device = 0;
    CUDA_CHECK(cudaGetDevice(&original_device));

    ncclUniqueId nccl_id = get_or_create_nccl_id(0);

    for (int i = 0; i < ctx.n_devices; ++i) {
        CUDA_CHECK(cudaSetDevice(ctx.device_ids[i]));
        NCCL_CHECK(ncclCommInitRank(&ctx.nccl_comm, ctx.n_devices, nccl_id, i));
    }

    CUDA_CHECK(cudaSetDevice(original_device));

    GGML_LOG_INFO("EP initialized with %d devices, %ld total experts, ~%ld experts/device (NCCL enabled)\n",
                  ctx.n_devices, long(n_experts), long(ctx.n_experts_per_device));
#else
    GGML_LOG_INFO("EP initialized with %d devices, %ld total experts, ~%ld experts/device (NCCL disabled, using fallback)\n",
                  ctx.n_devices, long(n_experts), long(ctx.n_experts_per_device));
#endif
}

// Cleanup Expert Parallelism context
void ggml_cuda_ep_free(ggml_cuda_ep_context& ctx)
{
#ifdef GGML_CUDA_EP_USE_NCCL
    if (ctx.nccl_comm != nullptr) {
        int original_device = 0;
        CUDA_CHECK(cudaGetDevice(&original_device));

        for (int i = 0; i < ctx.n_devices; ++i) {
            CUDA_CHECK(cudaSetDevice(ctx.device_ids[i]));
            ncclCommDestroy(ctx.nccl_comm);
        }

        CUDA_CHECK(cudaSetDevice(original_device));
        ctx.nccl_comm = nullptr;
    }
#endif

    if (ctx.peer_access_enabled) {
        for (int i = 0; i < ctx.n_devices; ++i) {
            for (int j = 0; j < ctx.n_devices; ++j) {
                if (i != j) {
                    cudaDeviceDisablePeerAccess(ctx.device_ids[j]);
                }
            }
        }
        ctx.peer_access_enabled = false;
    }

    ctx.device_ids.clear();
    ctx.expert_offsets.clear();
    ctx.n_devices = 0;
    ctx.n_experts_total = 0;
    ctx.n_experts_per_device = 0;
}

// Enable P2P access between all participating GPUs
void ggml_cuda_ep_enable_peer_access(ggml_cuda_ep_context& ctx)
{
    if (ctx.peer_access_enabled) {
        return;
    }

    int original_device = 0;
    CUDA_CHECK(cudaGetDevice(&original_device));

    for (int i = 0; i < ctx.n_devices; ++i) {
        CUDA_CHECK(cudaSetDevice(ctx.device_ids[i]));

        for (int j = 0; j < ctx.n_devices; ++j) {
            if (i == j) continue;

            int can_access_peer = 0;
            cudaError_t err = cudaDeviceCanAccessPeer(&can_access_peer, ctx.device_ids[i], ctx.device_ids[j]);
            if (err == cudaSuccess && can_access_peer) {
                err = cudaDeviceEnablePeerAccess(ctx.device_ids[j], 0);
                if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
                    GGML_LOG_DEBUG("Warning: Failed to enable peer access from device %d to %d\n",
                                   ctx.device_ids[i], ctx.device_ids[j]);
                }
            }
        }
    }

    CUDA_CHECK(cudaSetDevice(original_device));
    ctx.peer_access_enabled = true;

    GGML_LOG_INFO("P2P access enabled across %d devices\n", ctx.n_devices);
}

// Get the device ID for a specific expert
int ggml_cuda_ep_get_expert_device(const ggml_cuda_ep_context& ctx, int64_t expert_id)
{
    if (ctx.n_devices == 1) {
        return ctx.device_ids[0];
    }

    for (int i = 0; i < ctx.n_devices - 1; ++i) {
        if (expert_id < ctx.expert_offsets[i + 1]) {
            return ctx.device_ids[i];
        }
    }

    return ctx.device_ids[ctx.n_devices - 1];
}

// Helper: calculate expert distribution across devices
void ggml_cuda_ep_calculate_distribution(
    int64_t n_experts,
    int n_devices,
    const float* tensor_split,
    std::vector<int64_t>& expert_offsets,
    std::vector<int64_t>& experts_per_device)
{
    expert_offsets.resize(n_devices + 1, 0);
    experts_per_device.resize(n_devices, 0);

    if (n_devices == 1) {
        expert_offsets[0] = 0;
        expert_offsets[1] = n_experts;
        experts_per_device[0] = n_experts;
        return;
    }

    // Calculate normalized split points
    std::vector<float> splits(n_devices);
    bool all_zero = tensor_split == nullptr ||
                    std::all_of(tensor_split, tensor_split + n_devices, [](float x) { return x == 0.0f; });

    if (all_zero) {
        // Equal distribution
        for (int i = 0; i < n_devices; ++i) {
            splits[i] = float(i + 1) / n_devices;
        }
    } else {
        // Use provided tensor split ratios
        float split_sum = 0.0f;
        for (int i = 0; i < n_devices; ++i) {
            split_sum += tensor_split[i];
        }

        float cumulative = 0.0f;
        for (int i = 0; i < n_devices; ++i) {
            cumulative += tensor_split[i];
            splits[i] = cumulative / split_sum;
        }
    }

    // Calculate expert boundaries
    expert_offsets[0] = 0;
    for (int i = 0; i < n_devices; ++i) {
        expert_offsets[i + 1] = static_cast<int64_t>(splits[i] * n_experts);
        experts_per_device[i] = expert_offsets[i + 1] - expert_offsets[i];
    }

    // Ensure all experts are assigned
    expert_offsets[n_devices] = n_experts;
    experts_per_device[n_devices - 1] = n_experts - expert_offsets[n_devices - 1];
}

// CUDA kernel for gathering expert outputs
template <typename T>
__global__ void gather_expert_outputs_kernel(
    const T** expert_outputs,
    T* gathered_output,
    const int64_t* expert_offsets,
    const int32_t* token_expert_ids,
    int64_t n_tokens,
    int64_t n_expert_used,
    int64_t hidden_size)
{
    const int64_t token_idx = blockIdx.x;
    const int64_t expert_idx = blockIdx.y;
    const int64_t hidden_idx = threadIdx.x + blockDim.x * blockIdx.z;

    if (token_idx >= n_tokens || expert_idx >= n_expert_used || hidden_idx >= hidden_size) {
        return;
    }

    const int32_t expert_id = token_expert_ids[token_idx * n_expert_used + expert_idx];
    const int64_t expert_offset = expert_offsets[expert_id];

    const T* expert_out = expert_outputs[expert_id];
    if (expert_out == nullptr) return;

    const T val = expert_out[(expert_idx * hidden_size + hidden_idx)];
    atomicAdd(&gathered_output[token_idx * hidden_size + hidden_idx], val);
}

// Perform AllReduce across EP devices using NCCL
void ggml_cuda_ep_allreduce(
    const ggml_cuda_ep_context& ctx,
    void* data,
    size_t size,
    cudaStream_t stream,
    ggml_type type)
{
    if (ctx.n_devices == 1) {
        return;  // No reduction needed for single device
    }

#ifdef GGML_CUDA_EP_USE_NCCL
    if (ctx.nccl_comm == nullptr) {
        return;  // NCCL not initialized
    }

    // Map ggml_type to NCCL data type
    ncclDataType_t nccl_type;
    switch (type) {
        case GGML_TYPE_F32:
            nccl_type = ncclFloat32;
            break;
        case GGML_TYPE_F16:
            nccl_type = ncclFloat16;
            break;
        case GGML_TYPE_BF16:
            nccl_type = ncclBfloat16;
            break;
        default:
            // Fall back to float32 for unsupported types
            nccl_type = ncclFloat32;
            break;
    }

    // Use NCCL AllReduce for efficient multi-GPU communication
    // This leverages NVLink for high-bandwidth transfer
    NCCL_CHECK(ncclAllReduce(data, data, size / ggml_type_size(type), nccl_type,
                             ncclSum, ctx.nccl_comm, stream));
#else
    // Fallback: simple AllReduce using cudaMemcpy (for development/testing without NCCL)
    const size_t chunk_size = size / ctx.n_devices;
    const size_t remainder = size % ctx.n_devices;

    int original_device = 0;
    CUDA_CHECK(cudaGetDevice(&original_device));

    // Reduce-scatter phase
    for (int step = 0; step < ctx.n_devices - 1; ++step) {
        int send_dev = (original_device + step) % ctx.n_devices;
        int recv_dev = (original_device + step + 1) % ctx.n_devices;

        CUDA_CHECK(cudaSetDevice(ctx.device_ids[send_dev]));

        size_t send_offset = send_dev * chunk_size + (static_cast<size_t>(send_dev) < remainder ? send_dev : remainder);
        size_t recv_offset = recv_dev * chunk_size + (static_cast<size_t>(recv_dev) < remainder ? recv_dev : remainder);
        size_t send_size = chunk_size + (static_cast<size_t>(send_dev) < remainder ? 1 : 0);

        char* send_ptr = static_cast<char*>(data) + send_offset;
        char* recv_ptr = static_cast<char*>(data) + recv_offset;

        if (ctx.peer_access_enabled) {
            CUDA_CHECK(cudaMemcpyAsync(recv_ptr, send_ptr, send_size,
                                       cudaMemcpyDeviceToDevice, stream));
        } else {
            std::vector<char> host_buffer(send_size);
            CUDA_CHECK(cudaMemcpyAsync(host_buffer.data(), send_ptr, send_size,
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaSetDevice(ctx.device_ids[recv_dev]));
            CUDA_CHECK(cudaMemcpyAsync(recv_ptr, host_buffer.data(), send_size,
                                       cudaMemcpyHostToDevice, stream));
        }
    }

    CUDA_CHECK(cudaSetDevice(original_device));
    CUDA_CHECK(cudaStreamSynchronize(stream));
#endif
}

// Launch MoE forward pass with expert distribution
void ggml_cuda_moe_ep_forward(
    ggml_cuda_ep_context& ctx,
    const ggml_tensor* experts,
    const ggml_tensor* input,
    const ggml_tensor* expert_ids,
    ggml_tensor* output,
    cudaStream_t stream)
{
    // This is a placeholder for the full MoE EP forward pass
    // The actual implementation would:
    // 1. Route tokens to appropriate experts based on expert_ids
    // 2. Launch expert computation on respective devices
    // 3. Gather and combine results

    GGML_UNUSED(ctx);
    GGML_UNUSED(experts);
    GGML_UNUSED(input);
    GGML_UNUSED(expert_ids);
    GGML_UNUSED(output);
    GGML_UNUSED(stream);

    // Full implementation requires integration with ggml's compute graph
    GGML_LOG_DEBUG("MoE EP forward pass (placeholder)\n");
}
