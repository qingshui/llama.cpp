#pragma once

#include "common.cuh"
#include "ggml.h"

#include <cstdint>

#ifdef GGML_CUDA_EP_USE_NCCL
#include <nccl.h>
#endif

// Expert Parallelism context for managing distributed experts across GPUs
struct ggml_cuda_ep_context {
    int n_devices;                          // Number of GPUs participating in EP
    int64_t n_experts_total;                // Total number of experts
    int64_t n_experts_per_device;           // Experts per GPU (may vary)
    std::vector<int> device_ids;            // GPU device IDs
    std::vector<int64_t> expert_offsets;    // Expert start index per device
    bool peer_access_enabled;               // Whether P2P access is enabled
#ifdef GGML_CUDA_EP_USE_NCCL
    ncclComm_t nccl_comm;                   // NCCL communicator for AllReduce
#endif

    ggml_cuda_ep_context()
        : n_devices(0)
        , n_experts_total(0)
        , n_experts_per_device(0)
        , peer_access_enabled(false)
#ifdef GGML_CUDA_EP_USE_NCCL
        , nccl_comm(nullptr)
#endif
    {}
};

// Initialize Expert Parallelism context
void ggml_cuda_ep_init(
    ggml_cuda_ep_context& ctx,
    int64_t n_experts,
    const float* tensor_split,
    int n_devices_requested);

// Cleanup Expert Parallelism context
void ggml_cuda_ep_free(ggml_cuda_ep_context& ctx);

// Enable P2P access between all participating GPUs
void ggml_cuda_ep_enable_peer_access(ggml_cuda_ep_context& ctx);

// Get the device ID for a specific expert
int ggml_cuda_ep_get_expert_device(const ggml_cuda_ep_context& ctx, int64_t expert_id);

// Perform AllReduce across EP devices
void ggml_cuda_ep_allreduce(
    const ggml_cuda_ep_context& ctx,
    void* data,
    size_t size,
    cudaStream_t stream,
    ggml_type type = GGML_TYPE_F32);

// Launch MoE forward pass with expert distribution
void ggml_cuda_moe_ep_forward(
    ggml_cuda_ep_context& ctx,
    const ggml_tensor* experts,
    const ggml_tensor* input,
    const ggml_tensor* expert_ids,
    ggml_tensor* output,
    cudaStream_t stream);

// Helper: calculate expert distribution across devices
void ggml_cuda_ep_calculate_distribution(
    int64_t n_experts,
    int n_devices,
    const float* tensor_split,
    std::vector<int64_t>& expert_offsets,
    std::vector<int64_t>& experts_per_device);
