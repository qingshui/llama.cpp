#include "common.cuh"
#include "ggml.h"
#include "moe-ep.cuh"

#include <initializer_list>

struct ggml_cuda_topk_moe_args {
    bool sigmoid{};
    bool softmax{};
    bool delayed_softmax{};
    bool prob_bias{};
    bool norm{};
    bool scale{};
};

void ggml_cuda_op_topk_moe(ggml_backend_cuda_context &     ctx,
                           const ggml_tensor *             logits,
                           ggml_tensor *                   weights,
                           ggml_tensor *                   ids,
                           const ggml_tensor *             clamp,
                           const ggml_tensor *             scale,
                           const ggml_tensor *             bias,
                           const ggml_cuda_topk_moe_args & args);

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * gating_op,
                                   const ggml_tensor * weights,
                                   const ggml_tensor * logits,
                                   const ggml_tensor * ids);

// Expert Parallelism extensions for distributed MoE
void ggml_cuda_op_topk_moe_ep(ggml_backend_cuda_context &     ctx,
                              ggml_cuda_ep_context &          ep_ctx,
                              const ggml_tensor *             logits,
                              ggml_tensor *                   weights,
                              ggml_tensor *                   ids,
                              const ggml_tensor *             clamp,
                              const ggml_tensor *             scale,
                              const ggml_tensor *             bias,
                              const ggml_cuda_topk_moe_args & args);

bool ggml_cuda_should_use_topk_moe_ep(const ggml_tensor * gating_op,
                                      const ggml_tensor * weights,
                                      const ggml_tensor * logits,
                                      const ggml_tensor * ids,
                                      const ggml_cuda_ep_context & ep_ctx);
