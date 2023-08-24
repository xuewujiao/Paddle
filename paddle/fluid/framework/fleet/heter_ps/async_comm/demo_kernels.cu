#include "demo_kernels.h"

#include <assert.h>
#include <cuda_runtime_api.h>
#include <stdint.h>

#include "check_macros.h"

#define DIV_UP(X, Y)  (((X) + (Y) - 1) / (Y))

__global__ void GenerateRandomIdsKernel(int* output_ids, int output_count, int max_id, int random_seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= output_count) return;
  uint64_t rand_value = random_seed + 958333ULL * blockIdx.x + threadIdx.x * 5011ULL;
  output_ids[idx] = rand_value % max_id;
}

void GenerateRandomIds(int* output_ids, int output_count, int max_id, int random_seed, cudaStream_t stream) {
  int block_size = 64;
  int block_count = DIV_UP(output_count, block_size);
  GenerateRandomIdsKernel<<<block_count, block_size, 0, stream>>>(output_ids, output_count, max_id, random_seed);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

__device__ __forceinline__ int DeviceGetNextRandomWalkId(int input_index) {
  return input_index ^ 0xFFFF;
}

__global__ void GetNextRandomWalkIdKernel(const int *input_indices, int node_count, int *output_indices) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= node_count) return;
  output_indices[idx] = DeviceGetNextRandomWalkId(input_indices[idx]);
}

void GetNextRandomWalkId(const int *input_indices, int node_count, int *output_indices, cudaStream_t stream) {
  int block_size = 64;
  int block_count = DIV_UP(node_count, block_size);
  GetNextRandomWalkIdKernel<<<block_count, block_size, 0, stream>>>(input_indices, node_count, output_indices);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

__device__ int d_random_walk_id_check_error_count;

__global__ void CheckNextRandomWalkIdKernel(const int *input_indices, int node_count, const int *result_indices) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= node_count) return;
  int ref_value = DeviceGetNextRandomWalkId(input_indices[idx]);
  if (result_indices[idx] != ref_value) {
    int diff_count = atomicAdd(&d_random_walk_id_check_error_count, 1);
    if (diff_count <= 10) {
      printf("[CheckNextRandomWalkIdKernel] get error idx=%d, node_id=%d, %d should be %d\n",
             idx, input_indices[idx], result_indices[idx], ref_value);
    }
    assert(false);
  }
}

void CheckNextRandomWalkId(const int *input_indices, int node_count, const int *result_indices, cudaStream_t stream) {
  int block_size = 64;
  int block_count = DIV_UP(node_count, block_size);
  int *random_walk_id_check_error_count;
  CUDA_CHECK(cudaGetSymbolAddress(reinterpret_cast<void **>(&random_walk_id_check_error_count),
                                  d_random_walk_id_check_error_count));
  CUDA_CHECK(cudaMemsetAsync(random_walk_id_check_error_count, 0, sizeof(int)));
  CheckNextRandomWalkIdKernel<<<block_count, block_size, 0, stream>>>(input_indices, node_count, result_indices);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

__device__ __forceinline__ float DeviceGetNodeEmbedding(int input_id, int dim_idx) {
  float emb_value = input_id;
  emb_value += 0.5f * dim_idx;
  return emb_value;
}

__global__ void GetNodeEmbeddingKernel(const int *input_indices,
                                       int node_count,
                                       int embedding_dim,
                                       float *output_embedding) {
  int node_idx = blockIdx.x;
  int node_id = input_indices[node_idx];
  float *output_ptr = output_embedding + (size_t) embedding_dim * node_idx;
  for (int dim_idx = threadIdx.x; dim_idx < embedding_dim; dim_idx += blockDim.x) {
    output_ptr[dim_idx] = DeviceGetNodeEmbedding(node_id, dim_idx);
  }
}

__device__ int d_node_embedding_check_error_count;

__global__ void CheckNodeEmbeddingKernel(const int *input_indices,
                                         int node_count,
                                         int embedding_dim,
                                         const float *result_embedding) {
  int node_idx = blockIdx.x;
  int node_id = input_indices[node_idx];
  const float *result_ptr = result_embedding + (size_t) embedding_dim * node_idx;
  for (int dim_idx = threadIdx.x; dim_idx < embedding_dim; dim_idx += blockDim.x) {
    float ref_value = DeviceGetNodeEmbedding(node_id, dim_idx);
    if (result_ptr[dim_idx] != ref_value) {
      int diff_count = atomicAdd(&d_node_embedding_check_error_count, 1);
      if (diff_count <= 10) {
        printf("[CheckNodeEmbedding] get error node_idx=%d, node_id=%d, dim_idx=%d, %f should be %f\n",
               node_idx, node_id, dim_idx, result_ptr[dim_idx], ref_value);
      }
      assert(false);
    }
  }
}

void GetNodeEmbedding(const int *input_indices,
                      int node_count,
                      int embedding_dim,
                      float *output_embedding,
                      cudaStream_t stream) {
  GetNodeEmbeddingKernel<<<node_count, embedding_dim, 0, stream>>>(input_indices,
                                                                   node_count,
                                                                   embedding_dim,
                                                                   output_embedding);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

void CheckNodeEmbedding(const int *input_indices,
                        int node_count,
                        int embedding_dim,
                        const float *result_embedding,
                        cudaStream_t stream) {
  int *node_embedding_check_error_count;
  CUDA_CHECK(cudaGetSymbolAddress(reinterpret_cast<void **>(&node_embedding_check_error_count),
                                  d_node_embedding_check_error_count));
  CUDA_CHECK(cudaMemsetAsync(node_embedding_check_error_count, 0, sizeof(int)));
  CheckNodeEmbeddingKernel<<<node_count, embedding_dim, 0, stream>>>(input_indices,
                                                                     node_count,
                                                                     embedding_dim,
                                                                     result_embedding);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

__device__ __forceinline__ float DeviceGetNodeGradient(int input_id, int dim_idx) {
  float emb_value = input_id;
  emb_value += 3.0f * dim_idx;
  return emb_value;
}

__global__ void GetNodeGradientsKernel(const int *input_indices,
                                       int node_count,
                                       int embedding_dim,
                                       float *output_gradient) {
  int node_idx = blockIdx.x;
  int node_id = input_indices[node_idx];
  float *output_ptr = output_gradient + (size_t) embedding_dim * node_idx;
  for (int dim_idx = threadIdx.x; dim_idx < embedding_dim; dim_idx += blockDim.x) {
    output_ptr[dim_idx] = DeviceGetNodeGradient(node_id, dim_idx);
  }
}

__device__ int d_node_gradients_check_error_count;

__global__ void CheckNodeGradientsKernel(const int *input_indices,
                                         int node_count,
                                         int embedding_dim,
                                         const float *result_gradient) {
  int node_idx = blockIdx.x;
  int node_id = input_indices[node_idx];
  const float *result_ptr = result_gradient + (size_t) embedding_dim * node_idx;
  for (int dim_idx = threadIdx.x; dim_idx < embedding_dim; dim_idx += blockDim.x) {
    float ref_value = DeviceGetNodeGradient(node_id, dim_idx);
    if (result_ptr[dim_idx] != ref_value) {
      int diff_count = atomicAdd(&d_node_gradients_check_error_count, 1);
      if (diff_count <= 10) {
        printf("[CheckNodeGradientsKernel] get error node_idx=%d, node_id=%d, dim_idx=%d, %f should be %f\n",
               node_idx, node_id, dim_idx, result_ptr[dim_idx], ref_value);
      }
      assert(false);
    }
  }
}

void GetNodeGradients(const int *input_indices,
                      int node_count,
                      int embedding_dim,
                      float *output_gradients,
                      cudaStream_t stream) {
  GetNodeGradientsKernel<<<node_count, embedding_dim, 0, stream>>>(input_indices,
                                                                   node_count,
                                                                   embedding_dim,
                                                                   output_gradients);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

void CheckNodeGradients(const int *input_indices,
                        int node_count,
                        int embedding_dim,
                        const float *result_gradients,
                        cudaStream_t stream) {
  int *node_gradients_check_error_count;
  CUDA_CHECK(cudaGetSymbolAddress(reinterpret_cast<void **>(&node_gradients_check_error_count),
                                  d_node_gradients_check_error_count));
  CUDA_CHECK(cudaMemsetAsync(node_gradients_check_error_count, 0, sizeof(int)));
  CheckNodeGradientsKernel<<<node_count, embedding_dim, 0, stream>>>(input_indices,
                                                                     node_count,
                                                                     embedding_dim,
                                                                     result_gradients);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}
