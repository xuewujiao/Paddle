// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/fleet/heter_ps/graph_bucket.h"
__global__ void add_new_keys(uint64_t *keys,
                             size_t set_size,
                             uint64_t *new_keys,
                             size_t n,
                             uint64_t unused_key) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    size_t index = new_keys[idx] % set_size;
    size_t counter = 0;
    while (counter < set_size) {
      uint64_t old = atomicCAS(keys + index, unused_key, new_keys[idx]);
      if (old == unused_key || old == new_keys[idx]) break;
      counter++;
      index++;
      if (index >= set_size) {
        index = 0;
      }
    }
  }
}
}

__global__ void initialize_keys(uint64_t *keys,
                                size_t set_size,
                                uint64_t unused_key) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < set_size) {
    keys[idx] = unused_key;
  }
}
}

__global__ void relocate_keys(uint64_t *keys,
                              size_t set_size,
                              uint64_t *new_keys,
                              size_t new_set_size,
                              uint64_t unused_key) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < set_size && keys[idx] != unused_key) {
    size_t index = keys[idx] % new_set_size;
    size_t counter = 0;
    while (counter < new_set_size) {
      uint64_t old = atomicCAS(new_keys + index, unused_key, keys[idx]);
      if (old == unused_key || old == keys[idx]) break;
      counter++;
      index++;
      if (index >= new_set_size) {
        index = 0;
      }
    }
  }
}

__global__ void cal_emb_dist(int n,
                             int *type,
                             float *emb,
                             int dim,
                             float *aggregated_emb,
                             float *output) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int emb_type = type[idx];
    int aggregated_pos = emb_type * dim;
    int pos = dim * idx;
    float res = 0;
    for (int i = 0; i < dim; i++) {
      res += fabs(aggregated_emb[pos + i] - emb[pos + i]);
    }
    output[idx] = res;
  }
}

__global__ void export_sum(uint64_t *keys,
                           size_t set_size,
                           uint64_t unused_key,
                           size_t *sum,
                           size_t *block_offset) {
  __shared__ size_t local_num;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x == 0) {
    local_num = 0;
  }
  __syncthreads();
  if (keys[idx] != unused_key) {
    atomicAdd(&local_num, 1);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    block_offset[blockIdx.x] = atomicAdd(sum, local_num);
  }
}

void arrage_reduce_one_dim_data(int n, int *data, int *type) {
  thrust::device_ptr<int> data_ptr = thrust::device_pointer_cast(data);
  thrust::device_ptr<int> type_ptr = thrust::device_pointer_cast(type);
  thrust::sort_by_key(thrust::device, type_ptr, type_ptr + n, data_ptr);
}
void reduce_one_dim(
    int n, int type_len, int *data, int *type, int *res_data, int *res_type) {
  int len = n;
  int *dist_data[2], *dist_type[2];
  if (len == type_len) {
    vector<int> ind(type_len, 0);
    for (int i = 0; i < type_len; i++) ind[i] = i;
    sort(ind.begin(), ind.end(), [&](int a, int b) {
      return type[a] < type[b];
    });
    for (int i = 0; i < type_len; i++) {
      res_data[i] = data[ind[i]];
      res_type[i] = type[ind[i]];
    }
    return;
  }
  int cur = 0;
  cudaMalloc((void **)&dist_data[0], sizeof(int) * (n + type_len));
  cudaMalloc((void **)&dist_type[0], sizeof(int) * (n + type_len));
  cudaMemcpy(dist_data[0], data, n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dist_type[0], type, n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&dist_data[1], sizeof(int) * (n + type_len));
  cudaMalloc((void **)&dist_type[1], sizeof(int) * (n + type_len));
  arrage_reduce_one_dim_data(n, dist_data[0], dist_type[0]);
  int *raw_ptr;
  cudaMalloc((void **)&raw_ptr, sizeof(int));
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  int thread_num = 512;
  printf("start to test\n");
  while (len != type_len) {
    int block_num = (len - 1) / thread_num + 1;
    int p = block_num + type_len;
    int block_num2 = (p - 1) / thread_num + 1;
    fill_zero<<<block_num2, thread_num, 0, stream>>>(dist_data[1 - cur], p);
    cudaMemcpyAsync(
        raw_ptr, &block_num, sizeof(int), cudaMemcpyHostToDevice, stream);
    reduce_emb<<<block_num, thread_num, 0, stream>>>(dist_data[cur],
                                                     dist_type[cur],
                                                     len,
                                                     dist_data[1 - cur],
                                                     dist_type[1 - cur],
                                                     raw_ptr);
    cudaMemcpyAsync(&len, raw_ptr, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cur = 1 - cur;
  }
  cudaMemcpy(
      res_data, dist_data[cur], len * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(
      res_type, dist_type[cur], len * sizeof(int), cudaMemcpyDeviceToHost);
}

__global__ void reduce_emb(
    int *data, int *type, int n, int *dist_data, int *dist_type, int *tot) {
  __shared__ int first_type;
  __shared__ int last_type;
  __shared__ int last_pos;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    if (threadIdx.x == 0) {
      first_type = type[idx];
    } else if (idx == n - 1 || threadIdx.x == blockDim.x - 1) {
      last_type = type[idx];
      last_pos = idx;
    }
    int cur_type = type[idx];
    __syncthreads();
    size_t size = blockDim.x / 2;
    int temp;
    while (size != 0) {
      if (threadIdx.x >= size && cur_type == type[idx - size]) {
        temp = data[idx];
      }
      __syncthreads();
      if (threadIdx.x >= size && cur_type == type[idx - size]) {
        data[idx - size] += temp;
      }
      size /= 2;
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      dist_data[blockIdx.x] = data[idx];
      dist_type[blockIdx.x] = type[idx];
    }
    if (first_type != last_type) {
      if (threadIdx.x > 0 && type[idx] != type[idx - 1]) {
        if (type[idx] == last_type && last_pos < n - 1 &&
            type[last_pos + 1] == last_type) {
          int sum, old;
          do {
            sum = data[last_pos + 1];
            old = atomicCAS(data + last_pos + 1, sum, sum + data[idx]);
          } while (old != sum);
        } else {
          int t = atomicAdd(tot, 1);
          dist_data[t] = data[idx];
          dist_type[t] = type[idx];
        }
      }
    }
  }
}

__global__ void export_keys(uint64_t *keys,
                            size_t set_size,
                            uint64_t unused_key,
                            uint64_t *exported_keys,
                            size_t *block_offset) {
  __shared__ size_t local_num;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x == 0) {
    local_num = 0;
  }
  __syncthreads();
  if (keys[idx] != unused_key) {
    size_t pos = atomicAdd(&local_num, 1);
    export_keys[block_offset[blockIdx.x] + pos] = keys[idx];
  }
}

__global__ void find_keys(int n,
                          uint64_t *query_keys,
                          uint64_t *keys,
                          int *output) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = 0;
    size_t index = query_keys[idx] % set_size;
    size_t counter = 0;
    while (counter < set_size) {
      if (keys[index] == unused_key) break;
      if (keys[index] == query_keys[idx]) {
        output[idx] = 1;
        break;
      }
      counter++;
      index++;
      if (index >= set_size) {
        index = 0;
      }
    }
  }
}

__global__ void record_sum(int n,
                           int *output,
                           int *range,
                           int *neighbor_count) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    if (idx == 0) {
      neighbor_count[idx] = output[range[idx] - 1];
    } else {
      neighbor_count[idx] = output[range[idx] - 1] - output[range[idx - 1] - 1];
    }
  }
}

void GraphBucket::init_keys() {
  keys_alloc_ = memory::AllocShared(place_, capacity * sizeof(uint64_t));
  keys_ = reinterpret_cast<int64_t *>(keys_alloc_->ptr());
  initialize_keys<<<GET_BLOCKS(capacity), CUDA_NUM_THREADS>>>(
      keys, capacity, unused_key);
  emb_alloc_ =
      memory::AllocShared(place_, type_size_ * emb_size_ * sizeof(float));
  aggregated_emb = reinterpret_cast<float *>(emb_alloc_->ptr());
  thrust::device_ptr<int> dev_ptr = thrust::device_pointer_cast(aggregated_emb);
  const auto &exec_policy = thrust::cuda::par(allocator).on(stream_);
  thrust::fill(exec_policy, dev_ptr, dev_ptr + type_size_ * emb_size_, (int)0);
}

void GraphBucket::get_neighbor_count(int unique_count,
                                     int n,
                                     uint64_t *neighbors,
                                     int *range,
                                     int *neighbor_count) {
  auto res = memory::AllocShared(place_, n * sizeof(int));
  int *output = reinterpret_cast<int64_t *>(res->ptr());
  find_keys<<<GET_BLOCKS(n), CUDA_NUM_THREADS>>>(n, neighbors, keys, output);
  thrust::device_ptr<int> dev_ptr = thrust::device_pointer_cast(output);
  const auto &exec_policy = thrust::cuda::par(allocator).on(stream_);
  thrust::inclusive_scan(exec_policy, dev_ptr, dev_ptr + n, dev_ptr);
  record_sum<<<GET_BLOCKS(unique_count), CUDA_NUM_THREADS>>>(
      unique_count, output, range, neighbor_count);
}
