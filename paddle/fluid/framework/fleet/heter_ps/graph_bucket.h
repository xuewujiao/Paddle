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

#pragma once
#include <time.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>
#include "paddle/fluid/distributed/ps/table/common_graph_table.h"
#include "paddle/fluid/framework/fleet/heter_ps/gpu_graph_node.h"
#include "paddle/fluid/framework/fleet/heter_ps/graph_gpu_ps_table.h"
#include "paddle/fluid/string/printf.h"
#include "paddle/fluid/string/string_helper.h"
//#ifdef PADDLE_WITH_HETERPS
namespace paddle {
namespace framework {
struct CachedAllocator {
  typedef char value_type;
  CacheAllocator(platform::Place place) {
    VLOG(2) << "construct allocator";
    place_ = place;
  }

  ~CacheAllocator() { VLOG(2) << "destory allocator"; }

  char *allocate(std::ptrdiff_t num_bytes) {
    VLOG(2) << "allocate " << num_bytes << " bytes";
    auto storage = memory::AllocShared(place_, num_bytes);
    char *ptr = reinterpret_cast<char *>(storage->ptr());
    busy_allocation_.emplace(std::make_pair(ptr, storage));
    return ptr;
  }

  void deallocate(char *ptr, size_t) {
    VLOG(2) << "deallocate ";
    allocation_map_type::iterator iter = busy_allocation_.find(ptr);
    CHECK(iter != busy_allocation_.end());
    busy_allocation_.erase(iter);
  }

 private:
  typedef std::unordered_map<char *, std::shared_ptr<phi::Allocation>>
      allocation_map_type;
  allocation_map_type busy_allocation_;
  platform::Place place_;
  // const auto &exec_policy =
  // thrust::cuda::par(allocator).on(dev_ctx.stream());
};

class GraphBucket {
 public:
  GraphBucket(int dev_id, size_t capacity, int emb_size, int type_size)
      : capacity_(capacity), emb_size_(emb_size), type_size_(type_size) {
    size_ = 0;
    unused_key = (uint64_t)-1;
    place_ = paddle::platform::CUDAPlace(dev_id);
    allocator_ = CacheAllocator(place_);
    // stream_ =
    //     dynamic_cast<phi::GPUContext *>(
    //         paddle::platform::DeviceContextPool::Instance().Get(this->place_))
    //         ->stream();
    platform::CUDADeviceGuard guard(dev_id_);
    comm_streams_.resize(dev_ids_.size());
    for (size_t i = 0; i < dev_ids_.size(); ++i) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaStreamCreateWithFlags(&comm_streams_[i], cudaStreamNonBlocking));
    }
    stream_ = comm_streams_[dev_id];
    init_keys();
  }
  ~GraphBucket() {}
  std::vector<gpuStream_t> comm_streams_;
  size_t capacity_, size_;
  uint64_t unused_key_;
  uint64_t *keys_;
  float *aggregated_emb;
  CacheAllocator allocator_;
  cudaStream_t stream_;
  std::shared_ptr<phi::Allocation> keys_alloc_;
  std::shared_ptr<phi::Allocation> emb_alloc_;
  platform::Place place_;
  int emb_size_;
  int type_size_;
  const int CUDA_NUM_THREADS = 512;
  // CUDA: number of blocks for threads.
  int GET_BLOCKS(const int N) {
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
  }
  virtual void init_keys();
  virtual void get_neighbor_count(int unique_count,
                                  int n,
                                  uint64_t *neighbors,
                                  int *range,
                                  int *neighbor_count);

}

class BucketGroup {
 public:
  BucketGroup(int dev_num, int emb_size, int type_size) {
    for (int i = 0; i < dev_num; i++) {
      resources.push_back(std::shared_ptr<GraphBucket>(
          new GraphBucket(capacity, i, emb_size, type_size)));
    }
  }
  std::vector<std::shared_ptr<GraphBucket>> resources;
  const size_t capacity = 1000000000ll;

} enum GraphSamplerStatus { waiting = 0, running = 1, terminating = 2 };
class GraphSampler {
 public:
  GraphSampler() {
    status = GraphSamplerStatus::waiting;
    thread_pool.reset(new ::ThreadPool(1));
  }
  virtual int start_service(std::string path) {
    load_from_ssd(path);
    VLOG(0) << "load from ssd over";
    std::promise<int> prom;
    std::future<int> fut = prom.get_future();
    graph_sample_task_over = thread_pool->enqueue([&prom, this]() {
      VLOG(0) << " promise set ";
      prom.set_value(0);
      status = GraphSamplerStatus::running;
      return run_graph_sampling();
    });
    return fut.get();
    return 0;
  }
  virtual int end_graph_sampling() {
    if (status == GraphSamplerStatus::running) {
      status = GraphSamplerStatus::terminating;
      return graph_sample_task_over.get();
    }
    return -1;
  }
  ~GraphSampler() { end_graph_sampling(); }
  virtual int load_from_ssd(std::string path) = 0;
  ;
  virtual int run_graph_sampling() = 0;
  ;
  virtual void init(GpuPsGraphTable *gpu_table,
                    std::vector<std::string> args_) = 0;
  std::shared_ptr<::ThreadPool> thread_pool;
  GraphSamplerStatus status;
  std::future<int> graph_sample_task_over;
};

class CommonGraphSampler : public GraphSampler {
 public:
  CommonGraphSampler() {}
  virtual ~CommonGraphSampler() {}
  GpuPsGraphTable *g_table;
  virtual int load_from_ssd(std::string path);
  virtual int run_graph_sampling();
  virtual void init(GpuPsGraphTable *g, std::vector<std::string> args);
  GpuPsGraphTable *gpu_table;
  paddle::distributed::GraphTable *table;
  std::vector<uint64_t> gpu_edges_count;
  uint64_t cpu_edges_count;
  uint64_t gpu_edges_limit, cpu_edges_limit, gpu_edges_each_limit;
  std::vector<std::unordered_set<uint64_t>> gpu_set;
  int gpu_num;
};

class AllInGpuGraphSampler : public GraphSampler {
 public:
  AllInGpuGraphSampler() {}
  virtual ~AllInGpuGraphSampler() {}
  // virtual pthread_rwlock_t *export_rw_lock();
  virtual int run_graph_sampling();
  virtual int load_from_ssd(std::string path);
  virtual void init(GpuPsGraphTable *g, std::vector<std::string> args_);

 protected:
  paddle::distributed::GraphTable *graph_table;
  GpuPsGraphTable *gpu_table;
  std::vector<std::vector<uint64_t>> sample_node_ids;
  std::vector<std::vector<paddle::framework::GpuPsNodeInfo>> sample_node_infos;
  std::vector<std::vector<uint64_t>> sample_neighbors;
  std::vector<GpuPsCommGraph> sample_res;
  // std::shared_ptr<std::mt19937_64> random;
  int gpu_num;
};

}  // namespace framework
};  // namespace paddle
//#include "paddle/fluid/framework/fleet/heter_ps/graph_sampler_inl.h"
//#endif
