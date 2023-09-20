#include <algorithm>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "async_communicator.h"
#include "check_macros.h"
#include "config.h"
#include "cuda_utils.h"
#include "ib_utils.h"
#include "log_macros.h"
#include "demo_memory_allocator.h"
#include "demo_runner.h"
#include "demo_kernels.h"
#include "merged_handler.h"

int node_rank = -1;
std::string config_file;

int global_node_count = -1;
int ranks_per_node = -1;

void PringUsage(const char *program_name) {
  std::cout << "Usage: " << program_name << " -r <node_rank> -c <config_file>" << std::endl;
}

bool ParseArgs(int argc, char **argv) {
  try {
    if (argc < 5) {
      throw std::runtime_error("Missing arguments");
    }

    for (int i = 1; i < argc; i++) {
      if (std::string(argv[i]) == "-r") {
        if (i + 1 < argc) {
          node_rank = std::stoi(argv[++i]);
        } else {
          throw std::runtime_error("Missing value for node rank");
        }
      } else if (std::string(argv[i]) == "-c") {
        if (i + 1 < argc) {
          config_file = argv[++i];
        } else {
          throw std::runtime_error("Missing value for config_file");
        }
      } else {
        throw std::runtime_error("Unknown option: " + std::string(argv[i]));
      }
    }
  } catch (const std::runtime_error &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    PringUsage(argv[0]);
    return false;
  }
  return true;
}

static inline int IntScale(int num, float scale) {
  return static_cast<int>(num * scale);
}

void SingleRequestWorkFunction(int global_rank,
                               DemoRandomWalkRunner *random_walk_runner,
                               DemoEmbeddingPsRunner *embedding_ps_runner,
                               AsyncCommunicator *async_communicator) {
  int local_rank = global_rank % ranks_per_node;
  CUDA_CHECK(cudaSetDevice(local_rank));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  auto *allocator = dynamic_cast<DemoMemoryAllocator *>(async_communicator->GetMemoryAllocator());
  std::default_random_engine engine(std::random_device{}());
  constexpr int kMaxVertexId = 500000;
  constexpr int kAvgLoopCount = 10000;
  constexpr int kMaxVertexCount = 1000000;
  std::uniform_int_distribution<int> random_seed_dist(0, INT32_MAX);
  std::uniform_int_distribution<int> loop_dist(IntScale(kAvgLoopCount, 0.9), IntScale(kAvgLoopCount, 1.1));
  std::uniform_int_distribution<int> vertex_count_dist(IntScale(kMaxVertexCount, 0.5), kMaxVertexCount);
  std::uniform_int_distribution<int> rank_id_dist(0, global_node_count * ranks_per_node - 1);
  int loop_count = loop_dist(engine);

  for (int iter = 0; iter < loop_count; iter++) {
    int vertex_count = vertex_count_dist(engine);
    int target_rank = rank_id_dist(engine);
    DemoTensor vertex_tensor = allocator->CreateTensor(ML_DEVICE, DT_INT32, vertex_count);
    int random_seed = random_seed_dist(engine);
    GenerateRandomIds((int *) vertex_tensor.DataPtr(), vertex_count, kMaxVertexId, random_seed, stream);
    auto *vertex_context = allocator->ToMemoryContext(vertex_tensor);
    auto *walk_request = random_walk_runner->MakeRandomWalkRequest(vertex_context, target_rank);
    auto *walk_response = CreateAsyncReqRes();
    async_communicator->PutRequestSync(walk_request, walk_response);
    DemoTensor walked_vertex_tensor = allocator->FromMemoryContext(walk_response->memory_contexts[0]);
    allocator->FreeReqRes(walk_response);
    walk_response = nullptr;
    CheckNextRandomWalkId((int *) vertex_tensor.DataPtr(),
                          vertex_count,
                          (int *) walked_vertex_tensor.DataPtr(),
                          stream);

    int embedding_dim = embedding_ps_runner->GetEmbeddingDim();
    int ps_get_target = rank_id_dist(engine);
    auto *ps_vertex_context = allocator->ToMemoryContext(walked_vertex_tensor);
    auto *ps_get_request = embedding_ps_runner->MakePsGetRequest(ps_vertex_context, ps_get_target);
    auto *ps_get_response = CreateAsyncReqRes();
    async_communicator->PutRequestSync(ps_get_request, ps_get_response);
    DemoTensor fetch_ps_embedding_tensor = allocator->FromMemoryContext(ps_get_response->memory_contexts[0]);
    allocator->FreeReqRes(ps_get_response);
    CheckNodeEmbedding((int *) walked_vertex_tensor.DataPtr(),
                       vertex_count,
                       embedding_dim,
                       (float *) fetch_ps_embedding_tensor.DataPtr(),
                       stream);
    DemoTensor grad_tensor = allocator->CreateTensor(ML_DEVICE, DT_FLOAT, (size_t) vertex_count * embedding_dim);
    GetNodeGradients((int *) walked_vertex_tensor.DataPtr(),
                     vertex_count,
                     embedding_dim,
                     (float *) grad_tensor.DataPtr(),
                     stream);

    // need ToMemoryContext again as previous context is deleted after send.
    ps_vertex_context = allocator->ToMemoryContext(walked_vertex_tensor);
    auto *ps_grad_context = allocator->ToMemoryContext(grad_tensor);
    auto *ps_update_request =
        embedding_ps_runner->MakePsUpdateRequest(ps_vertex_context, ps_grad_context, ps_get_target);
    RequestHandle request_handle;
    request_handle.request_ = ps_update_request;
    request_handle.response_ = nullptr;
    async_communicator->PutRequestAsync(&request_handle);

    LOG_INFO("rank=%d iter=%d vertex_count=%lu success.",
             global_rank, iter, vertex_tensor.EltCount());
  }
  CUDA_CHECK(cudaStreamDestroy(stream));

  async_communicator->SendStopSignal();
}

void AllRequestWorkFunction(int global_rank,
                            DemoRandomWalkRunner *random_walk_runner,
                            DemoEmbeddingPsRunner *embedding_ps_runner,
                            AsyncCommunicator *async_communicator) {
  int local_rank = global_rank % ranks_per_node;
  int global_size = global_node_count * ranks_per_node;
  CUDA_CHECK(cudaSetDevice(local_rank));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  auto *allocator = dynamic_cast<DemoMemoryAllocator *>(async_communicator->GetMemoryAllocator());
  std::default_random_engine engine(std::random_device{}());
  constexpr int kMaxVertexId = 5000000;
  constexpr int kAvgLoopCount = 1000;
  constexpr int kMaxVertexCount = 2000000;
  int max_vertex_count_per_rank = kMaxVertexCount / global_size;
  std::uniform_int_distribution<int> random_seed_dist(0, INT32_MAX);
  std::uniform_int_distribution<int> loop_dist(IntScale(kAvgLoopCount, 0.9), IntScale(kAvgLoopCount, 1.1));
  std::uniform_int_distribution<int>
      vertex_count_dist(IntScale(max_vertex_count_per_rank, 0.5), max_vertex_count_per_rank);
  int loop_count = loop_dist(engine);

  int embedding_dim = embedding_ps_runner->GetEmbeddingDim();

  DemoTensor raw_vertex_tensor = allocator->CreateTensor(ML_DEVICE, DT_INT32, kMaxVertexCount);
  DemoTensor walked_vertex_tensor = allocator->CreateTensor(ML_DEVICE, DT_INT32, kMaxVertexCount);
  DemoTensor embedding_tensor = allocator->CreateTensor(ML_DEVICE, DT_FLOAT, kMaxVertexCount * embedding_dim);
  DemoTensor gradient_tensor = allocator->CreateTensor(ML_DEVICE, DT_FLOAT, kMaxVertexCount * embedding_dim);
  std::vector<int> rank_counts(global_size);
  std::vector<int> rank_offsets(global_size);
  std::vector<int> node_counts(global_node_count, 0);
  std::vector<int> node_offsets(global_node_count, 0);

  std::mt19937 perm_gen(std::random_device{}());
  std::vector<int> rand_idx_mapping(global_size);
  for (int i = 0; i < global_size; i++) {
    rand_idx_mapping[i] = i;
  }

  for (int iter = 0; iter < loop_count; iter++) {
    int offset = 0;
    for (int r = 0; r < global_size; r++) {
      rank_counts[r] = vertex_count_dist(engine);
      rank_offsets[r] = offset;
      if (r % ranks_per_node == 0) {
        node_counts[r / ranks_per_node] = rank_counts[r];
        node_offsets[r / ranks_per_node] = offset;
      } else {
        node_counts[r / ranks_per_node] += rank_counts[r];
      }
      offset += rank_counts[r];
    }
    int random_seed = random_seed_dist(engine);
    int vertex_count = offset;
    int *rank_raw_ptr = (int *) raw_vertex_tensor.DataPtr();
    int *rank_walked_ptr = (int *) walked_vertex_tensor.DataPtr();
    float *embedding_ptr = (float *) embedding_tensor.DataPtr();
    float *gradient_ptr = (float *) gradient_tensor.DataPtr();
    GenerateRandomIds(rank_raw_ptr, vertex_count, kMaxVertexId, random_seed, stream);
    std::vector<MergeHeader> merge_headers(global_node_count);
    std::vector<DemoTensor> rank_raw_tensors(global_size);
    std::vector<DemoTensor> rank_walked_tensors(global_size);
    std::vector<DemoTensor> node_raw_tensors(global_node_count);
    std::vector<DemoTensor> node_walked_tensors(global_node_count);
    std::vector<DemoTensor> node_header_tensors(global_node_count);
    std::vector<DemoTensor> rank_embedding_tensors(global_size);
    std::vector<DemoTensor> rank_gradient_tensors(global_size);
    std::vector<RequestHandle> walk_request_handles(global_size);
    std::vector<RequestHandle> get_embedding_request_handles(global_size);
    std::vector<RequestHandle> update_embedding_request_handles(global_size);
    std::shuffle(rand_idx_mapping.begin(), rand_idx_mapping.end(), perm_gen);
    int64_t int_para[4] = {global_rank, global_size, vertex_count, vertex_count + 1};
    auto int_para_tensor = MakeNotOwnedTensor(&int_para[0], ML_HOST, DT_INT64, 4);

    for (int i = 0; i < global_size; i++) {
      int r = i;
      rank_raw_tensors[r] = MakeNotOwnedTensor(rank_raw_ptr + rank_offsets[r], ML_DEVICE, DT_INT32, rank_counts[r]);
      rank_walked_tensors[r] =
          MakeNotOwnedTensor(rank_walked_ptr + rank_offsets[r], ML_DEVICE, DT_INT32, rank_counts[r]);
      rank_embedding_tensors[r] = MakeNotOwnedTensor(embedding_ptr + rank_offsets[r] * (size_t) embedding_dim,
                                                     ML_DEVICE,
                                                     DT_FLOAT,
                                                     rank_counts[r] * (size_t) embedding_dim);
      rank_gradient_tensors[r] = MakeNotOwnedTensor(gradient_ptr + rank_offsets[r] * (size_t) embedding_dim,
                                                    ML_DEVICE,
                                                    DT_FLOAT,
                                                    rank_counts[r] * (size_t) embedding_dim);

    }
    for (int i = 0; i < global_node_count; i++) {
      int r = i;
      node_raw_tensors[r] = MakeNotOwnedTensor(rank_raw_ptr + node_offsets[r], ML_DEVICE, DT_INT32, node_counts[r]);
      node_walked_tensors[r] =
          MakeNotOwnedTensor(rank_walked_ptr + node_offsets[r], ML_DEVICE, DT_INT32, node_counts[r]);
      node_header_tensors[r] = MakeNotOwnedTensor(&merge_headers[r], ML_HOST, DT_INT8, sizeof(MergeHeader));
      merge_headers[r].response_data_count = 1;

      size_t raw_tensor_offset = 0;
      size_t walked_tensor_offset = 0;
      for (int lr = 0; lr < ranks_per_node; lr++) {
        size_t rank_count = rank_counts[lr + i * ranks_per_node];
        merge_headers[r].ranks_info[lr].request_offset[0] = raw_tensor_offset * sizeof(int);
        merge_headers[r].ranks_info[lr].request_size[0] = rank_count * sizeof(int);
        merge_headers[r].ranks_info[lr].response_offset[0] = walked_tensor_offset * sizeof(int);
        merge_headers[r].ranks_info[lr].response_size[0] = rank_count * sizeof(int);
        raw_tensor_offset += rank_count;
        walked_tensor_offset += rank_count;
      }
      merge_headers[r].response_info[0].location = ML_DEVICE;
      merge_headers[r].response_info[0].dtype = DT_INT32;
      merge_headers[r].response_info[0].data_size = walked_tensor_offset * sizeof(int);
    }
    for (int nr = 0; nr < global_node_count; nr++) {
      int r = nr * ranks_per_node + local_rank;
      walk_request_handles[r].request_ =
          random_walk_runner->MakeRandomWalkRequest(allocator->ToMemoryContext(node_raw_tensors[nr]),
                                                    r);
      walk_request_handles[r].request_->memory_contexts[1] = allocator->ToMemoryContext(node_header_tensors[nr]);
      walk_request_handles[r].request_->meta.data_sizes[1] = sizeof(MergeHeader);
      walk_request_handles[r].request_->meta.data_types[1] = DT_INT8;
      walk_request_handles[r].request_->meta.locations[1] = ML_HOST;
      walk_request_handles[r].request_->meta.valid_data_count++;
      walk_request_handles[r].request_->meta.merged_flag = 1;
      walk_request_handles[r].response_ = CreateAsyncReqRes();
      walk_request_handles[r].response_->memory_contexts[0] = allocator->ToMemoryContext(node_walked_tensors[nr]);
      async_communicator->PutRequestAsync(&walk_request_handles[r]);
    }
    for (int nr = 0; nr < global_node_count; nr++) {
      int r = nr * ranks_per_node + local_rank;
      walk_request_handles[r].Wait();
      allocator->FreeReqRes(walk_request_handles[r].response_);
      walk_request_handles[r].response_ = nullptr;
    }
#if 0
    for (int i = 0; i < global_size; i++) {
      int r = rand_idx_mapping[i];
      walk_request_handles[r].request_ =
          random_walk_runner->MakeRandomWalkRequestWithPara(allocator->ToMemoryContext(rank_raw_tensors[r]),
                                                            allocator->ToMemoryContext(int_para_tensor),
                                                            r);
      walk_request_handles[r].response_ = CreateAsyncReqRes();
      walk_request_handles[r].response_->memory_contexts[0] = allocator->ToMemoryContext(rank_walked_tensors[r]);
      async_communicator->PutRequestAsync(&walk_request_handles[r]);
    }

    for (int r = 0; r < global_size; r++) {
      walk_request_handles[r].Wait();
      allocator->FreeReqRes(walk_request_handles[r].response_);
      walk_request_handles[r].response_ = nullptr;
    }
#endif
    CheckNextRandomWalkId(rank_raw_ptr, vertex_count, rank_walked_ptr, stream);

    std::shuffle(rand_idx_mapping.begin(), rand_idx_mapping.end(), perm_gen);
    for (int i = 0; i < global_size; i++) {
      int r = rand_idx_mapping[i];
      get_embedding_request_handles[r].request_ =
          embedding_ps_runner->MakePsGetRequest(allocator->ToMemoryContext(rank_walked_tensors[r]), r);
      get_embedding_request_handles[r].response_ = CreateAsyncReqRes();
      get_embedding_request_handles[r].response_->memory_contexts[0] =
          allocator->ToMemoryContext(rank_embedding_tensors[r]);
      async_communicator->PutRequestAsync(&get_embedding_request_handles[r]);
    }
    for (int r = 0; r < global_size; r++) {
      get_embedding_request_handles[r].Wait();
      allocator->FreeReqRes(get_embedding_request_handles[r].response_);
      get_embedding_request_handles[r].response_ = nullptr;
    }
    CheckNodeEmbedding(rank_walked_ptr,
                       vertex_count,
                       embedding_dim,
                       embedding_ptr,
                       stream);

    GetNodeGradients(rank_walked_ptr,
                     vertex_count,
                     embedding_dim,
                     gradient_ptr,
                     stream);

    std::shuffle(rand_idx_mapping.begin(), rand_idx_mapping.end(), perm_gen);
    for (int i = 0; i < global_size; i++) {
      int r = rand_idx_mapping[i];
      auto *ps_update_request = embedding_ps_runner->MakePsUpdateRequest(
          allocator->ToMemoryContext(rank_walked_tensors[r]),
          allocator->ToMemoryContext(rank_gradient_tensors[r]),
          r);
      RequestHandle& request_handle = update_embedding_request_handles[r];
      request_handle.request_ = ps_update_request;
      request_handle.response_ = nullptr;
      async_communicator->PutRequestAsync(&request_handle, true);
    }
    for (int r = 0; r < global_size; r++) {
      update_embedding_request_handles[r].Wait();
    }

    LOG_INFO("rank=%d iter=%d vertex_count=%lu success.",
             global_rank, iter, vertex_count);
  }
  CUDA_CHECK(cudaStreamDestroy(stream));

  async_communicator->SendStopSignal();
}

void RankFunction(int global_rank, Config *pconfig) {
  int node_id = global_rank / ranks_per_node;
  int local_rank = global_rank % ranks_per_node;
  Partitioner partitioner(global_node_count, ranks_per_node, node_id, local_rank);
  DemoMemoryAllocator memory_allocator(local_rank);
  RunnerRegistry runner_registry;
  DemoRandomWalkRunner random_walk_runner(&partitioner, &memory_allocator);
  DemoEmbeddingPsRunner embedding_ps_runner(&partitioner, &memory_allocator);
  runner_registry.Register(1, &random_walk_runner);
  runner_registry.Register(2, &embedding_ps_runner);
  random_walk_runner.StartProcessLoop();
  embedding_ps_runner.StartProcessLoop();

  std::unique_ptr<AsyncCommunicator>
      async_communicator(new AsyncCommunicator(&partitioner, &memory_allocator, &runner_registry, pconfig));
  async_communicator->CreateResources();
  async_communicator->Start();

  //SingleRequestWorkFunction(global_rank, &random_walk_runner, &embedding_ps_runner, async_communicator.get());
  AllRequestWorkFunction(global_rank, &random_walk_runner, &embedding_ps_runner, async_communicator.get());

  async_communicator->WaitStopped();
  async_communicator->DestroyResources();
}

int main(int argc, char **argv) {
  if (!ParseArgs(argc, argv)) {
    return 1;
  }
  Config config;
  ParseConfigFile(config_file, &config);
  Config *pconfig = &config;

  global_node_count = config.GetNodeCount();
  ranks_per_node = config.GetRanksPerNode();

  if (global_node_count > 1) {
    IbInit();
  }
  EnableAllPeerAccess();

  int local_size = ranks_per_node;
  std::vector<std::unique_ptr<std::thread>> rank_threads(local_size);
  for (int i = 0; i < local_size; i++) {
    int global_rank = i + local_size * node_rank;
    rank_threads[i] = std::make_unique<std::thread>([global_rank, pconfig]() {
      RankFunction(global_rank, pconfig);
    });
  }
  for (int i = 0; i < local_size; i++) {
    rank_threads[i]->join();
  }
  PrintRefCounts();
  PrintReqResCount();

  if (global_node_count > 1) {
    IbDeInit();
  }

  return 0;
}

