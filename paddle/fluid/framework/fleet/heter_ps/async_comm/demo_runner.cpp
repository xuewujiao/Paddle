#include "demo_runner.h"

#include <cstring>

#include "async_communicator.h"
#include "check_macros.h"
#include "demo_kernels.h"
#include "log_macros.h"

void DemoRandomWalkRunner::RegisterFunctions() {
  FunctionInfo function_info;
  function_info.function_id = function_info_table_.size();
  function_info.input_data_count = 1;
  function_info.output_data_count = 1;
  function_info.need_response = true;
  function_info.func_ = [this](AsyncReqRes* request, AsyncReqRes* response) {
    GetRandomWalkResult(request, response);
  };
  function_info.input_locations[0] = ML_DEVICE;
  function_info.output_locations[0] = ML_DEVICE;
  function_info_table_.push_back(function_info);
}

AsyncReqRes* DemoRandomWalkRunner::MakeRandomWalkRequest(MemoryContextBase* memory_context, int target_global_rank) {
  AsyncReqRes* request = CreateAsyncReqRes();
  request->meta.valid_data_count = 1;
  request->memory_contexts[0] = memory_context;
  request->FillMetaByMemoryContext();
  static constexpr int kRandomWalkFuncId = 0;
  CreateRequestMeta(&request->meta, target_global_rank, kRandomWalkFuncId);
  return request;
}

void DemoRandomWalkRunner::ProcessSetup() {
  QueuedRunner::ProcessSetup();
  CUDA_CHECK(cudaSetDevice(partitioner_->GetLocalRank()));
  CUDA_CHECK(cudaStreamCreate(&stream_));
}
void DemoRandomWalkRunner::ProcessCleanUp() {
  CUDA_CHECK(cudaStreamDestroy(stream_));
  QueuedRunner::ProcessCleanUp();
}
void DemoRandomWalkRunner::GetRandomWalkResult(AsyncReqRes *request, AsyncReqRes *response) {
  DataType dt = static_cast<DataType>(request->meta.data_types[0]);
  size_t elt_count = request->meta.data_sizes[0] / GetElementSize(dt);
  void* ptr = allocator_->AllocateOrUseExistContextPointer(response, 0, ML_DEVICE, dt, elt_count);
  LOG_INFO("[RandWalk] rank=%d processing request from requester_node_id=%d, requester_lane_id=%d,"
           " runner_node_id=%d, runner_lane_id=%d, elt_count=%ld",
           partitioner_->GetGlobalRank(),
           (int)request->meta.requester_node_id,
           (int)request->meta.requester_lane_id,
           (int)request->meta.runner_node_id,
           (int)request->meta.runner_lane_id,
           elt_count);
  GetNextRandomWalkId((int*)request->memory_contexts[0]->GetPointer(),
                      elt_count,
                      (int*)ptr,
                      stream_);
  response->meta.valid_data_count = 1;
  response->FillMetaByMemoryContext();
}
void DemoEmbeddingPsRunner::RegisterFunctions() {
  FunctionInfo function_info;
  function_info.function_id = function_info_table_.size();
  function_info.input_data_count = 1;
  function_info.output_data_count = 1;
  function_info.need_response = true;
  function_info.func_ = [this](AsyncReqRes* request, AsyncReqRes* response) {
    GetEmbedding(request, response);
  };
  function_info.input_locations[0] = ML_DEVICE;
  function_info.output_locations[0] = ML_DEVICE;
  function_info_table_.push_back(function_info);

  function_info.function_id = function_info_table_.size();
  function_info.input_data_count = 2;
  function_info.output_data_count = 0;
  function_info.need_response = false;
  function_info.func_ = [this](AsyncReqRes* request, AsyncReqRes* response) {
    UpdateEmbedding(request, response);
  };
  function_info.input_locations[0] = ML_DEVICE;
  function_info.input_locations[1] = ML_DEVICE;
  function_info_table_.push_back(function_info);
}

AsyncReqRes *DemoEmbeddingPsRunner::MakePsGetRequest(MemoryContextBase *memory_context, int target_global_rank) {
  AsyncReqRes *request = CreateAsyncReqRes();
  request->meta.valid_data_count = 1;
  request->memory_contexts[0] = memory_context;
  request->FillMetaByMemoryContext();
  static constexpr int kPsGetFuncId = 0;
  CreateRequestMeta(&request->meta, target_global_rank, kPsGetFuncId);
  return request;
}

AsyncReqRes *DemoEmbeddingPsRunner::MakePsUpdateRequest(MemoryContextBase *indice_context,
                                                        MemoryContextBase *grad_context,
                                                        int target_global_rank) {
  AsyncReqRes *request = CreateAsyncReqRes();
  request->meta.valid_data_count = 2;
  request->memory_contexts[0] = indice_context;
  request->memory_contexts[1] = grad_context;
  request->FillMetaByMemoryContext();
  static constexpr int kPsUpdateFuncId = 1;
  CreateRequestMeta(&request->meta, target_global_rank, kPsUpdateFuncId);
  return request;
}

void DemoEmbeddingPsRunner::ProcessSetup() {
  QueuedRunner::ProcessSetup();
  CUDA_CHECK(cudaSetDevice(partitioner_->GetLocalRank()));
  CUDA_CHECK(cudaStreamCreate(&stream_));
}
void DemoEmbeddingPsRunner::ProcessCleanUp() {
  CUDA_CHECK(cudaStreamDestroy(stream_));
  QueuedRunner::ProcessCleanUp();
}
void DemoEmbeddingPsRunner::GetEmbedding(AsyncReqRes *request, AsyncReqRes *response) {
  auto input_dt = static_cast<DataType>(request->meta.data_types[0]);
  size_t elt_count = request->meta.data_sizes[0] / GetElementSize(input_dt);
  LOG_INFO("[GetEmbedding] rank=%d processing request from requester_node_id=%d, requester_lane_id=%d,"
           " runner_node_id=%d, runner_lane_id=%d, elt_count=%ld",
           partitioner_->GetGlobalRank(),
           (int)request->meta.requester_node_id,
           (int)request->meta.requester_lane_id,
           (int)request->meta.runner_node_id,
           (int)request->meta.runner_lane_id,
           elt_count);
  DataType embedding_dt = DT_FLOAT;
  int embedding_dim = embedding_dim_;
  void* input_idx_ptr = request->memory_contexts[0]->GetPointer();
  auto *output_ptr = static_cast<float *>(allocator_->AllocateOrUseExistContextPointer(response,
                                                                                       0,
                                                                                       ML_DEVICE,
                                                                                       embedding_dt,
                                                                                       elt_count * embedding_dim));
  GetNodeEmbedding((int*)input_idx_ptr, elt_count, embedding_dim_, output_ptr, stream_);
#if 0
  for (size_t input_idx = 0; input_idx < elt_count; input_idx++) {
    int64_t input_id = -1;
    if (input_dt == DT_INT32) {
      input_id = static_cast<int*>(input_idx_ptr)[input_idx];
    } else {
      input_id = static_cast<int64_t*>(input_idx_ptr)[input_idx];
    }
    for (int embedding_idx = 0; embedding_idx < embedding_dim; embedding_idx++) {
      output_ptr[input_idx * embedding_dim + embedding_idx] = (float)input_id;
    }
  }
#endif
  response->meta.valid_data_count = 1;
  response->FillMetaByMemoryContext();
}
void DemoEmbeddingPsRunner::UpdateEmbedding(AsyncReqRes *request, AsyncReqRes *response) {
  auto input_dt = static_cast<DataType>(request->meta.data_types[0]);
  size_t elt_count = request->meta.data_sizes[0] / GetElementSize(input_dt);
  LOG_INFO("[UpdateEmbedding] rank=%d processing request from requester_node_id=%d, requester_lane_id=%d,"
           " runner_node_id=%d, runner_lane_id=%d, elt_count=%ld",
           partitioner_->GetGlobalRank(),
           (int)request->meta.requester_node_id,
           (int)request->meta.requester_lane_id,
           (int)request->meta.runner_node_id,
           (int)request->meta.runner_lane_id,
           elt_count);
  CheckNodeGradients((int*)request->memory_contexts[0]->GetPointer(),
                     elt_count,
                     embedding_dim_,
                     (float*)request->memory_contexts[1]->GetPointer(),
                     stream_);
}
