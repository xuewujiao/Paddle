#include "async_communicator.h"

#include <atomic>
#include <cstring>
#include <memory>

#include "check_macros.h"
#include "log_macros.h"

static std::atomic<int64_t> reqres_count{0};

AsyncReqRes::AsyncReqRes() {
  reqres_count.fetch_add(1);
  Init();
}

AsyncReqRes::~AsyncReqRes() {
  reqres_count.fetch_add(-1);
}

void AsyncReqRes::Init() {
  InitMeta(&meta);
  for (int i = 0; i < MAX_VEC_COUNT; i++) {
    memory_contexts[i] = nullptr;
  }
}

void AsyncReqRes::FillMetaByMemoryContext() {
  for (int i = 0; i < meta.valid_data_count; i++) {
    if (memory_contexts[i] == nullptr || memory_contexts[i]->GetDataType() == DT_UNDEFINED) {
      LOG_FATAL("memory_context[%d] is nullptr or data_type is undefined, maybe not set or already consumed.", i);
    }
    meta.data_types[i] = memory_contexts[i]->GetDataType();
    meta.data_sizes[i] = memory_contexts[i]->GetSize();
    meta.locations[i] = memory_contexts[i]->GetMemoryLocation();
  }
}

AsyncReqRes *CreateAsyncReqRes() {
  return new AsyncReqRes();
}

void DestroyAsyncReqRes(AsyncReqRes* req_res) {
  delete req_res;
}

void PrintReqResCount() {
  LOG_INFO("AsyncReqRes Count = %ld", reqres_count.load());
}

RequestHandle::RequestHandle() : started_(false), completed_(false) {
  response_ = nullptr;
}

RequestHandle::~RequestHandle() {
  Recycle();
}

void RequestHandle::SetStarted() {
  std::unique_lock<std::mutex> mlock(mutex_);
  BOOL_CHECK(completed_ == false);
  BOOL_CHECK(started_ == false);
  started_ = true;
}

void RequestHandle::SetCompleted() {
  std::unique_lock<std::mutex> mlock(mutex_);
  BOOL_CHECK(started_ == true);
  BOOL_CHECK(completed_ == false);
  completed_ = true;
  cv_.notify_all();
}

bool RequestHandle::IsCompleted() {
  std::unique_lock<std::mutex> mlock(mutex_);
  BOOL_CHECK(started_ == true);
  return completed_;
}

void RequestHandle::Wait() {
  std::unique_lock<std::mutex> mlock(mutex_);
  BOOL_CHECK(started_ == true);
  while(!completed_) {
    cv_.wait(mlock);
  }
}

void RequestHandle::Recycle() {
  std::unique_lock<std::mutex> mlock(mutex_);
  BOOL_CHECK(started_ == completed_);
  started_ = false;
  completed_ = false;
  response_ = nullptr;
}

AsyncCommunicator::AsyncCommunicator(Partitioner *partitioner,
                                     MemoryAllocatorBase *allocator,
                                     RunnerRegistry *runner_registry,
                                     Config* config)
    : partitioner_(partitioner), allocator_(allocator), runner_registry_(runner_registry), config_(config) {
}

AsyncCommunicator::~AsyncCommunicator() {

}

void AsyncCommunicator::CreateResources() {
  intra_node_communicator_ = std::make_unique<IntraNodeCommunicator>(partitioner_, allocator_);
  merged_handler_ = std::make_unique<MergedHandler>(partitioner_, allocator_, this);
  if (partitioner_->GetNodeCount() > 1) {
    inter_node_communicator_ = std::make_unique<InterNodeCommunicator>(partitioner_, allocator_, config_);
  }
  sideband_communicator_ = std::make_unique<SideBandCommunicator>(partitioner_,
                                                                  config_->sideband_server_name,
                                                                  config_->sideband_server_port);
  sideband_communicator_->Start();
  intra_node_communicator_->SetAsyncCommunicator(this);
  intra_node_communicator_->CreateResources();
  if (partitioner_->GetNodeCount() > 1) {
    inter_node_communicator_->SetAsyncCommunicator(this);
    inter_node_communicator_->CreateResources();
  }
  runner_registry_->Traverse([this](RunnerBase* runner_base) {
    runner_base->SetAsyncCommunicator(this);
  });
  sideband_communicator_->Barrier();
  intra_node_communicator_->ConnectFifos();
  sideband_communicator_->Barrier();
  if (partitioner_->GetNodeCount() > 1) {
    inter_node_communicator_->Connect();
  }
  sideband_communicator_->Stop();
}

void AsyncCommunicator::DestroyResources() {
  intra_node_communicator_->DestroyResources();
  intra_node_communicator_.reset();
  if (partitioner_->GetNodeCount() > 1) {
    inter_node_communicator_->DestroyResources();
    inter_node_communicator_.reset();
  }
  merged_handler_.reset();
}

void AsyncCommunicator::Start() {
  merged_handler_->Start();
  intra_node_communicator_->Start();
  if (partitioner_->GetNodeCount() > 1) {
    inter_node_communicator_->Start();
  }
}

void AsyncCommunicator::PutRequestAsync(RequestHandle *request_handle, bool need_notify_oneway_req) {
  auto* request = request_handle->request_;
  bool need_response = false;
  if (!IsStopMeta(&request->meta)) {
    RunnerBase *runner = runner_registry_->Find(request->meta.runner_id);
    need_response = runner->FuncNeedResponse(&request->meta);
  }
  request_handle->SetStarted();
  if (!need_response && need_notify_oneway_req) {
    request->complete_cb = [request_handle]() {
      request_handle->SetCompleted();
    };
  }
  if (need_response) {
    BOOL_CHECK(request_handle->response_ != nullptr);
    std::unique_lock<std::mutex> mlock(mutex_);
    pending_mapping_.insert(std::make_pair(request->meta.request_id, request_handle));
  }
  if (IsLocalNode(&request->meta)) {
    intra_node_communicator_->Send(request);
  } else {
    inter_node_communicator_->Send(request);
  }
  if (!need_response && !need_notify_oneway_req) {
    request_handle->SetCompleted();
  }
}

void AsyncCommunicator::PutRequestSync(AsyncReqRes *request, AsyncReqRes *response) {
  RequestHandle request_handle;
  request_handle.request_ = request;
  request_handle.response_ = response;
  PutRequestAsync(&request_handle);
  request_handle.Wait();
}

void AsyncCommunicator::PutResponseAsync(AsyncReqRes *response) {
  if (IsLocalNode(&response->meta)) {
    intra_node_communicator_->Send(response);
  } else {
    inter_node_communicator_->Send(response);
  }
}

void AsyncCommunicator::SendStopSignal() {
  for (int r = 0; r < partitioner_->GetGlobalSize(); r++) {
    SendStopSignalToSingleRank(r);
  }
}

void AsyncCommunicator::SendStopSignalToSingleRank(int global_rank) {
  auto* request = CreateAsyncReqRes();
  RequestHandle request_handle;
  CreateStopMetaToRank(&request->meta, global_rank, partitioner_);
  request_handle.request_ = request;
  request_handle.response_ = nullptr;
  PutRequestAsync(&request_handle);
}

void AsyncCommunicator::WaitStopped() {
  runner_registry_->Traverse([](RunnerBase* runner) {
    runner->WaitStopped();
  });
  response_queue_.Flush();
  response_queue_.WaitUntilFinished();
  BOOL_CHECK(pending_mapping_.empty());
  intra_node_communicator_->Stop();
  intra_node_communicator_->WaitStopped();
  if (partitioner_->GetNodeCount() > 1) {
    inter_node_communicator_->Stop();
    inter_node_communicator_->WaitStopped();
  }
  merged_handler_->Stop();
  merged_handler_->WaitStopped();
}

void AsyncCommunicator::OnReceiveRequest(AsyncReqRes* request) {
  if (IsStopMeta(&request->meta)) {
    runner_registry_->Traverse([request](RunnerBase* runner){
      runner->Process(request, nullptr);
    });
    allocator_->FreeReqRes(request);
    return;
  }
  if (IsMergedMeta(&request->meta)) {
    merged_handler_->PutRequestAsync(request);
    return;
  }
  RunnerBase* runner = runner_registry_->Find(request->meta.runner_id);
  bool need_response = runner->FuncNeedResponse(&request->meta);
  runner->Process(request, [this, need_response](AsyncReqRes* response){
    if (need_response) {
      this->PutResponseAsync(response);
    }
  });
}

void AsyncCommunicator::OnReceiveResponse(AsyncReqRes *response) {
  if (IsMergedSplittedMeta(&response->meta)) {
    merged_handler_->OnReceiveResponse(response);
    return;
  }
  RequestHandle* request_handle = nullptr;
  {
    std::unique_lock<std::mutex> mlock(mutex_);
    auto it = pending_mapping_.find(response->meta.request_id);
    BOOL_CHECK(it != pending_mapping_.end());
    request_handle = it->second;
    pending_mapping_.erase(response->meta.request_id);
  }
  BOOL_CHECK(request_handle->response_ == response);
  request_handle->SetCompleted();
}

AsyncReqRes* AsyncCommunicator::OnRunnerGetResponse(AsyncReqRes* request) {
  AsyncReqRes* response = nullptr;
  if (IsLocalGPU(&request->meta)) {
    if (IsMergedSplittedMeta(&request->meta)) {
      return merged_handler_->OnRunnerGetResponse(&request->meta);
    } else {
      // [Optimization] Local GPU can just pick response from pending_map_ to save one copy.
      std::unique_lock<std::mutex> mlock(mutex_);
      auto it = pending_mapping_.find(request->meta.request_id);
      BOOL_CHECK(it != pending_mapping_.end());
      auto *request_handle = it->second;
      response = request_handle->response_;
    }
  }
  if (response == nullptr) {
    response = CreateAsyncReqRes();
  }
  MakeResponseMeta(&response->meta, &request->meta);
  return response;
}

AsyncReqRes* AsyncCommunicator::OnReceiveGetResponse(Meta* meta) {
  if (IsMergedSplittedMeta(meta)) {
    return merged_handler_->OnReceiveGetResponse(meta);
  } else {
    std::unique_lock<std::mutex> mlock(mutex_);
    auto it = pending_mapping_.find(meta->request_id);
    BOOL_CHECK(it != pending_mapping_.end());
    auto *request_handle = it->second;
    return request_handle->response_;
  }
}
