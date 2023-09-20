#include "merged_handler.h"

#include <algorithm>
#include <memory>
#include <random>

#include "async_communicator.h"
#include "check_macros.h"
#include "intranode_communicator.h"
#include "log_macros.h"

class RawPtrMemoryContext : public MemoryContextBase {
 public:
  RawPtrMemoryContext() {
    need_allocator_ = false;
  }
  ~RawPtrMemoryContext() override = default;
  void *GetPointer() override {
    return pointer_;
  }
  void SetPointer(void *p) {
    pointer_ = p;
  }
 protected:
  void *pointer_ = nullptr;
};

MergedHandler::MergedHandler(Partitioner *partitioner,
                             MemoryAllocatorBase *allocator,
                             AsyncCommunicator *async_communicator)
    : allocator_(allocator), partitioner_(partitioner), async_communicator_(async_communicator) {
  BOOL_CHECK(partitioner_->GetRanksPerNode() <= MAX_GPUS_PER_NODE);
  pending_frees_.store(0);
}
MergedHandler::~MergedHandler() {
}
void MergedHandler::Start() {
  dispatch_thread_ = std::make_unique<std::thread>([this]() {
    DispatchThreadFunc();
  });
  free_thread_ = std::make_unique<std::thread>([this]() {
    FreeThreadFunc();
  });
}
void MergedHandler::Stop() {
  request_message_queue_.Flush();
}
void MergedHandler::WaitStopped() {
  dispatch_thread_->join();
  dispatch_thread_.reset();
  free_thread_->join();
  free_thread_.reset();
}
void MergedHandler::PutRequestAsync(AsyncReqRes* request) {
  request_message_queue_.Push(request);
}
static bool MakeSplittedRequest(AsyncReqRes *request,
                                AsyncReqRes *response,
                                AsyncReqRes *s_req,
                                AsyncReqRes *s_res,
                                MergeHeader *header,
                                int lr) {
  s_req->meta = request->meta;
  s_req->meta.valid_data_count--;
  s_req->meta.merged_flag = 2;
  s_req->meta.requester_node_id = s_req->meta.runner_node_id = request->meta.runner_node_id;
  s_req->meta.requester_lane_id = request->meta.runner_lane_id;
  s_req->meta.runner_lane_id = lr;
  size_t total_req_size = 0;
  for (int i = 0; i < s_req->meta.valid_data_count; i++) {
    auto req_data_type = static_cast<DataType>(request->meta.data_types[i]);
    auto req_ml = static_cast<MemoryLocation>(request->meta.locations[i]);
    s_req->meta.data_types[i] = req_data_type;
    s_req->meta.locations[i] = request->meta.locations[i];
    s_req->meta.data_sizes[i] = header->ranks_info[lr].request_size[i];
    auto *ptr_ctx = new RawPtrMemoryContext();
    s_req->memory_contexts[i] = ptr_ctx;
    ptr_ctx->SetPointer(
        static_cast<char *>(request->memory_contexts[i]->GetPointer()) + header->ranks_info[lr].request_offset[i]);
    BOOL_CHECK(header->ranks_info[lr].request_size[i] % GetElementSize(req_data_type) == 0);
    ptr_ctx->SetContextInfo(header->ranks_info[lr].request_size[i] / GetElementSize(req_data_type),
                            req_ml,
                            req_data_type);
    total_req_size += header->ranks_info[lr].request_size[i];
  }
  if (s_res == nullptr) return total_req_size > 0;
  MakeResponseMeta(&s_res->meta, &s_req->meta);
  s_res->meta.valid_data_count = header->response_data_count;
  for (int i = 0; i < s_res->meta.valid_data_count; i++) {
    auto res_data_type = static_cast<DataType>(header->response_info[i].dtype);
    auto res_ml = static_cast<MemoryLocation>(header->response_info[i].location);
    s_res->meta.data_types[i] = res_data_type;
    s_res->meta.locations[i] = res_ml;
    s_res->meta.data_sizes[i] = header->ranks_info[lr].response_size[i];
    auto* ptr_ctx = new RawPtrMemoryContext();
    s_res->memory_contexts[i] = ptr_ctx;
    ptr_ctx->SetPointer(
        static_cast<char *>(response->memory_contexts[i]->GetPointer()) + header->ranks_info[lr].response_offset[i]);
    BOOL_CHECK(header->ranks_info[lr].response_size[i] % GetElementSize(res_data_type) == 0);
    ptr_ctx->SetContextInfo(header->ranks_info[lr].response_size[i] / GetElementSize(res_data_type),
                            res_ml,
                            res_data_type);
  }
  return total_req_size > 0;
}
void MergedHandler::DispatchThreadFunc() {
  AsyncReqRes *request = nullptr;
  std::vector<bool> should_send(partitioner_->GetRanksPerNode());
  std::random_device rd;
  std::default_random_engine engine(rd());
  std::vector<int> send_lr_lb(partitioner_->GetRanksPerNode());
  for (int i = 0; i < partitioner_->GetRanksPerNode(); i++) {
    send_lr_lb[i] = i;
  }
  std::shuffle(send_lr_lb.begin(), send_lr_lb.end(), engine);
  CUDA_CHECK(cudaSetDevice(partitioner_->GetLocalRank()));
  while (request_message_queue_.WaitAndPop(&request)) {
    BOOL_CHECK(request->meta.valid_data_count >= 1);
    LOG_INFO("[MergedHandler::DispatchThreadFunc] splitting request %lx", request->meta.request_id);
    int merge_header_idx = request->meta.valid_data_count - 1;
    auto *merge_header = static_cast<MergeHeader *>(request->memory_contexts[merge_header_idx]->GetPointer());
    bool need_response = merge_header->response_data_count > 0;
    BOOL_CHECK(need_response == true);
    auto *merged_handle = new MergedHandle();
    merged_handle->request = request;
    int need_free_count = 1;                              // raw request (raw response freed by copy thread)
    if (need_response) need_free_count += partitioner_->GetRanksPerNode();                // splitted response
    pending_frees_.fetch_add(need_free_count);
    merged_handle->pending_count = partitioner_->GetRanksPerNode();
    AsyncReqRes *response = nullptr;
    if (need_response) {
      response = async_communicator_->OnRunnerGetResponse(request);
      response->meta.valid_data_count = merge_header->response_data_count;
      for (int i = 0; i < response->meta.valid_data_count; i++) {
        response->meta.data_sizes[i] = merge_header->response_info[i].data_size;
        response->meta.locations[i] = merge_header->response_info[i].location;
        response->meta.data_types[i] = merge_header->response_info[i].dtype;
      }
      allocator_->AllocateReqResByMeta(response);
    }
    merged_handle->response = response;
    int real_send_count = 0;
    for (int lr = 0; lr < partitioner_->GetRanksPerNode(); lr++) {
      auto *s_req = CreateAsyncReqRes();
      merged_handle->splitted_requests[lr] = s_req;
      AsyncReqRes* s_res = nullptr;
      if (need_response) {
        s_res = CreateAsyncReqRes();
        merged_handle->splitted_responses[lr] = s_res;
      }
      should_send[lr] = MakeSplittedRequest(request, response, s_req, s_res, merge_header, lr);
      if (should_send[lr]) real_send_count++;
    }
    {
      std::lock_guard<std::mutex> mlock(mutex_);
      BOOL_CHECK(request_map_.find(request->meta.request_id) == request_map_.end());
      request_map_.insert(std::pair<uint64_t, MergedHandle*>(request->meta.request_id, merged_handle));
    }
    merged_handle->done_cb = [this, need_response](MergedHandle *handle) {
      {
        std::lock_guard<std::mutex> mlock(mutex_);
        request_map_.erase(handle->response->meta.request_id);
      }
      to_free_queue_.Push(handle->request);
      async_communicator_->PutResponseAsync(handle->response);
      for (int i = 0; i < partitioner_->GetRanksPerNode(); i++) {
        if (need_response) {
          to_free_queue_.Push(handle->splitted_responses[i]);
        }
      }
      delete handle;
    };
    merged_handle->pending_count = partitioner_->GetRanksPerNode();
    for (int i = 0; i < partitioner_->GetRanksPerNode(); i++) {
      int tgt_localid = send_lr_lb[i];
      if (!should_send[tgt_localid]) continue;
      async_communicator_->intra_node_communicator_->Send(merged_handle->splitted_requests[tgt_localid]);
    }
    std::shuffle(send_lr_lb.begin(), send_lr_lb.end(), engine);
    if (real_send_count == 0) {
      merged_handle->done_cb(merged_handle);
    }
  }
  while(pending_frees_.load() > 0) {
    usleep(5 * 1000);
  }
  to_free_queue_.Flush();
}

// should be called with local GPU
AsyncReqRes* MergedHandler::OnRunnerGetResponse(Meta* meta) {
  MergedHandle *handle = nullptr;
  {
    std::lock_guard<std::mutex> mlock(mutex_);
    auto it = request_map_.find(meta->request_id);
    if (it != request_map_.end()) {
      handle = it->second;
    }
  }
  int local_id = GetSrcLocalRankFromMeta(meta);
  return handle == nullptr ? nullptr : handle->splitted_responses[local_id];
}

AsyncReqRes *MergedHandler::OnReceiveGetResponse(Meta *meta) {
  MergedHandle *handle = nullptr;
  {
    std::lock_guard<std::mutex> mlock(mutex_);
    auto it = request_map_.find(meta->request_id);
    BOOL_CHECK(it != request_map_.end());
    handle = it->second;
  }
  int local_id = GetSrcLocalRankFromMeta(meta);
  return handle->splitted_responses[local_id];
}

void MergedHandler::OnReceiveResponse(AsyncReqRes *response) {
  MergedHandle *handle = nullptr;
  {
    std::lock_guard<std::mutex> mlock(mutex_);
    auto it = request_map_.find(response->meta.request_id);
    BOOL_CHECK(it != request_map_.end());
    handle = it->second;
  }
  handle->FinishRequest();
}
void MergedHandler::FreeThreadFunc() {
  AsyncReqRes* req_res = nullptr;
  CUDA_CHECK(cudaSetDevice(partitioner_->GetLocalRank()));
  while (to_free_queue_.WaitAndPop(&req_res)) {
    allocator_->FreeReqRes(req_res);
    pending_frees_.fetch_add(-1);
    req_res = nullptr;
  }
}
