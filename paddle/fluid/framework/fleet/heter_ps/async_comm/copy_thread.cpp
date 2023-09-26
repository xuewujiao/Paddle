#include "copy_thread.h"

#include <string.h>

#include <algorithm>
#include <chrono>
#include <functional>
#include <memory>
#include <random>

#include "agent_copy_message.h"
#include "check_macros.h"
#include "cuda_utils.h"
#include "fifo_utils.h"
#include "log_macros.h"
#include "partitioner.h"
#include "memory_allocator.h"
//#include "gflags/gflags.h"
#include "async_communicator.h"
#include "meta.h"
#include "paddle/phi/core/flags.h"
DECLARE_uint64(async_buffer_size);
DECLARE_uint64(async_buffer_count);


CopyThread::CopyThread(Partitioner *partitioner, MemoryAllocatorBase *allocator) :
    partitioner_(partitioner), allocator_(allocator) {
  stopped_.store(false);
  sender_exited_.store(false);
  total_credit_count_.store(0);
  send_credits_.resize(partitioner->GetNodeCount());
}
CopyThread::~CopyThread() {

}
void CopyThread::CreateResources() {
  send_queues_.resize(partitioner_->GetNodeCount());
  recv_fifo_fd_ = CreateFifo(GetAgentCopyFifoName("copy_recv",
                                                  partitioner_->GetNodeID(),
                                                  partitioner_->GetLocalRank()));
}
void CopyThread::DestroyResources() {
  for (int i = 0; i < partitioner_->GetNodeCount(); i++) {
    if (!send_queues_[i].empty()) {
      LOG_WARN("Rank=%d CopyThread send_queues_[%d].size()=%d",
               partitioner_->GetGlobalRank(), i, send_queues_[i].size());
    }
  }
  UnlinkFifo(GetAgentCopyFifoName("copy_recv", partitioner_->GetNodeID(), partitioner_->GetLocalRank()));
}
void CopyThread::ConnectFifo() {
  send_fifo_fds_.resize(partitioner_->GetRanksPerNode(), -1);
  for (int lr = 0; lr < partitioner_->GetRanksPerNode(); lr++) {
    send_fifo_fds_[lr] =
        WaitAndOpenFifo(GetAgentCopyFifoName("agent_recv", partitioner_->GetNodeID(), lr));
  }
  send_to_rail_fifo_fd_ = send_fifo_fds_[partitioner_->GetLocalRank()];
}
void CopyThread::Start() {
  send_thread_ = std::make_unique<std::thread>([this]() {
    this->SendThreadFunc();
  });
  recv_thread_ = std::make_unique<std::thread>([this]() {
    this->RecvThreadFunc();
  });
}
void CopyThread::Stop() {
  stopped_.store(true);
}
void CopyThread::WaitStopped() {
  send_thread_->join();
  send_thread_.reset();
  recv_thread_->join();
  recv_thread_.reset();
}
void CopyThread::Send(AsyncReqRes *req_res) {
  // Local node request should be optimized by IntraNodeCommunicator.
  BOOL_CHECK(!IsLocalNode(&req_res->meta));
  LOG_DEBUG(" {1.CopyThread::Send} got one req_res");
  int target_node_id = GetTargetNodeIdFromMeta(&req_res->meta);
  {
    std::unique_lock<std::mutex> mlock(send_mutex_);
    send_queues_[target_node_id].push(req_res);
    if (send_queues_[target_node_id].size() == 1) {
      send_cv_.notify_all();
    }
  }
}
struct CopyToBufferState {
  CopyToBufferState() {
    req_res = nullptr;
    total_package_needed = 0;
  }
  void Clear() {
    req_res = nullptr;
    total_package_needed = 0;
    current_package_id = 0;
  }
  void SetReqRes(AsyncReqRes *new_req_res) {
    BOOL_CHECK(req_res == nullptr);
    req_res = new_req_res;
    size_t total_data_size = 0;
    for (int i = 0; i < req_res->meta.valid_data_count; i++) {
      total_data_size += req_res->meta.data_sizes[i];
    }
    size_t kDataSizePerBuffer = FLAGS_async_buffer_size - sizeof(Header);
    total_package_needed = (total_data_size + kDataSizePerBuffer - 1) / kDataSizePerBuffer;
    if (total_package_needed == 0) {
      // Header need at least 1 package.
      total_package_needed = 1;
    }
  }
  size_t CopyToBuffer(IntraNodeCredit credit, cudaStream_t stream) {
    char *buffer_ptr = (char *) credit.pointer;
    size_t kDataSizePerBuffer = FLAGS_async_buffer_size - sizeof(Header);
    size_t data_offset = current_package_id * kDataSizePerBuffer;
    MemcpyUnique(buffer_ptr, &req_res->meta, sizeof(Meta), stream);
    static_assert(sizeof(Header) == sizeof(Meta), "Header size not equal with Meta, should add addition copy.");
    buffer_ptr = buffer_ptr + sizeof(Header);
    size_t buffer_left = kDataSizePerBuffer;
    for (int i = 0; i < req_res->meta.valid_data_count; i++) {
      if (data_offset >= req_res->meta.data_sizes[i]) {
        data_offset -= req_res->meta.data_sizes[i];
        continue;
      }
      size_t tensor_size = req_res->meta.data_sizes[i] - data_offset;
      size_t copy_size = std::min(tensor_size, buffer_left);
      MemcpyUnique(buffer_ptr, (char *) req_res->memory_contexts[i]->GetPointer() + data_offset, copy_size, stream);
      buffer_ptr += copy_size;
      buffer_left -= copy_size;
      data_offset = 0;
      if (buffer_left == 0) break;
    }
    current_package_id++;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return FLAGS_async_buffer_size - buffer_left;
  }
  bool Finished() const {
    return current_package_id == total_package_needed;
  }
  AsyncReqRes *req_res = nullptr;
  int total_package_needed = 0;
  int current_package_id = 0;
};
void CopyThread::SendThreadFunc() {
  CUDA_CHECK(cudaSetDevice(partitioner_->GetLocalRank()));
  cudaStream_t send_copy_stream = CreateHighPriorityStream(cudaStreamNonBlocking);
  std::vector<CopyToBufferState> states(partitioner_->GetNodeCount());
  auto has_work_fn = [this]() {
    for (int i = 0; i < partitioner_->GetNodeCount(); i++) {
      if (!send_queues_[i].empty()) {
        return true;
      }
    }
    return false;
  };
  auto can_do_work_fn = [this]() {
    for (int i = 0; i < partitioner_->GetNodeCount(); i++) {
      if (!send_queues_[i].empty() && !send_credits_[i].empty()) {
        return true;
      }
    }
    return false;
  };
  struct WorkItem {
    AsyncReqRes *req_res;
    int local_rank;
    IntraNodeCredit credit;
  };
  auto get_work_fn = [this](std::vector<WorkItem> *p_work_items) {
    p_work_items->clear();
    for (int i = 0; i < partitioner_->GetNodeCount(); i++) {
      if (!send_queues_[i].empty() && !send_credits_[i].empty()) {
        WorkItem work_item;
        work_item.req_res = send_queues_[i].front();
        BOOL_CHECK(work_item.req_res != nullptr);
        work_item.local_rank = i;
        work_item.credit = send_credits_[i].front();
        send_credits_[i].pop();
        total_credit_count_.fetch_add(-1);
        p_work_items->push_back(work_item);
      }
    }
  };

  std::vector<WorkItem> work_items;
  work_items.reserve(partitioner_->GetNodeCount());

  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, INT32_MAX);
  std::mt19937 perm_gen(std::random_device{}());

  std::unique_lock<std::mutex> send_lock(send_mutex_);
  // just use stopped_ signal and no pending work to exit is OK for send
  while (!stopped_.load() || has_work_fn()) {
    send_cv_.wait_for(send_lock, std::chrono::milliseconds(50), can_do_work_fn);
    get_work_fn(&work_items);
    send_lock.unlock();
    if (!work_items.empty()) {
      std::shuffle(work_items.begin(), work_items.end(), perm_gen);
      int start_idx = distribution(generator) % (int) work_items.size();
      for (int idx = start_idx; idx < start_idx + (int)work_items.size(); idx++) {
        int real_idx = idx % work_items.size();
        auto &work_item = work_items[real_idx];
        int target_node_id = GetTargetNodeIdFromMeta(&work_item.req_res->meta);
        auto &copy_state = states[target_node_id];
        if (copy_state.req_res == nullptr) {
          copy_state.SetReqRes(work_item.req_res);
        } else {
          BOOL_CHECK(copy_state.req_res == work_item.req_res);
        }
        size_t data_size = copy_state.CopyToBuffer(work_item.credit, send_copy_stream);
        // send buffer filled message
        AgentCopyMessage agent_copy_message;
        agent_copy_message.MakeMsg(work_item.credit.pointer,
                                   data_size,
                                   partitioner_->GetGlobalRank(),
                                   GetTargetGlobalRankFromMeta(&work_item.req_res->meta,
                                                               partitioner_->GetRanksPerNode()),
                                   1,
                                   IsStopMeta(&work_item.req_res->meta));
        LOG_DEBUG(" {2.CopyThread::SendThreadFunc} sending to Agent through fifo.");
        SingleFifoWrite(send_to_rail_fifo_fd_, &agent_copy_message, sizeof(AgentCopyMessage));
        if (copy_state.Finished()) {
          copy_state.Clear();
          send_lock.lock();
          send_queues_[target_node_id].pop();
          send_lock.unlock();
          allocator_->FreeReqRes(work_item.req_res);
        }
      }
    }
    send_lock.lock();
  }
  send_lock.unlock();
  sender_exited_.store(true);
  // no need to close as RecvThreadFunc will close all send_fifo_fds_
  send_to_rail_fifo_fd_ = -1;
  CUDA_CHECK(cudaStreamDestroy(send_copy_stream));
  LOG_INFO("Rank=%d, CopyThread::SendThreadFunc exited.", partitioner_->GetGlobalRank());
}
struct CopyFromBufferState {
  CopyFromBufferState() {
    req_res = nullptr;
    total_package_count = 0;
    current_package_id = 0;
  }
  void Clear() {
    req_res = nullptr;
    total_package_count = 0;
    current_package_id = 0;
  }
  void SetReqRes(AsyncReqRes *new_req_res) {
    BOOL_CHECK(req_res == nullptr);
    req_res = new_req_res;
    size_t total_data_size = 0;
    for (int i = 0; i < req_res->meta.valid_data_count; i++) {
      total_data_size += req_res->meta.data_sizes[i];
    }
    size_t kDataSizePerBuffer = FLAGS_async_buffer_size - sizeof(Header);
    total_package_count = (total_data_size + kDataSizePerBuffer - 1) / kDataSizePerBuffer;
    if (total_package_count == 0) {
      // Header need at least 1 package.
      total_package_count = 1;
    }
    LOG_DEBUG("CopyFromBufferState total_package_count=%d", total_package_count);
  }
  void AddData(Header *header, void *data_ptr, size_t data_size, cudaStream_t stream) {
    BOOL_CHECK(SameMeta(&header->meta, &req_res->meta));
    if (current_package_id != total_package_count - 1) {
      BOOL_CHECK(data_size == (size_t)FLAGS_async_buffer_size);
    }
    char *buffer_ptr = (char *) data_ptr;
    size_t kDataSizePerBuffer = FLAGS_async_buffer_size - sizeof(Header);
    size_t data_offset = current_package_id * kDataSizePerBuffer;
    buffer_ptr = buffer_ptr + sizeof(Header);
    size_t buffer_left = data_size - sizeof(Header);
    for (int i = 0; i < req_res->meta.valid_data_count && buffer_left > 0; i++) {
      if (data_offset >= req_res->meta.data_sizes[i]) {
        data_offset -= req_res->meta.data_sizes[i];
        continue;
      }
      size_t tensor_size = req_res->meta.data_sizes[i] - data_offset;
      size_t copy_size = std::min(tensor_size, buffer_left);
      MemcpyUnique((char *) req_res->memory_contexts[i]->GetPointer() + data_offset, buffer_ptr, copy_size, stream);
      buffer_ptr += copy_size;
      buffer_left -= copy_size;
      data_offset = 0;
    }
    current_package_id++;
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  bool Finished() const {
    return current_package_id == total_package_count;
  }
  AsyncReqRes *req_res;
  int total_package_count;
  int current_package_id;
};
void CopyThread::RecvThreadFunc() {
  CUDA_CHECK(cudaSetDevice(partitioner_->GetLocalRank()));
  cudaStream_t recv_copy_stream = CreateHighPriorityStream(cudaStreamNonBlocking);
  std::vector<CopyFromBufferState> copy_states(partitioner_->GetGlobalSize());
  Header header{};
  int full_credit_count = FLAGS_async_buffer_count * (partitioner_->GetNodeCount() - 1);
  // Sender exit depends on stop signal, which means no request / response will come here.
  // Just make sure we collected all credits, then it will be OK to exit.
  while (!sender_exited_.load() || total_credit_count_.load() != full_credit_count) {
    AgentCopyMessage agent_copy_message;
    ssize_t read_bytes;
    auto timeout = SingleFifoTimedRead(recv_fifo_fd_, &agent_copy_message, sizeof(AgentCopyMessage), &read_bytes, 50);
    if (timeout) {
      continue;
    }
    BOOL_CHECK(read_bytes == 0 || read_bytes == sizeof(AgentCopyMessage));
    if (read_bytes == 0) {
      break;
    }
    if (agent_copy_message.has_data == (int)1) {
      // response data send from any agent
      int src_global_rank = agent_copy_message.src_global_rank;
      BOOL_CHECK(src_global_rank >= 0 && src_global_rank < partitioner_->GetGlobalSize());
      BOOL_CHECK(agent_copy_message.data_size >= sizeof(Header));
      BOOL_CHECK(agent_copy_message.dst_global_rank == partitioner_->GetGlobalRank());
      auto &copy_state = copy_states[src_global_rank];
      MemcpyUnique(&header, agent_copy_message.buffer_ptr, sizeof(Header), recv_copy_stream);
      CUDA_CHECK(cudaStreamSynchronize(recv_copy_stream));
      if (copy_state.req_res == nullptr) {
        AsyncReqRes *req_res = nullptr;
        if (header.meta.is_response) {
          req_res = async_communicator_->OnReceiveGetResponse(&header.meta);
        } else {
          req_res = CreateAsyncReqRes();
        }
        memcpy(&req_res->meta, &header.meta, sizeof(Meta));
        allocator_->AllocateReqResByMeta(req_res);
        copy_state.SetReqRes(req_res);
      }
      copy_state.AddData(&header, agent_copy_message.buffer_ptr, agent_copy_message.data_size, recv_copy_stream);
      AgentCopyMessage consumed_message;
      consumed_message.MakeMsg(agent_copy_message.buffer_ptr,
                               0,
                               src_global_rank,
                               agent_copy_message.dst_global_rank,
                               0,
                               false);
      LOG_DEBUG(" {7.CopyThread::RecvThreadFunc} CopyThread got notified on data.");
      int target_local_rank = partitioner_->GetLocalrankFromGlobalRank(src_global_rank);
      SingleFifoWrite(send_fifo_fds_[target_local_rank], &consumed_message, sizeof(AgentCopyMessage));
      if (copy_state.Finished()) {
        if (header.meta.is_response) {
          async_communicator_->OnReceiveResponse(copy_state.req_res);
        } else {
          async_communicator_->OnReceiveRequest(copy_state.req_res);
        }
        copy_state.Clear();
      }
    } else {
      // credit send from rail agent, dst_global_rank should be the target global rank of the credit
      int node_rank = partitioner_->GetNodeIDFromGlobalRank(agent_copy_message.dst_global_rank);
      BOOL_CHECK(node_rank >= 0 && node_rank < partitioner_->GetNodeCount());
      BOOL_CHECK(node_rank != partitioner_->GetNodeID());
      BOOL_CHECK(agent_copy_message.data_size == (size_t)FLAGS_async_buffer_size);
      BOOL_CHECK(agent_copy_message.is_stop_signal == 0);
      IntraNodeCredit credit{};
      credit.pointer = agent_copy_message.buffer_ptr;
      {
        std::unique_lock<std::mutex> send_lock(send_mutex_);
        send_credits_[node_rank].push(credit);
        total_credit_count_.fetch_add(1);
        if (send_credits_[node_rank].size() == 1) {
          send_cv_.notify_all();
        }
      }
    }
  }
  for (int i = 0; i < partitioner_->GetRanksPerNode(); i++) {
    CALL_CHECK(close(send_fifo_fds_[i]));
    send_fifo_fds_[i] = -1;
  }
  CALL_CHECK(close(recv_fifo_fd_));
  recv_fifo_fd_ = -1;
  CUDA_CHECK(cudaStreamDestroy(recv_copy_stream));
  LOG_INFO("Rank=%d, CopyThread::RecvThreadFunc exited.", partitioner_->GetGlobalRank());
}
