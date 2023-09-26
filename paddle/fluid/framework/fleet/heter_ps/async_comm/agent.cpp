#include "agent.h"

#include <string.h>

#include <algorithm>
#include <chrono>
#include <functional>
#include <memory>
#include <random>

#include "agent_copy_message.h"
#include "check_macros.h"
#include "fifo_utils.h"
#include "ib_utils.h"
#include "log_macros.h"
#include "partitioner.h"
#include "sideband_communicator.h"
#include "paddle/phi/core/flags.h"

#include "meta.h"
DECLARE_bool(async_use_gpu_driect_rdma);
DECLARE_uint64(async_buffer_size);
DECLARE_uint64(async_buffer_count);

Agent::Agent(Partitioner *partitioner, Config *config) :
    partitioner_(partitioner), config_(config) {
  no_more_sends_.store(false);
  pending_sends_.store(0);
  sender_exited_.store(false);
  uncompleted_data_sends_.store(0);
  total_credit_count_.store(0);
  to_send_credits_.resize(partitioner->GetNodeCount());
  internode_to_send_data_.resize(partitioner->GetNodeCount());
  internode_remote_recv_credits_.resize(partitioner->GetNodeCount());
}
Agent::~Agent() {
}
static bool SupportGPUDirectRDMA(struct ibv_pd *pd) {
  void* cuda_ptr = nullptr;
  const size_t buf_size = 128;
  CUDA_CHECK(cudaMalloc(&cuda_ptr, 128));
  auto* ib_mr = TryRegisterIbMr(pd, cuda_ptr, buf_size);
  bool support_gdr = ib_mr != nullptr;
  if (support_gdr) {
    DeRegIbMr(ib_mr);
  } else {
    void* cpu_ptr = malloc(buf_size);
    ib_mr = TryRegisterIbMr(pd, cpu_ptr, buf_size);
    if (ib_mr == nullptr) {
      LOG_INFO("CPU RDMA is also not supported!");
    } else {
      DeRegIbMr(ib_mr);
    }
    free(cpu_ptr);
  }
  CUDA_CHECK(cudaFree(cuda_ptr));
  return support_gdr;
}
void Agent::CreateResources() {
  recv_fifo_fd_ = CreateFifo(GetAgentCopyFifoName("agent_recv",
                                                  partitioner_->GetNodeID(),
                                                  partitioner_->GetLocalRank()));
  std::string ib_dev_name = config_->ib_device_name[partitioner_->GetLocalRank()];
  int ib_port = config_->ib_port[partitioner_->GetLocalRank()];
  CreateLocalContext(&ib_local_context_, ib_dev_name, ib_port);
  send_qps_.resize(partitioner_->GetNodeCount(), nullptr);
  recv_qps_.resize(partitioner_->GetNodeCount(), nullptr);
  auto *pd = ib_local_context_.pd;
  auto *cq = ib_local_context_.default_cq;

  use_gpu_direct_rdma_ = SupportGPUDirectRDMA(ib_local_context_.pd);
  use_gpu_direct_rdma_ = use_gpu_direct_rdma_ && FLAGS_async_use_gpu_driect_rdma;

  // allocate register memory
  size_t all_reg_mem_size = FLAGS_async_buffer_size * FLAGS_async_buffer_count * partitioner_->GetNodeCount();
  int old_dev_id;
  CUDA_CHECK(cudaGetDevice(&old_dev_id));
  CUDA_CHECK(cudaSetDevice(config_->agent_local_rank[partitioner_->GetLocalRank()]));
  if (use_gpu_direct_rdma_) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&send_reg_mem_), all_reg_mem_size));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&recv_reg_mem_), all_reg_mem_size));
  } else {
    LOG_WARN("GPU Direct RDMA is not supported, degrading to CPU RDMA, performance may suffer!");
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&send_reg_mem_), all_reg_mem_size));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&recv_reg_mem_), all_reg_mem_size));
  }
  CUDA_CHECK(cudaSetDevice(old_dev_id));
  credit_reg_mem_ = (char *) malloc(kCreditRegMemSize);
  size_t all_imm_size = 2 * FLAGS_async_buffer_count * partitioner_->GetNodeCount() * sizeof(int64_t);
  imm_mem_ = (int64_t*) malloc(all_imm_size);
  // reg memory regions.
  send_mr_ = RegisterIbMr(pd, send_reg_mem_, all_reg_mem_size);
  recv_mr_ = RegisterIbMr(pd, recv_reg_mem_, all_reg_mem_size);
  credit_mr_ = RegisterIbMr(pd, credit_reg_mem_, kCreditRegMemSize);
  imm_mr_ = RegisterIbMr(pd, imm_mem_, all_imm_size);
  // create qp
  for (int n = 0; n < partitioner_->GetNodeCount(); n++) {
    if (n == partitioner_->GetNodeID()) continue;
    send_qps_[n] = CreateIbvRcQp(pd, cq, cq, 0);
    recv_qps_[n] = CreateIbvRcQp(pd, cq, cq, 1);
    send_qpn_to_node_id_.insert(std::pair<uint32_t, int>(send_qps_[n]->qp_num, n));
    recv_qpn_to_node_id_.insert(std::pair<uint32_t, int>(recv_qps_[n]->qp_num, n));
    QpInit(send_qps_[n], ib_local_context_.port_id);
    QpInit(recv_qps_[n], ib_local_context_.port_id);
  }
  local_node_infos_.resize(partitioner_->GetNodeCount());
  remote_node_infos_.resize(partitioner_->GetNodeCount());
  for (int i = 0; i < partitioner_->GetNodeCount(); i++) {
    if (i == partitioner_->GetNodeID()) continue;
    FillIbPeerInfo(&local_node_infos_[i].peer_recv_info, ib_port, &ib_local_context_.port_attr, recv_qps_[i], &ib_local_context_);
    FillIbPeerInfo(&local_node_infos_[i].peer_send_info, ib_port, &ib_local_context_.port_attr, send_qps_[i], &ib_local_context_);
    FillIbMemInfo(&local_node_infos_[i].recv_mem_info, recv_mr_);
    FillIbMemInfo(&local_node_infos_[i].credit_mem_info, credit_mr_);
    PrintIbPeerInfo("recv_qp", partitioner_->GetGlobalRank(), i, &local_node_infos_[i].peer_recv_info);
    PrintIbPeerInfo("send_qp", partitioner_->GetGlobalRank(), i, &local_node_infos_[i].peer_send_info);
    PrintIbMemInfo("recv_mem_info", partitioner_->GetGlobalRank(), i, &local_node_infos_[i].recv_mem_info);
    PrintIbMemInfo("credit_mem_info", partitioner_->GetGlobalRank(), i, &local_node_infos_[i].credit_mem_info);
  }
  side_band_communicator_->RailAllToAll(local_node_infos_.data(),
                                        remote_node_infos_.data(),
                                        sizeof(RemoteNodeInfo));
  for (int i = 0; i < partitioner_->GetNodeCount(); i++) {
    if (i == partitioner_->GetNodeID()) continue;
    PrintIbPeerInfo("remote.recv_qp", partitioner_->GetGlobalRank(), i, &remote_node_infos_[i].peer_recv_info);
    PrintIbPeerInfo("remote.send_qp", partitioner_->GetGlobalRank(), i, &remote_node_infos_[i].peer_send_info);
    PrintIbMemInfo("remote.recv_mem_info", partitioner_->GetGlobalRank(), i, &remote_node_infos_[i].recv_mem_info);
    PrintIbMemInfo("remote.credit_mem_info", partitioner_->GetGlobalRank(), i, &remote_node_infos_[i].credit_mem_info);
  }
}
void Agent::DestroyResources() {
  for (int n = 0; n < partitioner_->GetNodeCount(); n++) {
    if (n == partitioner_->GetNodeID()) continue;
    DestroyIbvRcQp(send_qps_[n]);
    DestroyIbvRcQp(recv_qps_[n]);
    send_qps_[n] = nullptr;
    recv_qps_[n] = nullptr;
  }
  DeRegIbMr(send_mr_);
  DeRegIbMr(recv_mr_);
  DeRegIbMr(credit_mr_);
  DeRegIbMr(imm_mr_);
  DestroyLocalContext(&ib_local_context_);

  int old_dev_id;
  CUDA_CHECK(cudaGetDevice(&old_dev_id));
  CUDA_CHECK(cudaSetDevice(config_->agent_local_rank[partitioner_->GetLocalRank()]));
  if (use_gpu_direct_rdma_) {
    CUDA_CHECK(cudaFree(send_reg_mem_));
    CUDA_CHECK(cudaFree(recv_reg_mem_));
  } else {
    CUDA_CHECK(cudaFreeHost(send_reg_mem_));
    CUDA_CHECK(cudaFreeHost(recv_reg_mem_));
  }
  CUDA_CHECK(cudaSetDevice(old_dev_id));
  free(credit_reg_mem_);
  free(imm_mem_);
  UnlinkFifo(GetAgentCopyFifoName("agent_recv", partitioner_->GetNodeID(), partitioner_->GetLocalRank()));
}
void Agent::ConnectFifo() {
  send_fifo_fds_.resize(partitioner_->GetRanksPerNode(), -1);
  for (int lr = 0; lr < partitioner_->GetRanksPerNode(); lr++) {
    send_fifo_fds_[lr] =
        WaitAndOpenFifo(GetAgentCopyFifoName("copy_recv", partitioner_->GetNodeID(), lr));
  }
  send_to_rail_fifo_fd_ = send_fifo_fds_[partitioner_->GetLocalRank()];
}
void Agent::ConnectNetwork() {
  for (int i = 0; i < partitioner_->GetNodeCount(); i++) {
    if (i == partitioner_->GetNodeID()) continue;
    QpRtr(send_qps_[i], &local_node_infos_[i].peer_send_info, &remote_node_infos_[i].peer_recv_info);
    QpRtr(recv_qps_[i], &local_node_infos_[i].peer_recv_info, &remote_node_infos_[i].peer_send_info);
  }
  LOG_INFO("Rank=%d all QPs RTR finished.", partitioner_->GetGlobalRank());
  for (int i = 0; i < partitioner_->GetNodeCount(); i++) {
    if (i == partitioner_->GetNodeID()) continue;
    QpRts(send_qps_[i]);
    QpRts(recv_qps_[i]);
  }
  LOG_INFO("Rank=%d all QPs RTS finished.", partitioner_->GetGlobalRank());
}
void Agent::Start() {
  read_fifo_thread_ = std::make_unique<std::thread>([this]() {
    this->ReadFifoThreadFunc();
  });
  poll_cq_thread_ = std::make_unique<std::thread>([this]() {
    this->PollCQThreadFunc();
  });
  rdma_send_thread_ = std::make_unique<std::thread>([this]() {
    this->SendToRemoteThreadFunc();
  });
}
void Agent::Stop() {
  // No need to set stop as stop depends on CopyThread's Fifo.
}
void Agent::WaitStopped() {
  read_fifo_thread_->join();
  read_fifo_thread_.reset();
  poll_cq_thread_->join();
  poll_cq_thread_.reset();
  rdma_send_thread_->join();
  rdma_send_thread_.reset();
}
int Agent::GetBufferIndex(char *ptr, bool is_recv) {
  char *reg_buffer = is_recv ? recv_reg_mem_ : send_reg_mem_;
  int64_t ptr_diff = ptr - reg_buffer;
  BOOL_CHECK(ptr_diff >= 0 && 0 < partitioner_->GetNodeCount() * FLAGS_async_buffer_size * FLAGS_async_buffer_count - ptr_diff);
  // BOOL_CHECK(ptr_diff >= 0 && ptr_diff < partitioner_->GetNodeCount() * FLAGS_async_buffer_size * FLAGS_async_buffer_count);
  BOOL_CHECK(ptr_diff % FLAGS_async_buffer_size == 0);
  return ptr_diff / FLAGS_async_buffer_size;
}
int Agent::GetNodeIdFromQpn(uint32_t qpn, bool is_recv) {
  auto& qpn_to_node_id_map = is_recv ? recv_qpn_to_node_id_ : send_qpn_to_node_id_;
  auto it = qpn_to_node_id_map.find(qpn);
  BOOL_CHECK(it != qpn_to_node_id_map.end());
  return it->second;
}
void Agent::AddInterNodeSendData(AgentCopyMessage agent_copy_message) {
  InterNodeToSendData send_data;
  send_data.data_pointer = (char *) agent_copy_message.buffer_ptr;
  send_data.imm_data.is_stop_signal = agent_copy_message.is_stop_signal;
  send_data.imm_data.has_data = agent_copy_message.has_data;
  int target_node_id = -1;
  if (agent_copy_message.has_data == (int)1) {
    // requester data sent from rail copy thread, buffer filled message
    // should send data to remote node
    send_data.imm_data.src_local_rank = partitioner_->GetLocalrankFromGlobalRank(agent_copy_message.src_global_rank);
    send_data.imm_data.dst_local_rank = partitioner_->GetLocalrankFromGlobalRank(agent_copy_message.dst_global_rank);
    BOOL_CHECK(partitioner_->GetLocalRank() == send_data.imm_data.src_local_rank);
    send_data.data_size = agent_copy_message.data_size;
    int buffer_idx = GetBufferIndex(send_data.data_pointer, false);
    target_node_id = buffer_idx / FLAGS_async_buffer_count;
    send_data.imm_data.buffer_idx = -1; // buffer_idx should be set according to recv buffer idx, not here
    std::unique_lock<std::mutex> mlock(send_mutex_);
    pending_sends_.fetch_add(1);
    internode_to_send_data_[target_node_id].push(send_data);
    if (internode_to_send_data_[target_node_id].size() == 1) {
      send_cv_.notify_all();
    }
    LOG_DEBUG(" {3.Agent::AddInterNodeSendData} adding to internode_to_send_data_[%d].", target_node_id);
  } else {
    // buffer consumed message from any copy thread
    // should return credit (recv_mr) to remote node
    send_data.data_size = 0;
    send_data.imm_data.src_local_rank = partitioner_->GetLocalrankFromGlobalRank(agent_copy_message.src_global_rank);
    // as all local ranks share same recv buffer, just replace with agent's global rank
    send_data.imm_data.dst_local_rank = partitioner_->GetGlobalRank();
    int buffer_idx = GetBufferIndex(send_data.data_pointer, true);
    send_data.imm_data.buffer_idx = buffer_idx;
    target_node_id = buffer_idx / FLAGS_async_buffer_count;
    BOOL_CHECK(target_node_id == partitioner_->GetNodeIDFromGlobalRank(agent_copy_message.src_global_rank));
    std::unique_lock<std::mutex> mlock(send_mutex_);
    pending_sends_.fetch_add(1);
    to_send_credits_[target_node_id].push(send_data);
    if (to_send_credits_[target_node_id].size() == 1) {
      send_cv_.notify_all();
    }
  }
}
void Agent::ReadFifoThreadFunc() {
  // recv fifo closed is enough to exit.
  while (true) {
    AgentCopyMessage agent_copy_message;
    auto read_bytes = SingleFifoRead(recv_fifo_fd_, &agent_copy_message, sizeof(AgentCopyMessage));
    BOOL_CHECK(read_bytes == 0 || read_bytes == sizeof(AgentCopyMessage));
    if (read_bytes == 0) {
      break;
    }
    AddInterNodeSendData(agent_copy_message);
  }
  no_more_sends_.store(true);
  LOG_INFO("Rank=%d, Agent::ReadFifoThreadFunc exited.", partitioner_->GetGlobalRank());
}
void Agent::PollCQThreadFunc() {
  for (int n = 0; n < partitioner_->GetNodeCount(); n++) {
    if (n == partitioner_->GetNodeID()) continue;
    for (int i = 0; i < 2 *(int)FLAGS_async_buffer_count; i++) {
      uint64_t wr_id = n * 2 * FLAGS_async_buffer_count + i;
      IbRecv(imm_mem_ + wr_id, sizeof(int64_t), recv_qps_[n], imm_mr_, wr_id);
    }
    for (int i = 0; i < (int)FLAGS_async_buffer_count; i++) {
      AgentCopyMessage agent_copy_message;
      int buffer_idx = n * FLAGS_async_buffer_count + i;
      char* ptr = send_reg_mem_ + FLAGS_async_buffer_size * buffer_idx;
      int target_global_rank = partitioner_->MakeGlobalRank(n, partitioner_->GetLocalRank());
      agent_copy_message.MakeMsg(ptr, FLAGS_async_buffer_size, partitioner_->GetGlobalRank(), target_global_rank, 0, false);
      // send to copy the released credit
      SingleFifoWrite(send_to_rail_fifo_fd_, &agent_copy_message, sizeof(AgentCopyMessage));
    }
  }
  const int max_wc_count = 16;
  ibv_wc wcs[max_wc_count];
  int full_credit_count = FLAGS_async_buffer_count * (partitioner_->GetNodeCount() - 1);
  while (!sender_exited_.load() || uncompleted_data_sends_.load() > 0
      || total_credit_count_.load() < full_credit_count) {
    // Poll the completion queue
    int ret = ibv_poll_cq(ib_local_context_.default_cq, max_wc_count, wcs);
    if (ret < 0) {
      LOG_FATAL("ibv_poll_cq failed ret=%d", ret);
    }
    if (ret > 0) {
      LOG_DEBUG("Rank=%d cq got completion events, ret=%d", partitioner_->GetGlobalRank(), ret);
    }

    for (int idx = 0; idx < ret; idx++) {
      auto& wc = wcs[idx];
      // 3 types of completion event may be received
      // (completion event of add credit from send side is not used, because only imm is used in this case)
      if (wc.opcode != IBV_WC_RDMA_WRITE && wc.opcode != IBV_WC_SEND && wc.opcode != IBV_WC_RECV_RDMA_WITH_IMM) {
        LOG_FATAL("wc.opcode=%d, not IBV_WC_RDMA_WRITE(%d), IBV_WC_SEND(%d) or IBV_WC_RECV_RDMA_WITH_IMM(%d)",
                  wc.opcode, IBV_WC_RDMA_WRITE, IBV_WC_SEND, IBV_WC_RECV_RDMA_WITH_IMM);
      }
      BOOL_CHECK(wc.opcode == IBV_WC_RDMA_WRITE || wc.opcode == IBV_WC_SEND || wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM);
      if (wc.status != IBV_WC_SUCCESS) {
        LOG_FATAL("wc.opcode=%d, imm.u32=%u wc.status=%s", wc.opcode, GetImmDataFromWc(&wc), ibv_wc_status_str(wc.status));
      }
      if (wc.opcode == IBV_WC_RDMA_WRITE || wc.opcode == IBV_WC_SEND) {
        // 1. completion of data send (required by ibv_post_send)
        AgentCopyMessage agent_copy_message;
        WrId wr_id;
        wr_id.id = wc.wr_id;
        ImmData imm_data = wr_id.imm_data;
        if (imm_data.has_data != 0) {
          BOOL_CHECK(partitioner_->GetLocalRank() == imm_data.src_local_rank);
          BOOL_CHECK(
              wr_id.send_buffer_idx >= 0 && wr_id.send_buffer_idx < (int)FLAGS_async_buffer_count * partitioner_->GetNodeCount());
          void *ptr = send_reg_mem_ + wr_id.send_buffer_idx * FLAGS_async_buffer_size;
          int target_node_rank = wr_id.send_buffer_idx / FLAGS_async_buffer_count;
          int target_global_rank = partitioner_->MakeGlobalRank(target_node_rank, imm_data.dst_local_rank);
          agent_copy_message.MakeMsg(ptr, FLAGS_async_buffer_size, partitioner_->GetGlobalRank(), target_global_rank, 0, false);
          // send to CopyThread the released send credit
          SingleFifoWrite(send_to_rail_fifo_fd_, &agent_copy_message, sizeof(AgentCopyMessage));
          uncompleted_data_sends_.fetch_add(-1);
        } else {
          LOG_DEBUG("Rank=%d Completion event of send credit got", partitioner_->GetGlobalRank());
        }
      } else {
        ImmData imm_data;
        imm_data.imm_u32 = GetImmDataFromWc(&wc);
        if (imm_data.has_data) {
          // 2. completion of data received from remote (IBV_WR_RDMA_WRITE_WITH_IMM)
          int buffer_idx = imm_data.buffer_idx;
          BOOL_CHECK(buffer_idx >= 0 && buffer_idx < (int)FLAGS_async_buffer_count * partitioner_->GetNodeCount());
          int remote_node_id = buffer_idx / FLAGS_async_buffer_count;
          int src_global_rank = partitioner_->MakeGlobalRank(remote_node_id, partitioner_->GetLocalRank());
          int dst_global_rank = partitioner_->MakeGlobalRank(partitioner_->GetNodeID(), imm_data.dst_local_rank);
          char* ptr = recv_reg_mem_ + FLAGS_async_buffer_size * buffer_idx;
          AgentCopyMessage agent_copy_message;
          int temp = (int) imm_data.is_stop_signal;
          agent_copy_message.MakeMsg(ptr,
                                     wc.byte_len,
                                     src_global_rank,
                                     dst_global_rank,
                                     1,
                                     (temp == 1));
          LOG_DEBUG(" {6.Agent::PollCQThreadFunc} Received data from node %d, rank=%d.", remote_node_id, src_global_rank);
          SingleFifoWrite(send_fifo_fds_[imm_data.dst_local_rank], &agent_copy_message, sizeof(AgentCopyMessage));
        } else {
          // 3. completion of add credit from (IBV_WR_RDMA_WRITE_WITH_IMM)
          LOG_DEBUG("Rank=%d received credit", partitioner_->GetGlobalRank());
          InterNodeCredit credit;
          credit.buffer_idx = imm_data.buffer_idx;
          int target_node_id = GetNodeIdFromQpn(wc.qp_num, true);
          credit.target_node_id = target_node_id;
          std::unique_lock<std::mutex> mlock(send_mutex_);
          total_credit_count_.fetch_add(1);
          internode_remote_recv_credits_[target_node_id].push(credit);
          if (internode_remote_recv_credits_[target_node_id].size() == 1) {
            send_cv_.notify_all();
          }
        }
        int node_id = GetNodeIdFromQpn(wc.qp_num, true);
        BOOL_CHECK((uint64_t)node_id == wc.wr_id / (2 * FLAGS_async_buffer_count));
        // post recv again
        IbRecv(imm_mem_ + wc.wr_id, sizeof(int64_t), recv_qps_[node_id], imm_mr_, wc.wr_id);
      }
    }
  }
  LOG_INFO("Rank=%d, Agent::PollCQThreadFunc exited.", partitioner_->GetGlobalRank());
}
void Agent::SendToRemoteThreadFunc() {
  // first send out credits to all remote nodes
  int pre_send_credit = 0;
  for (int n = 0; n < partitioner_->GetNodeCount(); n++) {
    if (n == partitioner_->GetNodeID()) continue;
    ImmData imm_data;
    imm_data.has_data = 0;
    imm_data.is_stop_signal = 0;
    imm_data.src_local_rank = imm_data.dst_local_rank = partitioner_->GetLocalRank();
    for (int i = 0; i < (int)FLAGS_async_buffer_count; i++) {
      imm_data.buffer_idx = n * FLAGS_async_buffer_count + i;
      WrId wr_id{};
      wr_id.send_buffer_idx = imm_data.buffer_idx;
      wr_id.imm_data.imm_u32 = imm_data.imm_u32;
      IbvRdmaWriteImm(credit_reg_mem_,
                      kCreditRegMemSize,
                      send_qps_[n],
                      credit_mr_->lkey,
                      &remote_node_infos_[n].credit_mem_info,
                      imm_data.imm_u32,
                      wr_id.id,
                      true);
      pre_send_credit++;
    }
  }
  LOG_INFO("Rank=%d all remote credit send out, pre_send_credit=%d.", partitioner_->GetGlobalRank(), pre_send_credit);
  auto can_send_data = [this]() -> bool {
    for (int i = 0; i < partitioner_->GetNodeCount(); i++) {
      if (i == partitioner_->GetNodeID()) {
        BOOL_CHECK(internode_to_send_data_[i].empty() && internode_remote_recv_credits_[i].empty());
        BOOL_CHECK(to_send_credits_[i].empty());
      }
      if (!to_send_credits_[i].empty()) return true;
      if (!internode_to_send_data_[i].empty() && !internode_remote_recv_credits_[i].empty()) return true;
    }
    return false;
  };
  std::vector<int> to_send_nodes;
  to_send_nodes.reserve(partitioner_->GetNodeCount() * FLAGS_async_buffer_count * 2);
  auto get_send_data = [this, &to_send_nodes]() {
    to_send_nodes.clear();
    for (int i = 0; i < partitioner_->GetNodeCount(); i++) {
      if (i == partitioner_->GetNodeID()) continue;
      size_t add_count = to_send_credits_[i].size();
      add_count += std::min(internode_to_send_data_[i].size(), internode_remote_recv_credits_[i].size());
      if (add_count == 0) continue;
      for (size_t idx = 0; idx < add_count; idx++) {
        LOG_DEBUG("getting %d send task for node %d, credit=%ld", add_count, i, internode_remote_recv_credits_[i].size());
        to_send_nodes.push_back(i);
      }
    }
  };
  std::random_device rd;
  std::default_random_engine engine(rd());
  std::unique_lock<std::mutex> mlock(send_mutex_);
  mlock.unlock();
  // if no additional send will come and all pending send are send out, then it is OK to exit.
  while (!no_more_sends_.load() || pending_sends_.load() > 0) {
    mlock.lock();
    send_cv_.wait_for(mlock, std::chrono::milliseconds(50), can_send_data);
    get_send_data();
    mlock.unlock();
    // random shuffle to do load balance
    std::shuffle(to_send_nodes.begin(), to_send_nodes.end(), engine);
    for (auto to_send_node: to_send_nodes) {
      InterNodeToSendData send_data;
      InterNodeCredit credit;
      bool is_credit = false;
      mlock.lock();
      if (!to_send_credits_[to_send_node].empty()) {
        is_credit = true;
        send_data = to_send_credits_[to_send_node].front();
        to_send_credits_[to_send_node].pop();
      } else {
        is_credit = false;
        send_data = internode_to_send_data_[to_send_node].front();
        internode_to_send_data_[to_send_node].pop();
        credit = internode_remote_recv_credits_[to_send_node].front();
        internode_remote_recv_credits_[to_send_node].pop();
      }
      mlock.unlock();
      if (send_data.data_size != 0) {
        BOOL_CHECK(credit.target_node_id == to_send_node);
        BOOL_CHECK(!is_credit);
        // fill buffer index
        BOOL_CHECK(send_data.imm_data.buffer_idx == -1);
        send_data.imm_data.buffer_idx = credit.buffer_idx;
        IbMemInfo remote_mem_info = remote_node_infos_[to_send_node].recv_mem_info;
        remote_mem_info.addr += credit.buffer_idx * FLAGS_async_buffer_size;
        remote_mem_info.length = send_data.data_size;
        WrId wr_id{};
        wr_id.send_buffer_idx = GetBufferIndex(send_data.data_pointer, false);
        wr_id.imm_data.imm_u32 = send_data.imm_data.imm_u32;
        IbvRdmaWriteImm(send_data.data_pointer,
                        send_data.data_size,
                        send_qps_[to_send_node],
                        send_mr_->lkey,
                        &remote_mem_info,
                        send_data.imm_data.imm_u32,
                        wr_id.id,
                        true);
        uncompleted_data_sends_.fetch_add(1);
        total_credit_count_.fetch_add(-1);
        LOG_DEBUG(" {4.Agent::SendToRemoteThreadFunc} IbvRdmaWriteImm to node %d.", to_send_node);
      } else {
        BOOL_CHECK(is_credit);
        WrId wr_id{};
        wr_id.send_buffer_idx = send_data.imm_data.buffer_idx;
        wr_id.imm_data.imm_u32 = send_data.imm_data.imm_u32;
        IbvRdmaWriteImm(credit_reg_mem_,
                        kCreditRegMemSize,
                        send_qps_[to_send_node],
                        credit_mr_->lkey,
                        &remote_node_infos_[to_send_node].credit_mem_info,
                        send_data.imm_data.imm_u32,
                        wr_id.id,
                        true);
      }
      pending_sends_.fetch_add(-1);
    }
    to_send_nodes.clear();
  }
  sender_exited_.store(true);
  LOG_INFO("Rank=%d, Agent::SendToRemoteThreadFunc exited.", partitioner_->GetGlobalRank());
}

