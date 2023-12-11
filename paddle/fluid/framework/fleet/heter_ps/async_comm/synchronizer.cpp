#include "synchronizer.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include "check_macros.h"
#include "ib_utils.h"
#include "log_macros.h"
#include "sideband_communicator.h"

class ReaderWriterSemaphore {
 public:
  ReaderWriterSemaphore() : active_readers_(0), active_writers_(0), waiting_writers_(0) {}

  // Reader request for the semaphore
  void ReadLock() {
    std::unique_lock<std::mutex> lock(mutex_);
    // Wait until there are no active or waiting writers
    reader_condition_.wait(lock, [this]() {
      return active_writers_ == 0 && waiting_writers_ == 0;
    });
    // Increment the count of active readers
    ++active_readers_;
  }

  // Reader releases the semaphore
  void ReadUnlock() {
    std::unique_lock<std::mutex> lock(mutex_);
    --active_readers_;
    // If this was the last reader, notify the writers
    if (active_readers_ == 0) {
      writer_condition_.notify_all();
    }
  }

  // Writer request for the semaphore
  void WriteLock() {
    std::unique_lock<std::mutex> lock(mutex_);
    ++waiting_writers_;
    // Wait until there are no active readers or writers
    writer_condition_.wait(lock, [this]() {
      return active_readers_ == 0 && active_writers_ == 0;
    });
    --waiting_writers_;
    // Mark that there is an active writer
    active_writers_ = 1;
  }

  // Writer releases the semaphore
  void WriteUnlock() {
    std::unique_lock<std::mutex> lock(mutex_);
    active_writers_ = 0;
    // If there are waiting writers, notify one
    if (waiting_writers_ > 0) {
      writer_condition_.notify_one();
    } else {
      // Otherwise, notify all readers
      reader_condition_.notify_all();
    }
  }

 private:
  std::mutex mutex_;
  std::condition_variable reader_condition_;
  std::condition_variable writer_condition_;
  int active_readers_;    // Count of readers currently accessing the resource
  int active_writers_;    // Flag to indicate whether a writer is accessing the resource
  int waiting_writers_;   // Count of writers waiting to access the resource
};

class IntraNodeAWBarrier {
 public:
  IntraNodeAWBarrier() = default;
  ~IntraNodeAWBarrier() = default;
  void SetNumRanks(int num_ranks) {
    num_ranks_ = num_ranks;
    arrive_mask_.store(0);
  }
  bool Arrive(int rank) {
    int mask = (1 << rank);
    auto old_value = arrive_mask_.fetch_or(mask);
    BOOL_CHECK((old_value & mask) == 0);
    uint64_t new_value = (old_value | mask);
    uint64_t full_mask = (1 << num_ranks_) - 1;
    if (new_value == full_mask) {
      arrive_mask_.store(0);
      return true;
    }
    return false;
  }
  void GenWaitCredit() {
    auto full_mask = (1 << num_ranks_) - 1;
    wait_credit_.store(full_mask);
  }
  void Wait(int rank) {
    int mask = (1 << rank);
    while ((wait_credit_.load() & mask) == 0) {
      std::atomic_thread_fence(std::memory_order_acquire);
    }
    wait_credit_.fetch_sub(mask);
  }
 private:
  std::atomic<int> arrive_mask_;
  std::atomic<int> wait_credit_;
  int num_ranks_ = 1;
};

class Synchronizer {
 public:
  Synchronizer() = default;
  ~Synchronizer();

  void Enable(int node_count, int ranks_per_node, int node_rank, const char *ib_dev_name, int ib_port);
  bool IsEnabled() {
    std::lock_guard<std::mutex> lock_guard(init_mutex_);
    return sync_mode_enabled_;
  }
  bool IsEnabledUnSafe() const {
    return sync_mode_enabled_;
  }

  void StartThread(bool dummy);
  void StopThread(bool dummy);

  void ReadyToEnter(int local_rank);

  void EnterSynchronizeZone(int local_rank);
  void LeaveSynchronizeZone(int local_rank);

  void EnterSensitiveZone(int local_rank);
  void LeaveSensitiveZone(int local_rank);

  bool NeedIb() const {
    return sync_mode_enabled_ && sync_node_count_ > 1;
  }
  void CreateIbResources(SideBandCommunicator *side_band_communicator, bool dummy);
  void DestroyIbResources(bool dummy);

 private:
  static constexpr int64_t kStageCheckMask = 0xFFLL;

  void IbThreadFunc();
  
  void SignalAllNodes();

  void Arrive(int local_rank);
  void Wait(int local_rank);

  bool AllNodesArrive() {
    auto current_stage = arrive_stage_.load();
    for (int r = 0; r < sync_node_count_; r++) {
      if (rank_stages_[r]->load() < current_stage) return false;
    }
    return true;
  }

  struct InternodeSynchronizeData {
    uint64_t data;
  };

  InternodeSynchronizeData *p_sync_data_ = nullptr;

  struct SynchronizerIbExchangeData {
    IbPeerInfo peer_info;
    IbMemInfo mem_info;
  };

  union SyncImmData {
    uint32_t imm_u32;
    struct {
      int rank : 16;
      int stage_id : 16;
    };
  };

  std::vector<SynchronizerIbExchangeData> all_ib_ex_data_;

  int sync_ranks_per_node_ = 0;
  int sync_node_count_ = 0;
  int sync_node_rank_ = 0;
  bool sync_mode_enabled_ = false;
  std::mutex init_mutex_{};

  std::thread ib_thread_{};
  std::mutex ib_mutex_{};
  std::atomic<bool> ib_thread_stop_signal_{};

  IntraNodeAWBarrier intra_node_aw_barrier_;

  std::string ib_dev_name_;
  int ib_port_ = 1;

  IbLocalContext ib_local_context_;
  struct ibv_mr *reg_mr_ = nullptr;
  std::vector<ibv_qp *> qps_;

  std::atomic<bool> local_node_signal_;

  std::atomic<int> pending_ib_send_;

  std::atomic<int64_t> arrive_stage_;
  std::vector<std::unique_ptr<std::atomic<int64_t>>> rank_stages_;

  std::vector<std::unique_ptr<ReaderWriterSemaphore>> rw_sems_;
};

Synchronizer::~Synchronizer() {
}

void Synchronizer::StartThread(bool dummy) {
  if (!dummy && IsEnabledUnSafe() && NeedIb()) {
    ib_thread_ = std::thread([this]() {
      this->IbThreadFunc();
    });
  }
}
void Synchronizer::StopThread(bool dummy) {
  if (!dummy && IsEnabledUnSafe() && NeedIb()) {
    ib_thread_stop_signal_.store(true);
    ib_thread_.join();
  }
}

static Synchronizer *synchronizer = nullptr;

void Synchronizer::Enable(int node_count, int ranks_per_node, int node_rank, const char *ib_dev_name, int ib_port) {
  std::lock_guard<std::mutex> lock_guard(init_mutex_);
  BOOL_CHECK(sync_mode_enabled_ == false);
  sync_mode_enabled_ = true;
  sync_node_count_ = node_count;
  sync_node_rank_ = node_rank;
  sync_ranks_per_node_ = ranks_per_node;
  intra_node_aw_barrier_.SetNumRanks(sync_ranks_per_node_);
  ib_thread_stop_signal_.store(false);
  arrive_stage_.store(0);
  pending_ib_send_.store(0);
  local_node_signal_.store(false);
  rank_stages_.resize(sync_node_count_);
  for (int r = 0; r < sync_node_count_; r++) {
    rank_stages_[r] = std::make_unique<std::atomic<int64_t>>();
    rank_stages_[r]->store(-1LL);
  }
  rw_sems_.resize(sync_ranks_per_node_);
  for (int i = 0; i < sync_ranks_per_node_; i++) {
    rw_sems_[i] = std::make_unique<ReaderWriterSemaphore>();
  }

  if (NeedIb()) {
    ib_dev_name_ = ib_dev_name;
    BOOL_CHECK(SelectIbDeviceByName(ib_dev_name_) != nullptr);
  }
}

void Synchronizer::IbThreadFunc() {
  for (int r = 0; r < sync_node_count_; r++) {
    if (r == sync_node_rank_) continue;
    IbRecv(p_sync_data_ + r, sizeof(InternodeSynchronizeData), qps_[r], reg_mr_, r);
  }
  const int max_wc_count = 16;
  ibv_wc wcs[max_wc_count];
  while (!ib_thread_stop_signal_.load() || pending_ib_send_.load() > 0) {
    // Poll the completion queue
    int ret = ibv_poll_cq(ib_local_context_.default_cq, max_wc_count, wcs);
    if (ret < 0) {
      LOG_FATAL("Synchronizer ibv_poll_cq failed ret=%d", ret);
    }
    if (ret > 0) {
      LOG_DEBUG("Rank=%d Synchronizer cq got completion events, ret=%d", sync_node_rank_, ret);
    }
    if (local_node_signal_.load()) {
      // process local node signal
      rank_stages_[sync_node_rank_]->fetch_add(1);
      bool expected_value = true;
      BOOL_CHECK(local_node_signal_.compare_exchange_strong(expected_value, false));
      if (AllNodesArrive()) {
        arrive_stage_.fetch_add(1);
        intra_node_aw_barrier_.GenWaitCredit();
      }
    }
    for (int idx = 0; idx < ret; idx++) {
      auto &wc = wcs[idx];
      if (wc.opcode != IBV_WC_RDMA_WRITE && wc.opcode != IBV_WC_SEND && wc.opcode != IBV_WC_RECV_RDMA_WITH_IMM) {
        LOG_FATAL("Synchronizer wc.opcode=%d, not IBV_WC_RDMA_WRITE(%d), IBV_WC_SEND(%d) or IBV_WC_RECV_RDMA_WITH_IMM(%d)",
                  wc.opcode, IBV_WC_RDMA_WRITE, IBV_WC_SEND, IBV_WC_RECV_RDMA_WITH_IMM);
      }
      BOOL_CHECK(wc.opcode == IBV_WC_RDMA_WRITE || wc.opcode == IBV_WC_SEND || wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM);
      if (wc.status != IBV_WC_SUCCESS) {
        LOG_FATAL("Synchronizer wc.opcode=%d, imm.u32=%u wc.status=%s",
                  wc.opcode, GetImmDataFromWc(&wc), ibv_wc_status_str(wc.status));
      }
      if (wc.opcode == IBV_WC_RDMA_WRITE || wc.opcode == IBV_WC_SEND) {
        pending_ib_send_.fetch_add(-1);
        continue;
      }
      SyncImmData sync_imm_data;
      sync_imm_data.imm_u32 = GetImmDataFromWc(&wc);
      int peer_rank = sync_imm_data.rank;

      rank_stages_[peer_rank]->fetch_add(1);
      if (AllNodesArrive()) {
        arrive_stage_.fetch_add(1);
        intra_node_aw_barrier_.GenWaitCredit();
      }

      // post recv again
      IbRecv(p_sync_data_ + peer_rank, sizeof(InternodeSynchronizeData), qps_[peer_rank], reg_mr_, peer_rank);
    }
  }
}

void Synchronizer::CreateIbResources(SideBandCommunicator *side_band_communicator, bool dummy) {
  if (!IsEnabledUnSafe() || !NeedIb()) return;
  std::vector<SynchronizerIbExchangeData> send_ib_exch_data(sync_node_count_);
  std::vector<SynchronizerIbExchangeData> dummy_recv_ib_exch_data(sync_node_count_);
  if (!dummy) {
    p_sync_data_ = (InternodeSynchronizeData *) malloc(sizeof(InternodeSynchronizeData));
    CreateLocalContext(&ib_local_context_, ib_dev_name_, ib_port_);
    reg_mr_ = RegisterIbMr(ib_local_context_.pd, p_sync_data_, sizeof(InternodeSynchronizeData));
    qps_.resize(sync_node_count_, nullptr);
    for (int r = 0; r < sync_node_count_; r++) {
      if (r == sync_node_rank_) continue;
      qps_[r] = CreateIbvRcQp(ib_local_context_.pd, ib_local_context_.default_cq, ib_local_context_.default_cq, true);
      QpInit(qps_[r], ib_port_);
      FillIbPeerInfo(&send_ib_exch_data[r].peer_info,
                     ib_local_context_.port_id,
                     &ib_local_context_.port_attr,
                     qps_[r],
                     &ib_local_context_);
      FillIbMemInfo(&send_ib_exch_data[r].mem_info, reg_mr_);
    }

    all_ib_ex_data_.resize(sync_node_count_);
  }
  if (!dummy) {
    side_band_communicator->RailAllToAll(send_ib_exch_data.data(),
                                         all_ib_ex_data_.data(),
                                         sizeof(SynchronizerIbExchangeData));
  } else {
    side_band_communicator->RailAllToAll(send_ib_exch_data.data(),
                                         dummy_recv_ib_exch_data.data(),
                                         sizeof(SynchronizerIbExchangeData));
  }
  if (!dummy) {
    for (int r = 0; r < sync_node_count_; r++) {
      if (r == sync_node_rank_) continue;
      QpRtr(qps_[r], &send_ib_exch_data[r].peer_info, &all_ib_ex_data_[r].peer_info);
    }
    for (int r = 0; r < sync_node_count_; r++) {
      if (r == sync_node_rank_) continue;
      QpRts(qps_[r]);
    }
  }
}

void Synchronizer::DestroyIbResources(bool dummy) {
  if (!IsEnabledUnSafe() || !NeedIb()) return;
  if (!dummy) {
    DeRegIbMr(reg_mr_);
    for (int r = 0; r < sync_node_count_; r++) {
      if (r == sync_node_rank_) continue;
      DestroyIbvRcQp(qps_[r]);
    }
    DestroyLocalContext(&ib_local_context_);
  }
}

void Synchronizer::ReadyToEnter(int local_rank) {
  Arrive(local_rank);
}

void Synchronizer::EnterSynchronizeZone(int local_rank) {
  Wait(local_rank);
  rw_sems_[local_rank]->WriteLock();
  Arrive(local_rank);
  Wait(local_rank);
}

void Synchronizer::LeaveSynchronizeZone(int local_rank) {
  rw_sems_[local_rank]->WriteUnlock();
}

void Synchronizer::EnterSensitiveZone(int local_rank) {
  rw_sems_[local_rank]->ReadLock();
}

void Synchronizer::LeaveSensitiveZone(int local_rank) {
  rw_sems_[local_rank]->ReadUnlock();
}

void Synchronizer::Arrive(int local_rank) {
  if (intra_node_aw_barrier_.Arrive(local_rank)) {
    SignalAllNodes();
  }
}

void Synchronizer::Wait(int local_rank) {
  intra_node_aw_barrier_.Wait(local_rank);
}

void Synchronizer::SignalAllNodes() {
  if (sync_node_count_ == 1) {
    arrive_stage_.fetch_add(1);
    intra_node_aw_barrier_.GenWaitCredit();
    return;
  }
  SyncImmData sync_imm_data;
  sync_imm_data.rank = sync_node_rank_;
  sync_imm_data.stage_id = (arrive_stage_.load() & kStageCheckMask);
  for (int r = 0; r < sync_node_count_; r++) {
    if (r == sync_node_rank_) continue;
    pending_ib_send_.fetch_add(1);
    IbvRdmaWriteImm(p_sync_data_,
                    sizeof(InternodeSynchronizeData),
                    qps_[r],
                    reg_mr_->lkey,
                    &all_ib_ex_data_[r].mem_info,
                    sync_imm_data.imm_u32,
                    r,
                    true);
  }
  bool expected_value = false;
  BOOL_CHECK(local_node_signal_.compare_exchange_strong(expected_value, true));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void EnableSynchronizeMode(int node_count, int ranks_per_node, int node_rank, const char *ib_dev_name, int ib_port) {
  BOOL_CHECK(synchronizer == nullptr);
  synchronizer = new Synchronizer();
  synchronizer->Enable(node_count, ranks_per_node, node_rank, ib_dev_name, ib_port);
}

bool SynchronizeModeEnabled() {
  return synchronizer != nullptr && synchronizer->IsEnabled();
}

bool SynchronizeModeEnabledUnsafe() {
  return synchronizer != nullptr && synchronizer->IsEnabledUnSafe();
}

void InitSynchronizerHelperFunc(int local_rank, SideBandCommunicator *side_band_communicator) {
  if (synchronizer == nullptr) return;
  synchronizer->CreateIbResources(side_band_communicator, local_rank != 0);
  synchronizer->StartThread(local_rank != 0);
}

void DeinitSynchronizerHelperFunc(int local_rank) {
  if (synchronizer == nullptr) return;
  synchronizer->StopThread(local_rank != 0);
  synchronizer->DestroyIbResources(local_rank != 0);
}

void ReadyToEnterSynchronizeZone(int local_rank) {
  if (synchronizer != nullptr) {
    synchronizer->ReadyToEnter(local_rank);
  }
}

void EnterSynchronizeZone(int local_rank) {
  if (synchronizer != nullptr) {
    synchronizer->EnterSynchronizeZone(local_rank);
  }
}

void LeaveSynchronizeZone(int local_rank) {
  if (synchronizer != nullptr) {
    synchronizer->LeaveSynchronizeZone(local_rank);
  }
}

SensitiveZoneGuard::SensitiveZoneGuard(int local_rank) : local_rank_(local_rank) {
  EnterSensitiveOpZone();
}

SensitiveZoneGuard::~SensitiveZoneGuard() {
  if (in_sensitive_zone_) {
    LeaveSensitiveOpZone();
  }
}

void SensitiveZoneGuard::EnterSensitiveOpZone() {
  BOOL_CHECK(in_sensitive_zone_ == false);
  if (synchronizer != nullptr) {
    synchronizer->EnterSensitiveZone(local_rank_);
  }
  in_sensitive_zone_ = true;
}

void SensitiveZoneGuard::LeaveSensitiveOpZone() {
  BOOL_CHECK(in_sensitive_zone_ == true);
  if (synchronizer) {
    synchronizer->LeaveSensitiveZone(local_rank_);
  }
  in_sensitive_zone_ = false;
}

