#pragma once

// Should be called before creating any Communicator, Runner and Allocator.
// And after IbInit.
void EnableSynchronizeMode(int node_count, int ranks_per_node, int node_rank, const char* ib_dev_name, int ib_port);

// Function For Allocator, Runner to check if Synchronize Mode is enabled with lock.
bool SynchronizeModeEnabled();

// Function For Allocator, Runner to check if Synchronize Mode is enabled without lock.
// Calling this after EnableSynchronizeMode will be safe.
bool SynchronizeModeEnabledUnsafe();

class SideBandCommunicator;

// should be called by each rank to initialize
void InitSynchronizerHelperFunc(int local_rank, SideBandCommunicator* side_band_communicator);

// should be called by each rank to deinitialize
void DeinitSynchronizerHelperFunc(int local_rank);

/*
 * We divide timeline into two kinds of Zones. One is SynchronizeZone, the other is non-SynchronizeZone.
 * In SynchronizeZone, NCCL can be called with the guarantee that NCCL kernels can run so not hang.
 * To achieve this, we may need to disable all possible interlocks, such as GPU memory allocation.
 * As there may be some libraries like thrust or something else may also allocate GPU memory.
 * So to make this save, actually we Stop all Runners and all ReqRes allocation/deallocation.
 * User should also be sure that no other thread run something that may block NCCL outside the communication library.
 *
 * Synchronize has three steps:
 * 1. ReadyToEnterSynchronizeZone
 *    When current rank has no data dependence to resolve. ReadyToEnterSynchronizeZone can be called.
 *    We should wait until all ranks are ready before any rank can EnterSynchronizeZone.
 *    If any ranks are not ready, it may need data from other ranks so other ranks also can't EnterSynchronizeZone
 *    because EnterSynchronizeZone will also disable Runner which will cause the rank that need data will never get data
 *    from this rank .
 * 2. EnterSynchronizeZone
 *    When all ranks are ready, then they can EnterSynchronizeZone.
 *    After this, anything that may cause NCCL hang will be stopped.
 * 3. LeaveSynchronizeZone
 *    When synchronize is done. E.g. NCCL call is done. All ranks can LeaveSynchronizeZone.
 *    Every thing that are blocked will be unblocked.
 *
 * Allocators member function Malloc/Free should use the read lock.
 * Runners should also take read lock.
 * EnterSynchronizeZone should use write lock
 */

void ReadyToEnterSynchronizeZone(int local_rank);

void EnterSynchronizeZone(int local_rank);

void LeaveSynchronizeZone(int local_rank);

// Class that used for Sensitive Op Zone.
// It is RAII style, creating SynchronizeGuard will Enter Sensitive Op Zone.
class SensitiveZoneGuard {
 public:
  explicit SensitiveZoneGuard(int local_rank);
  SensitiveZoneGuard() = delete;
  ~SensitiveZoneGuard();
  // Enter sensitive op zone that may cause NCCL problems.
  // Will get internal read lock
  void EnterSensitiveOpZone();
  // Leave sensitive op zone that may cause NCCL problems.
  // Will release internal read lock
  void LeaveSensitiveOpZone();
 private:
  int local_rank_ = -1;
  bool in_sensitive_zone_ = false;
};

