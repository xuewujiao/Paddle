#include "demo_memory_allocator.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <queue>

#include "check_macros.h"
#include "log_macros.h"
#include "partitioner.h"
#include "synchronizer.h"

static std::atomic<int64_t> storage_count_{0};
static std::atomic<int64_t> context_count_{0};

void PrintRefCounts() {
  LOG_INFO("\nStorage counts=%ld\nContext counts=%ld",
           storage_count_.load(), context_count_.load());
}

class ChunkedMemoryPool {
 public:
  ChunkedMemoryPool();
  ~ChunkedMemoryPool();
  void* CachedMalloc(size_t size);
  void CachedFree(void* ptr, size_t size);
  void EmptyCache();
  virtual void* MallocFnImpl(size_t size) = 0;
  virtual void FreeFnImpl(void* ptr) = 0;
 private:
  static constexpr int kBucketCount = 64;
  std::vector<std::unique_ptr<std::mutex>> mutexes_;
  std::vector<std::queue<void*>> sized_pool_;
};
static size_t GetChunkIndex(size_t size) {
  int power = 0;
  size_t shifted_size = size;
  while (shifted_size) {
    shifted_size >>= 1;
    power++;
  }
  if ((size & (size - 1)) == 0) {
    return power;
  } else {
    return power + 1;
  }
}
ChunkedMemoryPool::ChunkedMemoryPool() {
  sized_pool_.resize(kBucketCount);
  mutexes_.resize(kBucketCount);
  for (int i = 0; i < kBucketCount; i++) {
    mutexes_[i] = std::make_unique<std::mutex>();
  }
}
ChunkedMemoryPool::~ChunkedMemoryPool() {
}
void *ChunkedMemoryPool::CachedMalloc(size_t size) {
  size_t chunked_index = GetChunkIndex(size);
  std::unique_lock<std::mutex> mlock(*mutexes_[chunked_index]);
  if (!sized_pool_[chunked_index].empty()) {
    void* ptr = sized_pool_[chunked_index].front();
    sized_pool_[chunked_index].pop();
    return ptr;
  } else {
    return MallocFnImpl(1ULL << chunked_index);
  }
  return nullptr;
}
void ChunkedMemoryPool::CachedFree(void* ptr, size_t size) {
  size_t chunked_index = GetChunkIndex(size);
  std::unique_lock<std::mutex> mlock(*mutexes_[chunked_index]);
  sized_pool_[chunked_index].push(ptr);
}
void ChunkedMemoryPool::EmptyCache() {
  for (int i = 0; i < kBucketCount; i++) {
    std::unique_lock<std::mutex> mlock(*mutexes_[i]);
    while (!sized_pool_[i].empty()) {
      FreeFnImpl(sized_pool_[i].front());
      sized_pool_[i].pop();
    }
  }
}

class DeviceChunkedMemoryPool : public ChunkedMemoryPool {
 public:
  explicit DeviceChunkedMemoryPool(int device_id);
  ~DeviceChunkedMemoryPool();
  void* MallocFnImpl(size_t size) override;
  void FreeFnImpl(void* ptr) override;
 protected:
  int device_id_ = -1;
};
DeviceChunkedMemoryPool::DeviceChunkedMemoryPool(int device_id) :
    device_id_(device_id) {
}
DeviceChunkedMemoryPool::~DeviceChunkedMemoryPool() {

}
void *DeviceChunkedMemoryPool::MallocFnImpl(size_t size) {
  int old_dev;
  void* ptr;
  CUDA_CHECK(cudaGetDevice(&old_dev));
  CUDA_CHECK(cudaSetDevice(device_id_));
  CUDA_CHECK(cudaMalloc(&ptr, size));
  CUDA_CHECK(cudaSetDevice(old_dev));
  return ptr;
}
void DeviceChunkedMemoryPool::FreeFnImpl(void *ptr) {
  int old_dev;
  CUDA_CHECK(cudaGetDevice(&old_dev));
  CUDA_CHECK(cudaSetDevice(device_id_));
  CUDA_CHECK(cudaFree(ptr));
  CUDA_CHECK(cudaSetDevice(old_dev));
}

class PinnedChunkedMemoryPool : public ChunkedMemoryPool {
 public:
  PinnedChunkedMemoryPool() = default;
  ~PinnedChunkedMemoryPool() = default;
  void* MallocFnImpl(size_t size) override;
  void FreeFnImpl(void* ptr) override;
};
void *PinnedChunkedMemoryPool::MallocFnImpl(size_t size) {
  void *ptr;
  CUDA_CHECK(cudaMallocHost(&ptr, size));
  return ptr;
}
void PinnedChunkedMemoryPool::FreeFnImpl(void *ptr) {
  CUDA_CHECK(cudaFreeHost(ptr));
}

class HostChunkedMemoryPool : public ChunkedMemoryPool {
 public:
  HostChunkedMemoryPool() = default;
  ~HostChunkedMemoryPool() = default;
  void* MallocFnImpl(size_t size) override;
  void FreeFnImpl(void* ptr) override;
};
void *HostChunkedMemoryPool::MallocFnImpl(size_t size) {
  return malloc(size);
}
void HostChunkedMemoryPool::FreeFnImpl(void *ptr) {
  free(ptr);
}

class GlobalDemoChunkedMemoryPool {
 public:
  ~GlobalDemoChunkedMemoryPool();
  void* Malloc(size_t size, MemoryLocation location);
  void Free(void* ptr, size_t size, MemoryLocation location);
  void EmptyCache();
  static GlobalDemoChunkedMemoryPool* GetInst();
 private:
  static GlobalDemoChunkedMemoryPool m_inst_;
  GlobalDemoChunkedMemoryPool();
  std::vector<std::unique_ptr<DeviceChunkedMemoryPool>> device_chunked_mem_pools_;
  std::unique_ptr<PinnedChunkedMemoryPool> pinned_chunked_mem_pool_;
  std::unique_ptr<HostChunkedMemoryPool> host_chunked_mem_pool_;

  static constexpr int kMaxSupportedDeviceCount = 16;
};

GlobalDemoChunkedMemoryPool GlobalDemoChunkedMemoryPool::m_inst_;

GlobalDemoChunkedMemoryPool::~GlobalDemoChunkedMemoryPool() {
}

GlobalDemoChunkedMemoryPool* GlobalDemoChunkedMemoryPool::GetInst() {
  return &m_inst_;
}

GlobalDemoChunkedMemoryPool::GlobalDemoChunkedMemoryPool() {
  device_chunked_mem_pools_.resize(kMaxSupportedDeviceCount);
  for (int i = 0; i < kMaxSupportedDeviceCount; i++) {
    device_chunked_mem_pools_[i] = std::make_unique<DeviceChunkedMemoryPool>(i);
  }
  pinned_chunked_mem_pool_ = std::make_unique<PinnedChunkedMemoryPool>();
  host_chunked_mem_pool_ = std::make_unique<HostChunkedMemoryPool>();
}

void *GlobalDemoChunkedMemoryPool::Malloc(size_t size, MemoryLocation location) {
  if (location == ML_DEVICE) {
    int dev_id;
    CUDA_CHECK(cudaGetDevice(&dev_id));
    return device_chunked_mem_pools_[dev_id]->CachedMalloc(size);
  } else if (location == ML_PINNED) {
    return pinned_chunked_mem_pool_->CachedMalloc(size);
  } else if (location == ML_HOST) {
    return host_chunked_mem_pool_->CachedMalloc(size);
  } else {
    LOG_FATAL("invalid location = %d", location);
    return nullptr;
  }
}
void GlobalDemoChunkedMemoryPool::Free(void *ptr, size_t size, MemoryLocation location) {
  if (location == ML_DEVICE) {
    int dev_id;
    CUDA_CHECK(cudaGetDevice(&dev_id));
    device_chunked_mem_pools_[dev_id]->CachedFree(ptr, size);
  } else if (location == ML_PINNED) {
    pinned_chunked_mem_pool_->CachedFree(ptr, size);
  } else if (location == ML_HOST) {
    host_chunked_mem_pool_->CachedFree(ptr, size);
  } else {
    LOG_FATAL("invalid location = %d", location);
  }
}

void GlobalDemoChunkedMemoryPool::EmptyCache() {
  for (int i = 0; i < kMaxSupportedDeviceCount; i++) {
    device_chunked_mem_pools_[i]->EmptyCache();
  }
  pinned_chunked_mem_pool_->EmptyCache();
  host_chunked_mem_pool_->EmptyCache();
}

void DemoStorage::Allocate(size_t size, MemoryLocation location) {
  BOOL_CHECK(owned_);
  ptr_ = GlobalDemoChunkedMemoryPool::GetInst()->Malloc(size, location);
  storage_count_.fetch_add(1);
  size_ = size;
  memory_location_ = location;
}

void DemoStorage::MakeFromExistMemory(void* ptr, size_t size, MemoryLocation location) {
  owned_ = false;
  ptr_ = ptr;
  size_ = size;
  memory_location_ = location;
}

void DemoStorage::Free() {
  if (owned_) {
    GlobalDemoChunkedMemoryPool::GetInst()->Free(ptr_, size_, memory_location_);
    storage_count_.fetch_add(-1);
  }
  owned_ = true;
  memory_location_ = ML_UNDEFINED;
  ptr_ = nullptr;
  size_ = 0;
}

DemoStorage::~DemoStorage() {
  Free();
}

void DemoTensor::Reset() {
  storage_.reset();
  data_type_ = DT_UNDEFINED;
  elt_count_ = 0;
}

DemoTensor MakeNotOwnedTensor(void* ptr, MemoryLocation memory_location, DataType data_type, size_t elt_count) {
  BOOL_CHECK(data_type != DT_UNDEFINED);
  BOOL_CHECK(memory_location != ML_UNDEFINED);
  DemoTensor demo_tensor;
  demo_tensor.data_type_ = data_type;
  demo_tensor.elt_count_ = elt_count;
  demo_tensor.storage_.reset(new DemoStorage);
  demo_tensor.storage_->MakeFromExistMemory(ptr, elt_count * GetElementSize(data_type), memory_location);
  return demo_tensor;
}

void* DemoMemoryContext::GetPointer() {
  BOOL_CHECK(data_type_ != DT_UNDEFINED);
  return tensor_.DataPtr();
}

DemoMemoryAllocator::DemoMemoryAllocator(int dev_id) : dev_id_(dev_id) {
}

MemoryContextBase* DemoMemoryAllocator::CreateMemoryContext() {
  context_count_.fetch_add(1);
  return new DemoMemoryContext;
}
void DemoMemoryAllocator::DestroyMemoryContext(MemoryContextBase* memory_context) {
  context_count_.fetch_add(-1);
  delete memory_context;
}

void *DemoMemoryAllocator::Malloc(MemoryContextBase *memory_context,
                                  MemoryLocation location,
                                  DataType data_type,
                                  size_t element_count) {
  SensitiveZoneGuard sensitive_zone_guard(partitioner_->GetLocalRank());
  if (location == ML_DEVICE) {
    int dev_id;
    CUDA_CHECK(cudaGetDevice(&dev_id));
    BOOL_CHECK(dev_id == dev_id_);
  }
  BOOL_CHECK(memory_context != nullptr);
  auto* demo_context = static_cast<DemoMemoryContext*>(memory_context);
  demo_context->tensor_ = CreateTensor(location, data_type, element_count);
  demo_context->SetContextInfo(element_count, location, data_type);
  return demo_context->tensor_.DataPtr();
}

void DemoMemoryAllocator::Free(MemoryContextBase* memory_context) {
  SensitiveZoneGuard sensitive_zone_guard(partitioner_->GetLocalRank());
  BOOL_CHECK(memory_context != nullptr);
  if (!memory_context->IsValid()) {
    return;
  }
  auto* demo_context = static_cast<DemoMemoryContext*>(memory_context);
  if (demo_context->memory_location_ == ML_DEVICE) {
    int dev_id;
    CUDA_CHECK(cudaGetDevice(&dev_id));
    BOOL_CHECK(dev_id == dev_id_);
  }
  demo_context->tensor_.Reset();
  demo_context->Clear();
}

DemoTensor DemoMemoryAllocator::CreateTensor(MemoryLocation location,
                                             DataType data_type,
                                             size_t element_count) {
  DemoTensor tensor;
  tensor.data_type_ = data_type;
  tensor.elt_count_ = element_count;
  auto* storage = new DemoStorage();
  storage->Allocate(element_count * GetElementSize(data_type), location);
  tensor.storage_.reset(storage);
  return tensor;
}

MemoryContextBase* DemoMemoryAllocator::ToMemoryContext(DemoTensor& demo_tensor) {
  auto* demo_context = static_cast<DemoMemoryContext*>(CreateMemoryContext());
  demo_context->tensor_ = demo_tensor;
  demo_context->SetContextInfo(demo_tensor.EltCount(), demo_tensor.Location(), demo_tensor.GetDateType());
  return demo_context;
}

DemoTensor DemoMemoryAllocator::FromMemoryContext(MemoryContextBase* memory_context) {
  BOOL_CHECK(memory_context != nullptr);
  auto* demo_context = static_cast<DemoMemoryContext*>(memory_context);
  DemoTensor demo_tensor = demo_context->tensor_;
  Free(memory_context);
  return demo_tensor;
}