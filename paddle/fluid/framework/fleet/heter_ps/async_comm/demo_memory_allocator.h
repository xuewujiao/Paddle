#pragma once

#include <atomic>
#include <memory>

#include "memory_allocator.h"

class DemoStorage {
 public:
  DemoStorage() = default;
  ~DemoStorage();
  void Allocate(size_t size, MemoryLocation location);
  void MakeFromExistMemory(void* ptr, size_t size, MemoryLocation location);
  void Free();
 private:
  MemoryLocation memory_location_ = ML_UNDEFINED;
  void* ptr_ = nullptr;
  size_t size_ = 0;
  bool owned_ = true;
  friend class DemoTensor;
};

class DemoTensor {
 public:
  DemoTensor() = default;
  ~DemoTensor() = default;
  void* DataPtr() {
    return storage_->ptr_;
  }
  size_t EltCount() const {
    return elt_count_;
  }
  MemoryLocation Location() const {
    return storage_->memory_location_;
  }
  DataType GetDateType() const {
    return data_type_;
  }
  void Reset();
 private:
  std::shared_ptr<DemoStorage> storage_;
  DataType data_type_ = DT_UNDEFINED;
  size_t elt_count_ = 0;
  friend class DemoMemoryContext;
  friend class DemoMemoryAllocator;
  friend DemoTensor MakeNotOwnedTensor(void* ptr, MemoryLocation memory_location, DataType data_type, size_t elt_count);
};

DemoTensor MakeNotOwnedTensor(void* ptr, MemoryLocation memory_location, DataType data_type, size_t elt_count);

class DemoMemoryContext : public MemoryContextBase {
 public:
  DemoMemoryContext() = default;
  ~DemoMemoryContext() override = default;
  void* GetPointer() override;
 private:
  DemoTensor tensor_;
  friend class DemoMemoryAllocator;
};

class DemoMemoryAllocator : public MemoryAllocatorBase {
 public:
  explicit DemoMemoryAllocator(int dev_id);
  ~DemoMemoryAllocator() override = default;
  MemoryContextBase* CreateMemoryContext() override;
  void DestroyMemoryContext(MemoryContextBase* memory_context) override;

  void *Malloc(MemoryContextBase *memory_context,
               MemoryLocation location,
               DataType data_type,
               size_t element_count) override;

  void Free(MemoryContextBase* memory_context) override;

  DemoTensor CreateTensor(MemoryLocation location,
                          DataType data_type,
                          size_t element_count);

  // This may be called in Requester before send request or in Runner before send back response.
  MemoryContextBase* ToMemoryContext(DemoTensor& demo_tensor);
  // This may be called in Requester after receive response or in Runner after receive request.
  DemoTensor FromMemoryContext(MemoryContextBase* memory_context);
 protected:
  int dev_id_ = -1;
};

void PrintRefCounts();

