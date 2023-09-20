#pragma once

#include <unistd.h>

enum MemoryLocation {
  ML_UNDEFINED = 0,
  ML_DEVICE = 1,
  ML_PINNED = 2,
  ML_HOST = 3,
};

enum DataType {
  DT_UNDEFINED = 0,
  DT_INT8 = 8,
  DT_UINT8 = 9,
  DT_FP8_E4M3 = 12,
  DT_FP8_E5M2 = 13,
  DT_INT16 = 16,
  DT_UINT16 = 17,
  DT_HALF = 20,
  DT_BF16 = 21,
  DT_INT32 = 32,
  DT_UINT32 = 33,
  DT_FLOAT = 36,
  DT_INT64 = 64,
  DT_UINT64 = 65,
  DT_DOUBLE = 68,
};

inline size_t GetElementSize(DataType data_type) {
  auto data_type_size_t = static_cast<size_t>(data_type);
  return (data_type_size_t >> 3LL);
}

// Lifetime of MemoryContextBase, four kind of life cycles
// 1. Created by Requester (using ToMemoryContext API)
//    Deleted by Communicator lib (deleted by FreeReqRes with data held)
// 2. Created by Communicator lib
//    Forwarded to Runner (using FromMemoryContext API) into Tensor, data owner transfers to Tensor.
//    Context object without data ownership deleted by FreeReqRes if forwarded
// 3. Created by Runner (using ToMemoryContext API)
//    Deleted by Communicator lib (deleted by FreeReqRes with data held)
// 4. Created by Communicator lib
//    Forwarded to Requester (using FromMemoryContext API) into Tensor, data owner transfers to Tensor.
//    Context object without data ownership deleted by FreeReqRes if forwarded
class MemoryContextBase {
 public:
  MemoryContextBase() = default;
  virtual ~MemoryContextBase() = default;
  virtual void* GetPointer() = 0;
  void SetContextInfo(size_t elt_count, MemoryLocation memory_location, DataType data_type) {
    elt_count_ = elt_count;
    memory_location_ = memory_location;
    data_type_ = data_type;
    size_ = elt_count * GetElementSize(data_type);
    valid_ = true;
  }
  size_t GetSize() const {
    return size_;
  }
  size_t GetEltCount() const {
    return elt_count_;
  }
  MemoryLocation GetMemoryLocation() const {
    return memory_location_;
  }
  DataType GetDataType() const {
    return data_type_;
  }
  void Clear() {
    size_ = 0;
    memory_location_ = ML_UNDEFINED;
    data_type_ = DT_UNDEFINED;
    elt_count_ = 0;
    valid_ = false;
  }
  bool IsValid() const {
    return valid_;
  }
  bool NeedAllocator() const {
    return need_allocator_;
  }
 protected:
  bool valid_ = false;
  size_t size_ = 0;
  MemoryLocation memory_location_ = ML_UNDEFINED;
  DataType data_type_ = DT_UNDEFINED;
  size_t elt_count_ = 0;
  bool need_allocator_ = true;
};

struct AsyncReqRes;

class MemoryAllocatorBase {
 public:
  MemoryAllocatorBase() = default;
  virtual ~MemoryAllocatorBase() = default;
  virtual MemoryContextBase* CreateMemoryContext() = 0;
  virtual void DestroyMemoryContext(MemoryContextBase* memory_context) = 0;

  virtual void *Malloc(MemoryContextBase *memory_context,
                       MemoryLocation location,
                       DataType data_type,
                       size_t element_size) = 0;

  virtual void Free(MemoryContextBase* memory_context) = 0;

  virtual void AllocateReqResByMeta(AsyncReqRes* req_res);
  virtual void *AllocateOrUseExistContextPointer(AsyncReqRes *req_res,
                                                 int ctx_idx,
                                                 MemoryLocation location,
                                                 DataType data_type,
                                                 size_t elt_count);
  virtual void FreeReqRes(AsyncReqRes* req_res);
};
