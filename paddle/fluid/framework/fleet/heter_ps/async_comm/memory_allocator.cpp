#include "memory_allocator.h"

#include "async_communicator.h"
#include "check_macros.h"
#include "log_macros.h"
#include "glog/logging.h"

void MemoryAllocatorBase::AllocateReqResByMeta(AsyncReqRes* req_res) {
  int data_count = req_res->meta.valid_data_count;
  for (int i = 0; i < data_count; i++) {
    auto dt = static_cast<DataType>(req_res->meta.data_types[i]);
    auto location = static_cast<MemoryLocation>(req_res->meta.locations[i]);
    size_t elt_count = req_res->meta.data_sizes[i] / GetElementSize(dt);
    BOOL_CHECK(req_res->meta.data_sizes[i] % GetElementSize(dt) == 0);
    AllocateOrUseExistContextPointer(req_res, i, location, dt, elt_count);
  }
}

void *MemoryAllocatorBase::AllocateOrUseExistContextPointer(AsyncReqRes *req_res,
                                                            int ctx_idx,
                                                            MemoryLocation location,
                                                            DataType data_type,
                                                            size_t elt_count) {
  BOOL_CHECK(req_res != nullptr);
  if (req_res->memory_contexts[ctx_idx] == nullptr) {
    req_res->memory_contexts[ctx_idx] = CreateMemoryContext();
    Malloc(req_res->memory_contexts[ctx_idx], location, data_type, elt_count);
  } else {
    BOOL_CHECK(req_res->memory_contexts[ctx_idx]->GetMemoryLocation() == location);
    BOOL_CHECK(req_res->memory_contexts[ctx_idx]->GetDataType() == data_type);
    BOOL_CHECK(req_res->memory_contexts[ctx_idx]->GetEltCount() == elt_count);
  }
  return req_res->memory_contexts[ctx_idx]->GetPointer();
}

void MemoryAllocatorBase::FreeReqRes(AsyncReqRes* req_res) {
  int data_count = req_res->meta.valid_data_count;
  for (int i = 0; i < data_count; i++) {
    MemoryContextBase* memory_context = req_res->memory_contexts[i];
    BOOL_CHECK(memory_context != nullptr);
    if (memory_context->NeedAllocator()) {
      Free(memory_context);
      DestroyMemoryContext(memory_context);
    } else {
      delete memory_context;
    }
    req_res->memory_contexts[i] = nullptr;
  }
  if (req_res->complete_cb != nullptr) {
    req_res->complete_cb();
    req_res->complete_cb = nullptr;
  }
  InitMeta(&req_res->meta);
  DestroyAsyncReqRes(req_res);
}