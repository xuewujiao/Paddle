#include "runner.h"
#include "batched_queued_runner.h"

#include <cuda_runtime_api.h>

class DemoRandomWalkRunner : public BatchedQueuedRunner {
 public:
  DemoRandomWalkRunner(Partitioner *partitioner, MemoryAllocatorBase *allocator)
      : BatchedQueuedRunner(partitioner, allocator) {}
  ~DemoRandomWalkRunner() override = default;
  void RegisterFunctions() override;

  AsyncReqRes* MakeRandomWalkRequest(MemoryContextBase* memory_context, int target_global_rank);
  AsyncReqRes *MakeRandomWalkRequestWithPara(MemoryContextBase *memory_context,
                                             MemoryContextBase *para_context,
                                             int target_global_rank);
 protected:
  void ProcessSetup() override;
  void ProcessCleanUp() override;

  void GetRandomWalkResult(AsyncReqRes *request, AsyncReqRes *response);
  void GetRandomWalkResultBatched(AsyncReqRes **request, AsyncReqRes **response, size_t batch_size);
  void GetRandomWalkResultWithPara(AsyncReqRes *request, AsyncReqRes *response);
  void GetRandomWalkResultWithParaBatched(AsyncReqRes **request, AsyncReqRes **response, size_t batch_size);

  cudaStream_t stream_ = nullptr;
};

class DemoEmbeddingPsRunner : public QueuedRunner {
 public:
  DemoEmbeddingPsRunner(Partitioner *partitioner, MemoryAllocatorBase *allocator)
      : QueuedRunner(partitioner, allocator) {}
  ~DemoEmbeddingPsRunner() override = default;
  void RegisterFunctions() override;

  int GetEmbeddingDim() const {
    return embedding_dim_;
  }

  AsyncReqRes* MakePsGetRequest(MemoryContextBase* memory_context, int target_global_rank);
  AsyncReqRes *MakePsUpdateRequest(MemoryContextBase *indice_context,
                                   MemoryContextBase *grad_context,
                                   int target_global_rank);

  void ProcessSetup() override;
  void ProcessCleanUp() override;

  void GetEmbedding(AsyncReqRes *request, AsyncReqRes *response);
  void UpdateEmbedding(AsyncReqRes *request, AsyncReqRes *response);
 private:
  int embedding_dim_ = 69;
  cudaStream_t stream_ = nullptr;
};
