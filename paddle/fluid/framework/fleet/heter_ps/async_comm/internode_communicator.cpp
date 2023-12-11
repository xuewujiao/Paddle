#include "internode_communicator.h"

#include <memory>

#include "async_communicator.h"
#include "synchronizer.h"

InterNodeCommunicator::InterNodeCommunicator(Partitioner *partitioner, MemoryAllocatorBase *allocator, Config* config) :
    partitioner_(partitioner), allocator_(allocator), config_(config) {
  copy_thread_ = std::make_unique<CopyThread>(partitioner, allocator);
  agent_ = std::make_unique<Agent>(partitioner, config);
}
InterNodeCommunicator::~InterNodeCommunicator() {

}
void InterNodeCommunicator::CreateResources() {
  InitSynchronizerHelperFunc(partitioner_->GetLocalRank(), async_communicator_->GetSideBandCommunicator());
  copy_thread_->SetAsyncCommunicator(async_communicator_);
  copy_thread_->CreateResources();
  agent_->SetSideBandCommunicator(async_communicator_->GetSideBandCommunicator());
  agent_->CreateResources();
}
void InterNodeCommunicator::DestroyResources() {
  copy_thread_->DestroyResources();
  agent_->DestroyResources();
  DeinitSynchronizerHelperFunc(partitioner_->GetLocalRank());
}
void InterNodeCommunicator::Connect() {
  copy_thread_->ConnectFifo();
  agent_->ConnectFifo();
  agent_->ConnectNetwork();
}
void InterNodeCommunicator::Start() {
  copy_thread_->Start();
  agent_->Start();
}
void InterNodeCommunicator::Stop() {
  copy_thread_->Stop();
  agent_->Stop();
}
void InterNodeCommunicator::WaitStopped() {
  copy_thread_->WaitStopped();
  agent_->WaitStopped();
}
void InterNodeCommunicator::Send(AsyncReqRes *req_res) {
  copy_thread_->Send(req_res);
}
