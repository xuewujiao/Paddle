/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/fleet/heter_ps/heter_async_comm.h"
#include <vector>

#ifdef PADDLE_WITH_HETERPS

namespace paddle {
namespace framework {

std::shared_ptr<AsyncContext> AsyncContext::s_instance_;
Config AsyncContext::config;

AsyncReqRes* RequestRunner::MakePullRequest(MemoryContextBase* memory_context, int target_global_rank) {
	AsyncReqRes *request = CreateAsyncReqRes();
    InitMeta(&request->meta);
    request->meta.valid_data_count = 1;
	request->memory_contexts[0] = memory_context;
	request->FillMetaByMemoryContext();
	static constexpr int kPsGetFuncId = 0;
	CreateRequestMeta(&request->meta, target_global_rank, kPsGetFuncId);
	return request;
}

AsyncReqRes* RequestRunner::MakePullRequest(MemoryContextBase* memory_context, int node_id, int gpu_id) {
	int target_global_rank = partitioner_->MakeGlobalRank(node_id, gpu_id);
	return MakePullRequest(memory_context, target_global_rank);
}

AsyncReqRes* RequestRunner::MakePushRequest(MemoryContextBase *grad_context,
									  int target_global_rank) {
	  AsyncReqRes *request = CreateAsyncReqRes();
	  InitMeta(&request->meta);
	  request->meta.valid_data_count = 1;
	  request->memory_contexts[0] = grad_context;
	  request->FillMetaByMemoryContext();
	  static constexpr int kPsUpdateFuncId = 1;
	  CreateRequestMeta(&request->meta, target_global_rank, kPsUpdateFuncId);
	  return request;
}

AsyncReqRes* RequestRunner::MakePushRequest(MemoryContextBase *grad_context,
									  int node_id, int gpu_id) {
	 int target_global_rank = partitioner_->MakeGlobalRank(node_id, gpu_id);
	 return MakePushRequest(grad_context, target_global_rank);
}

}  // end namespace framework
}  // end namespace paddle
#endif


