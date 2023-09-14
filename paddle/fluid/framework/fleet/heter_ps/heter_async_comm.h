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

#pragma once
#include <memory>
#include <vector>
#if defined(PADDLE_WITH_CUDA)
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/timer.h"
#elif defined(PADDLE_WITH_XPU_KP)
#include <xpu/runtime.h>

#include "paddle/fluid/platform/device/xpu/enforce_xpu.h"
#endif

#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/framework/fleet/heter_ps/heter_resource.h"
#include "paddle/fluid/framework/fleet/heter_ps/async_comm/async_communicator.h"
#include "paddle/fluid/framework/fleet/heter_ps/async_comm/config.h"
#include "paddle/fluid/framework/fleet/heter_ps/async_comm/ib_utils.h"

#ifdef PADDLE_WITH_HETERPS
namespace paddle {
namespace framework {
class AsyncComAllocator;
class  AsyncComMemContext : public MemoryContextBase {
 public:
     AsyncComMemContext() : MemoryContextBase() {
      raw_point = nullptr;
      poll_point = nullptr;
      cpu_poll_point = nullptr;
  };
  ~ AsyncComMemContext()  override = default;

  void clear() {
	raw_point = nullptr;
	poll_point = nullptr;
	cpu_poll_point = nullptr;
  }

  void* GetPointer() override {
      if (raw_point != nullptr) {
          return raw_point;
      }
      else if(poll_point != nullptr) {
          return poll_point->ptr();
      }
      else{
          return (void*) (cpu_poll_point.get());
      }
  }
  void * raw_point;
  std::shared_ptr<phi::Allocation> poll_point;
  std::shared_ptr<char> cpu_poll_point;

 private:
  // TODO(Baidu): Need to check if this is enough
  friend class  AsyncComAllocator;
};
class AsyncComAllocator : public MemoryAllocatorBase {
 public:
 	AsyncComAllocator(int dev_id) : MemoryAllocatorBase() {
	  place_ = platform::CUDAPlace(dev_id);
	  cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
	  gpu_id_ = dev_id;
  };
  ~AsyncComAllocator() {
	  platform::CUDADeviceGuard guard(gpu_id_);
	  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamDestroy(stream_));
  }
  MemoryContextBase* CreateMemoryContext() override {
	  return new AsyncComMemContext();
  }

  cudaStream_t GetCudaMemoryStream() {
	  return stream_;
  }

  void DestroyMemoryContext(MemoryContextBase* memory_context) {
	  delete memory_context;
  }

  void *Malloc(MemoryContextBase *memory_context,
               MemoryLocation location,
               ::DataType data_type,
               size_t element_count) {
	  if (location == ML_PINNED) {
	  	VLOG(0) << "don not support ML_PINNED type";
	  }
      auto* async_context = static_cast<AsyncComMemContext*>(memory_context);
	  if (location == ML_DEVICE) {
	      platform::CUDADeviceGuard guard(gpu_id_);
	      size_t need_mem = element_count * GetElementSize(data_type);
	      async_context->poll_point = memory::AllocShared(place_, need_mem, phi::Stream(reinterpret_cast<phi::StreamId>(stream_)));
	      async_context->SetContextInfo(element_count, location, data_type);
	      return async_context->poll_point->ptr();
	  } else {
		  size_t need_mem = element_count * GetElementSize(data_type);
		  async_context->cpu_poll_point.reset(new char[need_mem], [](char * p) {delete[] p;});
		  async_context->SetContextInfo(element_count, location, data_type);
		  return (void *) async_context->cpu_poll_point.get();
	  }
  }

  void Free(MemoryContextBase* memory_context) {
	  platform::CUDADeviceGuard guard(gpu_id_);
	  auto* async_context = static_cast<AsyncComMemContext*>(memory_context);
	  async_context->poll_point = nullptr;
	  async_context->cpu_poll_point = nullptr;
  }

  //data can free by comlib
  MemoryContextBase* ToMemoryContext(std::shared_ptr<phi::Allocation>& data, size_t elt_count,
		  ::DataType data_type) {
	  MemoryLocation location = ML_DEVICE;
	  auto* async_context = static_cast<AsyncComMemContext*>(CreateMemoryContext());
	  async_context->poll_point = data;
	  async_context->SetContextInfo(elt_count, location, data_type);
	  return async_context;
  };
  //data can not free by comlib
  MemoryContextBase* ToMemoryContext(void * data, size_t elt_count,
		  ::DataType data_type, MemoryLocation location = ML_DEVICE) {
	  auto* async_context = static_cast<AsyncComMemContext*>(CreateMemoryContext());
	  async_context->raw_point = data;
	  async_context->SetContextInfo(elt_count, location, data_type);
	  return async_context;
  }

private:
#if defined(PADDLE_WITH_CUDA)
    platform::CUDAPlace place_;
#elif defined(PADDLE_WITH_XPU_KP)
    platform::XPUPlace place_;
#endif
    cudaStream_t stream_;
    int gpu_id_;
};

class AsyncContext {
public:
	 static std::shared_ptr<AsyncContext> s_instance_;
	 static std::shared_ptr<AsyncContext> GetInstance() {
	    if (NULL == s_instance_) {
	      s_instance_.reset(new AsyncContext());
	    }
	    return s_instance_;
	 }
	 static void Destroy() {
	   if (s_instance_ != NULL) {
		   s_instance_ = NULL;
	   }
	 }

	 ~AsyncContext() {
	   VLOG(0) << " call ~AsyncContext ";
	   if (_start) {
	    std::vector<std::unique_ptr<std::thread>> init_threads(_card_num);
	    for (int i = 0; i < _card_num; i++) {
	         init_threads[i] = std::make_unique<std::thread>([this,i]() {
	           platform::CUDADeviceGuard guard(i);
	           async_coms[i]->SendStopSignal();
	           async_coms[i]->WaitStopped();
	           async_coms[i]->DestroyResources();
	       });
	    }
	    for (int i = 0; i < _card_num; i++) {
	       init_threads[i]->join();
	    }
	   }
	   VLOG(0) << " call IbDeInit ";
	   if (_is_init) {
	     IbDeInit();
	   }
	   VLOG(0) << " call IbDeInit end";
	 }
	 static Config config;
	 std::vector<std::shared_ptr<RunnerRegistry>> registrys;
	 std::vector<std::shared_ptr<Partitioner>>  partitioners;
	 std::vector<std::shared_ptr<AsyncComAllocator>> async_com_allocators;
	 std::vector<std::shared_ptr<AsyncCommunicator>> async_coms;
	 bool _is_init = false;
	 bool _start = false;
	 int _card_num = 0;
     void init (int node_num, int card_num, int node_id) {
       if (!_is_init) {
    	 _is_init = true;
         _card_num = card_num;
         auto rdma_checker = GpuRDMAChecker::get(card_num);
         config.agent_local_rank.clear();
         config.ib_device_name.clear();
         if (rdma_checker->need_rdma_trans()) {
             for (int gpu_id = 0; gpu_id < card_num; gpu_id++) {
               if (!rdma_checker->is_device_support_rdma(gpu_id)){
                  int trans_id = (gpu_id + 4) % card_num;
                  config.agent_local_rank.push_back(trans_id);
              }
             else{
            	 config.agent_local_rank.push_back(gpu_id);
             }
           }
         } else{
        	 for (int gpu_id = 0; gpu_id < card_num; gpu_id++) {
        		 config.agent_local_rank.push_back(gpu_id);
        	 }
         }
         auto gpu_mlxs = rdma_checker->get_gpu_mlxs();
         for (int gpu_id = 0; gpu_id < card_num; gpu_id++) {
             auto agent_id = config.agent_local_rank[gpu_id];
             config.ib_device_name.push_back(gpu_mlxs[agent_id]);
         }
    	 IbInit();
    	 config.node_count = node_num;
    	 config.sideband_server_name = "localhost";
    	 config.sideband_server_port = 24680;
    	 config.ib_port = {1, 1, 1, 1, 1, 1, 1, 1};
    	 partitioners.reserve(card_num);
         async_com_allocators.reserve(card_num);
         registrys.reserve(card_num);
         async_coms.reserve(card_num);
         VLOG(0) << "node num is " << node_num << " card num is " << card_num << " node id is " << node_id;
         for (int i = 0; i < card_num; i++) {
        	 VLOG(0) << "card: " << i << " use network "
        	   << config.ib_device_name[i] << " agent " << config.agent_local_rank[i];
         }

    	 for (int i = 0; i < card_num; i++) {
    	   partitioners.push_back(std::shared_ptr<Partitioner>(new Partitioner(node_num, card_num, node_id, i)));
    	   async_com_allocators.push_back(std::shared_ptr<AsyncComAllocator>( new AsyncComAllocator(i)));
    	   registrys.push_back(std::shared_ptr<RunnerRegistry>(new RunnerRegistry()));
    	   async_coms.push_back(std::shared_ptr<AsyncCommunicator>(new AsyncCommunicator(partitioners[i].get(), async_com_allocators[i].get(), registrys[i].get(), &config)));
    	 }
       }
     }

     void start() {
       if (!_start) {
         _start = true;
         VLOG(0) << "async com begin start";
         std::vector<std::unique_ptr<std::thread>> init_threads(_card_num);
         for (int i = 0; i < _card_num; i++) {
           init_threads[i] = std::make_unique<std::thread>([this,i]() {
             platform::CUDADeviceGuard guard(i);
             async_coms[i]->CreateResources();
             async_coms[i]->Start();
           });
         }
         for (int i = 0; i < _card_num; i++) {
           init_threads[i]->join();
         }
         VLOG(0) <<"async com start end";
       }
     }

     RunnerRegistry * get_registry (int gpu_id) {
       return registrys[gpu_id].get();
     }

     Partitioner * get_partitioner(int gpu_id) {
    	 return partitioners[gpu_id].get();
     }

     AsyncComAllocator * get_alloctor(int gpu_id) {
         return async_com_allocators[gpu_id].get();
     }

     AsyncCommunicator * get_async_com(int gpu_id) {
       return async_coms[gpu_id].get();
     }
};


//runner 较轻量，只用于发送请求，对于采样等  可以在此类加虚函数，并继承此类
class RequestRunner : public QueuedRunner {
public:
	RequestRunner(Partitioner *partitioner, MemoryAllocatorBase *allocator)
      : QueuedRunner(partitioner, allocator) {
	}
  virtual ~RequestRunner() {};
  MemoryAllocatorBase* get_allocator() {
     return allocator_;
  }
  virtual std::string get_runner_name() {return std::string("RequestRunner");}

  AsyncReqRes* MakePullRequest(MemoryContextBase* memory_context, int node_id, int gpu_id);
  AsyncReqRes* MakePullRequest(MemoryContextBase* memory_context, int target_global_rank);
  AsyncReqRes* MakePushRequest(MemoryContextBase *grad_context,
									  int node_id, int gpu_id);
  AsyncReqRes* MakePushRequest(MemoryContextBase *grad_context,
									  int target_global_rank);

  virtual AsyncReqRes* MakeDeepWalkRequest(MemoryContextBase *node_key_context,
                                           MemoryContextBase *para_int_context,
                                           int target_global_rank){ 
  PADDLE_ENFORCE_NE(-1,
                    -1,
                    platform::errors::InvalidArgument(
                      "enter base virtual function."));
  return nullptr; 
}
  virtual AsyncReqRes* MakeFloatFeaturePullRequest(MemoryContextBase *node_key_context,
                                              int target_global_rank){ 
  PADDLE_ENFORCE_NE(-1,
                    -1,
                    platform::errors::InvalidArgument(
                      "enter base virtual function."));
  return nullptr; 
}
  virtual AsyncReqRes* MakeUintFeaturePullRequest(MemoryContextBase *node_key_context,
                                              int target_global_rank){ 
  PADDLE_ENFORCE_NE(-1,
                    -1,
                    platform::errors::InvalidArgument(
                      "enter base virtual function."));
  return nullptr; 
}
  virtual AsyncReqRes* MakeSageSampleRequest(MemoryContextBase *node_key_context,
                                             MemoryContextBase *para_int_context,
                                             int target_global_rank) { 
  PADDLE_ENFORCE_NE(-1,
                    -1,
                    platform::errors::InvalidArgument(
                      "enter base virtual function."));
  return nullptr; 
}
  
  virtual AsyncReqRes* MakeDegreeGetRequest(MemoryContextBase *node_key_context,
                                            MemoryContextBase *para_int_context,
                                            int target_global_rank){ 
  PADDLE_ENFORCE_NE(-1,
                    -1,
                    platform::errors::InvalidArgument(
                      "enter base virtual function."));
  return nullptr; 
}
};

}  // end namespace framework
}  // end namespace paddle


#endif
