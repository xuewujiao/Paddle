if(WITH_GPU)
  find_library(IBVERBS_LIB ibverbs)
  if(NOT IBVERBS_LIB)
    message(FATAL_ERROR "ibverbs library not found")
  endif()
  message(INFO " IBVERBS_LIB=${IBVERBS_LIB}")
  include_directories(${CUDA_INCLUDE_DIRS})

  cc_library(
    async_comm
    SRCS agent.h agent_copy_message.h config.h copy_thread.h cuda_utils.h fifo_utils.h
         ib_utils.h intranode_communicator.h internode_communicator.h log_macros.h memory_allocator.h
         meta.h net_utils.h partitioner.h runner.h sideband_communicator.h async_communicator.h
         batched_queued_runner.h merged_handler.h synchronizer.h
         agent.cpp agent_copy_message.cpp config.cpp copy_thread.cpp cuda_utils.cpp fifo_utils.cpp
         ib_utils.cpp intranode_communicator.cpp internode_communicator.cpp log_macros.cpp memory_allocator.cpp
         meta.cpp net_utils.cpp partitioner.cpp runner.cpp sideband_communicator.cpp async_communicator.cpp
         batched_queued_runner.cpp merged_handler.cpp synchronizer.cpp
    DEPS glog flags ${IBVERBS_LIB}
   )
endif()
