#include "agent_copy_message.h"

#include "fifo_utils.h"

std::string GetAgentCopyFifoName(const std::string &type,
                                 int node_id,
                                 int local_rank) {
  std::string name = GetFifoNamePrefix();
  name.append("internode_node_").append(type).append("_").append(std::to_string(node_id))
      .append("_lr_").append(std::to_string(local_rank));
  return name;
}
