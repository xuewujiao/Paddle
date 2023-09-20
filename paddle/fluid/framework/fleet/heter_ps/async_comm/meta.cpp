#include "meta.h"

#include <cstring>

#include "partitioner.h"

static constexpr uint32_t kMetaMagic = 0xA54CCC0E;

void InitMeta(Meta* meta) {
  meta->magic_number = kMetaMagic;
  meta->status_code = 0;
  meta->request_id = -1;
  meta->requester_node_id = meta->runner_node_id = -1;
  meta->requester_lane_id = meta->runner_lane_id = -1;
  meta->is_response = 0;
  meta->merged_flag = 0;
  meta->runner_id = -1;
  meta->function_id = -1;
  meta->valid_data_count = 0;
  for (int i = 0; i < MAX_VEC_COUNT; i++) {
    meta->locations[i] = 0;
    meta->data_types[i] = 0;
    meta->data_sizes[i] = 0;
  }
}

void MakeResponseMeta(Meta* response_meta, const Meta* request_meta) {
  memcpy(response_meta, request_meta, sizeof(Meta));
  response_meta->valid_data_count = 0;
  response_meta->is_response = 1;
  for (int i = 0; i < MAX_VEC_COUNT; i++) {
    response_meta->locations[i] = 0;
    response_meta->data_types[i] = 0;
    response_meta->data_sizes[i] = 0;
  }
}

bool IsStopMeta(const Meta* meta) {
  return meta->runner_id == -1 && meta->function_id == -1 && meta->status_code == -1;
}

bool IsMergedMeta(const Meta* meta) {
  return meta->merged_flag == 1;
}

bool IsMergedSplittedMeta(const Meta* meta) {
  return meta->merged_flag == 2;
}

bool SameMeta(const Meta* meta_lhs, const Meta* meta_rhs) {
  if (meta_lhs->magic_number != meta_rhs->magic_number) return false;
  if (meta_lhs->status_code != meta_rhs->status_code) return false;
  if (meta_lhs->request_id != meta_rhs->request_id) return false;
  if (meta_lhs->is_response != meta_rhs->is_response) return false;
  if (meta_lhs->runner_id != meta_rhs->runner_id) return false;
  if (meta_lhs->function_id != meta_rhs->function_id) return false;
  if (meta_lhs->requester_node_id != meta_rhs->requester_node_id) return false;
  if (meta_lhs->runner_node_id != meta_rhs->runner_node_id) return false;
  if (meta_lhs->requester_lane_id != meta_rhs->requester_lane_id) return false;
  if (meta_lhs->runner_lane_id != meta_rhs->runner_lane_id) return false;
  if (meta_lhs->valid_data_count != meta_rhs->valid_data_count) return false;
  for (int i = 0; i < meta_lhs->valid_data_count; i++) {
    if (meta_lhs->locations[i] != meta_rhs->locations[i] || meta_lhs->data_types[i] != meta_rhs->data_types[i]
        || meta_lhs->data_sizes[i] != meta_rhs->data_sizes[i])
      return false;
  }
  return true;
}

int GetTargetGlobalRankFromMeta(const Meta* meta, int ranks_per_node) {
  int target_rank = -1;
  if (meta->is_response) {
    target_rank = meta->requester_node_id * ranks_per_node + meta->requester_lane_id;
  } else {
    target_rank = meta->runner_node_id * ranks_per_node + meta->runner_lane_id;
  }
  return target_rank;
}

int GetSrcGlobalRankFromMeta(const Meta* meta, int ranks_per_node) {
  int src_rank = -1;
  if (meta->is_response) {
    src_rank = meta->runner_node_id * ranks_per_node + meta->runner_lane_id;
  } else {
    src_rank = meta->requester_node_id * ranks_per_node + meta->requester_lane_id;
  }
  return src_rank;
}

int GetTargetLocalRankFromMeta(const Meta* meta) {
  return meta->is_response ? meta->requester_lane_id : meta->runner_lane_id;
}

int GetSrcLocalRankFromMeta(const Meta* meta) {
  return meta->is_response ? meta->runner_lane_id : meta->requester_lane_id;
}

int GetTargetNodeIdFromMeta(const Meta* meta) {
  return meta->is_response ? meta->requester_node_id : meta->runner_node_id;
}

int GetSrcNodeIdFromMeta(const Meta* meta) {
  return meta->is_response ? meta->runner_node_id : meta->requester_node_id;
}

bool IsLocalGPU(const Meta* meta) {
  return meta->requester_node_id == meta->runner_node_id && meta->requester_lane_id == meta->runner_lane_id;
}

bool IsLocalNode(const Meta* meta) {
  return meta->requester_node_id == meta->runner_node_id;
}

void CreateRequestMetaToRank(Meta* meta, int target_global_rank, Partitioner* partitioner) {
  int target_node_id = partitioner->GetNodeIDFromGlobalRank(target_global_rank);
  int target_local_rank = partitioner->GetLocalrankFromGlobalRank(target_global_rank);
  meta->status_code = 0;
  meta->is_response = 0;
  meta->runner_id = -1;
  meta->function_id = -1;
  meta->requester_node_id = static_cast<int16_t>(partitioner->GetNodeID());
  meta->requester_lane_id = static_cast<int8_t>(partitioner->GetLocalRank());
  meta->runner_node_id = static_cast<int16_t>(target_node_id);
  meta->runner_lane_id = static_cast<int8_t>(target_local_rank);
  meta->request_id = -1;
}

void CreateStopMetaToRank(Meta* meta, int target_global_rank, Partitioner* partitioner) {
  CreateRequestMetaToRank(meta, target_global_rank, partitioner);
  meta->status_code = -1;
  meta->request_id = -1;
}

