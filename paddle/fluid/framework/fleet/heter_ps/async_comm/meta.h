#pragma once

#include <stdint.h>
#include <unistd.h>

#define MAX_VEC_COUNT (4)

struct Meta {
  uint32_t magic_number;              // magic number for Meta package
  int32_t status_code;                // for stop signal -1, otherwise,  for request 0, for response it is return code.
  uint64_t request_id;                // id for request / response (per runner), -1 for stop signal
  int8_t is_response;                 // 0 for request, 1 for response
  int8_t merged_flag;                 // 0 for normal, 1 for merged request
  int8_t runner_id;                   // requested runner id, or -1 for stop signal
  int8_t function_id;                 // requested function id, or -1 for stop signal
  int16_t requester_node_id;          // requester node id
  int16_t runner_node_id;             // runner node id
  int8_t requester_lane_id;           // requester lane id
  int8_t runner_lane_id;              // runner lane id
  int16_t valid_data_count;           // data count of this package
  int8_t locations[MAX_VEC_COUNT];    // locations for each data
  int8_t data_types[MAX_VEC_COUNT];   // data types for each data
  size_t data_sizes[MAX_VEC_COUNT];   // data (memory) size of each data
};

void InitMeta(Meta* meta);

void MakeResponseMeta(Meta* response_meta, const Meta* request_meta);

bool IsStopMeta(const Meta* meta);

bool IsMergedMeta(const Meta* meta);

bool IsMergedSplittedMeta(const Meta* meta);

bool SameMeta(const Meta* meta_lhs, const Meta* meta_rhs);

int GetTargetGlobalRankFromMeta(const Meta* meta, int ranks_per_node);

int GetSrcGlobalRankFromMeta(const Meta* meta, int ranks_per_node);

int GetTargetLocalRankFromMeta(const Meta* meta);

int GetSrcLocalRankFromMeta(const Meta* meta);

int GetTargetNodeIdFromMeta(const Meta* meta);

int GetSrcNodeIdFromMeta(const Meta* meta);

bool IsLocalGPU(const Meta* meta);

bool IsLocalNode(const Meta* meta);

class Partitioner;

void CreateRequestMetaToRank(Meta* meta, int target_global_rank, Partitioner* partitioner);

void CreateStopMetaToRank(Meta* meta, int target_global_rank, Partitioner* partitioner);
