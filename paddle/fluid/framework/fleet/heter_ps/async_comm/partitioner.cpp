#include "partitioner.h"

#include "check_macros.h"

int64_t RankSet::MakeKey(int node_id, int local_rank) {
  BOOL_CHECK(local_rank >= 0);
  BOOL_CHECK(node_id >= 0);
  int64_t key = node_id;
  key <<= 32LL;
  key |= local_rank;
  return key;
}

bool RankSet::TryAddRank(int node_id, int local_rank) {
  BOOL_CHECK(local_rank >= 0 && local_rank < ranks_per_node_);
  BOOL_CHECK(node_id >= 0 && node_id < node_count_);
  auto key = MakeKey(node_id, local_rank);
  auto it = rank_set_.find(key);
  if (it == rank_set_.end()) {
    rank_set_.insert(key);
    return true;
  }
  return false;
}

void RankSet::AddRank(int node_id, int local_rank) {
  BOOL_CHECK(TryAddRank(node_id, local_rank));
}

bool RankSet::TryRemoveRank(int node_id, int local_rank) {
  auto key = MakeKey(node_id, local_rank);
  auto it = rank_set_.find(key);
  if (it != rank_set_.end()) {
    rank_set_.erase(key);
    return true;
  }
  return false;
}

void RankSet::RemoveRank(int node_id, int local_rank) {
  BOOL_CHECK(TryRemoveRank(node_id, local_rank));
}

bool RankSet::HasRank(int node_id, int local_rank) {
  auto key = MakeKey(node_id, local_rank);
  auto it = rank_set_.find(key);
  return it != rank_set_.end();
}

void RankSet::Clear() {
  rank_set_.clear();
}

void RankSet::MakeFullSet() {
  rank_set_.clear();
  BOOL_CHECK(ranks_per_node_ > 0);
  BOOL_CHECK(node_count_ > 0);
  for (int node_id = 0; node_id < node_count_; node_id++) {
    for (int local_rank = 0; local_rank < ranks_per_node_; local_rank++) {
      auto key = MakeKey(node_id, local_rank);
      rank_set_.insert(key);
    }
  }
}
