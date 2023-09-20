#pragma once

#include <cstdint>
#include <set>

class RankSet {
 public:
  RankSet() = delete;
  RankSet(int node_count, int ranks_per_node) :
      node_count_(node_count), ranks_per_node_(ranks_per_node) {}
  ~RankSet() = default;
  // return true if success, false if already exist.
  bool TryAddRank(int node_id, int local_rank);
  void AddRank(int node_id, int local_rank);
  // return true if success, false if not exist.
  bool TryRemoveRank(int node_id, int local_rank);
  void RemoveRank(int node_id, int local_rank);
  bool HasRank(int node_id, int local_rank);
  size_t Size() const {
    return rank_set_.size();
  }
  bool Empty() const {
    return rank_set_.empty();
  };
  bool Full() const {
    return int(rank_set_.size()) == node_count_ * ranks_per_node_;
  }
  void Clear();
  void MakeFullSet();
 private:
  static int64_t MakeKey(int node_id, int local_rank);
  int node_count_ = 0;
  int ranks_per_node_ = 0;
  std::set<int64_t> rank_set_;
};

class Partitioner {
 public:
  Partitioner(int node_count, int ranks_per_node, int node_id, int local_rank) :
      node_count_(node_count), ranks_per_node_(ranks_per_node), node_id_(node_id), local_rank_(local_rank) {}
  ~Partitioner() = default;
  inline int GetNodeID() const {
    return node_id_;
  }
  inline int GetLocalRank() const {
    return local_rank_;
  }
  inline int GetNodeCount() const {
    return node_count_;
  }
  inline int GetRanksPerNode() const {
    return ranks_per_node_;
  }
  inline int GetGlobalRank() const {
    return node_id_ * ranks_per_node_ + local_rank_;
  }
  inline int GetGlobalSize() const {
    return node_count_ * ranks_per_node_;
  }
  inline bool IsSameNode(int node_id) const {
    return node_id == node_id_;
  }
  inline bool IsSameRank(int node_id, int local_rank) const {
    return node_id == node_id_ && local_rank == local_rank_;
  }
  inline bool IsSameLocalrank(int local_rank) const {
    return local_rank == local_rank_;
  }
  inline int GetNodeIDFromGlobalRank(int global_rank) const {
    return global_rank / ranks_per_node_;
  }
  inline int GetLocalrankFromGlobalRank(int global_rank) const {
    return global_rank % ranks_per_node_;
  }
  inline int MakeGlobalRank(int node_id, int local_rank) const {
    return node_id * ranks_per_node_ + local_rank;
  }
 private:
  int node_count_ = 0;
  int ranks_per_node_ = 0;
  int node_id_ = -1;
  int local_rank_ = -1;
};