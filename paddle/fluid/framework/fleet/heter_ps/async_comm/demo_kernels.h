#pragma once

void GenerateRandomIds(int* output_ids, int output_count, int max_id, int random_seed, cudaStream_t stream);

void GetNextRandomWalkId(const int *input_indices, int node_count, int *output_indices, cudaStream_t stream);

void CheckNextRandomWalkId(const int *input_indices, int node_count, const int *result_indices, cudaStream_t stream);

void GetNodeEmbedding(const int *input_indices,
                      int node_count,
                      int embedding_dim,
                      float *output_embedding,
                      cudaStream_t stream);

void CheckNodeEmbedding(const int *input_indices,
                        int node_count,
                        int embedding_dim,
                        const float *result_embedding,
                        cudaStream_t stream);

void GetNodeGradients(const int *input_indices,
                      int node_count,
                      int embedding_dim,
                      float *output_gradients,
                      cudaStream_t stream);

void CheckNodeGradients(const int *input_indices,
                        int node_count,
                        int embedding_dim,
                        const float *result_gradients,
                        cudaStream_t stream);
