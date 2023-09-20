#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#include <string>

#include <infiniband/verbs.h>

#include "check_macros.h"
#include "ib_utils.h"
#include "net_utils.h"

static struct ProcessInfo {
  bool is_server = false;
  std::string server_addr;
  int server_port = 12123;
} g_process_info;

static std::string g_ib_device_name = "mlx5_0";
static int g_ib_port = 1;
static int g_issue_gpu_id = -1;
static int g_memory_gpu_id = -1;
static int repeat_count = 100;

void PringUsage(const char* program_name) {
  printf("Usage: %s -c/-s [-n IB_name] [-p IB_port] [-g issue_GPU] [-m memory_GPU] [-r repeat_count] server_name server_port\n",
         program_name);
}

void ParseCommandLineArgments(int argc, char** argv) {
  if (argc < 4) {
    PringUsage(argv[0]);
    exit(-1);
  }
  bool has_client_flag = false;
  bool has_server_flag = false;
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "-c") {
      has_client_flag = true;
    } else if (std::string(argv[i]) == "-s") {
      has_server_flag = true;
    } else if (std::string(argv[i]) == "-n") {
      g_ib_device_name = std::string(argv[++i]);
    } else if (std::string(argv[i]) == "-p") {
      g_ib_port = std::stoi(argv[++i]);
    } else if (std::string(argv[i]) == "-g") {
      g_issue_gpu_id = std::stoi(argv[++i]);
    } else if (std::string(argv[i]) == "-m") {
      g_memory_gpu_id = std::stoi(argv[++i]);
    } else if (std::string(argv[i]) == "-r") {
      repeat_count = std::stoi(argv[++i]);
    } else {
        if (i != argc - 2) {
          PringUsage(argv[0]);
          exit(-1);
        }
        g_process_info.server_addr = argv[i];
        g_process_info.server_port = std::stoi(argv[++i]);
    }
  }
  if (g_issue_gpu_id == -1 && g_memory_gpu_id == -1) {
    g_issue_gpu_id = 0;
    g_memory_gpu_id = 0;
  }
  if (g_issue_gpu_id == -1) {
    g_issue_gpu_id = g_memory_gpu_id;
  }
  if (g_memory_gpu_id == -1) {
    g_memory_gpu_id = g_issue_gpu_id;
  }
  if (has_client_flag == has_server_flag) {
    printf("Should specified either -c or -s, and not both.\n");
    PringUsage(argv[0]);
    exit(-1);
  }
  g_process_info.is_server = has_server_flag;
}

static int g_cq_count = 64;

static int g_gpu_count = 0;

static IbLocalContext g_ib_local_context{};

static struct ibv_qp *g_qp = nullptr;
static struct ibv_mr *g_mr = nullptr;

static const size_t kMemorySize = 1 * 1024LL * 1024LL;

int* InitMemory() {
  if (g_issue_gpu_id != g_memory_gpu_id) {
    int can_access_peer = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, g_issue_gpu_id, g_memory_gpu_id));
    BOOL_CHECK(can_access_peer == 1);
  }
  int* data_ptr = nullptr;
  CUDA_CHECK(cudaSetDevice(g_memory_gpu_id));
  BOOL_CHECK(kMemorySize % sizeof(int) == 0);
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&data_ptr), kMemorySize));
  CUDA_CHECK(cudaMemset(data_ptr, 0, kMemorySize));
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaSetDevice(g_issue_gpu_id));
  if (g_issue_gpu_id != g_memory_gpu_id) {
    CUDA_CHECK(cudaDeviceEnablePeerAccess(g_memory_gpu_id, 0));
  }
  return data_ptr;
}

void DeallocateMemory(int* data) {
  CUDA_CHECK(cudaSetDevice(g_memory_gpu_id));
  CUDA_CHECK(cudaFree(data));
  CUDA_CHECK(cudaSetDevice(g_issue_gpu_id));
}

int server_client_fd = -1;

void SetupConnection() {
  if (g_process_info.is_server) {
    int server_listen_fd = CreateServerListenFd(g_process_info.server_port);
    ServerListen(server_listen_fd);
    sockaddr_in client_addr;
    socklen_t client_addr_len = sizeof(client_addr);
    server_client_fd = ServerAccept(server_listen_fd, &client_addr, &client_addr_len);
    CALL_CHECK(close(server_listen_fd));
  } else {
    server_client_fd = CreateClientFd(g_process_info.server_addr, g_process_info.server_port);
  }
  BOOL_CHECK(server_client_fd >= 0);
}

static struct IbPeerInfo local_info, remote_info;

void FillAndExchangeIbPeerInfo() {
  FillIbPeerInfo(&local_info, g_ib_port, &g_ib_local_context.port_attr, g_qp, &g_ib_local_context);
  SingleSend(server_client_fd, &local_info, sizeof(local_info));
  SingleRecv(server_client_fd, &remote_info, sizeof(remote_info));
}

static struct IbMemInfo local_mr, remote_mr;

void FillAndExchangeIbMemInfo() {
  FillIbMemInfo(&local_mr, g_mr);
  SingleSend(server_client_fd, &local_mr, sizeof(local_mr));
  SingleRecv(server_client_fd, &remote_mr, sizeof(remote_mr));
}

void DoPerfTest(int* data_ptr, bool read_test, bool imm = false) {
  ibv_cq* cq = g_ib_local_context.default_cq;
  struct timeval tv_start, tv_end;
  printf("All in DoPerfTest.\n");
  if (g_process_info.is_server) {
    struct ibv_recv_wr rwr, *bad_wr;
    memset(&rwr, 0, sizeof(ibv_recv_wr));
    printf("Server in DoPerfTest.\n");
    if (read_test || !imm) return;
    printf("Server before for.\n");
    for (int r = 0; r < repeat_count; r++) {
      printf("Server waiting for request %d\n", r);
      rwr.wr_id = 0;
      struct ibv_sge sg_entry;
      sg_entry.lkey = g_mr->lkey;
      sg_entry.addr = (uint64_t) data_ptr;
      sg_entry.length = kMemorySize;
      rwr.sg_list = &sg_entry;
      rwr.num_sge = 1;
      rwr.next = nullptr;
      CALL_CHECK(ibv_post_recv(g_qp, &rwr, &bad_wr));

      const int kIBVWcEntryCount = 2;
      struct ibv_wc wcs[kIBVWcEntryCount];
      memset(&wcs[0], 0, sizeof(struct ibv_wc) * kIBVWcEntryCount);
      int wr_done = 0;
      while (wr_done == 0) {
        wr_done = ibv_poll_cq(cq, kIBVWcEntryCount, wcs);
        BOOL_CHECK(wr_done >= 0);
      }
      if (wcs[0].status != IBV_WC_SUCCESS) {
        printf("wcs[0].status=%d\n", (int)wcs[0].status);
      }
      BOOL_CHECK(wcs[0].status == IBV_WC_SUCCESS);
      printf("Server got request %d\n", r);
    }
    return;
  }
  struct ibv_send_wr swr, *bad_wr;
  memset(&swr, 0, sizeof(ibv_send_wr));
  struct ibv_sge sg_entry;
  sg_entry.lkey = g_mr->lkey;
  sg_entry.addr = (uint64_t) data_ptr;
  sg_entry.length = kMemorySize;
  swr.wr_id = 0;
  swr.next = nullptr;
  swr.num_sge = 1;
  swr.sg_list = &sg_entry;
  if (read_test) {
    swr.opcode = IBV_WR_RDMA_READ;
  } else {
    if (imm) {
      swr.imm_data = 0x1111;
      swr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    } else {
      swr.opcode = IBV_WR_RDMA_WRITE;
    }
  }
  swr.wr.rdma.remote_addr = (uint64_t) (remote_mr.addr);
  swr.wr.rdma.rkey = remote_mr.rkey;
  gettimeofday(&tv_start, nullptr);
  for (int r = 0; r < repeat_count; r++) {
    CALL_CHECK(ibv_post_send(g_qp, &swr, &bad_wr));
    printf("request %d send.\n", r);
    const int kIBVWcEntryCount = 2;
    struct ibv_wc wcs[kIBVWcEntryCount];
    memset(&wcs[0], 0, sizeof(struct ibv_wc) * kIBVWcEntryCount);
    int wr_done = 0;
    while (wr_done == 0) {
      wr_done = ibv_poll_cq(cq, kIBVWcEntryCount, wcs);
      BOOL_CHECK(wr_done >= 0);
    }
    if (wcs[0].status != IBV_WC_SUCCESS) {
      printf("wcs[0].status=%d\n", (int)wcs[0].status);
    }
    BOOL_CHECK(wcs[0].status == IBV_WC_SUCCESS);
    printf("request %d done.\n", r);
  }
  gettimeofday(&tv_end, nullptr);
  int64_t elapsed_us = 0;
  elapsed_us = (tv_end.tv_sec - tv_start.tv_sec) * 1000LL * 1000LL + tv_end.tv_usec - tv_start.tv_usec;
  double bandwidth_gbps = (double) kMemorySize * 8 * repeat_count / elapsed_us / 1000.0f;
  printf("Action %s, bandwidth=%7.2f\n", read_test ? "RDMA read" : "RDMA write",  bandwidth_gbps);
}

void Barrier() {
  int value_send = 0;
  int value_recv = 0;
  SingleSend(server_client_fd, &value_send, sizeof(value_send));
  SingleRecv(server_client_fd, &value_recv, sizeof(value_recv));
}

int main(int argc, char** argv) {
  ParseCommandLineArgments(argc, argv);
  IbInit();

  CUDA_CHECK(cudaGetDeviceCount(&g_gpu_count));
  BOOL_CHECK(g_gpu_count > g_issue_gpu_id);
  BOOL_CHECK(g_gpu_count > g_memory_gpu_id);

  PrintAllIbDevices();
  CreateLocalContext(&g_ib_local_context, g_ib_device_name, g_ib_port);
  g_qp = CreateIbvRcQp(g_ib_local_context.pd, g_ib_local_context.default_cq, g_ib_local_context.default_cq, 1);
  BOOL_CHECK(g_qp != nullptr);
  QpInit(g_qp, g_ib_port);
  int* data_ptr = InitMemory();
  g_mr = RegisterIbMr(g_ib_local_context.pd, data_ptr, kMemorySize);

  SetupConnection();
  printf("Connected.\n");
  FillAndExchangeIbPeerInfo();
  printf("Peer Info done.\n");
  FillAndExchangeIbMemInfo();
  printf("Memory Info done.\n");
  QpRtr(g_qp, &local_info, &remote_info);
  printf("RTR done.\n");
  QpRts(g_qp);
  printf("RTS done.\n");

  Barrier();

  DoPerfTest(data_ptr, true);
  DoPerfTest(data_ptr, false);
  DoPerfTest(data_ptr, false, true);

  Barrier();

  DeRegIbMr(g_mr);
  DeallocateMemory(data_ptr);
  CALL_CHECK(ibv_destroy_qp(g_qp));
  DestroyLocalContext(&g_ib_local_context);
  IbDeInit();
  return 0;
}