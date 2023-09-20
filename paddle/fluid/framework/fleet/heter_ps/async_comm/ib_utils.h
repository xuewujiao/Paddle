#pragma once

#include <infiniband/verbs.h>

#include <string>

void IbInit();

void IbDeInit();

struct IbPeerInfo {
  uint32_t lid;
  uint32_t qpn;
  uint64_t spn;
  uint64_t iid;
  uint8_t ib_port;
  uint8_t link_layer;
  enum ibv_mtu mtu = IBV_MTU_4096;
};

void PrintIbPeerInfo(const char* prefix, int rank, int id, IbPeerInfo* ib_peer_info);

struct IbMemInfo {
  uint32_t rkey;
  uint64_t addr;
  size_t length;
};

void PrintIbMemInfo(const char* prefix, int rank, int id, IbMemInfo* ib_mem_info);

struct IbLocalContext{
  ibv_device* device = nullptr;
  ibv_context* context = nullptr;
  ibv_device_attr device_attr{};
  int port_id = 0;
  ibv_port_attr port_attr{};
  ibv_pd* pd = nullptr;
  ibv_cq* default_cq = nullptr;
  void clear() {
    device = nullptr;
    context = nullptr;
    memset(&device_attr, 0, sizeof(ibv_device_attr));
    port_id = 0;
    memset(&port_attr, 0, sizeof(ibv_port_attr));
    pd = nullptr;
    default_cq = nullptr;   
  }
};

void CreateLocalContext(IbLocalContext* ib_local_context, const std::string& dev_name, int port);

void DestroyLocalContext(IbLocalContext* ib_local_context);

void FillIbPeerInfo(IbPeerInfo *peer_info, int port_id, struct ibv_port_attr *port_attr, struct ibv_qp *qp, IbLocalContext* ib_local_context);

void FillIbMemInfo(IbMemInfo *mem_info, struct ibv_mr *mr);

void PrintAllIbDevices();

void PrintDeviceInfo(struct ibv_device_attr *attr);

void PrintPortInfo(struct ibv_port_attr *attr);

struct ibv_device *SelectIbDeviceByName(const std::string &name);

struct ibv_mr* TryRegisterIbMr(struct ibv_pd *pd, void* data, size_t size);

struct ibv_mr* RegisterIbMr(struct ibv_pd *pd, void* data, size_t size);

void DeRegIbMr(struct ibv_mr* mr);

struct ibv_qp *CreateIbvRcQp(struct ibv_pd *pd, struct ibv_cq *send_cq, struct ibv_cq *recv_cq, int sig_all);

void DestroyIbvRcQp(struct ibv_qp* qp);

void QpInit(ibv_qp *qp, int port_id);

void QpRtr(ibv_qp *qp, IbPeerInfo *local_info, IbPeerInfo *remote_info);

void QpRts(struct ibv_qp *qp);

void WaitDone(struct ibv_cq *cq);

void IbRecv(void *data, size_t length, struct ibv_qp *qp, struct ibv_mr *mr, uint64_t wr_id);

void IbSend(void *data, size_t length, struct ibv_qp *qp, struct ibv_mr *mr, uint64_t wr_id);

void IbvRdmaRead(void *data,
                 size_t length,
                 struct ibv_qp *qp,
                 uint32_t lkey,
                 IbMemInfo *remote_mr,
                 uint64_t wr_id,
                 bool need_to_be_signaled);

void IbvRdmaWrite(void *data,
                  size_t length,
                  struct ibv_qp *qp,
                  uint32_t lkey,
                  IbMemInfo *remote_mr,
                  uint64_t wr_id,
                  bool need_to_be_signaled);

void IbvRdmaWriteImm(void *data,
                     size_t length,
                     struct ibv_qp *qp,
                     uint32_t lkey,
                     IbMemInfo *remote_mr,
                     uint32_t imm,
                     uint64_t wr_id,
                     bool need_to_be_signaled);

uint32_t GetImmDataFromWc(ibv_wc* wc);