#include "ib_utils.h"

#include <string.h>
#include <netinet/in.h>

#include "check_macros.h"

static int g_num_ib_devices = 0;
static struct ibv_device **g_ib_devices_list = nullptr;

struct IbEnvs {
  int gid_index = 0;
  int service_level = 0;
  int traffic_class = 0;
};

static IbEnvs g_ib_envs{};

static int GetIntEnv(const char* env_name, const char* second_env_name, int default_value) {
  char* str_env = nullptr;
  int value;
  str_env = getenv(env_name);
  if (str_env && strlen(str_env) > 0) {
    errno = 0;
    value = strtol(str_env, nullptr, 0);
    if (errno) {
      printf("Invalid value %s for %s.", str_env, env_name);
    } else {
      printf("%s set by environment to %d", env_name, value);
      return value;
    }
  }
  str_env = getenv(second_env_name);
  if (str_env && strlen(str_env) > 0) {
    errno = 0;
    value = strtol(str_env, nullptr, 0);
    if (errno) {
      printf("Invalid value %s for %s.", str_env, second_env_name);
    } else {
      printf("%s set by environment to %d", second_env_name, value);
      return value;
    }
  }

  return default_value;
}

static void GetIbEnvsFromEnv() {
  g_ib_envs.gid_index = GetIntEnv("AC_IB_GID_INDEX", "NCCL_IB_GID_INDEX", 0);
  g_ib_envs.service_level = GetIntEnv("AC_IB_SL", "NCCL_IB_SL", 0);
  g_ib_envs.traffic_class = GetIntEnv("AC_IB_TC", "NCCL_IB_TC", 0);
}

void IbInit() {
  CALL_CHECK(ibv_fork_init());
  g_ib_devices_list = ibv_get_device_list(&g_num_ib_devices);
  BOOL_CHECK(g_num_ib_devices > 0);
  GetIbEnvsFromEnv();
}

void IbDeInit() {
  ibv_free_device_list(g_ib_devices_list);
}

void PrintIbPeerInfo(const char* prefix, int rank, int id, IbPeerInfo* ib_peer_info) {
  printf("Rank=%d, %s[%d] IbPeerInfo:\n"
         "    lid=%u, qpn=%u, spn=%lu, iid=%lu, ib_port=%u\n",
         rank, prefix, id,
         ib_peer_info->lid, ib_peer_info->qpn, ib_peer_info->spn, ib_peer_info->iid, (uint32_t) ib_peer_info->ib_port);
}

void PrintIbMemInfo(const char* prefix, int rank, int id, IbMemInfo* ib_mem_info) {
  printf("Rank=%d, %s[%d] IbMemInfo:\n"
         "    rkey=%u, addr=%lu, length=%lu\n",
         rank, prefix, id, ib_mem_info->rkey, ib_mem_info->addr, ib_mem_info->length);
}

void CreateLocalContext(IbLocalContext* ib_local_context, const std::string& dev_name, int port) {
  BOOL_CHECK(port >= 1);
  ib_local_context->port_id = port;
  BOOL_CHECK(ib_local_context != nullptr);
  ib_local_context->device = SelectIbDeviceByName(dev_name);
  BOOL_CHECK(ib_local_context->device != nullptr);
  ib_local_context->context = ibv_open_device(ib_local_context->device);
  BOOL_CHECK(ib_local_context->context != nullptr);
  CALL_CHECK(ibv_query_device(ib_local_context->context, &ib_local_context->device_attr));
  printf("num_comp_vectors=%d\n", ib_local_context->context->num_comp_vectors);
  PrintDeviceInfo(&ib_local_context->device_attr);
  int port_count = ib_local_context->device_attr.phys_port_cnt;
  BOOL_CHECK(port_count >= port); // IB port starts from 1.
  CALL_CHECK(ibv_query_port(ib_local_context->context, port, &ib_local_context->port_attr));
  //PrintPortInfo(&ib_local_context->port_attr);
  ib_local_context->pd = ibv_alloc_pd(ib_local_context->context);
  BOOL_CHECK(ib_local_context->pd != nullptr);
  ib_local_context->default_cq = ibv_create_cq(ib_local_context->context, 1024, nullptr, nullptr, 0);
  BOOL_CHECK(ib_local_context->default_cq != nullptr);
}

void DestroyLocalContext(IbLocalContext* ib_local_context) {
  CALL_CHECK(ibv_destroy_cq(ib_local_context->default_cq));
  CALL_CHECK(ibv_dealloc_pd(ib_local_context->pd));
  CALL_CHECK(ibv_close_device(ib_local_context->context));
  ib_local_context->clear();
}

void FillIbPeerInfo(IbPeerInfo *peer_info, int port_id, struct ibv_port_attr *port_attr, struct ibv_qp *qp, IbLocalContext* ib_local_context) {
  peer_info->lid = port_attr->lid;
  peer_info->qpn = qp->qp_num;
  peer_info->ib_port = port_id;
  peer_info->mtu = port_attr->active_mtu;
  peer_info->link_layer = port_attr->link_layer;
  BOOL_CHECK(port_attr->link_layer == IBV_LINK_LAYER_INFINIBAND || port_attr->link_layer == IBV_LINK_LAYER_ETHERNET);
  if (port_attr->link_layer == IBV_LINK_LAYER_INFINIBAND) {
    peer_info->spn = 0;
    peer_info->iid = 0;
  } else {
    ibv_gid local_gid;
    CALL_CHECK(ibv_query_gid(ib_local_context->context, ib_local_context->port_id, g_ib_envs.gid_index, &local_gid));
    peer_info->spn = local_gid.global.subnet_prefix;
    peer_info->iid = local_gid.global.interface_id;
  }
}

void FillIbMemInfo(IbMemInfo *mem_info, struct ibv_mr *mr) {
  mem_info->rkey = mr->rkey;
  mem_info->addr = reinterpret_cast<uint64_t>(mr->addr);
  mem_info->length = mr->length;
}

void PrintAllIbDevices() {
  struct ibv_device **ib_device_list = g_ib_devices_list;
  int num_ib_devices = g_num_ib_devices;
  static const char
      *kIbDeviceNodeTypeArray[] = {"?", "CA", "SWITCH", "ROUTER", "RNIC", "USNIC", "USNIC_UDP", "UNSPECIFIED"};
  static const char *kIbDeviceTransportTypeArray[] = {"IB", "IWARP", "USNIC", "USNIC_UDP", "UNSPECIFIED"};
  static const char *kIbUnknownNodeOrTransport = "UNKNOWN";
  for (int i = 0; i < num_ib_devices; i++) {
    struct ibv_device *ib_dev = ib_device_list[i];
    const char *node_type_str =
        (ib_dev->node_type == IBV_NODE_UNKNOWN) ? kIbUnknownNodeOrTransport : kIbDeviceNodeTypeArray[ib_dev->node_type];
    const char *transport_type_str =
        (ib_dev->transport_type == IBV_TRANSPORT_UNKNOWN) ? kIbUnknownNodeOrTransport
                                                          : kIbDeviceTransportTypeArray[ib_dev->transport_type];
    printf("Ib device index=%d, name=%s, dev_name=%s, node_type=%s, transport_type=%s\n",
           i, ib_dev->name, ib_dev->dev_name, node_type_str, transport_type_str);
  }
}

void PrintDeviceInfo(struct ibv_device_attr *attr) {
  printf("%10c%20s : %lx\n", ' ', "max_mr_size", attr->max_mr_size);
  printf("%10c%20s : %lx\n", ' ', "page_size_cap", attr->page_size_cap);
  printf("%10c%20s : %d \n", ' ', "max_qp", attr->max_qp);
  printf("%10c%20s : %d \n", ' ', "max_qp_wr", attr->max_qp_wr);
  printf("%10c%20s : %x \n", ' ', "device_cap_flags", attr->device_cap_flags);
  printf("%10c%20s : %d \n", ' ', "max_sge", attr->max_sge);
  printf("%10c%20s : %d \n", ' ', "max_sge_rd", attr->max_sge_rd);
  printf("%10c%20s : %d \n", ' ', "max_cq", attr->max_cq);
  printf("%10c%20s : %d \n", ' ', "max_cqe", attr->max_cqe);
  printf("%10c%20s : %d \n", ' ', "max_mr", attr->max_mr);
  printf("%10c%20s : %d \n", ' ', "max_pd", attr->max_pd);
  printf("%10c%20s : %d \n", ' ', "max_ah", attr->max_ah);
  printf("%10c%20s : %d \n", ' ', "max_srq", attr->max_srq);
  printf("%10c%20s : %d \n", ' ', "max_pkeys", (int) attr->max_pkeys);
  printf("%10c%20s : %d \n", ' ', "phys_port_cnt", (int) attr->phys_port_cnt);
}

void PrintPortInfo(struct ibv_port_attr *attr) {
  // https://www.rdmamojo.com/2012/07/21/ibv_query_port/
  // https://en.wikipedia.org/wiki/InfiniBand
  // https://github.com/NVIDIA/nccl/blob/ea38312273e5b9a19a224c9ff4c10b7fcf441eaf/src/transport/net_ib.cc
  static const char *kPortStatusArray[] = {"Nop", "Down", "Init", "Armed", "Active", "ActiveDefer"};
  static const int kPortMTU[] = {0, 256, 512, 1024, 2048, 4096};
  static const char *kPortPhyStatusArray[] =
      {"NotSpecified", "Sleep", "Polling", "Disabled", "PortConfigurationTraining", "LinkUp", "LinkErrorRecovery",
       "Phytest"};
  static const float kPortSpeedArray[] = {2.5, 5, 10, 10, 14, 25, 50, 100, -1, -1, -1, -1, -1};
  static const char
      *kPortSpeedNameArray[] = {"SDR", "DDR", "QDR", "FDR10", "FDR", "EDR", "HDR", "NDR", "XDR", "?GDR", "???", "???"};
  static const int kPortWidthArray[] = {1, 4, 8, 12, -1, -1, -1, -1};
  static const char *kPortLinkLayerArray[] = {"Unspecified", "InfiniBand", "Ethernet", "???", "???", "???"};

  printf("%20c%20s : %s\n", ' ', "State", kPortStatusArray[attr->state]);
  printf("%20c%20s : %d\n", ' ', "MaxMTU", kPortMTU[attr->max_mtu]);
  printf("%20c%20s : %d\n", ' ', "ActiveMTU", kPortMTU[attr->active_mtu]);
  printf("%20c%20s : %d\n", ' ', "GidTableLen", attr->gid_tbl_len);
  printf("%20c%20s : %s\n", ' ', "Physical state", kPortPhyStatusArray[attr->phys_state]);
  printf("%20c%20s : %.0f Gbps (%s)\n", ' ', "Rate",
         kPortSpeedArray[__builtin_ctz(attr->active_speed)] * kPortWidthArray[__builtin_ctz(attr->active_width)],
         kPortSpeedNameArray[__builtin_ctz(attr->active_speed)]);
  printf("%20c%20s : %s\n", ' ', "Link layer", kPortLinkLayerArray[attr->link_layer]);
}

struct ibv_device *SelectIbDeviceByName(const std::string &name) {
  struct ibv_device **ib_device_list = g_ib_devices_list;
  int num_ib_devices = g_num_ib_devices;
  for (int i = 0; i < num_ib_devices; i++) {
    if (strcmp(ib_device_list[i]->name, name.c_str()) == 0) {
      return ib_device_list[i];
    }
  }
  return nullptr;
}

struct ibv_mr* TryRegisterIbMr(struct ibv_pd *pd, void* data, size_t size) {
  struct ibv_mr *ib_mr = ibv_reg_mr(pd,
                                    data,
                                    size,
                                    IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE);
  return ib_mr;
}
struct ibv_mr* RegisterIbMr(struct ibv_pd *pd, void* data, size_t size) {
  auto* ib_mr = TryRegisterIbMr(pd, data, size);
  BOOL_CHECK(ib_mr != nullptr);
  return ib_mr;
}

void DeRegIbMr(struct ibv_mr* mr) {
  CALL_CHECK(ibv_dereg_mr(mr));
}

struct ibv_qp *CreateIbvRcQp(struct ibv_pd *pd, struct ibv_cq *send_cq, struct ibv_cq *recv_cq, int sig_all) {
  struct ibv_qp_init_attr qpInitAttr;
  memset(&qpInitAttr, 0, sizeof(struct ibv_qp_init_attr));
  qpInitAttr.send_cq = send_cq;
  qpInitAttr.recv_cq = recv_cq;
  qpInitAttr.qp_type = IBV_QPT_RC;
  qpInitAttr.sq_sig_all = sig_all;
  qpInitAttr.cap.max_send_wr = 1024;
  qpInitAttr.cap.max_recv_wr = 1024;
  qpInitAttr.cap.max_send_sge = 1;
  qpInitAttr.cap.max_recv_sge = 1;
  qpInitAttr.cap.max_inline_data = 0;
  return ibv_create_qp(pd, &qpInitAttr);
}

void DestroyIbvRcQp(struct ibv_qp* qp) {
  CALL_CHECK(ibv_destroy_qp(qp));
}

void QpInit(ibv_qp *qp, int port_id) {
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_INIT;
  qpAttr.pkey_index = 0;
  qpAttr.port_num = port_id;
  qpAttr.qp_access_flags =
      IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC | IBV_ACCESS_LOCAL_WRITE;
  CALL_CHECK(ibv_modify_qp(qp, &qpAttr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS));
}

void QpRtr(ibv_qp *qp, IbPeerInfo *local_info, IbPeerInfo *remote_info) {
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_RTR;
  qpAttr.path_mtu = std::min(local_info->mtu, remote_info->mtu);
  qpAttr.dest_qp_num = remote_info->qpn;
  qpAttr.rq_psn = 0;
  qpAttr.max_dest_rd_atomic = 1;
  qpAttr.min_rnr_timer = 12;
  BOOL_CHECK(local_info->link_layer == remote_info->link_layer);
  if (local_info->link_layer == IBV_LINK_LAYER_ETHERNET) {
    qpAttr.ah_attr.is_global = 1;
    qpAttr.ah_attr.grh.dgid.global.subnet_prefix = remote_info->spn;
    qpAttr.ah_attr.grh.dgid.global.interface_id = remote_info->iid;
    qpAttr.ah_attr.grh.flow_label = 0;
    qpAttr.ah_attr.grh.sgid_index = g_ib_envs.gid_index;  // IB Gid Index
    qpAttr.ah_attr.grh.hop_limit = 255;
    qpAttr.ah_attr.grh.traffic_class = g_ib_envs.traffic_class; // IB Traffic Class
  } else {
    qpAttr.ah_attr.is_global = 0;
    qpAttr.ah_attr.dlid = remote_info->lid;
  }

  qpAttr.ah_attr.sl = g_ib_envs.service_level;  // service level
  qpAttr.ah_attr.src_path_bits = 0;
  qpAttr.ah_attr.port_num = remote_info->ib_port;
  CALL_CHECK(ibv_modify_qp(qp,
                           &qpAttr,
                           IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN
                               | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER));
}

void QpRts(struct ibv_qp *qp) {
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_RTS;
  qpAttr.timeout = 22;
  qpAttr.retry_cnt = 7;
  qpAttr.rnr_retry = 7;
  qpAttr.sq_psn = 0;
  qpAttr.max_rd_atomic = 1;
  CALL_CHECK(ibv_modify_qp(qp,
                           &qpAttr,
                           IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN
                               | IBV_QP_MAX_QP_RD_ATOMIC));
}

void WaitDone(struct ibv_cq *cq) {
  int wr_done = 0;
  const int kIBVWcEntryCount = 2;
  struct ibv_wc wcs[kIBVWcEntryCount];
  while (wr_done == 0) {
    wr_done = ibv_poll_cq(cq, kIBVWcEntryCount, wcs);
    BOOL_CHECK(wr_done >= 0);
  }
  BOOL_CHECK(wr_done == 1);
  BOOL_CHECK(wcs[0].status == IBV_WC_SUCCESS);
}

void IbRecv(void *data, size_t length, struct ibv_qp *qp, struct ibv_mr *mr, uint64_t wr_id) {
  struct ibv_recv_wr rwr, *bad_wr;
  bad_wr = nullptr;
  memset(&rwr, 0, sizeof(ibv_recv_wr));
  struct ibv_sge sg_entry;
  sg_entry.lkey = mr->lkey;
  sg_entry.addr = (uint64_t) data;
  sg_entry.length = length;
  rwr.wr_id = wr_id;
  rwr.next = nullptr;
  rwr.num_sge = 1;
  rwr.sg_list = &sg_entry;
  CALL_CHECK(ibv_post_recv(qp, &rwr, &bad_wr));
}

void IbSend(void *data, size_t length, struct ibv_qp *qp, struct ibv_mr *mr, uint64_t wr_id) {
  struct ibv_send_wr swr, *bad_wr;
  bad_wr = nullptr;
  memset(&swr, 0, sizeof(ibv_send_wr));
  struct ibv_sge sg_entry;
  sg_entry.lkey = mr->lkey;
  sg_entry.addr = (uint64_t) data;
  sg_entry.length = length;
  swr.wr_id = wr_id;
  swr.next = nullptr;
  swr.num_sge = 1;
  swr.sg_list = &sg_entry;
  swr.opcode = IBV_WR_SEND;
  CALL_CHECK(ibv_post_send(qp, &swr, &bad_wr));
}

void IbvRdmaRead(void *data,
                 size_t length,
                 struct ibv_qp *qp,
                 uint32_t lkey,
                 IbMemInfo *remote_mr,
                 uint64_t wr_id,
                 bool need_to_be_signaled) {
  struct ibv_send_wr swr, *bad_wr;
  bad_wr = nullptr;
  memset(&swr, 0, sizeof(ibv_send_wr));
  struct ibv_sge sg_entry;
  sg_entry.lkey = lkey;
  sg_entry.addr = (uint64_t) data;
  sg_entry.length = length;
  swr.wr_id = wr_id;
  swr.next = nullptr;
  swr.num_sge = 1;
  swr.sg_list = &sg_entry;
  swr.opcode = IBV_WR_RDMA_READ;
  swr.send_flags = need_to_be_signaled ? IBV_SEND_SIGNALED : 0;
  swr.wr.rdma.remote_addr = (uint64_t) (remote_mr->addr);
  swr.wr.rdma.rkey = remote_mr->rkey;
  CALL_CHECK(ibv_post_send(qp, &swr, &bad_wr));
}

void IbvRdmaWrite(void *data,
                  size_t length,
                  struct ibv_qp *qp,
                  uint32_t lkey,
                  IbMemInfo *remote_mr,
                  uint64_t wr_id,
                  bool need_to_be_signaled) {
  struct ibv_send_wr swr, *bad_wr;
  bad_wr = nullptr;
  memset(&swr, 0, sizeof(ibv_send_wr));
  struct ibv_sge sg_entry;
  sg_entry.lkey = lkey;
  sg_entry.addr = (uint64_t) data;
  sg_entry.length = length;
  swr.wr_id = wr_id;
  swr.next = nullptr;
  swr.num_sge = 1;
  swr.sg_list = &sg_entry;
  swr.opcode = IBV_WR_RDMA_WRITE;
  swr.send_flags = need_to_be_signaled ? IBV_SEND_SIGNALED : 0;
  swr.wr.rdma.remote_addr = (uint64_t) (remote_mr->addr);
  swr.wr.rdma.rkey = remote_mr->rkey;
  CALL_CHECK(ibv_post_send(qp, &swr, &bad_wr));
}

void IbvRdmaWriteImm(void *data,
                     size_t length,
                     struct ibv_qp *qp,
                     uint32_t lkey,
                     IbMemInfo *remote_mr,
                     uint32_t imm,
                     uint64_t wr_id,
                     bool need_to_be_signaled) {
  struct ibv_send_wr swr, *bad_wr;
  bad_wr = nullptr;
  memset(&swr, 0, sizeof(ibv_send_wr));
  struct ibv_sge sg_entry;
  sg_entry.lkey = lkey;
  sg_entry.addr = (uint64_t) data;
  sg_entry.length = length;
  swr.wr_id = wr_id;
  swr.next = nullptr;
  swr.num_sge = 1;
  swr.sg_list = &sg_entry;
  swr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  swr.send_flags = need_to_be_signaled ? IBV_SEND_SIGNALED : 0;
  swr.imm_data = htonl(imm);
  swr.wr.rdma.remote_addr = (uint64_t) (remote_mr->addr);
  swr.wr.rdma.rkey = remote_mr->rkey;
  CALL_CHECK(ibv_post_send(qp, &swr, &bad_wr));
}

uint32_t GetImmDataFromWc(ibv_wc* wc) {
  return ntohl(wc->imm_data);
}


