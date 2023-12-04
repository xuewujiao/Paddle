#include "net_utils.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <iostream>
#include <string>

#include "check_macros.h"

static void ResolveHostName(sockaddr_in* saddr, const std::string& host_name, int port) {
  addrinfo hints = { 0, AF_INET, SOCK_STREAM, IPPROTO_TCP, 0, nullptr, nullptr, nullptr};
  addrinfo *res;
  char port_buf[16];
  snprintf(port_buf, 16, "%d", port);
  int ret = getaddrinfo(host_name.c_str(), port_buf, &hints, &res);
  if (ret != 0) {
    printf("Resolve IP for host %s failed.\n", host_name.c_str());
    abort();
  }
  *saddr = *(sockaddr_in*)(res->ai_addr);
}

int CreateServerListenFd(int port) {
  int server_sock = socket(AF_INET, SOCK_STREAM, 0);
  BOOL_CHECK(server_sock >= 0);
  int enable = 1;
  CALL_CHECK(setsockopt(server_sock, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)));

  // Binding
  sockaddr_in server_addr;
  memset(&server_addr, 0, sizeof(sockaddr_in));
  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(port);
  server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
  // int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
  if(setsockopt(server_sock, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) < 0){
    std::cerr << "Error setting socket option." << std::endl;
    abort();
  };
  // CALL_CHECK(bind(server_sock, (sockaddr*)&server_addr, sizeof(server_addr)));
  if(bind(server_sock, (sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
    std::cerr << "Bind port failed, errno=" << strerror(errno) << std::endl;
    // 打印 server_addr 的内容
    std::cerr << "Server Address: "
              << inet_ntoa(server_addr.sin_addr)   // 将 IP 地址转换为点分十进制格式
              << ":" << ntohs(server_addr.sin_port) // 将网络字节顺序的端口号转换为主机字节顺序
              << std::endl;
    abort();
  }

  return server_sock;
}

void ServerListen(int listen_fd, int backlog) {
  CALL_CHECK(listen(listen_fd, backlog));
}

int ServerAccept(int listen_fd, sockaddr_in* client_addr, socklen_t* client_addr_len) {
  int client_sock = accept(listen_fd, (sockaddr*)client_addr, client_addr_len);
  return client_sock;
}

int CreateClientFd(const std::string& server_name, int server_port) {
  int client_sock = socket(AF_INET, SOCK_STREAM, 0);
  BOOL_CHECK(client_sock >= 0);

  sockaddr_in server_addr;
  ResolveHostName(&server_addr, server_name, server_port);

  BOOL_CHECK(server_addr.sin_family == AF_INET);
  BOOL_CHECK(server_addr.sin_port == htons(server_port));
#if 0
  inet_pton(AF_INET, server_name.c_str(), &server_addr.sin_addr);
#endif

  while (connect(client_sock, (sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
    switch (errno) {
      case ECONNREFUSED:
        //std::cerr << "Server may not running, waiting..." << std::endl;
        break;
      case ETIMEDOUT:
        printf("Connecting timeout retrying...\n");
        break;
      case ENETUNREACH:
        printf("Network unreachable, retrying...\n");
        break;
      default:
        printf("unknow error %d, retrying...\n", errno);
        break;
    }
    usleep(500 * 1000);
  }

  return client_sock;
}

void SingleSend(int sock_fd, const void* send_data, size_t send_size) {
  ssize_t bytes_send = send(sock_fd, send_data, send_size, 0);
  if (bytes_send < 0) {
    printf("recv returned %ld, errno=%d %s\n", bytes_send, errno, strerror(errno));
  }
  BOOL_CHECK(bytes_send == (ssize_t)send_size);
}

void SingleRecv(int sock_fd, void* recv_data, size_t recv_size) {
  ssize_t bytes_received = recv(sock_fd, recv_data, recv_size, 0);
  if (bytes_received < 0) {
    printf("recv returned %ld, errno=%d %s\n", bytes_received, errno, strerror(errno));
  }
  BOOL_CHECK(bytes_received == (ssize_t)recv_size);
}