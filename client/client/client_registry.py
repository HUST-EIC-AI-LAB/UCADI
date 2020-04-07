from socket import *
import struct
import json
import os
import sys
import json
import time
import logging
from socket import SOL_SOCKET,SO_REUSEADDR,SO_SNDBUF,SO_RCVBUF

# get configurations of client side
with open('./config/config_client.json', 'r') as j:
    client_cfg = json.load(j)

if __name__ == "__main__":

    FORMAT = "%(asctime)s | %(message)s"
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger("client-registry")
    logger.setLevel(level=logging.DEBUG)

    username = client_cfg['username']
    password = client_cfg['password']
    aimServerIP = client_cfg['server_ip']
    aimServerPort = client_cfg['register_server_port']
    buffsize = client_cfg['buffsize']


    dirc = {
        'username': username,
        'password': password,
    }

    # connecting
    tcp_client = socket(AF_INET, SOCK_STREAM)
    ip_port = ((aimServerIP, aimServerPort))
    tcp_client.connect(ip_port)

    head_info = json.dumps(dirc)  # 将字典转换成字符串
    head_info_len = struct.pack('i', len(head_info))  # 将字符串的长度打包

    tcp_client.send(head_info_len)  # 发送head_info的长度
    tcp_client.send(head_info.encode('utf-8'))

    logger.info('registry successfully')
    while True:
        if(len(tcp_client.recv(1024)) <= 0):
            break
    tcp_client.close()
