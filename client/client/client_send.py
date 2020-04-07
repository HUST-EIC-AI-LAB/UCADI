import logging
from socket import *
import struct
import json
import os
import sys
import json
import time
from socket import SOL_SOCKET,SO_REUSEADDR,SO_SNDBUF,SO_RCVBUF

# get configurations of client side
with open('./config/config_client.json', 'r') as j:
    client_cfg = json.load(j)

FORMAT = "%(asctime)s | %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger("client-sending")
logger.setLevel(level=logging.DEBUG)

username = client_cfg['username']
password = client_cfg['password']
aimServerIP = client_cfg['server_ip']
aimServerPort = client_cfg['send_server_port']


file = sys.argv[1]
#file = 'weight.pth'

tcp_client = socket(AF_INET, SOCK_STREAM)
ip_port = ((aimServerIP, aimServerPort))
buffsize = client_cfg['buffsize']
tcp_client.connect(ip_port)
logger.info('Waiting for connecting server')

num_file = 0
filemesg = file.strip()
filesize_bytes = os.path.getsize(filemesg) # 得到文件的大小,字节
filename = file.split("/")[-1]
# filename = filemesg
dirc = {
    'filename': filename,
    'filesize_bytes': filesize_bytes,
    'username': username,
    'password': password,
}

#print('client header :',  dirc)

head_info = json.dumps(dirc)  # 将字典转换成字符串
head_info_len = struct.pack('i', len(head_info)) #  将字符串的长度打包
#   先将报头转换成字符串(json.dumps), 再将字符串的长度打包
#   发送报头长度,发送报头内容,最后放真是内容
#   报头内容包括文件名,文件信息,报头
#   接收时:先接收4个字节的报头长度,
#   将报头长度解压,得到头部信息的大小,在接收头部信息, 反序列化(json.loads)
#   最后接收真实文件
tcp_client.send(head_info_len)  # 发送head_info的长度
tcp_client.send(head_info.encode('utf-8'))

# if(len(tcp_client.recv(1024)) <= 0):
#     print('connection failed, fzw')
#     tcp_client.close()

#   发送真实信息
with open(filemesg, 'rb') as f:
    data = f.read()
    tcp_client.sendall(data)

logger.info('Sending successfully')
while True:
    if(len(tcp_client.recv(client_cfg['buffsize'])) <= 0):
        break
tcp_client.close()
    
