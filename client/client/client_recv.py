import logging
from socket import *
import struct
import json
import os
import sys
import time
from socket import SOL_SOCKET,SO_REUSEADDR,SO_SNDBUF,SO_RCVBUF


def process_bar(logger, precent, width=50):
    use_num = int(precent*width)
    space_num = int(width-use_num)
    precent = precent*100
    #
    # print('[%s%s]%d%%'%(use_num*'#', space_num*' ',precent))
    # print('[%s%s]%d%%'%(use_num*'#', space_num*' ',precent), end='\r')
    print('[%s%s]%d%%'%(use_num*'#', space_num*' ',precent),file=sys.stdout,flush=True, end='\r')


FORMAT = "%(asctime)s | %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger("client-receive")
logger.setLevel(level=logging.INFO)

with open('./config/config_client.json', 'r') as j:
    client_cfg = json.load(j)

aimServerIP = client_cfg['server_ip']
aimServerPort = client_cfg['recv_server_port']

tcp_client = socket(AF_INET, SOCK_STREAM)
ip_port = ((aimServerIP, aimServerPort))
buffsize = client_cfg['buffsize']
tcp_client.connect(ip_port)
logger.info('Waiting for connecting server')

num_file = 0
while True:
    head_struct = tcp_client.recv(4)
    if head_struct:
        logger.info('Connected with server, waiting for data ')
    else:
        logger.info('Connected with server failed.')
        logger.info('Maybe you has been download the model in this iteration.')
        logger.info('Maybe you has upload the updated model in this iteration.')
        tcp_client.close()
        break
    head_len = struct.unpack('i', head_struct)[0]
    data = tcp_client.recv(head_len)

    head_dir = json.loads(data.decode('utf-8'))
    filesize_b = head_dir['filesize_bytes']
    filename = './download/' + head_dir['filename']
    
    
    # receive the real info of file.
    recv_len = 0
    recv_mesg = b''
    old = time.time()
    f = open(filename, 'wb')
    while recv_len < filesize_b:
        percent = recv_len / filesize_b
        #process_bar(logger, percent)
        if filesize_b - recv_len > buffsize:

            recv_mesg = tcp_client.recv(buffsize)
            f.write(recv_mesg)
            recv_len += len(recv_mesg)
        else:
            recv_mesg = tcp_client.recv(filesize_b - recv_len)
            recv_len += len(recv_mesg)
            f.write(recv_mesg)
    #process_bar(logger, recv_len / filesize_b)
    logger.info("File number: " + str(num_file + 1))
    logger.info("File size: " + str(recv_len))
    now = time.time()
    stamp = int(now - old)
    logger.info('All time: %ds' % stamp)
    f.close()
    num_file += 1
    if num_file == 2:
        break
tcp_client.close()
