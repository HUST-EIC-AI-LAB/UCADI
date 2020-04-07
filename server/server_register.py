import struct
from socket import *
import json
import os
import multiprocessing
from socket import SOL_SOCKET,SO_REUSEADDR,SO_SNDBUF,SO_RCVBUF

with open('./server_config.json', 'r') as j:
    cfg_server = json.load(j)
buffsize = cfg_server['buffsize']

def echo(conn, addr):
    while True:
        if not conn:
            print('Connection failed')
            break
        head_struct = conn.recv(4)
        if head_struct:
            print('Connected with client, waiting for data ')
        head_len = struct.unpack('i', head_struct)[0] 
        print(head_len)
        # decode the size of head
        data = conn.recv(head_len)
        # receive the info of head(length = head_len), including file size and file info.

        head_dir = json.loads(data.decode('utf-8'))

        client_username = head_dir['username']
        client_password = head_dir['password']

        client_ip = addr[0]

        # generate the storing name as a fixed format
        # ip_port-originName

        # get the state dict
        #
        with open('./server_users.json', 'r') as j:
            validUsers = json.load(j)

        with open(cfg_server["register_json_path"], 'r') as j:
            ip_name_dict = json.load(j)

        # check if the current connection is valid
        if client_username not in validUsers.keys():
            print(client_username, 'is invalid users or password\n')
            break
        else:
            if client_password == validUsers[client_username]:
                print('successfully login')
                print('current user is ', client_username)
            else:
                print(client_username, 'is invalid users or password\n')
                break
        # when you use correct name and password, you could registry your machine ip.
        if client_ip not in ip_name_dict.keys():
            ip_name_dict[client_ip] = client_username
            print(client_ip, client_username, ' has been registered')
        else:
            ip_name_dict[client_ip] = client_username

        # update the register dict
        with open(cfg_server["register_json_path"], 'w') as j:
            json.dump(ip_name_dict, j)
        break
    conn.close()

if __name__ == '__main__':
    tcp_server = socket(AF_INET, SOCK_STREAM)

    server_ip = cfg_server['server_ip']
    register_port = cfg_server['register_port']
    ip_port = ((server_ip, register_port))
    buffsize = cfg_server['buffsize']

    #  Port Reused. 
    tcp_server.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    tcp_server.bind(ip_port)
    tcp_server.listen(5)

    while True:
        print('Waiting for clients to connect')
        # Waiting for client to connect, and other clients except the one on working will wait.
        conn, addr = tcp_server.accept()
        print('Info of client:', addr)
        p = multiprocessing.Process(target=echo, args=(conn, addr))
        p.start()
        conn.close()
    tcp_server.close()


