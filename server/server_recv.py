import struct
from socket import *
import json
import os
import multiprocessing
from socket import SOL_SOCKET,SO_REUSEADDR,SO_SNDBUF,SO_RCVBUF

with open('./server_config.json', 'r') as j:
    cfg_server = json.load(j)
buffsize = cfg_server['buffsize']
# recv_port = cfg_server['recv_server_port']

def echo(conn, addr):

    client_ip = addr[0]

    with open(cfg_server["recv_state_json_path"], 'r') as j:
        state_dict = json.load(j)

    # check the state dict
    if state_dict[client_ip] == 0:
        print('ip: ', client_ip, '--- will receive its model')
        # change the value to 1, which means have been received
        state_dict[client_ip] = 1
    elif state_dict[client_ip] == 1:
        print('ip: ', client_ip, '--- has received its updated model')
        conn.close()
    elif client_ip not in state_dict.keys():
        print('ip: ', client_ip, '--- has not been involved in the previous FL iteration')
        conn.close()
    else:
        raise KeyError
    while True:
        if not conn:
            print('Connection failed')
            break
        head_struct = conn.recv(4)  
        if head_struct:
            print('Connected with client, waiting for data ')
        head_len = struct.unpack('i', head_struct)[0]  
        data = conn.recv(head_len)  

        head_dir = json.loads(data.decode('utf-8'))
        filesize_b = head_dir['filesize_bytes']
        filename = head_dir['filename']

        client_username = head_dir['username']
        client_password = head_dir['password']

        with open('./server_users.json', 'r') as j:
            validUsers = json.load(j)


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

        #  
        recv_len = 0
        recv_mesg = b''
        filename = './client_data/' + filename
        f = open(filename, 'wb')
        while recv_len < filesize_b:
            if filesize_b - recv_len > buffsize:
                recv_mesg = conn.recv(buffsize)
                f.write(recv_mesg)
                recv_len += len(recv_mesg)
            else:
                recv_mesg = conn.recv(filesize_b - recv_len)
                recv_len += len(recv_mesg)
                f.write(recv_mesg)
        print('Received successfully\n')
        break
    with open(cfg_server["recv_state_json_path"], 'w') as j:
        json.dump(state_dict, j)

    conn.close()

if __name__ == '__main__':
    tcp_server = socket(AF_INET, SOCK_STREAM)
    ip_port = (('0.0.0.0', cfg_server['recv_server_port']))
    buffsize = cfg_server['buffsize']

  
    tcp_server.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    tcp_server.bind(ip_port)
    tcp_server.listen(5)

    while True:
        print('Waiting for clients to connect')
       
        conn, addr = tcp_server.accept()
        print('Info of client:', addr)
        p = multiprocessing.Process(target=echo, args=(conn, addr))
        p.start()
        conn.close()
    tcp_server.close()
