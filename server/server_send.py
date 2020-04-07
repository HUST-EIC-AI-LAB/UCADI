from socket import *
import struct
import json
import os
import multiprocessing
from socket import SOL_SOCKET,SO_REUSEADDR,SO_SNDBUF,SO_RCVBUF

with open('./server_config.json', 'r') as j:
        cfg_server = json.load(j)
        buffsize = cfg_server['buffsize']

def echo(conn, addr):
    
    with open('./server_config.json', 'r') as j:
        cfg_server = json.load(j)
        buffsize = cfg_server['buffsize']
   
    filename = [cfg_server['model'],
                cfg_server['weight']
                ]

    with open(cfg_server['register_json_path'], 'r') as j:
        registerDict = json.load(j)
    client_ip = addr[0]

    # check whether is valid
    if client_ip not in registerDict.keys():
        # not registered
        print("an unregistered ip attempts to login")
        conn.close()
        return
    else:
        print('ip:', client_ip, 'is logining')

    # check the state dict
    with open(cfg_server["recv_state_json_path"], 'r') as j:
        state_dict = json.load(j)

    if client_ip not in state_dict.keys():
        # set the state value to 0, which means has sent it the model,
        # but not received the updated model from the client
        state_dict[client_ip] = 0
        print('ip:', client_ip, '--- model will be sended to it')
    elif state_dict[client_ip] == 0:
        print('ip:', client_ip, '--- has been download the model in this iteration')
        conn.close()
        return
    elif state_dict[client_ip] == 1:
        print('ip:', client_ip, '--- has updated its model in this iteration')
        conn.close()
        return
    else:
        raise KeyError

    # send the files
    for file in filename:
        if not conn:
            print('Connection failed')
            break

       
        filemesg = file.strip()
        filesize_bytes = os.path.getsize(filemesg) 
        filename = file.split("/")[-1]
        dirc = {
            'filename': filename,
            'filesize_bytes': filesize_bytes,
        }
        print(dirc)
        head_info = json.dumps(dirc) 
        head_info_len = struct.pack('i', len(head_info)) 
        
        conn.send(head_info_len) 
        conn.send(head_info.encode('utf-8'))

      
        with open(filemesg, 'rb') as f:
            data = f.read()
            conn.sendall(data)
        print('Sending successfully')

        # update the state dict
        with open(cfg_server["recv_state_json_path"], 'w') as j:
            json.dump(state_dict, j)


    while True:
        if(len(conn.recv(1024)) <= 0):
            break
    conn.close()

if __name__ == '__main__':
    tcp_server = socket(AF_INET, SOCK_STREAM)
    ip_port = (('0.0.0.0', cfg_server['send_server_port']))
    # buffsize = 1024

    #  
    tcp_server.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    tcp_server.bind(ip_port)
    tcp_server.listen(5)
    #filename = ['model.py', 'DenseNet.pth']

    # with open(cfg_server['register_json_path'], 'r') as j:
    #     registerDict = json.load(j)

    while True:
        print('Waiting for clients to connect')
       
        conn, addr = tcp_server.accept()
        print('Info of client:', addr)

        # after sending the model and the weight
        # update the state_dict

        p = multiprocessing.Process(target=echo, args=(conn, addr))
        p.start()
        conn.close()

