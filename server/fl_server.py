# -*- coding: utf-8 -*-
import os
import pdb
import json
import torch
import logging
import threading
from time import sleep
from aggregation import aggregateWeight, getWeightList
from socket import socket, SOL_SOCKET, SO_REUSEADDR, AF_INET, SOCK_STREAM
from common.tcp_utils import send_head_dir, send_file, recv_head_dir, recv_and_write_file


class FL_Server(object):
    def __init__(self, config_path):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        with open(config_path, 'r') as j:
            self.configs = json.load(j)

        with open(self.configs["clients_path"], 'r') as j:
            self.all_clients = json.load(j)

        self.ip_port = (self.configs["ip"], self.configs["recv_port"])
        self.recv_socket = socket(AF_INET, SOCK_STREAM)
        self.clients_status = {}
        self.lock = threading.Lock()
        self.finish, self.max_delay = False, 100
        self.clients_ip_port, self.n_clients = [], 0
        self.map_loc = torch.device('cuda')
        self.model_path = self.configs['weight_path']
        self.merge_weight_dir = self.configs['merge_model_dir']
        self.client_weight_dir = self.configs['client_weight_dir']
        os.makedirs(self.merge_weight_dir, exist_ok=True)
        os.makedirs(self.client_weight_dir, exist_ok=True)
        # -1: no active training process;
        # 0: training in process;
        # 1: training completed;
        for client in self.all_clients:
            self.clients_status[client] = -1

    def set_map_loc(self, device):
        if device not in ['cuda', 'cpu']:
            print('DEVICE ERROR')
            raise KeyError
        self.map_loc = torch.device(device)

    def count_status(self, status):
        count = 0
        for s in self.clients_status.values():
            if status == s:
                count += 1
        return count

    def start(self):
        self.logger.info("server start...")
        self.recv_socket.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
        self.recv_socket.bind(self.ip_port)
        self.recv_socket.listen(10)
        t1 = threading.Thread(target=self.handle_request)
        t1.setDaemon(True)
        t1.start()

    def handle_request(self):
        while True:
            conn, addr = self.recv_socket.accept()
            try:
                if self.finish:
                    send_head_dir(conn=conn, head_dir=json.dumps({'msg': "finish"}))

                head_dir = recv_head_dir(conn=conn)
                msg = head_dir["msg"]
                if msg == "register":
                    self.register(conn=conn, head_dir=head_dir)
                elif msg == "request_model":
                    self.send_model(conn=conn, head_dir=head_dir)
                elif msg == "send_model":
                    self.recv_model(conn=conn, head_dir=head_dir)
            finally:
                conn.close()

    def stop(self):
        self.logger.info("server is ready to exit...")
        self.finish = True
        sleep(15)
        exit()

    def register(self, conn, head_dir):

        username, password = head_dir['username'], head_dir['password']
        if username not in self.all_clients or password != self.all_clients[username]:
            send_head_dir(conn=conn, head_dir=json.dumps({'msg': "error"}))
        else:
            send_head_dir(conn=conn, head_dir=json.dumps({'msg': "ok"}))
            self.logger.info(username + " successfully registered!")
            self.logger.info("Wait a bit longer for other clients to join.")
            send_file(conn=conn, file_path=self.configs["model_path"], new_file_name=None)
            sleep(20)

    def send_model(self, conn, head_dir, _model_path=None):
        username = head_dir["username"]
        status = self.clients_status[username]

        if status == 1 or (not self.lock.acquire(blocking=False)):
            send_head_dir(conn=conn, head_dir=json.dumps({'msg': "wait"}))
        else:
            try:
                if status == -1:
                    send_head_dir(conn=conn, head_dir=json.dumps({'msg': "ok"}))
                    model_path = self.model_path if _model_path is None else _model_path
                    send_file(conn=conn, file_path=model_path, new_file_name=None)
                    self.clients_status[username] = 0
                    self.logger.info("sent model to " + username)
                else:
                    send_head_dir(conn=conn, head_dir=json.dumps({'msg': "error"}))
                    self.clients_status[username] = -1
            finally:
                self.lock.release()

    def recv_model(self, conn, head_dir):
        username = head_dir["username"]
        status = self.clients_status[username]

        if status != 0 or (not self.lock.acquire(blocking=False)):
            send_head_dir(conn=conn, head_dir=json.dumps({'msg': "error"}))
        else:
            try:
                send_head_dir(conn=conn, head_dir=json.dumps({'msg': "ok"}))
                recv_and_write_file(conn=conn, file_dir=self.client_weight_dir,
                                    buff_size=self.configs['buff_size'])
                self.clients_status[username] = 1
                self.logger.info("received model from " + username)
            finally:
                self.lock.release()

    def aggregation(self, client_models_dir='./model/client_model/',):
        """
        simple summation aggregation over weighted parameters from clients
        :param client_models_dir: store the model_weight & model_state_dict
        :return: aggregated model state_dict (in list)
        """
        print("*** aggregation begin ***")
        weightDictList, weightList, client_num = getWeightList(client_models_dir, self.map_loc)
        new_parm = aggregateWeight(weightDictList, weightList)
        return new_parm, sum(weightList), client_num

    def pack_param(self, _model_state, _client_weight, _client_num,  save_path=None):
        ob = {"model_state_dict": _model_state,
              "client_weight": _client_weight,
              "client_num": _client_num}
        torch.save(ob, save_path) if save_path is not None else torch.save(ob, self.model_path)

    @staticmethod
    def unpack_param(_model_param_path):
        ob = torch.load(_model_param_path)
        return ob['model_state_dict'], ob['client_weight']

    @staticmethod
    def flush_client_weight_dir(client_models_dir='./model/client_model/'):
        """Clean up all files within the client_model_dir"""
        file_list = os.listdir(client_models_dir)
        for i in range(len(file_list)):
            os.remove(os.path.join(client_models_dir, file_list[i]))
        print("all clients .pth caches removed")
