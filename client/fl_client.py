# -*- coding: utf-8 -*-
import os
import pdb
import json
import torch
import logging
from time import sleep
from model.model import densenet3d
from common.LWE_based_PHE.cuda_test import KeyGen, Enc, Dec
from encrypt_decrypt import generate_shape, encrypt, decrypt
from socket import socket, AF_INET, SOCK_STREAM, SOL_SOCKET, SO_REUSEADDR
from common.tcp_utils import send_head_dir, recv_head_dir, recv_and_write_file, send_file


class FL_Client(object):
    def __init__(self, config_path, shape_param_path='./config/shape_parameter.pth'):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        with open(config_path, 'r') as j:
            self.configs = json.load(j)

        self.server_ip_port = (self.configs['server_ip'], self.configs['server_port'])
        self.ip_port = (self.configs["ip"], self.configs["send_port"])
        self.pk, self.sk = KeyGen(self.configs['seed'])
        self.model = densenet3d().cuda()

        if not os.path.exists(shape_param_path):
            model_weight = self.model.state_dict()
            generate_shape(shape_param_path, model_weight)
        self.shape_parameter = torch.load(shape_param_path)
        self.weight, self.model_path = None, self.configs["model_path"]
        self.weight_path = self.configs["weight_path"]

    def start(self):
        self.logger.info("client starts, username: " + self.configs['username'])
        self.register()

    def stop(self):
        self.logger.info("training finished, client quiting...")
        exit()

    def register(self):
        send_socket = socket(AF_INET, SOCK_STREAM)
        self.logger.info("registering with server ...")
        try:
            pdb.set_trace()
            send_socket.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
            send_socket.bind(self.ip_port)
            send_socket.connect(self.server_ip_port)

            head_dir = json.dumps({'username': self.configs['username'],
                                   'password': self.configs['password'],
                                   'msg': "register"})
            send_head_dir(conn=send_socket, head_dir=head_dir)
            recv_dir = recv_head_dir(conn=send_socket)

            if recv_dir["msg"] == "ok":  # successfully receive and save model.py from server
                recv_and_write_file(conn=send_socket,
                                    file_dir='/'.join(self.model_path.split("/")[:-1]) + "/",
                                    buff_size=self.configs["buff_size"])
                self.logger.info("successfully registered!")
            else:
                self.stop()
        finally:
            send_socket.close()
            sleep(5)

    def request_model(self):
        self.logger.info("requesting server to send model...")
        send_socket = socket(AF_INET, SOCK_STREAM)
        try:
            send_socket.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
            send_socket.bind(self.ip_port)
            send_socket.connect(self.server_ip_port)

            head_dir = json.dumps({'username': self.configs['username'],
                                   'msg': "request_model"})
            send_head_dir(conn=send_socket, head_dir=head_dir)
            recv_dir = recv_head_dir(conn=send_socket)

            if recv_dir["msg"] == "finish":
                self.stop()

            if recv_dir["msg"] == "ok":
                fileName = recv_and_write_file(conn=send_socket,
                                               file_dir='/'.join(self.model_path.split("/")[:-1]) + "/",
                                               buff_size=self.configs['buff_size'])
                self.weight_path = os.path.join('/'.join(self.model_path.split("/")[:-1]) + "/", fileName)
                self.logger.info("successfully received the model from the server!")
                return "ok"
            elif recv_dir["msg"] == "wait":
                self.logger.info("waiting...")
                return "wait"
            else:
                self.logger.warning("model request was rejected!")
                return "error"
        finally:
            send_socket.close()
            sleep(1)

    def send_model(self, weight_path=None, versionNum=0):
        send_socket = socket(AF_INET, SOCK_STREAM)
        try:
            send_socket.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
            send_socket.bind(self.ip_port)
            send_socket.connect(self.server_ip_port)

            head_dir = json.dumps({'username': self.configs['username'], 'msg': "send_model"})
            send_head_dir(conn=send_socket, head_dir=head_dir)
            recv_dir = recv_head_dir(conn=send_socket)

            if recv_dir["msg"] == "finish":
                self.stop()

            if recv_dir["msg"] == "ok":
                self.logger.info("sending model to server...")
                send_path = self.weight_path if weight_path is None else weight_path
                send_file(conn=send_socket, file_path=send_path,
                          new_file_name="model_Param_{}_v_{}.pth".format(
                              self.configs["username"], versionNum))
                self.logger.info("successfully sent the model to the server!")
                return "ok"
            else:
                self.logger.warning("sent model was rejected!")
                return "error"
        finally:
            send_socket.close()
            sleep(1)

    def set_weight(self, weight=1.0):
        self.weight = weight

    def pack_param(self, _model_state, _client_weight, save_path=None):
        ob = {"model_state_dict": _model_state,
              "client_weight": _client_weight}
        torch.save(ob, save_path) if save_path is not None else torch.save(ob, self.weight_path)

    @staticmethod
    def unpack_param(_model_param_path):
        ob = torch.load(_model_param_path)
        return ob['model_state_dict'], ob['client_weight'], ob['client_num']

    def enc_num(self, num):
        return Enc(self.pk, num)

    def dec_num(self, num):
        return Dec(self.sk, num)

    def _encrypt(self, model_weight):  # to avoid the same function name
        return encrypt(self.pk, model_weight)

    def _decrypt(self, encrypted_model_weight, client_num):
        return  decrypt(self.sk, encrypted_model_weight, client_num, self.shape_parameter)
