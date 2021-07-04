# -*- coding: utf-8 -*-

#  Copyright (c) 2021. Jiefeng, Ziwei and Hanchen
#  jiefenggan@gmail.com, ziwei@hust.edu.cn, hc.wang96@gmail.com

import os
import json
import struct


def send_head_dir(conn, head_dir):
    conn.send(struct.pack('i', len(head_dir)))
    conn.send(head_dir.encode('utf-8'))


def send_file(conn, file_path, new_file_name):
    """send the files"""
    if new_file_name is None:
        new_file_name = file_path.split("/")[-1]
    head_dir = json.dumps({'filename': new_file_name,
                           'file_size_bytes': os.path.getsize(file_path)})
    send_head_dir(conn, head_dir)
    with open(file_path, 'rb') as f:
        conn.sendall(f.read())


def recv_head_dir(conn):
    head_len = struct.unpack('i', conn.recv(4))[0]
    return json.loads(conn.recv(head_len).decode('utf-8'))


def recv_and_write_file(conn, file_dir, buff_size):
    """ receive and save file at the server side
    :return: saved filename """
    head_dir = recv_head_dir(conn)
    file_size_bytes, file_name = head_dir['file_size_bytes'], head_dir["filename"]

    with open(file_dir + file_name, 'wb') as f:
        count_len = 0
        while count_len < file_size_bytes:
            to_recv = min(file_size_bytes - count_len, buff_size)
            recv_meg = conn.recv(to_recv)
            count_len += len(recv_meg)
            f.write(recv_meg)
    return file_name
