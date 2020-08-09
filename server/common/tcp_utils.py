# -*- coding: utf-8 -*-
import os
import json
import struct


def send_head_dir(conn, head_dir):
    # 发送 4字节头
    conn.send(struct.pack('i', len(head_dir)))
    # 发送 head_dir
    conn.send(head_dir.encode('utf-8'))


def send_file(conn, file_path, new_file_name):
    """
    发送文件，可指定发送时的文件名，不指定为原文件名
    :param conn:
    :param file_path:
    :param new_file_name: 新文件名
    :return:
    """
    if new_file_name is None:
        new_file_name = file_path.split("/")[-1]
    head_dir = json.dumps({'filename': new_file_name,
                           'file_size_bytes': os.path.getsize(file_path)})
    send_head_dir(conn, head_dir)
    # 发送 文件
    with open(file_path, 'rb') as f:
        conn.sendall(f.read())


def recv_head_dir(conn):
    # 接收 4字节头
    head_len = struct.unpack('i', conn.recv(4))[0]
    # 接收 head_dir
    return json.loads(conn.recv(head_len).decode('utf-8'))


def recv_and_write_file(conn, file_dir, buff_size):
    """
    接收文件，并写入指定目录
    :param conn:
    :param file_dir:
    :param buff_size:
    :return: 返回文件名
    """
    head_dir = recv_head_dir(conn)
    # 接收 文件
    file_size_bytes = head_dir['file_size_bytes']
    file_name = head_dir["filename"]

    with open(file_dir + file_name, 'wb') as f:
        count_len = 0
        while count_len < file_size_bytes:
            to_recv = min(file_size_bytes - count_len, buff_size)
            recv_meg = conn.recv(to_recv)
            count_len += len(recv_meg)
            f.write(recv_meg)
    return file_name
