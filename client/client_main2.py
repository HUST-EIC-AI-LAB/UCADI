# -*- coding: utf-8 -*-
import os
import sys
import pdb
import json
import torch
import logging
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim

from apex import amp
from time import sleep
sys.path.append('common')
from fl_client import FL_Client
from model.model import densenet3d
from train import train, add_weight_decay
from common import TrainDataset, DataLoader, WarmUpLR, Logger

if __name__ == '__main__':
    print('all start')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description='Client Configuration')
    parser.add_argument('--client_config', type=str, default='client2_config.json')
    parser.add_argument('--train_config', type=str, default='train_config_client2.json')
    parser.add_argument('--logfile', type=str, default='train_valid_client2.log')
    args = parser.parse_args()

    # FL_Client() already includes GPU usage
    client = FL_Client(os.path.join('./config', args.client_config))
    client.start()
    client.register()

    ''' === set up training configs and load the model before querying from the server === '''
    with open(os.path.join('./config', args.train_config)) as j:
        train_config = json.load(j)

    train_data_train = TrainDataset(train_config['train_data_dir'],
                                    train_config['train_df_csv'],
                                    train_config['labels_train_df_csv'])
    train_data_loader = DataLoader(dataset=train_data_train,
                                   batch_size=train_config['train_batch_size'],
                                   shuffle=True,
                                   num_workers=train_config['num_workers'])

    device = 'cuda' if train_config['use_cuda'] else 'cpu'
    model = densenet3d().to(device)
    weight = torch.from_numpy(np.array([[0.2, 0.2, 0.4, 0.2]])).float()
    criterion = nn.CrossEntropyLoss(weight=weight).to(device)

    # add no l2 decay to the bias terms during optimisation
    params = add_weight_decay(model, 4e-5)
    optimizer = optim.SGD(params, lr=train_config['lr'], momentum=train_config['momentum'])
    model, optimizer = amp.initialize(model, optimizer)
    model = nn.DataParallel(model)
    iter_per_epoch, warm_epoch = len(train_data_loader), 5

    # scheduler is called per training steps/iterations
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warm_epoch)
    train_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, (train_config['epoch'] - warm_epoch) * iter_per_epoch)

    # set up the log file
    logfile = os.path.join('log', args.logfile)
    os.remove(logfile) if os.path.exists(logfile) else None
    sys.stdout = Logger(logfile)
    log = logging.getLogger()

    ''' === receive the initial model from the server, add aggregation weight, sent back === '''
    while True:
        request_model_result = client.request_model()
        if request_model_result == "ok":
            request_model_finish = True
            break
        elif request_model_result == "wait":
            sleep(10)
            continue
        elif request_model_result == "error":
            break

    _model_state, _weight_sum, _client_num = client.unpack_param(
        _model_param_path=client.weight_path)

    # aggregation weight when decode the models from server,
    # based on the amount of training data the client use
    client.set_weight(iter_per_epoch)
    _model_Param = {"model_state_dict": _model_state,
                    "client_weight":    client.weight}
    savePath = os.path.join(client.configs["models_dir"],
                            './model_Param_{}.pth'.format(client.configs['username']))
    torch.save(_model_Param, savePath)
    client.send_model(weight_path=savePath, versionNum=0)

    ''' === formally start training, receive and send model to the server every epoch ==='''
    logger.info("******\ntraining begin\n******")
    for epoch_num in range(client.configs['iteration']):
        request_model_finish = False

        while True:
            request_model_result = client.request_model()
            if request_model_result == "ok":
                request_model_finish = True
                break
            elif request_model_result == "wait":
                sleep(10)
                continue
            elif request_model_result == "error":
                break
        if not request_model_finish:
            continue

        # unpack the package, decrypt model parameters and aggregation weights
        logger.info('***** current epoch is {} *****'.format(epoch_num))
        _model_state, _weight_sum, _client_num = client.unpack_param(
            _model_param_path=client.weight_path)
        dec_model_state = client.decrypts(_model_state, _client_num)
        client.set_weight(iter_per_epoch)

        for key in dec_model_state.keys():
            if epoch_num == 0:  # first aggregation is simple addition
                dec_model_state[key] = dec_model_state[key]
            else:   # later aggregation is weight summation
                dec_model_state[key] = dec_model_state[key] / _client_num
        torch.save(dec_model_state, './model/{}_current.pth'.format(client.configs['username']))
        temp_key = list(dec_model_state.keys())[0]
        print("After Decryption\n", dec_model_state[temp_key][0])

        logger.info("{} weight is {}".format(client.configs['username'], client.weight/_weight_sum))
        logger.info("weight sum is {}\t client num is {}".format(_weight_sum, _client_num))

        # pdb.set_trace()
        model.load_state_dict(dec_model_state)
        fileName = 'model_state_{}.pth'.format(client.configs['username'])
        update_name = train(filename=fileName, device=device, train_data_loader=train_data_loader,
                            model=model, optimizer=optimizer, log=log, warm_epoch=warm_epoch,
                            epoch=epoch_num, criterion=criterion, warmup_scheduler=warmup_scheduler,
                            train_scheduler=train_scheduler, save_interval=5, save_folder='./model/')

        # encryption, then send the model back to the server
        trained_model_state_dict = torch.load(update_name)
        print("After training, some updated model parameters: ")
        print(trained_model_state_dict[temp_key][0])
        print("*****************************\n*****************************\n")
        for key in trained_model_state_dict.keys():
            trained_model_state_dict[key] = trained_model_state_dict[key] * client.weight / _weight_sum

        # pdb.set_trace()
        enc_model_state = client.encrypts(trained_model_state_dict)
        _model_Param = {"model_state_dict": enc_model_state,
                        "client_weight": client.weight}
        savePath = os.path.join(client.configs["models_dir"],
                                './model_Param_{}.pth'.format(client.configs['username'], ))
        torch.save(_model_Param, savePath)
        client.send_model(weight_path=savePath, versionNum=epoch_num)

    logger.info("training finished")
    client.stop()
