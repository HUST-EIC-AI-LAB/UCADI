# -*- coding: utf-8 -*-
import os
import sys
import pdb
import json
import torch
import logging
import numpy as np
from apex import amp
import torch.nn as nn
sys.path.append('common')
import torch.optim as optim
import torch.nn.functional as F
from model.model import densenet3d
from common import TrainDataset, DataLoader, WarmUpLR, Logger


def add_weight_decay(net, l2_value, skip_list=()):
    """no L2 regularisation on the bias of the model, in optimiser"""
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    print('add no l2 decay to the bias terms')
    return [{'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': l2_value}]


def train(filename, device, train_data_loader, model, optimizer, log,
          warm_epoch, epoch, criterion, warmup_scheduler, train_scheduler,
          save_interval=5, save_folder="./model/"):
    train_num = len(train_data_loader)

    if epoch == 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-5

    running_loss = 0.0
    model.train()

    # training process
    for index, (inputs, labels, patient_name) in enumerate(train_data_loader):

        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        pdb.set_trace()
        for idx, label in enumerate(labels):
            if label.size() == torch.Size([2]):  # == torch.tensor([[1, 1]]):
                labels[idx] = label[0]
                print(patient_name)

        inputs = inputs.unsqueeze(dim=1).float()
        inputs = F.interpolate(inputs, size=[16, 128, 128],
                               mode="trilinear", align_corners=False)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        pdb.set_trace()

        # add apex for "loss.backward()"
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        optimizer.step()
        train_scheduler.step(epoch) if epoch > warm_epoch else warmup_scheduler.step()

        running_loss += loss.item()
        print("{} epoch, {} iter, loss {}".format(epoch, index + 1, loss.item()))
        # break
    log.info("{} epoch, Average loss {}".format(epoch, running_loss / train_num))

    # save checkpoint
    saved_path = save_folder + str(epoch) + '_' + filename
    log.info("save {} epoch.pth".format(epoch))
    torch.save(model.state_dict(), saved_path)

    return saved_path


if __name__ == "__main__":

    # with open('./config/train_config_client1.json') as j:
    with open('./config/train_config_client1_hc.json') as j:
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

    model = densenet3d().to(device)
    weight = torch.from_numpy(np.array([[0.2, 0.2, 0.4, 0.2]])).float()
    params = add_weight_decay(model, 4e-5)
    optimizer = optim.SGD(params, lr=train_config['lr'], momentum=train_config['momentum'])
    model, optimizer = amp.initialize(model, optimizer)
    model = nn.DataParallel(model)

    pdb.set_trace()
    restore = False
    restore_path = 'model/model_Param_Bob.pth'
    if restore:
        ob = torch.load(restore_path)
        state = ob['model_state_dict']
        model.load_state_dict(state)
    pdb.set_trace()
    iter_per_epoch, warm_epoch = len(train_data_loader), 5
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warm_epoch)
    train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, train_config['epoch'] - warm_epoch)

    logfile = "./train_valid_client1.log"
    os.remove(logfile) if os.path.exists(logfile) else None
    sys.stdout = Logger(logfile)
    log = logging.getLogger()

    for epoch_num in range(train_config['epoch']):

        fileName = 'central_model.pth'
        update_name = train(filename=fileName, device=device, train_data_loader=train_data_loader,
                        model=model, optimizer=optimizer, log=log, warm_epoch=warm_epoch,
                        epoch=epoch_num, criterion=criterion, warmup_scheduler=warmup_scheduler,
                        train_scheduler=train_scheduler, save_interval=5, save_folder='./model/')

    print("Training Finished")
