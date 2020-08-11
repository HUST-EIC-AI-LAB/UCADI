import json
# from test_case import test_case
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from WarmUpLR import WarmUpLR
from apex import amp
from client.client_recv import *
from client.client_registry import *
from client.client_send import *
from data_raw import TrainDataset
from download.model import densenet3d
from torch.utils.data import DataLoader

from back_up.logger import *

# from warmup_scheduler import GradualWarmupScheduler
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

## no bias decay
def add_weight_decay(net, l2_value, skip_list=()):
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    print('add no bias decay')
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]

def train(filename, device, train_data_loader, model, optimizer, log, warm_epoch, epoch, warmup_scheduler, train_scheduler, save_interval = 5, save_folder = "./checkpoint/"):
    train_num = len(train_data_loader)
    print("start training")
    recall_two_max = 0

        # lr = get_lr(epoch)
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr
        # for train
    if epoch==1:
        lr=0.00001
        for param_group in optimizer.param_groups:
            param_group['lr']=lr

    running_loss = 0.0
    model.train()

    for index, (inputs, labels, patient_name) in enumerate(train_data_loader):

        lr=optimizer.param_groups[0]['lr']
        optimizer.zero_grad()

        inputs, labels = inputs.to(device), labels.to(device)#to(device), labels.to(device)
        for label in labels:
            if label.size() == torch.Size([2]):#== torch.tensor([[1, 1]]):
                label = label[0]
                print(patient_name)
        # foward
        inputs = inputs.unsqueeze(dim = 1).float()
        inputs = F.interpolate(inputs, size = [16, 128, 128], mode = "trilinear", align_corners = False)
        # print(inputs.shape,labels.shape)
        outputs = model(inputs)

        # backward
        loss = criterion(outputs, labels)
        #loss.backward()
        #optimize
        ## add apex
        with amp.scale_loss(loss,optimizer) as scaled_loss:
            scaled_loss.backward()

        optimizer.step()
        if epoch > warm_epoch:
            train_scheduler.step(epoch)
        if epoch <= warm_epoch:
            warmup_scheduler.step()
        # loss update
        running_loss += loss.item()

        print("{} epoch, {} iter, loss {}".format(epoch, index + 1, loss.item()))

    print("{} epoch, Average loss {}".format(epoch, running_loss / train_num))
    log.info("{} epoch, Average loss {}".format(epoch, running_loss / train_num))

    running_loss = 0.0
    #recall_two = test_case(test_data_loader, model)
    #if recall_two > recall_two_max:
    #    recall_two_max = recall_two
    PATH = save_folder + str(epoch) + '_' +filename
    log.info("save {} epoch.pth".format(epoch))
    torch.save(model.state_dict(), PATH)
    update_name = save_folder + str(epoch) + '_' + filename
    return update_name


if __name__ == "__main__":
    with open('./config/config_client.json') as f:
        config_client = json.load(f)
    #hostname = socket.gethostname()
    #ip = socket.gethostbyname(hostname)
    epoch = 0

    with open('./config/config.json') as f:
             config = json.load(f)
    device = 'cuda' if config['use_cuda'] else 'cpu'
    model = densenet3d().to(device)
    weight = torch.from_numpy(np.array([[0.2, 0.2, 0.4, 0.2]])).float()

    num_workers = config['num_workers']
    train_data_train = TrainDataset(config['train_data_dir'],
                                    config['train_df_csv'],
                                    config['labels_train_df_csv'])
    train_batch_size = config['train_batch_size']
    train_data_loader = DataLoader(dataset = train_data_train, batch_size = train_batch_size, shuffle = True, num_workers = num_workers)

    criterion = nn.CrossEntropyLoss(weight=weight).to(device)
    ## initial lr
    lr_rate = config['lr']
    num_epochs = config['epoch']
    #lr_rate = config['lr']
    momentum = config['momentum']
    #num_epochs = config['epoch']
    ## add no bias decay
    params = add_weight_decay(model, 4e-5)
    optimizer = optim.SGD(params, lr=lr_rate, momentum=momentum)
    model,optimizer=amp.initialize(model,optimizer)
    model=nn.DataParallel(model)
   # scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epoch)
   # scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=10, after_scheduler=scheduler_cosine)

    iter_per_epoch=len(train_data_loader)
    warm_epoch=5
    warmup_scheduler=WarmUpLR(optimizer,iter_per_epoch*warm_epoch)
    train_scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs-warm_epoch)

    logfile = "./train_valid.log"
    if os.path.exists(logfile):
        os.remove(logfile)
    sys.stdout = Logger(logfile)
    # sets.phase = 'train'
    log = logging.getLogger()

    while True:
        #name = config_client['weight_pth'].split("/")[-1]
        #save_name = ip + '_' + str(config_client['send_server_port']) + '_' + name
        registry()
        while True:
            filename = receive()
            if len(filename) == 2:
                break
            else:
                time.sleep(20)
        #from download.model import densenet3d
        name = './download/' + filename[1].split('/')[-1]
        print(name)
        update = name
        if os.path.exists(name):
             model.load_state_dict(torch.load(name))
        else:
             print('error when download initial weight from server')
             sys.exit(0)

        epoch = epoch + 1

        #update = train(name.split("/")[-1], device, train_data_loader, model, optimizer, log, warm_epoch, epoch, warmup_scheduler, train_scheduler)
        update = name
        original_state = torch.load(name)
        update_state = torch.load(update)
        diff_state = {}

        for key in original_state.keys():
            init = update_state[key] - original_state[key]
            diff_state[key] = init
        diff_name = config_client['username'] + '_differ.pth'
        path = './checkpoint/' + diff_name
        torch.save(diff_state, path)

        send(path)








