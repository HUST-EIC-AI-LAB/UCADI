import time
import socket
import data 
import numpy as np 
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import model 
from data import TrainDataset, TestDataset
from download.model import densenet3d
from torch.nn import DataParallel
from logger import *
from test_case import test_case  
import logging
import subprocess
import sys
from pathlib import Path
import os

#from warmup_scheduler import GradualWarmupScheduler

import os
import json
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"


def train(device, train_data_loader, test_data_loader, model, optimizer, log, num_epochs, save_interval = 5, save_folder = "./checkpoint_2/"):
    train_num = len(train_data_loader)
    print("start training")
    recall_two_max = 0

    for epoch in range(num_epochs):
        # lr = get_lr(epoch)
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr
        # for train
        running_loss = 0.0
        model.train()

        for index, (inputs, labels, patient_name) in enumerate(train_data_loader):
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
            loss.backward()
            #optimize
            optimizer.step()

            # loss update
            running_loss += loss.item()
     
            print("{} epoch, {} iter, loss {}".format(epoch + 1, index + 1, loss.item()))
            
        print("{} epoch, Average loss {}".format(epoch + 1, running_loss / train_num))
        log.info("{} epoch, Average loss {}".format(epoch + 1, running_loss / train_num))

        running_loss = 0.0
        #recall_two = test_case(test_data_loader, model)
        #if recall_two > recall_two_max:
        #    recall_two_max = recall_two
        save_name = 'weight_' + str(epoch+1) +'.pth'
        PATH = save_folder + save_name
        log.info("save {} epoch.pth".format(epoch+1))
        torch.save(model.state_dict(), PATH)
    return model


if __name__ == "__main__":

    with open('./config/config_client.json') as f:
         config_client = json.load(f)
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    #name = config_client['weight_pth'].split("/")[-1]
    #save_name = ip + '_' + str(config_client['send_server_port']) + '_' + name

    with open('./config/config.json') as f:
         config = json.load(f)
    device = 'cuda' if config['use_cuda'] else 'cpu'
    model = densenet3d().to(device)
    weight = torch.from_numpy(np.array([[0.2, 0.2, 0.4, 0.2]])).float()   
    
    if os.path.exists(config['load_model']):
         model.load_state_dict(torch.load(config['load_model']))
    else: 
         print('please run `python download.py` at first')
         sys.exit(0)
    criterion = nn.CrossEntropyLoss(weight=weight).to(device)
    lr_rate = config['lr']
    momentum = config['momentum']
    num_epochs = config['epoch'] 
    optimizer = optim.SGD(model.parameters(),lr=lr_rate, momentum=momentum)
   # scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epoch)
   # scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=10, after_scheduler=scheduler_cosine)

    logfile = "./train_valid.log"
    if os.path.exists(logfile):
        os.remove(logfile)
    sys.stdout = Logger(logfile)
           
    # sets.phase = 'train'
    num_workers = config['num_workers']
    train_data_train = TrainDataset(config['train_data_dir'],
                                    config['train_df_csv'],
                                    config['labels_train_df_csv'])
    train_batch_size = config['train_batch_size']
    train_data_loader = DataLoader(dataset = train_data_train, batch_size = train_batch_size, shuffle = True, num_workers = num_workers)
    test_data_test = TestDataset(config['test_data_dir'],
                                 config['test_df_csv'],
                                 config['labels_test_df_csv'])
    test_batch_size = config['test_batch_size']
    test_data_loader = DataLoader(dataset = test_data_test, batch_size = test_batch_size, shuffle = False, num_workers = num_workers)

    log = logging.getLogger()
    
    model = train(device, train_data_loader, test_data_loader, model, optimizer, log, num_epochs)
   

    model_original = densenet3d().to(device)
    model_diff = densenet3d().to(device)
    if os.path.exists(config['load_model']):
         model_original.load_state_dict(torch.load(config['load_model']))
    else:
         print('please run `python download.py` at first')
         sys.exit(0)
    for name, param in model_diff.named_parameters():
         model_diff.state_dict()[name].copy_(model.state_dict()[name] - model_original.state_dict()[name])
   
    name = config_client['weight_pth'].split("/")[-1]
    diff_name = ip + '_' + str(config_client['send_server_port']) + '_differ_' + name
    path = './checkpoint_2/' + diff_name
    torch.save(model_diff.state_dict(), path)
 
    python = Path(sys.executable).name
    FILE_PATH_SEND = Path(__file__).resolve().parents[0].joinpath("./client/client_send.py")
    send = [python,FILE_PATH_SEND, path]
    start = time.time()
    p = subprocess.Popen(send)
    while True:
        if p.poll() == 0:
            break;
        else:
            now = time.time()
            if (now - start) >= 20:
                print("send error, pleas try `python upload.py file_path`")
                p.terminate()
                sys.exit(0)
    sys.exit(0)
    

   
    




