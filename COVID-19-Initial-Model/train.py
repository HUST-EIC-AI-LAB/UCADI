
import os
import sys
import torch
import logging
import numpy as np 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from apex import amp
from logger import Logger
from test_case_roc import test_case 
from model import densenet3d
from data_raw import TrainDataset, TestDataset
from WarmUpLR import WarmUpLR

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

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

def train(train_data_loader, test_data_loader, model, optimizer, log,
            warm_epoch, warmup_scheduler, train_scheduler, criterion, num_epochs = 101,
            save_folder = "./checkpoint/"):
    print("start training")
    recall_two_max = 0

    for epoch in range(1, num_epochs + 1):
        # for train
        running_loss = 0.0
        model.train()

        for index, (inputs, labels, patient_name) in enumerate(train_data_loader):
            optimizer.zero_grad()
            
            inputs, labels = inputs.cuda(), labels.cuda()
            for label in labels:
                if label.size() == torch.Size([2]): # == torch.tensor([[1, 1]]):
                    label = label[0]
                    print(patient_name)

            inputs = inputs.unsqueeze(dim = 1).float()
            inputs = F.interpolate(inputs, size = [16, 128, 128], mode = "trilinear", align_corners = False)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
#             loss.backward()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            train_scheduler.step() if epoch > warm_epoch else warmup_scheduler.step()

            running_loss += loss.item()
            print("{} iter, loss {}".format(index + 1, loss.item()))

        print("{} epoch, loss {}".format(epoch + 1, running_loss / len(train_data_loader)))
        log.info("{} epoch, loss {}".format(epoch + 1, running_loss / len(train_data_loader)))

        running_loss = 0.0
        recall_two = test_case(test_data_loader, model)
        if recall_two > recall_two_max:
            recall_two_max = recall_two

        PATH = os.path.join(save_folder, "{}_epoch_{}.pth".format(epoch, recall_two))
        log.info("save {} epoch.pth".format(epoch))
        torch.save(model.state_dict(), PATH)
    return recall_two_max


if __name__ == "__main__":
    data_dir = ''
    train_csv = ''
    train_label_csv = ''
    test_csv = ''
    test_label_csv = ''
    train_data_train = TrainDataset(data_dir, train_csv, train_label_csv)
    train_data_loader = DataLoader(dataset = train_data_train, batch_size = 70, shuffle = True, num_workers = 12)
    test_data_test = TestDataset(data_dir, test_csv, test_label_csv)
    test_data_loader = DataLoader(dataset = test_data_test, batch_size = 70, shuffle = False, num_workers = 12)
    
    model = densenet3d().cuda()
    weight = torch.from_numpy(np.array([[0.2, 0.2, 0.4, 0.2]])).float()
    criterion = nn.CrossEntropyLoss(weight=weight).cuda()
    params = add_weight_decay(model, 4e-5)
    optimizer = optim.SGD(params, lr=0.01, momentum=0.9)
    model, optimizer = amp.initialize(model, optimizer)
    model = nn.DataParallel(model)

    num_epochs = 100
    iter_per_epoch, warm_epoch = len(train_data_loader), 5
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warm_epoch)
    train_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, (num_epochs - warm_epoch) * iter_per_epoch)
    
    logfile = "./log_n1.log"
    os.remove(logfile) if os.path.exists(logfile) else None
    sys.stdout = Logger(logfile)
    log = logging.getLogger()
    train(train_data_loader, test_data_loader, model, optimizer, log,
        warm_epoch, warmup_scheduler, train_scheduler, criterion, num_epochs)



