#import os
#os.chdir("/home/xyang/fl-frame2/FL-Test-2/")

import logging
from time import sleep
import json
from client.fl_client import FL_Client
from client.model.model import densenet3d
import torch.optim as optim
import torch
import numpy as np
from apex import amp
from client.common import *
from client.train import train, add_weight_decay

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    client = FL_Client('./config/client1_config.json')
    client.start()

    # *********************** train part-preprocess begin *******************************
    ## train configs and some prepartions before receive the model
    print('training')
    # initialize train params

    # train configurations
    with open('./config/train_config_client1.json') as j:
        train_config = json.load(j)

    device = 'cuda' if train_config['use_cuda'] else 'cpu'
    model = densenet3d().to(device)
    weight = torch.from_numpy(np.array([[0.2, 0.2, 0.4, 0.2]])).float()

    num_workers = train_config['num_workers']
    train_data_train = TrainDataset(train_config['train_data_dir'],
                                    train_config['train_df_csv'],
                                    train_config['labels_train_df_csv'])
    train_batch_size = train_config['train_batch_size']
    train_data_loader = DataLoader(dataset=train_data_train, batch_size=train_batch_size, shuffle=True,
                                   num_workers=num_workers)

    criterion = nn.CrossEntropyLoss(weight=weight).to(device)
    ## initial lr
    lr_rate = train_config['lr']
    num_epochs = train_config['epoch']
    # lr_rate = config['lr']
    momentum = train_config['momentum']
    # num_epochs = config['epoch']

    ## add no bias decay
    params = add_weight_decay(model, 4e-5)
    optimizer = optim.SGD(params, lr=lr_rate, momentum=momentum)
    model, optimizer = amp.initialize(model, optimizer)
    model = nn.DataParallel(model)
    iter_per_epoch = len(train_data_loader)
    print("iter per epoch".format(iter_per_epoch))
    warm_epoch = 5
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warm_epoch)
    train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs - warm_epoch)

    logfile = "./train_valid.log"
    if os.path.exists(logfile):
        os.remove(logfile)
    sys.stdout = Logger(logfile)
    log = logging.getLogger()

    # receive the initial.pth
    # *********************** train part-0 begin *******************************
    while True:
        request_model_result = client.request_model()
        if request_model_result == "ok":
            request_model_finish = True
            # if received successfully, client will get its newly updated model from server.
            break
        elif request_model_result == "wait":
            sleep(10)
            continue
        elif request_model_result == "error":
            break
    logger.info("preparing the first communication round")
    recv_filePath = client.train_model_path
    logger.info("Begin Training...")
    _model_state, _weight_sum, _client_num = client.unpack_param(_model_param_path=recv_filePath)
    ## assign weighted state to the model
    client.set_weight(iter_per_epoch)
    _model_Param = {"model_state_dict": _model_state,
                    "client_weight": client.weight}
    savePath = os.path.join(client.configs["models_dir"],
                            './model_Param_{}.pth'.format(client.configs['username'], ))
    torch.save(_model_Param, savePath)
    # print("encrpted param[0]", enc_model_state[0])
    # send the model
    # eg: model_Param_Alan_v0.pth
    client.send_model(model_weight_path=savePath, versionNum=0)
    # *********************** train part-0 end *******************************
    # *********************** train part-preprocess end *******************************

    logger.info("******\nTruely training begin\n******")
    while True:
        epoch_num = 0
        logger.info('******* new  epoch *******')
        request_model_finish = False

        while True:
            request_model_result = client.request_model()
            if request_model_result == "ok":
                request_model_finish = True
                # if received successfully, client will get its newly updated model from server.
                break
            elif request_model_result == "wait":
                sleep(10)
                continue
            elif request_model_result == "error":
                break
        if not request_model_finish:
            continue

        logger.info('***** current epoch is {} *****'.format(epoch_num))
        recv_filePath = client.train_model_path
        print(recv_filePath)
        # unpack the .pth file, and decrypt model_state & weight num
        _model_state, _weight_sum, _client_num = client.unpack_param(_model_param_path=recv_filePath)
        ## decryption
        dec_model_state = client.decrypt(_model_state, _client_num)
        # dec_weight_num = client.dec_num(_weight_sum)
        dec_weight_num = _weight_sum  # not encypted

        # print("weight num claculated from server:{}".format(1.0 / dec_weight_num))
        print("some of decrypted state:")
        temp_key = list(dec_model_state.keys())[0]
        print(dec_model_state[temp_key][0])
        print("{} weight is {}".format(client.configs['username'], client.weight))
        print("weight sum is {}\t client num is {}".format(_weight_sum, _client_num))

        for key in dec_model_state.keys():
            if epoch_num == 0:
                dec_model_state[key] = dec_model_state[key]
            else:
                dec_model_state[key] = dec_model_state[key] * _client_num
        torch.save(dec_model_state, './{}_current.pth'.format(client.configs['username']))

        print("After Decryption\n", dec_model_state[temp_key][0])

        # *********************** train part-2 begin *******************************
        # load the state dict of new model
        model.load_state_dict(dec_model_state)
        fileName = 'model_state_{}.pth'.format(client.configs['username'])
        update_name = train(filename=fileName, device=device, train_data_loader=train_data_loader,
                            model=model, optimizer=optimizer, log=log, warm_epoch=warm_epoch,
                            epoch=epoch_num,
                            # epoch=1,
                            criterion=criterion,
                            warmup_scheduler=warmup_scheduler, train_scheduler=train_scheduler,
                            save_interval=5, save_folder='./model/')
        # *********************** train part-2 end *******************************

        ## encryption, then send
        trained_model_state_dict = torch.load(update_name)
        # print("After training, some state")
        # print(trained_model_state_dict[temp_key][0])
        # print("**********************\n*****************************\n")
        # print("client weight {}\n".format(client.weight))
        for key in trained_model_state_dict.keys():
            trained_model_state_dict[key] = trained_model_state_dict[key] * client.weight / _weight_sum
        enc_model_state = client.encrypt(trained_model_state_dict)
        # dec_model_again = client.decrypt(enc_model_state, 1)
        # print("some test decrypted state:", dec_model_again[temp_key][0])
        # client.set_weight(float(iter_per_epoch))
        # enc_client_weight = client.enc_num(client.weight)
        _model_Param = {"model_state_dict": enc_model_state,
                        "client_weight": client.weight}
        savePath = os.path.join(client.configs["models_dir"],
                                './model_Param_{}.pth'.format(client.configs['username'], ))
        torch.save(_model_Param, savePath)
        # print("encrpted param[0]", enc_model_state[0])
        # send the model
        # eg: model_Param_Alan_v0.pth
        client.send_model(model_weight_path=savePath, versionNum=epoch_num + 1)
        epoch_num += 1

        logger.info("current training finished")





