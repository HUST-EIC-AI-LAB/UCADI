import os
import torch
from apex import amp
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


## no bias decay
def add_weight_decay(net, l2_value, skip_list=()):
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    print('add no bias decay')
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]


def train(filename, device, train_data_loader, model, optimizer, log,
          warm_epoch, epoch, criterion, warmup_scheduler, train_scheduler,
          save_interval=5, save_folder="./model/"):
    train_num = len(train_data_loader)
    print("start training")
    recall_two_max = 0

    # lr = get_lr(epoch)
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
    # for train
    if epoch == 1:
        lr = 0.00001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    running_loss = 0.0
    model.train()

    # training process
    for index, (inputs, labels, patient_name) in enumerate(train_data_loader):

        lr = optimizer.param_groups[0]['lr']
        optimizer.zero_grad()

        inputs, labels = inputs.to(device), labels.to(device)  # to(device), labels.to(device)
        for label in labels:
            if label.size() == torch.Size([2]):  # == torch.tensor([[1, 1]]):
                label = label[0]
                print(patient_name)
        # forward
        inputs = inputs.unsqueeze(dim=1).float()
        inputs = F.interpolate(inputs, size=[16, 128, 128], mode="trilinear", align_corners=False)
        # print(inputs.shape,labels.shape)
        outputs = model(inputs)

        # backward
        loss = criterion(outputs, labels)
        # loss.backward()
        # optimize
        ## add apex
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        optimizer.step()
        if epoch > warm_epoch:
            train_scheduler.step(epoch)
        if epoch <= warm_epoch:
            warmup_scheduler.step()
        # loss update
        running_loss += loss.item()

        print("{} epoch, {} iter, loss {}".format(epoch, index + 1, loss.item()))
        break
    print("{} epoch, Average loss {}".format(epoch, running_loss / train_num))
    log.info("{} epoch, Average loss {}".format(epoch, running_loss / train_num))

    running_loss = 0.0

    # save checkpoint
    PATH = save_folder + str(epoch) + '_' + filename
    log.info("save {} epoch.pth".format(epoch))
    torch.save(model.state_dict(), PATH)
    update_name = PATH
    return update_name


if __name__ == "__main__":
    pass
#  #hostname = socket.gethostname()
#  #ip = socket.gethostbyname(hostname)
#  client_json_path = './config/config_client.json'
#
#  with open('./config/config_client.json') as f:
#      config_client = json.load(f)
#  with open('./config/config.json') as f:
#  	config = json.load(f)
#
#  epoch = 0
#
#  # Homomophric encryot setting
#  prec = 32
#  bound = 2 ** 3
#
#  with open('./config/config.json') as f:
#           config = json.load(f)
#  device = 'cuda' if config['use_cuda'] else 'cpu'
#  model = densenet3d().to(device)
#  weight = torch.from_numpy(np.array([[0.2, 0.2, 0.4, 0.2]])).float()
#
#  num_workers = config['num_workers']
#  train_data_train = TrainDataset(config['train_data_dir'],
#                                  config['train_df_csv'],
#                                  config['labels_train_df_csv'])
#  train_batch_size = config['train_batch_size']
#  train_data_loader = DataLoader(dataset = train_data_train,
#  batch_size = train_batch_size, shuffle = True, num_workers = num_workers)
#
#  criterion = nn.CrossEntropyLoss(weight=weight).to(device)
#  ## initial lr
#  lr_rate = config['lr']
#  num_epochs = config['epoch']
#  #lr_rate = config['lr']
#  momentum = config['momentum']
#  #num_epochs = config['epoch']
#  ## add no bias decay
#  params = add_weight_decay(model, 4e-5)
#  optimizer = optim.SGD(params, lr=lr_rate, momentum=momentum)
#  model, optimizer=amp.initialize(model, optimizer)
#  model = nn.DataParallel(model)
#  client = FL_Client(client_json_path = client_json_path, model=model)
#
#  #model_parameters = model.state_dict()
#  #model_parameters_dict = collections.OrderedDict()
#  #for key, value in model_parameters.item():
#  #    model_parameters_dict[key] = torch.numel(value), value.shape
# # scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epoch)
# # scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=10, after_scheduler=scheduler_cosine)
#
#  iter_per_epoch=len(train_data_loader)
#  warm_epoch=5
#  warmup_scheduler=WarmUpLR(optimizer,iter_per_epoch*warm_epoch)
#  train_scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs-warm_epoch)
#
#  logfile = "./train_valid.log"
#  if os.path.exists(logfile):
#      os.remove(logfile)
#  sys.stdout = Logger(logfile)
#  # sets.phase = 'train'
#  log = logging.getLogger()
#
#  pk, sk = KeyGen()
#  shape_parameters = torch.load("../shape_parameter.pth")
#  client_number = 3
#  last_version = "./model_state/initial.pth"
#
#  while True:
#      client.registry()
#      while True:
#          filename = client.receive()
#          if len(filename) == 2:
#              break
#          else:
#              time.sleep(20)
#
#      #from download.model import densenet3d
#      name = './download/' + filename[1].split('/')[-1]
#      print(name)
#      if os.path.exists(name):
#          aggregation = torch.load(name)
#          decrypted_params = client.decrypt(aggregation)
#          last_model = torch.load(last_version)
#          update_model = dict()
#         # print(decrypted_params.keys())
#          for key in last_model.keys():
#              update_model[key] = last_model[key] + decrypted_params[key]
#          #print(decrypted_params)
#          model.load_state_dict(update_model)
#          torch.save(update_model, './model_state/' + filename[1].split('/')[-1])
#          last_version = './model_state/' + filename[1].split('/')[-1]
#      else:
#          print('error when download initial weight from server')
#          sys.exit(0)
#
#      epoch = epoch + 1
#
#      update_name = train(name.split("/")[-1],
#      device, train_data_loader, model, optimizer, log, warm_epoch, epoch, warmup_scheduler, train_scheduler)
#      # update_name = last_version
#      original_state = torch.load(last_version)
#      update_state = torch.load(update_name)
#      diff_state = {}
#
#      for key in original_state.keys():
#          init = update_state[key] - original_state[key]
#          diff_state[key] = init
#
#      encrypt_params = client.encrypt(diff_state)
#      diff_name = config_client['username'] + '_differ.pth'
#      path = './checkpoint/' + diff_name
#      torch.save(encrypt_params, path)
#
#      client.send(path)
