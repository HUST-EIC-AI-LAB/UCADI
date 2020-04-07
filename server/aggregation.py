import torch
import torchvision
import json
import os
from server_data.model import densenet3d

## This file will all the weights in the weights directory

with open('./server_config.json', 'r') as j:
    cfg_server = json.load(j)

def getWeightList(weights_store_directory, map_loc = torch.device('cuda')):
    fileList = os.listdir(weights_store_directory)

    weightDictList = []
    print(fileList)
    for file in fileList:
        file_path = os.path.join(weights_store_directory, file)
 
        temp_state_dict = torch.load(file_path, map_location=map_loc)

        weightDictList.append(temp_state_dict)

    return weightDictList

def aggregateWeight(weightDictList, origin):
    """

    :param weightDictList: transmitted diff
    :param origin:   original weights from server
    :return:
    """
    length = len(weightDictList)
    keyList = weightDictList[0].keys()
    new_dict = {}

    for key in keyList:
        ini_tensor = origin[key]

        for i in range(length):
            ini_tensor += weightDictList[i][key]

        new_dict[key] = ini_tensor / length

    return new_dict

def weightSave(weights_direc, origin, savePath, map_loc = torch.device('cpu')):

    # get weight dicts list
    weight_dict_list = getWeightList(weights_direc, map_loc=map_loc)

    # average aggregation
    res_state_dict = aggregateWeight(weight_dict_list, origin)

    # save
    torch.save(res_state_dict, savePath)


if __name__ == '__main__':
    ## This file will all the weights in the weights directory

    # directory storing weights
    weights_direc = cfg_server['weight_directory']

    print(weights_direc)
    print(os.listdir(weights_direc))

    model = densenet3d().to('cpu')
    snapshot = './weight.pth'
    state_dict = torch.load(snapshot, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    # get weight dicts list
    weight_dict_list = getWeightList(weights_direc)

    # average aggregation
    res_state_dict = aggregateWeight(weight_dict_list, state_dict)

    res_model = densenet3d().to('cpu')
    res_model.load_state_dict(res_state_dict)

    # for pram in model.parameters():
    #     # print(pram/2)
    #     print(pram)

    for para in res_model.parameters():
        print(para)
        break
    print('******************************************************\n')
    print('******************************************************\n')
    print('******************************************************\n')

    for para in model.parameters():
        print(para)
        break
