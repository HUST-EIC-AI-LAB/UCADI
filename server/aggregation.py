# -*- coding: utf-8 -*-
"""aggregate all the encrypted weights in the weights directory"""
import os
import pdb
import torch

# for loading the encrypted file, import the following:
# from LWE_based_PHE.cuda_test import KeyGen, Enc, Dec

def getWeightList(weights_store_directory, map_loc=torch.device('cuda')):
    """
    weights_store_directory contains .pth files with following contents:
        {"model_state_dict": model weights,
         "client_weight": tensor_num}

    ATTENTION: After each aggregation, the weights_store_directory will be flushed.
    """
    fileList = os.listdir(weights_store_directory)
    weightDictList, weight_sum, client_num = [], 0, len(fileList)

    for file in fileList:
        file_path = os.path.join(weights_store_directory, file)
        # the client's weights are set as the amount of training data
        _model_param = torch.load(file_path, map_location=map_loc)
        weight_sum += _model_param['client_weight']
        weightDictList.append(_model_param['model_state_dict'])

    return weightDictList, weight_sum, client_num

def aggregateWeight(weightDictList):
    """Aggregate the Weights by Simple Summation"""
    
    # print('length of dict:', len(weightDictList[0]))
    length, new_dict = len(weightDictList), []

    for index in range(len(weightDictList[0])):
        ini_tensor = weightDictList[0][index]
        for i in range(1, length):
            ini_tensor += weightDictList[i][index]
        new_dict.append(ini_tensor)

    return new_dict

def weightSave(weights_direc, origin, savePath, map_loc=torch.device('cpu')):

    weight_dict_list = getWeightList(weights_direc, map_loc=map_loc)
    res_state_dict = aggregateWeight(weight_dict_list, origin)
    torch.save(res_state_dict, savePath)


if __name__ == '__main__':

    weights_store_directory = './model/client_model/'
    state_dict_list, weight_num, _ = getWeightList(weights_store_directory)
    aggre = aggregateWeight(state_dict_list)
    torch.save(aggre, './aggregated.pth')
