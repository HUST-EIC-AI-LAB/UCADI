# -*- coding: utf-8 -*-
"""aggregate all the encrypted weights in the weights directory"""
import os
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
    weightDictList, weightList, client_num = [], [], len(fileList)

    for file in fileList:
        file_path = os.path.join(weights_store_directory, file)
        # weights are set as the clients' amount of training data
        _model_param = torch.load(file_path, map_location=map_loc)
        weightList.append(_model_param['client_weight'])
        weightDictList.append(_model_param['model_state_dict'])
    
    return weightDictList, weightList, client_num

def aggregateWeight(weightDictList, weightList):
    """Aggregate the Weights by Simple Summation (Multiplication of CypherText is not straightforward)
    weightDictList[i] -> i-th model states dict, all have same params order
    """
    new_dict, weight_sum = [], sum(weightList)

    for index in range(len(weightDictList[0])):
        ini_tensor = weightDictList[0][index]  # * weightList[0] / weight_sum
        for i in range(1, len(weightDictList)):
            ini_tensor += weightDictList[i][index]  # * weightList[i] / weight_sum
        new_dict.append(ini_tensor)

    return new_dict

def weightSave(weights_direc, savePath, map_loc=torch.device('cuda')):

    weightDictList, weightList, client_num = getWeightList(weights_direc, map_loc)
    res_state_dict = aggregateWeight(weightDictList, weightList)
    torch.save(res_state_dict, savePath)


if __name__ == '__main__':

    weights_store_directory = './model/merge_model'  # './model/client_model/'
    weightDictList, weightList, client_num = getWeightList(weights_store_directory)
    aggre = aggregateWeight(weightDictList, weightList)
    torch.save(aggre, './model/initial.pth')
