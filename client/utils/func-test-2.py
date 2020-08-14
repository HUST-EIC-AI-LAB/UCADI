# -*- coding: utf-8 -*-
import torch
from common.LWE_based_PHE.cuda_test import KeyGen, Enc, Dec

if __name__ == "__main__":
    # fileName = './server_data_four/weight_v1.pth'
    fileName = '../client2/model/model_Param_Alan.pth'
    print("loading {}".format(fileName))
    mm = torch.load(fileName)
    enc = mm['model_state_dict']
    dec = Dec(enc)
    temp_key = list(dec.keys())[0]
    print(dec[0])
    
    print("length of state: {}".format(len(mm)))
    print("type of the loaded file is {}".format(type(mm)))
    print(mm['client_weight'])
    print(mm['model_state_dict'][0])
