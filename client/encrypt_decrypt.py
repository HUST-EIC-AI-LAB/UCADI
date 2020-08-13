# -*- coding: utf-8 -*-
import os
import copy
import torch
import collections
from common.LWE_based_PHE.cuda_test import KeyGen, Enc, Dec


def encrypt(public_key, model_weight):
    """
    params:
    model_weight: torch.nn.Module.state_dict()
    return: list(encrypted params)
    Due to the max length is 65536, so we cut each weight to a fixed size = 65536,
    so that one tensor could to cut to many.
    """
    prec, bound = 32, 2 ** 3
    params_list = list()
    for key, value in model_weight.items():
        length = torch.numel(value) // 65536
        params = copy.deepcopy(value).view(-1).float()
        for ind in range(length):
                params_list.append(params[ind*65536: (ind+1)*65536])
        params_list.append(params[length*65536:])

    params_list = [((params + bound) * 2 ** prec).long().cuda() for params in params_list]
    encrypted_params = [Enc(public_key, params) for params in params_list]

    return encrypted_params


def decrypt(private_key, encrypted_params, num, shape_parameter):
    """
    params:
    encrypted_params: list()
    shape_parameter: dict(), shape of each layer about model.
    return: decrypted params, torch.nn.Module.state_dict()
    """
    prec = 32
    bound = 2 ** 3
    decrypted_params = [(Dec(private_key, params).float() / (2 ** prec) / num - bound) 
        for params in encrypted_params]

    ind = 0
    weight_params = dict()
    for key in shape_parameter.keys():
            params_size, params_shape = shape_parameter[key]
            length = params_size // 65536
            decrypted = list()
            for index in range(length):
                    decrypted.append(decrypted_params[ind])
                    ind += 1
            decrypted.append(decrypted_params[ind][0:(params_size - length*65536)])
            ind += 1
            weight_params[key] = torch.cat(decrypted, 0).reshape(params_shape)

    return weight_params


def generate_shape(path, model):
    """
    Record the concrete size of each layer about model.
    It will be used to decrypt.
    """
    if not os.path.exists(path):
        model_parameters_dict = collections.OrderedDict()
        for key, value in model.items():
            model_parameters_dict[key] = torch.numel(value), value.shape
            torch.save(model_parameters_dict, path)


if __name__ == '__main__':

    pk, sk = KeyGen()
    model0 = torch.load("weight.pth")
    generate_shape("../shape_parameter.pth", model0)
    shape_parameter = torch.load("../shape_parameter.pth")

    encrypt_params = torch.load('initial.pth')['model_state_dict']
    encrypt_params2 = torch.load('initial.pth')['model_state_dict']
    model1 = decrypt(sk, encrypt_params, 1, shape_parameter)
    model2 = decrypt(sk, encrypt_params2, 1, shape_parameter)

    newmodel0, newmodel1, newmodel2 = {}, {}, {}
    for key in model0.keys():
        newmodel0[key], newmodel1[key] = model0[key] * 3, model1[key] * 2
    encrypt_model0, encrypt_model1, encrypt_model2 \
        = encrypt(pk, newmodel0), encrypt(pk, newmodel1), encrypt(pk, model2)
    for i in range(len(encrypt_model1)):
        encrypt_model1[i] = encrypt_model1[i] + encrypt_model2[i] + encrypt_model0[i]

    decrypt_params = decrypt(sk, encrypt_model1, 3, shape_parameter)
    print(model0['module.classifier.bias'])
    print(model1['module.classifier.bias'])
    print(model2['module.classifier.bias'])
    print(decrypt_params['module.classifier.bias'])
