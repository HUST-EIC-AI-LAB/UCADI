import collections
import copy
import os

import torch

from back_up.LWE_based_PHE.cuda_test import KeyGen, Enc, Dec


def encrypt(public_key, model_weight):
    """
    params:
    model_weight: torch.nn.Module.state_dict()
    return: list(encrypted params)
    Due to the max length is 65536, so we cut each weight to a fixed size = 65536,
    so that one tensor could to cut to many.
    """
    prec = 32
    bound = 2 ** 3
    #model = torch.load(model_weight)

    # if not os.path.exists(shape_path):
    # 	model_parameters_dict = collection.OrderedDict()
    # 	for key, value in model:
    # 		model_parameters_dict[key] = torch.numel(value), value.shape
    # 	torch.save(model_parameters_dict, shape_path)

    # shape_parameter =  torch.load(shape_path)
    params_list = list()
    for key, value in model_weight.items():
        length = torch.numel(value) // 65536
        params = copy.deepcopy(value).view(-1).float()
        for ind in range(length):
                params_list.append(params[ind*65536 : (ind+1)*65536])
        params_list.append(params[length*65536 : ])

    params_list = [((params + bound) * 2 ** prec).long().cuda() for params in params_list]

    encrypted_params = [Enc(public_key, params) for params in params_list]

    return encrypted_params


def decrypt(private_key, encrypted_params, shape_parameter):
    """
    params:
    encrypted_params: list()
    shape_parameter: dict(), shape of each layer about model.
    return: decrypted params, torch.nn.Module.state_dict()
    """
    prec = 32
    bound = 2 ** 3
    dencrypted_params = [(Dec(private_key, params).float() / (2 ** prec) - bound) for params in encrypted_params]

    ind = 0
    weight_params = dict()
    for key in shape_parameter.keys():
            params_size, params_shape = shape_parameter[key]
            length = params_size // 65536
            #print(length)
            dencrypted = list()
            for index in range(length):
                    dencrypted.append(dencrypted_params[ind])
                    ind += 1
            dencrypted.append(dencrypted_params[ind][0 : (params_size - length*65536)])
            ind += 1
            weight_params[key] = torch.cat(dencrypted, 0).reshape(params_shape)

    #torch.save(weight_params, 'weight_encrypted.pth')
    return weight_params


def generate_shape(path, model):
    """
    Record the concret size of each layer about model.
    It will be used to decrypt.
    """
    #print(model)
    if not os.path.exists(path):
        model_parameters_dict = collections.OrderedDict()
        for key, value in model.items():
            model_parameters_dict[key] = torch.numel(value), value.shape
            torch.save(model_parameters_dict, path)


if  __name__ == '__main__':
    pk, sk = KeyGen()
    model = torch.load("./model_state/initial.pth")
    generate_shape("../shape_parameter.pth",model)
    shape_parameter = torch.load("../shape_parameter.pth")
    print(model)
    diff_model = dict()
    for key in model.keys():
        diff_model[key] = model[key] - model[key]
    encrypt_params = encrypt(pk, diff_model)
    #print(encrypt_params)
    #sum_encrypted_params = [(params + params) for params in encrypt_params]
    decrypt_params = decrypt(sk, encrypt_params, shape_parameter)
    #print(decrypt_params)
    #model = torch.load("weight_encrypted.pth")
    torch.save(encrypt_params, "initial.pth")
    #correct = 0
    #count = 0
    #for key in model.keys():
    #	correct += torch.sum(torch.eq(model[key], dencrypt_params[key])).item()
    #	count += torch.numel(model[key])
    #acc = float(correct / count)
    #print('Accuracy: %.2f%%' % (acc * 100))





