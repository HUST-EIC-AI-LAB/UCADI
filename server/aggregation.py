import os
import torch

# for loading the encrypted file, import this.
# from LWE_based_PHE.cuda_test import KeyGen, Enc, Dec
# from server_data.model import densenet3d

## This file will all the weights in the weights directory


def getWeightList(weights_store_directory, map_loc=torch.device('cuda')):
	"""
    weights_store_directory will Two type .pth file:
        _model_param : named as model_Name_Version.pth, store the model state_dict and client weight from the client
        _model_param = {"model_state_dict":[...state list...],  "client_weight": tensor_num}

    ATTENTION: After each aggregation, the weights_store_directory must be flushed.
    :param weights_store_directory:
    :param map_loc:
    :return:
    """
	fileList = os.listdir(weights_store_directory)

	client_num = len(fileList)
	weightDictList = []
	final_weight_sum = 0
	for file in fileList:
		file_path = os.path.join(weights_store_directory, file)
		_model_param = torch.load(file_path, map_location=map_loc)
		final_weight_sum += _model_param['client_weight']
		print('in func, client-X model_weight:', final_weight_sum)

		temp_state_dict = _model_param['model_state_dict']
		weightDictList.append(temp_state_dict)

	return weightDictList, final_weight_sum, client_num


def aggregateWeight(weightDictList):
	"""
    :param weightDictList: transmitted diff
    :return:
    """

	print('length of dict:', len(weightDictList[0]))
	length = len(weightDictList)
	# keyList = weightDictList[0].keys()
	new_dict = []

	for index in range(len(weightDictList[0])):
		ini_tensor = weightDictList[0][index]

		for i in range(1, length):
			ini_tensor += weightDictList[i][index]

		new_dict.append(ini_tensor)

	return new_dict


def weightSave(weights_direc, savePath, map_loc=torch.device('cpu')):
	# get weight dicts list
	weight_dict_list = getWeightList(weights_direc, map_loc=map_loc)

	# average aggregation
	res_state_dict = aggregateWeight(weight_dict_list)

	# save
	torch.save(res_state_dict, savePath)


if __name__ == '__main__':
	# This file will all the weights in the weights directory
	weights_direc = './model/client_model/'
	# directory storing weights
	# weights_direc = cfg_server['weight_directory']
	#
	# print(weights_direc)
	# print(os.listdir(weights_direc))
	#
	# # model = densenet3d().to('cpu')
	# snapshot = './weight.pth'
	# state_dict = torch.load(snapshot, map_location=torch.device('cpu'))
	# model.load_state_dict(state_dict)
	# state_dict = torch.load('./model/merge_model/initial.pth', map_location=torch.device('cpu'))
	# print(type(state_dict))
	# print(len(state_dict))
	# print(state_dict.keys())
	# test_initial = []
	# for i in range(10):
	#    test_initial.append(torch.tensor(i))

	# state = torch.load('./model/merge_model/initial.pth')
	# print(len(state))
	# print(state[1])
	# _model_param = {'model_state_dict': test_initial,
	#                'client_weight':torch.tensor([1])}
	# torch.save(_model_param, './model/merge_model/initial.pth')
	state_dict_list, weight_num, _ = getWeightList(weights_direc)
	aggre = aggregateWeight(state_dict_list)
	torch.save(aggre, './aggregated.pth')
	# # get weight dicts list
	# weight_dict_list, weight_sum = getWeightList(weights_direc, map_loc = torch.device('cpu'))
	# print('weight sum:', weight_sum)
	# #
	# # # average aggregation
	# res_state_dict = aggregateWeight(weight_dict_list,)
	# print(res_state_dict)

	# res_model = densenet3d().to('cpu')
	# res_model.load_state_dict(res_state_dict)

	# for pram in model.parameters():
	#     # print(pram/2)
	#     print(pram)

	# for para in res_model.parameters():
	#     print(para)
	#     break
	# print('******************************************************\n')
	# print('******************************************************\n')
	# print('******************************************************\n')
	#
	# for para in model.parameters():
	#     print(para)
	#     break

	# w1 = torch.tensor([0.3])
	# w2 = torch.tensor([0.7])
	# torch.save(w1, './server_data/client_model/model_weight_Alan_v1.pth')
	# torch.save(w2, './server_data/client_model/model_weight_Bob_v1.pth')
