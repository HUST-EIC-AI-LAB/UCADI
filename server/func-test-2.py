import torch
# from LWE_based_PHE.cuda_test import KeyGen, Enc, Dec

if __name__ == "__main__":
    # fileName = './server_data_four/weight_v1.pth'
    fileName = './model/merge_model/initial.pth'
    print("loading {}".format(fileName))
    mm = torch.load(fileName)
    for i in range(len(mm['model_state_dict'])):
        print(mm['model_state_dict'][i])
    # print("cuda type:",mm['model_state_dict'][0].is_cuda())
    print("length of state: {}".format(len(mm)))
    print("type of the loaded file is {}".format(type(mm)))
    print(mm['client_weight'])
    print(mm['model_state_dict'][0])
    
    print("keys:{}".format(mm.keys()))
    mm['client_num'] = 1
    print("keys:{}".format(mm.keys()))
    torch.save(mm, fileName) 
