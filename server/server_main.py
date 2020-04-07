import numpy as np
import torch
import torchvision
import subprocess
import json
import time
from aggregation import *
import shutil
# When server gather enough data from clients, begin aggregating.
# Before the aggregation finish, no client could join in .
# register -->
# server_send --> client_recv --> client_send --> server_recv

with open('./server_config.json', 'r') as j:
    cfg_server = json.load(j)

# initialize the model, after the some epchs of train
# begin to send the model and the weight to the clients.
# Count the clients num, so as to

if __name__ == '__main__':
    # begin register process
    proc_serv_regi = subprocess.Popen(['python', 'server_register.py'])

    # begin to send the model to each clients
    # start the sending service
    proc_serv_send = subprocess.Popen(['python', 'server_send.py'])

    # begin to receive the updated modelf from each clients
    # start the receiving process
    proc_serv_recv = subprocess.Popen(['python', 'server_recv.py'])

    print("parent process...\n")

    # FL process
    for epch_num in range(cfg_server['max_iterations']):
        # after max_iterations, the FL process will be ended

        # count the clients which has downloaded the model
        # if number is exceed the min_num &
        curr_time = time.time()
        while True:
            with open(cfg_server["recv_state_json_path"], 'r') as j:
                state_dict = json.load(j)

            len_dict = len(state_dict)

            # if there's no received file, waiting to the first
            if len_dict == 0:
                print('have not send a model')
                time.sleep(10)
                continue

            valuesSum = sum(state_dict.values())

            print("current recv:", valuesSum)

            # if all clients have updated their model
            if valuesSum == len_dict:
                break

            # some clients may not updated their model
            if valuesSum >= cfg_server["min_clients"]:
                # bigger than the min clients num to begin thee fl
                _delay_from = time.time() - curr_time
                print("_delat_from")
                if _delay_from >= cfg_server['max_delay']:
                    break
                else:
                    time.sleep(10)
            else:
                time.sleep(10)

        print('begin to update the local model...\n')

        # doing aggregation
        weights_direc = cfg_server["weight_directory"]

        weight_dict_list = getWeightList(weights_direc, map_loc=torch.device('cuda'))
        # need to set the param "map_loc"
        # map_loc = torch.device('cpu') (default) / 'gpu'

        # average aggregation
        origin_state_dict = torch.load(cfg_server["weight"], map_location=torch.device('cuda'))
        res_state_dict = aggregateWeight(weight_dict_list, origin=origin_state_dict)
        name = "./server_data/weight_v" + str(epch_num + 1) + ".pth" 
        torch.save(res_state_dict, name)
        # eg. "./server_store/aggregated_weights_v1.pth"
        cfg_server["weight"] = name
        with open('./server_config.json', 'w') as j:
            json.dump(cfg_server, j)


        # if the accuracy of new model is qualified, early stop the FL.

        # store the new weights, and named it with v1, v2 etc.

        # clean up the state dict
        state_dict = {}
        with open(cfg_server["recv_state_json_path"], 'w') as j:
            json.dump(state_dict, j)
        shutil.rmtree("./client_data")
        os.mkdir("./client_data")

    proc_serv_regi.kill()
    proc_serv_recv.kill()
    proc_serv_send.kill()

    print('all ended...')





