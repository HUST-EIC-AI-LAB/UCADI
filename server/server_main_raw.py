# -*- coding: utf-8 -*-

#  Copyright (c) 2021. Jiefeng, Ziwei and Hanchen
#  jiefenggan@gmail.com, ziwei@hust.edu.cn, hc.wang96@gmail.com

import os
import sys
import time
import logging
sys.path.insert(0, os.path.join(sys.path[0], 'common/'))
from common import FL_Server

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    server = FL_Server('./config/config.json')
    server.set_map_loc('cpu')
    server.start()

    for epoch_num in range(server.configs['iteration']):
        logger.info('***** Current Epoch is {} *****'.format(epoch_num))
       
        server.max_delay, current_time = 100, time.time()  # set up max_delay time
       
        while True:
            time.sleep(10)
            n_running, n_finish = server.count_status(0), server.count_status(1)
            logger.info("[epoch " + str(epoch_num) + "] n_running: "
                        + str(n_running) + " , n_finish: " + str(n_finish))
            if n_finish > 1 and (n_running == 0 or (time.time() - current_time) > server.max_delay):
                logger.info("Timeout, not received all clients' parameters, Start Aggregation")
                break

        print("model path:{} will be sent".format(server.model_path))

        # model aggregation
        server.lock.acquire()
        try:
            logger.info("Begin Models Aggregation......")
            server.set_map_loc('cuda')
            new_param, weight_sum, client_num = server.aggregation(client_models_dir='./model/client_model/')
            print('weight sum:', weight_sum, " client num is {}".format(client_num))

            # store then send
            new_model_path = os.path.join(server.merge_weight_dir,
                                          "merge_model_Param_v{}.pth".format(epoch_num + 1))
            server.pack_param(_model_state=new_param, _client_weight=weight_sum, _client_num=client_num,
                              save_path=new_model_path)
            print("packed model param saved at:", new_model_path)

            server.model_path = new_model_path  # newly aggregated model, will be sent at next epoch
            server.flush_client_weight_dir()  # clean up all .pth file at ./model/client_model

            # after aggregation, all status of client will be set to -1
            for username in server.clients_status:
                server.clients_status[username] = -1
            logger.info("Aggregation Over!")

        finally:
            server.lock.release()

    logger.info("All training finished!")
    server.stop()
