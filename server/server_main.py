# -*- coding: utf-8 -*-
import os
import sys
import time
import logging
from os.path import abspath, dirname
base_dir = dirname(abspath(__file__))
sys.path.append(dirname(base_dir))
from fl_server import FL_Server

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # some configs
    server = FL_Server('./config/config.json')
    server.set_map_loc('cpu')
    server.start()

    model_path = server.configs['weight_path']

    for epoch_num in range(server.configs['iteration']):
        logger.info('***** current epoch is {} *****'.format(epoch_num))

        # set max delay
        server.max_delay = 100
        current_time = time.time()
        while True:
            sleep(10)
            n_running = server.count_status(0)
            n_finish = server.count_status(1)
            logger.info("[epoch " + str(epoch_num) + "] n_running: " + str(n_running) + " , n_finish: " + str(n_finish))
            if n_finish > 0 and (n_running == 0 or (time.time() - current_time) > server.max_delay):
                logger.info("Not received all Clients' parameters, Aggregate Early")
                break

        print("model path:{} will be sent".format(server.model_path))

        # model aggregation
        server.lock.acquire()
        try:
            logger.info("Begin Models Aggregation......")

            # begin to aggregate
            server.set_map_loc('cuda')
            new_param, weight_sum, client_num = server.aggregation(client_models_dir='./model/client_model/')
            print('weight sum:', weight_sum)
            # print("aggregated param[0]", new_param[0])
            print("client num is {}".format(client_num))

            # store then send
            new_model_path = os.path.join(server.configs['merge_model_dir'],
                                          "merge_model_Param_v{}.pth".format(epoch_num + 1))
            server.pack_param(_model_state=new_param, _client_weight=weight_sum, _client_num=client_num,
                              save_path=new_model_path)
            model_path = new_model_path
            print("packed model param saved at:", new_model_path)

            # update the model path will be sent at next iterations
            server.model_path = model_path
            # clean up all .pth file at ./model/client_model
            server.flush_client_weight_dir()

            # after aggregation, all status of client will be set to -1
            for username in server.clients_status:
                server.clients_status[username] = -1
            logger.info("Aggregation Over !")

        finally:
            server.lock.release()

    logger.info("All training finished！")
    server.stop()
