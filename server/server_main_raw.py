<<<<<<< HEAD
import logging
import time
from time import sleep

from server.fl_server import FL_Server

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    server = FL_Server('./config/config.json')
    server.start()

    for epoch_num in range(server.configs['iteration']):
        logger.info('******* current epoch is {} *******'.format(epoch_num))

        current_time = time.time()
        while True:
            sleep(10)
            n_running = server.count_status(0)
            n_finish = server.count_status(1)
            logger.info("[epoch " + str(epoch_num) + "] n_running: " + str(n_running) + " , n_finish: " + str(n_finish))
            if n_finish > 0 and (n_running == 0 or (time.time() - current_time) > 50):
                break

        # Aggregation Here
        server.lock.acquire()
        try:
            logger.info("Start aggregation")

            # ****
            sleep(10) # Use the sleep function to simulate the aggregation process
            # ****


            # client status reset to -1 after aggregation
            for username in server.clients_status:
                server.clients_status[username] = -1
            logger.info("The aggregation is complete!")
        finally:
            server.lock.release()

    logger.info("The training is complete!")
    server.stop()
=======
import logging
import time
from time import sleep

from server.fl_server import FL_Server

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    server = FL_Server('./config/config.json')
    server.start()

    for epoch_num in range(server.configs['iteration']):
        logger.info('******* current epoch is {} *******'.format(epoch_num))

        current_time = time.time()
        while True:
            sleep(10)
            n_running = server.count_status(0)
            n_finish = server.count_status(1)
            logger.info("[epoch " + str(epoch_num) + "] n_running: " + str(n_running) + " , n_finish: " + str(n_finish))
            if n_finish > 0 and (n_running == 0 or (time.time() - current_time) > 50):
                break

        # Aggregation Here
        server.lock.acquire()
        try:
            logger.info("Start aggregation")

            # ****
            sleep(10) # Use the sleep function to simulate the aggregation process
            # ****


            # client status reset to -1 after aggregation
            for username in server.clients_status:
                server.clients_status[username] = -1
            logger.info("The aggregation is complete!")
        finally:
            server.lock.release()

    logger.info("The training is complete!")
    server.stop()
>>>>>>> 818be9365f619dae0b8228c2d7d4156d2988de7b
