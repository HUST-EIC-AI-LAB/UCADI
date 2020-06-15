import logging
from time import sleep
from client.fl_client import FL_Client

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    client = FL_Client('./config/client1_config.json')
    client.start()

    while True:
        logger.info('******* new  epoch *******')
        request_model_finish = False
        while True:
            # request and receive model
            request_model_result = client.request_model()
            if request_model_result == "ok":
                request_model_finish = True
                break
            elif request_model_result == "wait":
                sleep(10)
                continue
            elif request_model_result == "error":
                break
        if not request_model_finish:
            continue

        # train
        logger.info("Start training ...")
        sleep(15)
        logger.info("Training is finished!")

        # send model
        if client.send_model() != "ok":
            continue
        sleep(5)
