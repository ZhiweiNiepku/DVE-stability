import logging
import os
import sys

def setup_logger(name, save_path="logs", log_name="train_log.txt", if_train=True):
    """
        using in main py script, such as "logger = setup_logger("train", output_dir, if_train=True)"
        in other scripy, using logger = logging.getLogger("train")
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if if_train:
            fh = logging.FileHandler(os.path.join(save_path, log_name), mode='w')
        else:
            fh = logging.FileHandler(os.path.join(save_path, log_name), mode='w')

        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger