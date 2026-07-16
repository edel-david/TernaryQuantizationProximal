import logging
from datetime import datetime
import os
def init_loger_and_folder(run_name):
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger.setLevel(logging.DEBUG)
    # --- Console handler ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Print INFO and above
    console_handler.setFormatter(formatter)

    # --- File handler ---
    start_time_string = datetime.now().strftime('%Y_%m_%d_%H.%M.%S')
    log_filename = f"./logs/{start_time_string}.log"
    file_handler = logging.FileHandler(log_filename, mode='a')
    file_handler.setLevel(logging.DEBUG)  # Save DEBUG and above
    file_handler.setFormatter(formatter)

    # remove handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        if hasattr(handler, 'close'):
            handler.close()

    # --- Add handlers to logger ---
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.warning("Starting logging...")
    folder =  f"runs/{start_time_string}_{run_name}"
    os.makedirs(folder, exist_ok=True)
    return logger, folder