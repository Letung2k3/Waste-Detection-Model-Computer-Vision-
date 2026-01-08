# training/train.py

import os
import logging
from ultralytics import YOLO
from config import TRAIN_CONFIG
import sys
from tb_callback import on_train_epoch_end
sys.stdout.reconfigure(encoding='utf-8')


def setup_logger(save_dir):
    log_file = os.path.join(save_dir, "train.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("TRAIN")



def train():

    model = YOLO(TRAIN_CONFIG["model"])
    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    project_dir = os.path.join(
        TRAIN_CONFIG["project"],
        TRAIN_CONFIG["name"]
    )
    os.makedirs(project_dir, exist_ok=True)
    logger = setup_logger(project_dir)

    logger.info("Start YOLOv8 Training")
    logger.info(f"Config: {TRAIN_CONFIG}")

    results = model.train(**TRAIN_CONFIG)

    logger.info("Training Finished")
    logger.info(f"Results saved at: {project_dir}")

    return results


if __name__ == "__main__":
    train()