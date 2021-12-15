import torch

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer

# prepare dataset for detectron2
register_coco_instances("my_dataset_train", {}, "./mmdetection/data/coco/annotations/train_coco.json", "./mmdetection/data/coco/train2017")
register_coco_instances("my_dataset_val", {}, "./mmdetection/data/coco/annotations/val_coco.json", "./mmdetection/data/coco/val2017")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",) # training dataset that use register_coco_instances() to generate
cfg.INPUT.MASK_FORMAT = "bitmask" # important, if you are use RLE in annotation, must have change to this parameter
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
#cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.CHECKPOINT_PERIOD = 200 # Every how many iteration will save model weight
cfg.SOLVER.BASE_LR = 0.00025  # learning rate
cfg.SOLVER.MAX_ITER = 10000    # number of iterations
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # default: 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class 

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True) # log and weight will put in output/
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()