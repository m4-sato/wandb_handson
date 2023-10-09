import wandb
import pandas as pd
from fastai.vision.models as tvmodels
from fastai.vision.all import *
from fastai.callback.wandb import WandbCallback

import params
from utils import get_predictions, create_iou_table, MIOU, BackgroundIOU, RoadIOU,\
    TrafficLightIOU, TrafficSignIOU, PersonIOU, VehicleIOU, BicycleIOU
    
train_config = SimpleNamespace(
    framework="fastai",
    img_size=(180, 320),
    batch_size=8,
    augment=True,
    epochs=10, 
    lr = 2e-3, 
    arch="resnet18", 
    pretrained=True, 
    seed=42, 
    log_preds=True,
)

def lable_func(fname):
    return (fname.parent.parent/"labels")/f"{fname.stem}_mask.png"