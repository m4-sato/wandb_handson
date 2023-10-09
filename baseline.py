import wandb
import pandas as pd
from fastai.vision.all import *
from fastai.callback.wandb import WandbCallback

import params
from utils import get_predictions, create_iou_table, MIOU, BackgroundIOU, RoadIOU,\
    TrafficLightIOU, TrafficSignIOU, PersonIOU, VehicleIOU, BicycleIOU
    
train_config = SimpleNamespace(
    framework="fastai",
    img_size=(180, 320),
    batch_size = 8,
    arguments=True,
    epochs=10,
    lr=2e-3,
    pretrained=True,
    seed=42,
)

set_seed(train_config.seed, reproducible=True)

wandb.login(anonymous="allow")

run = wandb.init(project=params.WANDB_PROJECT, entity=params.ENTITY, job_type="training", config=train_config)

processed_data_at = run.use_artifact(f'{params.PROCESSED_DATA_AT}:latest')
processed_dataset_dir = Path(processed_data_at.download())
df = pd.read_csv(processed_dataset_dir/'data_split.csv')

df = df[df.Stage != 'test'].reset_index(drop=True)
df['is_valid'] = df.Stage == 'valid'

def label_func(fname):
    return (fname.parent.parent/"labels")/f"{fname.stem}_mask.png"

def get_data(df, bs=4, img_size=(180, 320), augment=True):
    block = DataBlock(blocks=(ImageBlock, MaskBlock(codes=params.BDD_CLASSES)),
                    get_x = ColReader("image_fname"),
                    get_y = ColReader("label_fname"),
                    splitter=ColSplitter(),
                    item_tfms=Resize(img_size),
                    batch_tfms=aug_transforms() if augument else None,
                    )
block.dataloaders(df, bs=bs)

config = wandb.config
dls = get_data(df, bs = config.batch_size, img_size=config.img_size, augment=config.augment)

metrics = [MIOU(), BackgroundIOU(), RoadIOU(), TrafficLightIOU(), \
    TrafficSignIOU, PersonIOU, VehicleIOU(), BicycleIOU()]

learn = unet_learner(dls, arch=resnet18, pretrained=config.pretrained, metrics=metrics)

callbacks = [
    SaveModelCallback(monitor='miou'),
    WandbCallback(log_preds=False, log_model=True)
]

learn.fit_one_cycle(config.epochs, config.lr, cbs=callbacks)

samples, outputs, predictions = get_predictions(learn)
table = create_iou_table(samples, outputs, predictions, params.BDD_CLASSES) 
wandb.log({"pred_table": table})

scores = learn.validate()
metric_names = ['final_loss'] + [f"final_{x.name}" for x in metrics]
final_results = {metric_names[i] : scores[i] for i in range(len(scores))}
for k, v in final_results.items():
    wandb.summary[k] = v

wandb.finish()