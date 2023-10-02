DEBUG = False
from fastai.vision.all import *
import params

import wandb

wandb.login(anonymous="allow")

params.WANDB_PROJECT

params.ENTITY

URL = 'https://storage.googleapis.com/wandb_course/bdd_simple_1k.zip'

path = Path(untar_data(URL, force_download=True))

path.ls()