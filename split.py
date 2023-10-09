import os, warnings
import wandb

import pandas as pd
from fastai.vision.all import *
from sklearn.model_selection import StratifiedGroupKFold

import params
warnings.filterwarnings("ignore")

wandb.login(anonymous="allow")

run = wandb.init(project=params.WANDB_PROJECT, entity=params.ENTITY, job_type="data_split")

raw_data_at = run.use_artifact(f'{params.RAW_DATA_AT}:latest')
path = Path(raw_data_at.download())

path.ls()

fnames = os.listdir(path/'images')
groups = [s.split('_')[0] for s in fnames]

orig_eda_table = raw_data_at.get('eda_table')

y = orig_eda_table.get_column('bicycle')

df = pd.DataFrame()
df['File_Name'] = fnames
df['fold'] = -1

cv = StratifiedGroupKFold(n_splits=10)
for i, (train_idxs, test_idxs) in enumerate(cv.split(fnames, y, groups)):
    df.loc[test_idxs, ['fold']] = i
    
df['Stage'] = 'train'
df.loc[df.fold == 0, ['Stage']] = 'test'
df.loc[df.fold == 1, ['Stage']] = 'valid'
del df['fold']
df.Stage.value_counts()

df.to_csv('data_split.csv', index = False)

processed_data_at = wandb.Aftifact(params.PROCESSED_DATA_AT, type="split_data")

processed_data_at.add_file('data_split.csv')
processed_data_at.add_dir(path)

data_split_table = wandb.Table(dataframe=df[['File_Name', 'Stage']])

join_table = wandb.JoinedTable(orig_eda_table, data_split_table, "File_Name")

processed_data_at.add(join_table, "eda_table_data_split")

run.log_artifact(processed_data_at)
run.finish()
