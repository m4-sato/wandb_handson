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

def label_func(fname):
    return (fname.parent.parent/"labels")/f"{fname.stem}_mask.png"

def get_classes_per_image(mask_data, class_labels):
    unique = list(np.unique(mask_data))
    result_dict = {}
    for _class in class_labels.keys():
        result_dict[class_labels[_class]] = int(_class in unique)
    return result_dict

def _create_table(image_files, class_labels):
    labels = [str(class_labels[_lab]) for _lab in list(class_labels)]
    table = wandb.Table(columns=["File_Name", "Images", "Split"] + labels)
    
    for i, image_file in progress_bar(enumerate(image_files), total=len(image_files)):
        image = Image.open(image_file)
        mask_data = np.array(Image.open(label_func(image_file)))
        class_in_image = get_classes_per_image(mask_data, class_labels)
        table.add_data(
            str(image_file.name), 
            wandb.Image(
                image,
                masks = {
                    "predictions": {
                        "mask_data": mask_data,
                        "class_labels": class_labels,
                    },
                },
            ),
            "None",
            *[class_in_image[_lab] for _lab in labels]
        )
    return table

run = wandb.init(project=params.WANDB_PROJECT, entity=params.ENTITY, job_type="upload")
raw_data_at = wandb.Artifact(params.RAW_DATA_AT, type="raw_data")

raw_data_at.add_file(path/"LICENSE.txt", name="LICENSE.txt")

raw_data_at.add_dir(path/"images", name="images")
raw_data_at.add_dir(path/"labels", name="labels")

image_files = get_image_files(path/"images", recurse=False)

if DEBUG:
    image_files = image_files[:10]

table = _create_table(image_files, params.BDD_CLASSES)

fnames = os.listdir(path/'images')

P1 = [s.split('-')[0] for s in fnames]
P2 = [s.split('-')[1] for s in fnames]

table.add_column("P1", P1)
table.add_column("P2", P2)

raw_data_at.add(table, "eda_table")

run.log_artifact(raw_data_at)
run.finish()

