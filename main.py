import os
import threading 

import torch
import torch.optim
import torch.utils.data

# pre process stuff
from utils.datasets import PascalDataset

from pre_process.create_data_lists import create_data_lists
from pre_process.convert_validation_data_to_coco import convert_validation_data_to_coco

from train_and_eval import train_and_eval

CONFIG = {
    "training_path": "./training_temp",
    "backbone": "mobilenetv2",  # vgg is also supported
    "input_dims": (300, 300),
    "classes": ["helmet", "head"],
    "device": torch.device("cuda" if torch.cuda.is_available(
    ) else "cpu"),  # can hard code to cpu if GPU doesn't have enough vram
    "batch_size": 2,  # eval batch size is always 1 regardless of this setting
    "epochs": 10,
    "learning_rate": 0.0003,
    "weight_decay": 0.00005,  # l2 regularization
    "detection_threshold": 0.3,
    "save_results": True,
    "save_results_path": "./training_temp/out"
}


def pre_process(config):

    # TODO: Right now I'm using a small dataset and splitting train-val manually. Do this w/ code later

    training_temp_path = config["training_path"]
    classes = config["classes"]

    if not os.path.exists(training_temp_path):
        os.mkdir(training_temp_path)

    train_path = os.path.join(training_temp_path, "train")
    val_path = os.path.join(training_temp_path, "val")
    coco_dir = os.path.join(training_temp_path, "coco_eval")
    coco_json_file = os.path.join(coco_dir, "annotations.json")

    # Making directories
    dirs = ["JPEGImages", "Annotations"]
    if not os.path.exists(train_path):
        os.mkdir(train_path)
        [os.mkdir(os.path.join(train_path, d)) for d in dirs]
        
    if not os.path.exists(val_path):
        os.mkdir(val_path)
        [os.mkdir(os.path.join(train_path, d)) for d in dirs]

    if not os.path.exists(coco_dir):
        os.mkdir(coco_dir)

    # CONVERSIONS
    xmls_dir = os.path.join(val_path, "Annotations")
    xml_files = [os.path.join(xmls_dir, x) for x in os.listdir(xmls_dir)]
    convert_validation_data_to_coco(xml_files, coco_json_file, classes)

    create_data_lists(train_path, training_temp_path, classes)
    create_data_lists(val_path, training_temp_path, classes)


def train(config):
    batch_size = config["batch_size"]
    training_temp_path = config["training_path"]
    input_dims = config["input_dims"]

    # Create the data loaders
    train_dataset = PascalDataset(training_temp_path,
                                  split='train',
                                  model_input_dims=input_dims)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=4,
        pin_memory=True)  # note that we're passing the collate function here

    test_dataset = PascalDataset(training_temp_path,
                                 split='test',
                                 model_input_dims=input_dims)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
        num_workers=4,
        pin_memory=True)

    train_and_eval(config, train_loader, test_loader)


if __name__ == "__main__":
    pre_process(CONFIG)
    train(CONFIG)
