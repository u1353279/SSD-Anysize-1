"""

"""

import os
import shutil 
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.optim
import torch.utils.data

# pre process stuff
from utils import utils
from utils.datasets import PascalDataset
from pre_process.create_data_lists import create_data_lists
from pre_process.convert_validation_data_to_coco import convert_validation_data_to_coco
from pre_process.comb_for_labels import comb_for_labels
from utils.train_and_eval import train_and_eval
from config import CONFIG as CONFIG
from models.backbones import MobileNetV1, MobileNetV2


def pre_process(config):

    shutil.rmtree(config["training_path"])
    train_path = os.path.join(config["training_path"], "train")
    val_path = os.path.join(config["training_path"], "val")
    coco_dir = os.path.join(config["training_path"], "coco_eval")
    coco_json_file = os.path.join(coco_dir, "annotations.json")
    os.mkdir(config["training_path"])
    dirs = ["JPEGImages", "Annotations"]
    os.mkdir(train_path)
    [os.mkdir(os.path.join(train_path, d)) for d in dirs]
    os.mkdir(val_path)
    [os.mkdir(os.path.join(val_path, d)) for d in dirs]
    os.mkdir(coco_dir)
    with open(os.path.join(config["training_path"], "EMPTY"), "w+") as f:
        f.write("")


    utils.extract_zipped_data(f"source_data/{config['zip_dataset']}", "training_temp")

    annos = os.path.join(config["training_path"], "Annotations")
    config["classes"], class_distribution = comb_for_labels(
        [os.path.join(annos, x) for x in os.listdir(annos)])  

    utils.make_train_val_split(train_val_ratio=config["train_val_ratio"], 
                                training_directory=config["training_path"], 
                                train_folder=train_path, 
                                val_folder=val_path)

    # CONVERSIONS
    xmls_dir = os.path.join(val_path, "Annotations")
    xml_files = [os.path.join(xmls_dir, x) for x in os.listdir(xmls_dir)]
    convert_validation_data_to_coco(xml_files, coco_json_file, config["classes"])

    create_data_lists(train_path, config["training_path"], config["classes"])
    create_data_lists(val_path, config["training_path"], config["classes"])


def train(config):
    batch_size = config["batch_size"]
    config["training_path"] = config["training_path"]
    input_dims = config["input_dims"]

    # Create the data loaders
    train_dataset = PascalDataset(config["training_path"],
                                  split='train',
                                  model_input_dims=input_dims)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=4,
        pin_memory=True)  # note that we're passing the collate function here

    test_dataset = PascalDataset(config["training_path"],
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

    if CONFIG["backbone"] == "mobilenetv2":
        backbone = MobileNetV2(CONFIG["input_dims"])
        # mobilenet v2 weights come with Pytorch

    elif CONFIG["backbone"] == "mobilenetv1":
        backbone = MobileNetV1(CONFIG["input_dims"])

        # TODO: Not sure if this works, need to spend some time working with the checkpoints
        backbone.load_state_dict(
            torch.load("weights/mobilenet-v1-ssd-mp-0_675.pth"), strict=False)

    CONFIG["backbone_model"] = backbone

    pre_process(CONFIG)
    # train(CONFIG)
