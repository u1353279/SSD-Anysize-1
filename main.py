import os
import multiprocessing 

import torch
import torch.optim
import torch.utils.data
from torchsummary import summary

# pre process stuff
from utils.datasets import PascalDataset

try: 
    from source_dataset import source_dataset
except:
    source_dataset = ""

from pre_process.create_data_lists import create_data_lists
from pre_process.convert_validation_data_to_coco import convert_validation_data_to_coco

from train_and_eval import train_and_eval

config = {
    "training_path" : "./training_temp",
    "source_dataset_path" : source_dataset,
    "backbone" : "mobilenetv2", # vgg is also supported
    "input_dims" : (300,300),
    "classes" : ["person"],
    "device" : torch.device("cuda" if torch.cuda.is_available() else "cpu"), # can hard code to cpu if GPU doesn't have enough vram
    "batch_size" : 4,  # eval batch size is always 1 regardless of this setting
    "epochs" : 10,
    "learning_rate" : 0.0003,
    "weight_decay" : 0.00005,  # l2 regularization
    "detection_threshold" : 0.3,
    "save_results" : True,
    "save_results_path" : "./training_temp/out"
}

def pre_process(config):
    from pre_process.convert_tf_record import convert_tf_record

    training_temp_path = config["training_path"]
    dataset_path = config["source_dataset_path"]
    classes = config["classes"]

    if not os.path.exists(training_temp_path):
        os.mkdir(training_temp_path)

    train_path = os.path.join(training_temp_path, "train")
    val_path = os.path.join(training_temp_path, "val")
    coco_dir = os.path.join(training_temp_path, "coco_eval")
    coco_json_file = os.path.join(coco_dir, "annotations.json")

    # Making directories
    if not os.path.exists(train_path):
        os.mkdir(train_path)
        os.mkdir(os.path.join(train_path, "JPEGImages"))
        os.mkdir(os.path.join(train_path, "Annotations"))

    if not os.path.exists(val_path):
        os.mkdir(val_path)
        os.mkdir(os.path.join(val_path, "JPEGImages"))
        os.mkdir(os.path.join(val_path, "Annotations"))

    if not os.path.exists(coco_dir):
        os.mkdir(coco_dir)

    # CONVERSIONS
    xmls_dir = os.path.join(val_path, "Annotations")

    p1 = multiprocessing.Process(target=convert_tf_record,
                     args=(dataset_path, training_temp_path))
    p1.start()
    p1.join()

    xml_files = [os.path.join(xmls_dir, x) for x in os.listdir(xmls_dir)]
    convert_validation_data_to_coco(xml_files, coco_json_file, classes)  

    create_data_lists(train_path, training_temp_path, classes)
    create_data_lists(val_path, training_temp_path, classes)



def train(config):
    device = config["device"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    classes = config["classes"]
    training_temp_path = config["training_path"]
    input_dims = config["input_dims"]

    # Create the data loaders
    train_dataset = PascalDataset(training_temp_path, split='train', model_input_dims=input_dims)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=4,
                                               pin_memory=True)  # note that we're passing the collate function here
    
    test_dataset = PascalDataset(training_temp_path, split='test', model_input_dims=input_dims)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                            collate_fn=test_dataset.collate_fn, num_workers=4, 
                                            pin_memory=True)

    train_and_eval(config, train_loader, test_loader)

if __name__ == "__main__":
#     pre_process(config)
    train(config)