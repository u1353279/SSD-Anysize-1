import torch

# MAX SIZE: 800,800

CONFIG = {
    "training_path": "./training_temp",
    "backbone": "mobilenetv1",
    "backbone_model": None,  # Loads at runtime
    "input_dims": (300, 300),
    "classes": ["license_plate", "vehicle"],
    "device": torch.device("cpu"), # cpu or cuda:0
    "batch_size": 2, # eval batch size is always 1 regardless of this setting
    "epochs": 10,
    "learning_rate": 0.0003,
    "weight_decay": 0.00005,  # l2 regularization
    "detection_threshold": 0.3,
    "save_results": True,
    "save_results_path": "./training_temp/out",
    "zip_dataset": "dataset_9.zip",
    "train_val_ratio": 0.7
}