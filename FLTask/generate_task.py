
from .models import *
from .data import data_distributer
import numpy as np
__all__ = ["create_models"]

MODELS = {
    "mnistcnn": FedAvgNetMNIST,
    "cifarcnn": FedAvgNetCIFAR,
    "tinycnn": FedAvgNetTiny,
}

NUM_CLASSES = {
    "mnist": 10,
    "cifar10": 10,
    "cifar100": 100,
    "cinic10": 10,
    "tinyimagenet": 200,
}


def create_models(model_name, dataset_name):
    """Create a network model"""

    num_classes = NUM_CLASSES[dataset_name]
    model = MODELS[model_name](num_classes=num_classes)

    return model

def generated_task(root,dataset_name,model_name,clients,batchsize,data_type,dir,shards):
    distributer = data_distributer(root,dataset_name,batchsize,clients,data_type,dir,shards)
    num_classes = NUM_CLASSES[dataset_name]
    model = MODELS[model_name](num_classes=num_classes)
    return distributer,model

