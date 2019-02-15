"""
Model library
==========================================================
Script to keep track of model architectures / hyperparameter sets used in experiments for ICEBERG seals use case.
Author: Bento Goncalves
License: MIT
Copyright: 2019-2020
"""

from utils.model_architectures import *
from utils.dataloaders import *


model_archs = {'U-Net': 224}

model_defs = {'U-Net': DynamicUnet(encoder=models.resnet34(pretrained=True), n_classes=3)}

hyperparameters = {'A': {'learning_rate': 1E-3, 'batch_size_train': 64, 'batch_size_val': 8, 'batch_size_test': 64,
                         'step_size': 1, 'gamma': 0.95, 'epochs': 5, 'num_workers_train': 16, 'num_workers_val': 1}}

