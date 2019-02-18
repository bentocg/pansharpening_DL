"""
Model library
==========================================================
Script to keep track of model architectures / hyperparameter sets used in experiments for ICEBERG seals use case.
Author: Bento Goncalves
License: MIT
Copyright: 2019-2020
"""
__all__ = {'model_archs', 'model_defs', 'hyperparameters', 'loss_functions'}
from utils.model_architectures import *
from utils.loss_functions import *
import torch.nn as nn

model_archs = {'U-Net': 224,
               'DynamicU-Net': 256}

model_defs = {'U-Net': UNet(scale=32, n_channels=4, n_classes=3)}

hyperparameters = {'A': {'learning_rate': 1E-3, 'batch_size_train': 32, 'batch_size_val': 8, 'batch_size_test': 64,
                         'step_size': 1, 'gamma': 0.95, 'epochs': 3, 'num_workers_train': 16, 'num_workers_val': 1},
                   'B': {'learning_rate': 1E-4, 'batch_size_train': 32, 'batch_size_val': 8, 'batch_size_test': 64,
                         'step_size': 1, 'gamma': 0.95, 'epochs': 3, 'num_workers_train': 16, 'num_workers_val': 1},
                   'C': {'learning_rate': 1E-5, 'batch_size_train': 32, 'batch_size_val': 8, 'batch_size_test': 64,
                         'step_size': 1, 'gamma': 0.95, 'epochs': 3, 'num_workers_train': 16, 'num_workers_val': 1}
                   }

loss_functions = {'BCE': nn.BCEWithLogitsLoss(),
                  'MSE': nn.MSELoss(),
                  'SL1': nn.SmoothL1Loss(),
                  'Dice': DiceLoss(),
                  'Focal': FocalLoss(gamma=2)

                  }