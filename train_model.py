import argparse
import datetime
import os
import time
import shutil

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler

from utils.dataloaders.dataloader_train import ImageFolderTrain
from utils.dataloaders.transforms import TransformPair
from utils.model_library import *


def parse_args():
    parser = argparse.ArgumentParser(description='trains a CNN to find seals in satellite imagery')
    parser.add_argument('--t_dir', type=str, help='base directory to recursively search for images in')
    parser.add_argument('--model_arch', type=str, help='model architecture, must be a member of models '
                                                       'dictionary')
    parser.add_argument('--hyp_set', type=str, help='combination of hyperparameters used, must be a member of '
                                                    'hyperparameters dictionary')
    parser.add_argument('--out_name', type=str,
                        help='name of output file from training, this name will also be used in '
                             'subsequent steps of the pipeline')
    parser.add_argument('--models_dir', type=str, default='saved_models', help='folder where the model will be saved')
    return parser.parse_args()


def lr_find(model, dataloader, optimizer):
    return None
    # TODO learning rate finder


def save_checkpoint(filename, state, is_best_loss):
    torch.save(state, filename + '.tar')
    if is_best_loss:
        shutil.copyfile(filename + '.tar', filename + '_best_loss.tar')


def train_model(model, dataloader, criterion, optimizer, scheduler, num_epochs,
                output_name, num_cycles):
    """

    :param model:
    :param data:
    :param criterion:
    :param optimizer:
    :param scheduler:
    :param num_epochs:
    :param num_cycles:
    :return:
    """
    # keep track of training time
    since = time.time()

    # create summary writer with tensorboardX
    writer = SummaryWriter(log_dir='./tensorboard_logs/{}_{}'.format(output_name, str(datetime.datetime.now())))

    # keep track of iterations
    global_step = 0

    # keep track of best loss
    best_loss = 10E8

    # set cuda
    use_gpu = torch.cuda.is_available()

    # run training cycles
    for cycle in range(num_cycles):
        # reset learning rate
        optim_cycle = optimizer

        # each cycle has n epochs
        for epoch in range(num_epochs):
            epoch_loss = 0
            exp_avg_loss = 0
            # training and validation loops
            for phase in ["training", "validation"]:
                for iter, data in enumerate(dataloader[phase]):
                    if phase == "training":
                        # add global step
                        global_step += 1

                        # zero gradients
                        optim_cycle.zero_grad()

                        # step with scheduler
                        scheduler.step()

                        # get input data
                        input_img, target_img = data

                        if use_gpu:
                            input_img, target_img = input_img.cuda(), target_img.cuda()

                        # get model predictions
                        preds = model(input_img)

                        # get loss
                        loss = criterion(preds.view(preds.numel()), target_img.view(target_img.numel()))
                        exp_avg_loss = 0.99 * exp_avg_loss + 0.1 * (loss.item / len(preds))

                        # update parameters
                        loss.backward()
                        optim_cycle.step()

                        # save stats
                        if iter > 0 and iter % 100 == 0:
                            writer.add_scalar("training loss", exp_avg_loss)

                    else:
                        with torch.no_grad():
                            # get input data
                            input_img, target_img = data

                            # get model predictions
                            preds = model(input_img)

                            # get loss
                            loss = criterion(preds, target_img)
                            epoch_loss += loss.item() / len(preds)

            if phase == "validation":
                epoch_loss /= (len(dataloader["validation"]))
                writer.add_scalar("validation loss", epoch_loss)
                is_best_loss = epoch_loss < best_loss
                best_loss = min(epoch_loss, best_loss)
                save_checkpoint(model.state_dict(), is_best_loss)

    return model


def main():
    # unroll arguments
    args = parse_args()
    hyp_set = args.hyp_set

    # set cuda
    use_gpu = torch.cuda.is_available()

    # augmentation
    patch_size = model_archs[args.model_arch]
    data_transforms = {
        'training': TransformPair(patch_size, train=True),
        'validation': TransformPair(patch_size, train=False)
    }

    # load images
    image_datasets = {x: ImageFolderTrain(root=os.path.join(args.t_dir, x),
                                          transform=data_transforms[x])
                      for x in ['training', 'validation']}

    dataloaders = {"training": torch.utils.data.DataLoader(image_datasets["training"],
                                                           batch_size=
                                                           hyperparameters[hyp_set]['batch_size_train'],
                                                           num_workers=
                                                           hyperparameters[hyp_set][
                                                               'num_workers_train']),
                   "validation": torch.utils.data.DataLoader(image_datasets["validation"],
                                                             batch_size=
                                                             hyperparameters[hyp_set]['batch_size_val'],
                                                             num_workers=
                                                             hyperparameters[hyp_set][
                                                                 'num_workers_val'])
                   }

    model = model_defs(args.model_arch)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.params(), lr=10E-2, weight_decay=0.1)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, len(dataloaders["training"] *
                                                              hyperparameters[hyp_set["batch_size_train"]]))

    if use_gpu:
        model = model.cuda()
        criterion = criterion.cuda()

    train_model(model=model, dataloader=dataloaders, criterion=criterion,
                optimizer=optimizer, scheduler=scheduler, num_cycles=3,
                num_epochs=3, output_name='coco')


if __name__ == "__main__":
    main()
