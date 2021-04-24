#!/usr/bin/env python3
#        This file is part of qdpy.
#
#        qdpy is free software: you can redistribute it and/or modify
#        it under the terms of the GNU Lesser General Public License as
#        published by the Free Software Foundation, either version 3 of
#        the License, or (at your option) any later version.
#
#        qdpy is distributed in the hope that it will be useful,
#        but WITHOUT ANY WARRANTY; without even the implied warranty of
#        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#        GNU Lesser General Public License for more details.
#
#        You should have received a copy of the GNU Lesser General Public
#        License along with qdpy. If not, see <http://www.gnu.org/licenses/>.


########## IMPORTS ########### {{{1
import gc
import copy
import pickle
import numpy as np
import warnings
import yaml

# Pytorch
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
#from torch.nn.utils import parameters_to_vector, vector_to_parameters
#from torch.autograd import Variable
#from torch.utils.data import DataLoader


# QDpy
import qdpy
from qdpy.base import *
from qdpy.experiment import QDExperiment
from qdpy.algorithms import LoggerStat, TQDMAlgorithmLogger, AlgorithmLogger, QDAlgorithmLike
from qdpy.containers import TorchAE, TorchFeatureExtractionContainerDecorator, ContainerLike, NoveltyArchive, Container
import qdpy.plots


import matplotlib.pyplot as plt

import curiosity
import nets
from nets import *
from metrics import *
import containers
from containers import *
#import sim
import main





########## TRAINING CLASSES ########### {{{1



def gather_data(config):
    with open(config.inputPath, "rb") as f:
        data = pickle.load(f)
    cont = data['algorithms'][0].container.container.parents[0]
    return data, cont



def create_trainer(config):
    trainer_type = config['trainer'].get(['trainer_type'], "NNTrainer")
    if trainer_type == "NNTrainer":
        trainer_class = NNTrainer
    elif trainer_type == "IterativeNNTrainer":
        trainer_class = IterativeNNTrainer
    else:
        raise ValueError("``trainer_type`` must be either 'NNTrainer' or 'IterativeNNTrainer'.")

    nb_training_sessions = config['trainer']['nb_training_sessions']
    nb_epochs = config['trainer']['nb_epochs']
    learning_rate = config['trainer']['learning_rate']
    batch_size = config['trainer']['batch_size']
    epochs_avg_loss = config['trainer']['epochs_avg_loss']
    validation_split = config['trainer']['validation_split']
    reset_model_every_training = config['trainer']['reset_model_every_training']
    diversity_loss_computation = config['trainer']['diversity_loss_computation']
    div_coeff = config['trainer']['div_coeff']

    trainer = trainer_class(nb_training_sessions=nb_training_sessions, nb_epochs=nb_epochs,
            learning_rate=learning_rate, batch_size=batch_size, epochs_avg_loss=epochs_avg_loss,
            validation_split=validation_split, reset_model_every_training=reset_model_every_training,
            diversity_loss_computation=diversity_loss_computation, div_coeff=div_coeff)

    return trainer


def create_models(config, trainer, cont) -> None:
    example_ind = cont[0]
    # Find default model parameters
    input_size = example_ind.scores['observations'].shape[-1] #len(base_scores)
    latent_size = config['nb_features']
    nb_models = config['nb_models']
    nn_models = []
    for name in config['model_names']:
        # Create models
        if config['model_type'] == "AE":
            model = AE(input_size, latent_size, config['tanh_encoder'])
        elif config['model_type'] == "ConvAE":
            model = ConvAE(input_size, latent_size, config['nb_filters'], config['batch_norm_before_latent'])
        else:
            raise ValueError(f"Unknown model_type: {config['model_type']}.")
        # Register model in global dict
        nn_models[name] = model
    # Create ensemble model
    trainer.create_ensemble_model(nn_models)
    return nn_models


def train(config, cont, trainer):
    trainer.train(cont)

    for name in config['model_names']:
        pass # TODO


def save_res(config, data, cont, trainer, nn_models):
    pass # TODO



########## BASE FUNCTIONS ########### {{{1

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputFile', type=str, default='results/test-ballistic/final_20210114122758.p', help = "Path of input data file")
    parser.add_argument('-c', '--configFilename', type=str, default='conf/test.yaml', help = "Path of configuration file")
    parser.add_argument('-o', '--resultsBaseDir', type=str, default='results/', help = "Path of results files")
#    parser.add_argument('-p', '--parallelismType', type=str, default='concurrent', help = "Type of parallelism to use")
    parser.add_argument('--seed', type=int, default=None, help="Numpy random seed")
    parser.add_argument('-v', '--verbose', default=False, action='store_true', help = "Enable verbose mode")
    return parser.parse_args()


def create_base_config(args):
    base_config = {}
    if len(args.inputFile) > 0:
        base_config['inputFile'] = args.inputFile
    if len(args.resultsBaseDir) > 0:
        base_config['resultsBaseDir'] = args.resultsBaseDir
    base_config['verbose'] = args.verbose
    return base_config



########## MAIN ########### {{{1
if __name__ == "__main__":
    import traceback
    args = parse_args()
    base_config = create_base_config(args)

    data, cont = gather_data(base_config)
    trainer = create_trainer(base_config)
    nn_models = create_models(base_config, trainer, cont)
    train(base_config, cont, trainer)
    save_res(base_config, data, cont, trainer, nn_models)




# MODELINE  "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
