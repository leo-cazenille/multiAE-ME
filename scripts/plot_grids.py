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
import pandas as pd
import warnings
import yaml
import glob
import sortedcollections
import gc
import traceback
from typing import Sized

# QDpy
import qdpy
from qdpy.base import *
from qdpy.experiment import QDExperiment
from qdpy.benchmarks import artificial_landscapes
from qdpy.algorithms import LoggerStat, TQDMAlgorithmLogger, AlgorithmLogger, QDAlgorithmLike
from qdpy.containers import TorchAE, TorchFeatureExtractionContainerDecorator, ContainerLike, NoveltyArchive, Container
import qdpy.plots


import matplotlib.pyplot as plt

#import curiosity
import nets
#from nets import *
import metrics
import containers
#from containers import *
#import sim
import main

import sortedcollections

#import warnings
#warnings.simplefilter('always', UserWarning)



########## PLOTS FUNCTIONS ########### {{{1

def plots_hardcoded_FD(inputFilename, resultsDir):
    grid0 = qdpy.containers.Grid(shape=[25, 25], fitness_score_names=["meanAvgReward"], fitness_domain=[[-200., 50.]], features_score_names=["meanDistance", "meanHeadStability"], features_domain=[[0., 50.], [0., 2.5]], only_add_accepted_inds_to_parents=True, storage_type=sortedcollections.IndexableSet)
    grid1 = qdpy.containers.Grid(shape=[25, 25], fitness_score_names=["meanAvgReward"], fitness_domain=[[-200., 50.]], features_score_names=["meanTorquePerStep", "meanJump"], features_domain=[[0., 25.], [0., 0.75]], only_add_accepted_inds_to_parents=True, storage_type=sortedcollections.IndexableSet)
    grid2 = qdpy.containers.Grid(shape=[25, 25], fitness_score_names=["meanAvgReward"], fitness_domain=[[-200., 50.]], features_score_names=["meanLeg0HipAngle", "meanLeg1HipAngle"], features_domain=[[0.0, 2.5], [0.0, 2.5]], only_add_accepted_inds_to_parents=True, storage_type=sortedcollections.IndexableSet)
    grid3 = qdpy.containers.Grid(shape=[25, 25], fitness_score_names=["meanAvgReward"], fitness_domain=[[-200., 50.]], features_score_names=["meanLeg0KneeAngle", "meanLeg1KneeAngle"], features_domain=[[0.0, 2.5], [0.0, 2.5]], only_add_accepted_inds_to_parents=True, storage_type=sortedcollections.IndexableSet)


    with open(args.inputFilename, "rb") as f:
        data = pickle.load(f)
    for i, a in enumerate(data['algorithms']):
        grid0.update(a.container)
        grid1.update(a.container)
        grid2.update(a.container)
        grid3.update(a.container)
    qdpy.plots.plotGridSubplots(grid0.quality_array[..., 0], os.path.join(resultsDir, f"performancesGrid-hardcoded-0.pdf"), plt.get_cmap("inferno"), grid0.features_domain, [-200, 50], nbTicks=None)
    qdpy.plots.plotGridSubplots(grid1.quality_array[..., 0], os.path.join(resultsDir, f"performancesGrid-hardcoded-1.pdf"), plt.get_cmap("inferno"), grid1.features_domain, [-200, 50], nbTicks=None)
    qdpy.plots.plotGridSubplots(grid2.quality_array[..., 0], os.path.join(resultsDir, f"performancesGrid-hardcoded-2.pdf"), plt.get_cmap("inferno"), grid2.features_domain, [-200, 50], nbTicks=None)
    qdpy.plots.plotGridSubplots(grid3.quality_array[..., 0], os.path.join(resultsDir, f"performancesGrid-hardcoded-3.pdf"), plt.get_cmap("inferno"), grid3.features_domain, [-200, 50], nbTicks=None)


########## BASE FUNCTIONS ########### {{{1

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('-c', '--configFilename', type=str, default='conf/stats/all.yaml', help = "Path of configuration file")
    parser.add_argument('-i', '--inputFilename', type=str, default='results/bipedalM30-preonline5000_100-recononly_0.0-4xAutoScalingGrid2x25x25-scoreprop_100000_cpu/final_20210204140207.p', help = "Path of input data file")
    parser.add_argument('-o', '--resultsDir', type=str, default='.', help = "Path of results plots")
    #parser.add_argument('-v', '--verbose', default=False, action='store_true', help = "Enable verbose mode")
    return parser.parse_args()



########## MAIN ########### {{{1
if __name__ == "__main__":
    import traceback
    args = parse_args()

    # Create grid plots using hardcoded features
    plots_hardcoded_FD(args.inputFilename, args.resultsDir)



# MODELINE  "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
