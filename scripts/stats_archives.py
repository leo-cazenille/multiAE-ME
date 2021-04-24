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
#import pickle
import numpy as np
import warnings
import yaml

# QDpy
import qdpy
from qdpy.base import *
from qdpy.experiment import QDExperiment
from qdpy.benchmarks import artificial_landscapes
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
import sim


#import warnings
#warnings.simplefilter('always', UserWarning)




########## PLOT FUNCTIONS ########### {{{1

# TODO
#def mk_plots(data)



########## MAIN ########### {{{1
if __name__ == "__main__":
    import traceback
    args = parse_args()
    base_config = create_base_config(args)
    exp = create_experiment(args, base_config)
    launch_experiment(exp)
    #model = train_AE(exp)
    make_plots(exp)




# MODELINE  "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
