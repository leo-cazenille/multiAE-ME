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
import os
import copy
import pickle
import numpy as np
import pandas as pd
#import warnings
#import yaml
#import glob
#import sortedcollections
#import gc
#import traceback
#from typing import Sized
import pathlib
import tabulate

# QDpy
#import qdpy
#from qdpy.base import *
#from qdpy.experiment import QDExperiment
#from qdpy.benchmarks import artificial_landscapes
#from qdpy.algorithms import LoggerStat, TQDMAlgorithmLogger, AlgorithmLogger, QDAlgorithmLike
#from qdpy.containers import TorchAE, TorchFeatureExtractionContainerDecorator, ContainerLike, NoveltyArchive, Container
#import qdpy.plots


#import matplotlib.pyplot as plt

#import curiosity
#import nets
#from nets import *
#import metrics
#import containers
#from containers import *
#import sim
#import ray

import stats


#import warnings
#warnings.simplefilter('always', UserWarning)



########## PLOT FUNCTIONS ########### {{{1


def create_table(comparisons, output_file, mean_key = "mean_klc", std_key = "std_klc"):
    vals_dict = {}
    for c_name, c_df in comparisons.items():
        mean_series = c_df[mean_key]
        std_series = c_df[std_key]
        idx = c_df.index
        entries = {}
        for n, m, s in zip(idx, mean_series, std_series):
            entries[n] = f"${m:.3f} \pm {s:.3f}$"
        vals_dict[c_name] = entries
    vals_df = pd.DataFrame.from_dict(vals_dict)

    table_str = tabulate.tabulate(vals_df, headers="keys", tablefmt="latex_raw")
    print(f"Creating table with keys '{mean_key}' and '{std_key}' to file '{output_file}':")
    print(table_str)
    print()
    with open(output_file, "w") as f:
        f.write(table_str)
    return table_str



########## BASE FUNCTIONS ########### {{{1

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('-c', '--configFilename', type=str, default='conf/stats/all.yaml', help = "Path of configuration file")
    #parser.add_argument('-i', '--inputFilename', type=str, default='results/stats.p', help = "Path of input stats file")
    parser.add_argument('-o', '--resultsDir', type=str, default='plots/figs/', help = "Path of results file")
    #parser.add_argument('-v', '--verbose', default=False, action='store_true', help = "Enable verbose mode")
    parser.add_argument('input_files', nargs=argparse.REMAINDER)
    return parser.parse_args()


def load_stats_files(filenames):
    comparisons = {}
    for filename in filenames:
        with open(filename, "rb") as f:
            stats = pickle.load(f)
        stats_name = stats['ref']['name']
        comparisons[stats_name] = stats['compare']
    return comparisons




########## MAIN ########### {{{1
if __name__ == "__main__":
    import traceback
    args = parse_args()

    # Create output dir
    pathlib.Path(args.resultsDir).mkdir(parents=True, exist_ok=True)

    # Create or retrieve stats
    comparisons = load_stats_files(args.input_files)

    # Create tables
    create_table(comparisons, os.path.join(args.resultsDir, "table-klc.tex"), "mean_klc", "std_klc")
    create_table(comparisons, os.path.join(args.resultsDir, "table-fullness.tex"), "mean_fullness", "std_fullness")
    create_table(comparisons, os.path.join(args.resultsDir, "table-qdscore.tex"), "mean_qdscore", "std_qdscore")



# MODELINE  "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
