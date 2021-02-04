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
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

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


import scipy.constants
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker
sns.set_style("ticks")

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


def create_table_pairwise(all_stats, output_file, mean_key = "mean_klc", std_key = "std_klc"):
    vals_dict = {}
    for c_stats in all_stats.values():
        c_case_name = c_stats['case_name']
        if not 'compare' in c_stats:
            continue
        c_df = c_stats['compare']
        mean_series = c_df[mean_key]
        std_series = c_df[std_key]
        idx = c_df.index
        idx_case_names = [all_stats[n]['case_name'] if n in all_stats else n for n in idx]
        entries = {}
        for n, m, s in zip(idx_case_names, mean_series, std_series):
            entries[n] = f"${m:.3f} \pm {s:.3f}$"
        vals_dict[c_case_name] = entries
    vals_df = pd.DataFrame.from_dict(vals_dict)

    table_str = tabulate.tabulate(vals_df, headers="keys", tablefmt="latex_raw")
    print(f"Creating table with keys '{mean_key}' and '{std_key}' to file '{output_file}':")
    print(table_str)
    print()
    with open(output_file, "w") as f:
        f.write(table_str)
    return table_str




def create_table_last_it_stats(all_stats, output_file):
    vals_dict = {}
    for c_stats in all_stats.values():
        c_case_name = c_stats['case_name']
        last_it_stats = c_stats['ref']['last_it_stats']
        corr_stats = c_stats['ref']['corr_stats']
        entries = {}
        entries['QD-Score'] = f"${last_it_stats['mean_mean_qd_score']:.3f} \pm {last_it_stats['std_mean_qd_score']:.3f}$"
        entries['Unique QD-Score'] = f"${last_it_stats['mean_all_unique_qd_score']:.3f} \pm {last_it_stats['std_all_unique_qd_score']:.3f}$"
        entries['Unique Coverage (%)'] = f"${last_it_stats['mean_all_unique_coverage']:.3f} \pm {last_it_stats['std_all_unique_coverage']:.3f}$"
        entries['Best Fitness'] = f"${last_it_stats['mean_max_best']:.3f} \pm {last_it_stats['std_max_best']:.3f}$"
        entries['FD Abs. Corr.'] = f"${corr_stats['mean_mean_abs_corr']:.3f} \pm {corr_stats['std_mean_abs_corr']:.3f}$"
        vals_dict[c_case_name] = entries

    vals_df = pd.DataFrame.from_dict(vals_dict, orient="index")
    table_str = tabulate.tabulate(vals_df, headers="keys", tablefmt="latex_raw")
    print(f"Creating table of last iteration statis to file '{output_file}':")
    print(table_str)
    print()
    with open(output_file, "w") as f:
        f.write(table_str)
    return table_str




# https://stackoverflow.com/questions/47581672/replacement-for-deprecated-tsplot
def tsplot(ax, data, label=None, **kw):
    x = np.arange(data.shape[1])
    med = np.median(data, axis=0)
    min_ = np.min(data, axis=0)
    max_ = np.max(data, axis=0)
    q25 = np.percentile(data, 25, axis=0)
    q75 = np.percentile(data, 75, axis=0)
    #sd = np.std(data, axis=0)
    #ax.fill_between(x, cis[0], cis[1], alpha=0.4, **kw)
    ax.fill_between(x,min_,max_,alpha=0.2, **kw)
    ax.fill_between(x,q25,q75,alpha=0.4, **kw)
    ax.plot(x, med, label=label, **kw)
    ax.margins(x=0)


def create_plot_it(all_stats, output_file, stats_key, y_label, cmap=plt.cm.jet, hack_always_increase=False, only_legend=False):
    y_all = [s['ref']['all_it_stats'][stats_key].T for s in all_stats.values()]
    y_names = [s['case_name'] for s in all_stats.values()]

    fig = plt.figure(figsize=(5.*scipy.constants.golden, 5.))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(bottom=0.3)

    colors = cmap(np.linspace(0., 1., len(y_all)))
    for y, c, name in zip(y_all, colors, y_names):
        if hack_always_increase:
            y2 = []
            prev = y[:,0]
            y2.append(prev)
            for val in y.T[1:]:
                val2 = np.max([val, prev], 0)
                prev = val2
                y2.append(val2)
            y2 = np.array(y2).T
        else:
            y2 = y
        tsplot(ax, y2, label=name, color=c)

    plt.xlabel("Iteration", fontsize=18)
    plt.xticks(fontsize=18)
    #plt.xticks(x, fontsize=18)
    #ax.set_xticklabels([str(i * args.nbEvalsPerIt) for i in x])
    plt.ylabel(y_label, fontsize=18)
    plt.yticks(fontsize=18)

    ##ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
    ##ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    #ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    if only_legend:
        legend = plt.legend(loc=3, framealpha=1, frameon=True)
        figleg = legend.figure
        figleg.canvas.draw()
        bbox  = legend.get_window_extent()
        expand=[0,0,0,-3]
        bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(output_file, dpi="figure", bbox_inches=bbox)

    else:
        sns.despine()
        #plt.tight_layout(rect=[0, 0, 1.0, 0.95])
        plt.tight_layout()
        fig.savefig(output_file)
    plt.close(fig)




########## BASE FUNCTIONS ########### {{{1

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('-c', '--configFilename', type=str, default='conf/stats/all.yaml', help = "Path of configuration file")
    #parser.add_argument('-i', '--inputFilename', type=str, default='results/stats.p', help = "Path of input stats file")
    parser.add_argument('-o', '--resultsDir', type=str, default='plots/figs/', help = "Path of results file")
    #parser.add_argument('-v', '--verbose', default=False, action='store_true', help = "Enable verbose mode")
    parser.add_argument('-t', '--type', type=str, default='all', help = "Type of plots to create")
    parser.add_argument('-n', '--names', type=str, default=None, help = "Names of the input data files")
    parser.add_argument('input_files', nargs=argparse.REMAINDER)
    return parser.parse_args()


def load_stats_files(filenames, case_names):
    all_stats = {}
    if case_names == None:
        case_names_lst = [None for _ in range(len(filenames))]
    else:
        case_names_lst = case_names.split(",")
    for filename, case_name in zip(filenames, case_names_lst):
        with open(filename, "rb") as f:
            stats = pickle.load(f)
        stats_name = stats['ref']['name']
        stats['case_name'] = case_name if case_name != None else stats_name
        all_stats[stats_name] = stats
    return all_stats




########## MAIN ########### {{{1
if __name__ == "__main__":
    import traceback
    args = parse_args()

    # Create output dir
    pathlib.Path(args.resultsDir).mkdir(parents=True, exist_ok=True)

    # Create or retrieve stats
    all_stats = load_stats_files(args.input_files, args.names)

    # Create tables
    if args.type == "table_last_it" or args.type == "all":
        create_table_last_it_stats(all_stats, os.path.join(args.resultsDir, "table-last_it.tex"))
    #if args.type == "table_pairwise" or args.type == "all":
    #    create_table_pairwise(all_stats, os.path.join(args.resultsDir, "table-klc.tex"), "mean_klc", "std_klc")
    #    create_table_pairwise(all_stats, os.path.join(args.resultsDir, "table-coverage.tex"), "mean_coverage", "std_coverage")
    #    create_table_pairwise(all_stats, os.path.join(args.resultsDir, "table-qdscore.tex"), "mean_qdscore", "std_qdscore")

    # Create plots
    if args.type == "plots_it" or args.type == "all":
        create_plot_it(all_stats, os.path.join(args.resultsDir, "qd-score.pdf"), "qd_score", "QD-Score", hack_always_increase=True)
        create_plot_it(all_stats, os.path.join(args.resultsDir, "unique-qd-score.pdf"), "all_unique_qd_score", "Unique QD-Score")
        create_plot_it(all_stats, os.path.join(args.resultsDir, "unique-coverage.pdf"), "all_unique_coverage", "Unique Coverage (\%)")
        create_plot_it(all_stats, os.path.join(args.resultsDir, "training-size.pdf"), "training_size", "Training size", hack_always_increase=True)
        create_plot_it(all_stats, os.path.join(args.resultsDir, "best.pdf"), "best", "Best Fitness", hack_always_increase=True)
        create_plot_it(all_stats, os.path.join(args.resultsDir, "legend.pdf"), "best", "", only_legend=True)



# MODELINE  "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
