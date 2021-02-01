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
#import ray


#import warnings
#warnings.simplefilter('always', UserWarning)



########## STATS FUNCTIONS ########### {{{1

def data_files_loader(config, path, max_data_files=None):
    base_path = config.get("resultsBaseDir", ".")
    if max_data_files == None:
        max_data_files = np.inf
    for i, filename in enumerate(glob.glob(os.path.join(base_path, path, "final*.p"))):
        if i >= max_data_files:
            break
        with open(filename, "rb") as f:
            data_file = pickle.load(f)
        yield data_file

#def added_inds_loader(config, data_file, max_inds=None):
#    if max_inds == None:
#        for a in data_file['algorithms']:
#            yield from a.container
#    else:
#        remaining_inds = max_inds
#        for a in data_file['algorithms']:
#            next_inds = sortedcollections.IndexableSet(a.container)[:remaining_inds]
#            yield from next_inds
#            remaining_inds -= len(next_inds)
#            if remaining_inds == 0:
#                break

def get_added_inds(config, data_file, max_inds = None, remove_extracted_scores = False):
    added_inds = sortedcollections.IndexableSet()
    if max_inds == None:
        for a in data_file['algorithms']:
            for i in a.container:
                if remove_extracted_scores:
                    for s in [k for k in i.scores.keys() if k.startswith("extracted_")]:
                        del i.scores[s]
                added_inds.add(i)
    else:
        remaining_inds = max_inds
        for a in data_file['algorithms']:
            for i in a.container:
                if remaining_inds <= 0:
                    return added_inds
                if remove_extracted_scores:
                    for s in [k for k in i.scores.keys() if k.startswith("extracted_")]:
                        del i.scores[s]
                added_inds.add(i)
                remaining_inds -= 1
    return added_inds


def get_empty_containers(config, data_file):
    containers = []
    for a in data_file['algorithms']:
        try:
            a.container.clear()
        except Exception as e:
            # XXX horrible hack... and ignore "also_from_parents"
            a.container.items.clear()
            a.container._init_grid()
            a.container.recentness = []
            a.container._size = 0
            a.container._best = None
            a.container._best_fitness = None
            a.container._best_features = None
        #print(f"DEBUG container len: {len(a.container)}")
        containers.append(a.container)
    return containers





##def recompute_latent(config, containers_case_name, inds_case_name, ):
#def recompute_latent(config, ref_stats, inds_case_name):
#    folders = config['klc']["data_dirs"]
#    max_data_files = config['klc'].get('max_data_files', None)
#    max_inds = config['klc'].get('max_inds', None)
#
#    # Create loaders
#    inds_folder = folders[inds_case_name]
#    loader_inds = data_files_loader(config, inds_folder, max_data_files)
#
#    # Recompute latent for all inds
#    all_scores_mats = []
#    for i, inds_data_file in enumerate(loader_inds):
#        cont_scores_mats = []
#        for j, cont in enumerate(ref_stats['containers']):
#            print(f"Recomputing latent scores of '{inds_case_name}'/{i} using containers of '{ref_stats['name']}'/{j}...")
#            orig_inds = get_added_inds(config, inds_data_file, max_inds, remove_extracted_scores=True)
#            added_inds = sortedcollections.IndexableSet()
#            # Add all inds to all containers:
#            for c in containers:
#                try:
#                    c.update(orig_inds)
#                    added_inds |= c
#                except Exception as e:
#                    print("Container UPDATE failed !")
#                    traceback.print_exc()
#                    #raise e
#            # Retrieve score matrix
#            scores_mat = metrics.inds_to_scores_mat(added_inds)
#            cont_scores_mats.append(scores_mat)
#        all_scores_mats.append(cont_scores_mats)
#
#    return all_scores_mats

def recompute_latent(config, inds_data_file, base_containers):
    max_inds = config['klc'].get('max_inds', None)
    scores_names = config['klc'].get('scores_names', None)
    if isinstance(scores_names, Sized) and len(scores_names) == 0:
        scores_names = None

    containers = copy.deepcopy(base_containers)
    orig_inds = get_added_inds(config, inds_data_file, max_inds, remove_extracted_scores=True)
    added_inds = sortedcollections.IndexableSet()
    # Add all inds to all containers:
    for c in containers:
        try:
            c.update(orig_inds)
            added_inds |= c
        except Exception as e:
            print("Container UPDATE failed !")
            traceback.print_exc()
            #raise e
    # Retrieve score matrix
    scores_mat = metrics.inds_to_scores_mat(added_inds, scores_names)

    return scores_mat




#@ray.remote
def _compute_klc_density(mat_refs, nb_bins, epsilon, ranges):
    # Compute refs extrema
    if ranges == None:
        refs_min = np.min(mat_refs, 0)
        refs_max = np.max(mat_refs, 0)
        refs_range = list(zip(refs_min, refs_max))
    else:
        if len(ranges) == 2 and not isinstance(ranges[0], Sequence):
            refs_range = [list(ranges)] * int(mat_refs.shape[1])
        else:
            refs_range = list(ranges)
    # Compute histograms
    density_refs = (np.histogramdd(mat_refs, nb_bins, range=refs_range, density=False)[0] / len(mat_refs)).ravel()
    density_refs[np.where(density_refs == 0.)] = epsilon
    return density_refs, refs_range






#@ray.remote
def _compute_klc(mat_inds, density_refs, refs_range, nb_bins, epsilon):
    print(f"DEBUG _compute_klc mat_inds: {mat_inds}")
    print(f"DEBUG _compute_klc density_refs: {density_refs}")
    print(f"DEBUG _compute_klc refs_range: {refs_range}")
    print(f"DEBUG _compute_klc nb_bins: {nb_bins}")
    # Compute histograms
    density_inds = (np.histogramdd(mat_inds, nb_bins, range=refs_range, density=False)[0] / len(mat_inds)).ravel()
    density_inds[np.where(density_inds == 0.)] = epsilon
    # Compute KLC
    return np.sum(density_inds * np.log(density_inds / density_refs))
    #return np.sum(density_refs * np.log(density_refs / density_inds))


def relative_kl_coverage_btw_two_cases(config, ref_stats, inds_case_name):
    nb_bins = config['klc'].get('nb_bins', 15)
    epsilon = config['klc'].get('epsilon', 1e-20)
    print(f"Computing KL coverage of case '{inds_case_name}' using case '{ref_stats['name']}' as reference.")

#    # Recompute latent scores
#    comp_scores_mats = recompute_latent(config, ref_stats, inds_case_name)
#    # Compute KL coverage for each score_mat
#    futures = []
#    for refs_d, refs_r, cont_scores_mat in zip(refs_density, refs_range, comp_scores_mats):
#        for sc_mat in cont_scores_mat:
#            futures.append(_compute_klc.remote(sc_mat, refs_d, refs_r, nb_bins, epsilon))

    # Create loaders
    folders = config['klc']["data_dirs"]
    max_data_files = config['klc'].get('max_data_files', None)
    max_inds = config['klc'].get('max_inds', None)
    inds_folder = folders[inds_case_name]
    loader_inds = data_files_loader(config, inds_folder, max_data_files)

    # Recompute latent scores
    klcs = []
    for i, inds_data_file in enumerate(loader_inds):
        for j, cont, density, ranges in enumerate(zip(ref_stats['containers'], ref_stats['density'], ref_stats['range'])):
            print(f"Recomputing latent scores of '{inds_case_name}'/{i} using containers of '{ref_stats['name']}'/{j}...")
            scores_mat = recompute_latent(config, inds_data_file, cont)
            klcs.append(_compute_klc(scores_mat, density, ranges, nb_bins, epsilon))
            gc.collect()


    #futures2 = [_compute_klc.remote(sc_mat, refs_d, refs_r, nb_bins, epsilon) for sc_mat in comp_scores_mats for refs_d, refs_r in zip(density_refs, refs_range)]
    #klcs = list(ray.get(futures))
    print(f"DEBUG1 klcs: {len(klcs)} {np.array(klcs).shape}")
    print(f"DEBUG2 klcs: {klcs}")
    print(f"DEBUG3 klcs: {np.mean(klcs)} {np.std(klcs)}")
    gc.collect()
    return klcs



def compute_relative_kl_coverage(config, ref_stats):
    dirs = config['klc']["data_dirs"]
    # Compute klc between each all cases
    mean_klc_mat = np.zeros(len(dirs))
    std_klc_mat = np.zeros(len(dirs))
    for i, comp_case_name in enumerate(dirs.keys()):
        klcs = relative_kl_coverage_btw_two_cases(config, ref_stats, comp_case_name)
        mean_klc_mat[i] = np.mean(klcs)
        std_klc_mat[i] = np.std(klcs)
        gc.collect()
    return {"mean": mean_klc_mat, "std": std_klc_mat}





def compute_ref(config):
    ref_name = config['klc']['refs']['name']
    ref_dir = config['klc']['refs']['dir']
    ref_ranges = config['klc']['refs'].get('ranges', None)
    print(f"Computing KL densities of reference case '{ref_name}'...", end=" ")

    max_data_files = config['klc'].get('max_data_files', None)
    max_inds = config['klc'].get('max_inds', None)
    nb_bins = config['klc'].get('nb_bins', 15)
    epsilon = config['klc'].get('epsilon', 1e-20)
    scores_names = config['klc'].get('scores_names', None)
    if isinstance(scores_names, Sized) and len(scores_names) == 0:
        scores_names = None

    #density_refs, refs_range = compute_ref_density(config, ref_dir, ref_ranges)
    # Compute densities and extract emptied containers
    loader = data_files_loader(config, ref_dir, max_data_files)
    futures = []
    containers = []
    density_refs = []
    refs_range = []
    for data_file in loader:
        inds = get_added_inds(config, data_file, max_inds, remove_extracted_scores=False)
        mat_inds = metrics.inds_to_scores_mat(inds, scores_names)
        #futures.append(_compute_klc_density.remote(mat_inds, nb_bins, epsilon, ref_ranges))
        r = _compute_klc_density(mat_inds, nb_bins, epsilon, ref_ranges)
        density_refs.append(r[0])
        refs_range.append(r[1])
        containers.append(get_empty_containers(config, data_file))
    #assert(len(futures) > 0)
    print(f"Found {len(containers)} data files.")

    # Compute density and ranges
    #futures = [_compute_klc_density.remote(sc_mat, nb_bins, epsilon, ref_ranges) for sc_mat in ref_sc_lst]
    #density_refs, refs_range = list(zip(*(ray.get(futures))))

    res = {"name": ref_name, "dir": ref_dir, "density": density_refs, "range": refs_range, "containers": containers}
    return res


#def gather_conts(config):
#    folders = config['klc'].get("data_dirs", {})
#    scores_names = config['klc'].get('scores_names', None)
#    if isinstance(scores_names, Sized) and len(scores_names) == 0:
#        scores_names = None
#    max_data_files = config['klc'].get('max_data_files', None)
#
#    # Gather all emptied containers for all data files
#    all_empty_containers = {}
#    for conf_name, base_folder in folders.items():
#        folder = os.path.join(config['resultsBaseDir'], base_folder)
#        #print(f"Gathering containers for case '{conf_name}' in folder '{folder}'...", end = " ")
#        print(f"Gathering containers for case '{conf_name}' in folder '{folder}'...")
#        loader = data_files_loader(config, folder, max_data_files)
#        case_empty_containers = []
#
#        for data in loader:
#            containers = []
#            for a in data['algorithms']:
#
#                try:
#                    a.container.clear()
#                except Exception as e:
#                    # XXX hack... and ignore "also_from_parents"
#                    a.container.items.clear()
#                    a.container._init_grid()
#                    a.recentness = []
#                    a._size = 0
#                    a._best = None
#                    a._best_fitness = None
#                    a._best_features = None
#                print(f"DEBUG container len: {len(a.container)}")
#
#                containers.append(a.container)
#            case_empty_containers.append(containers)
#            gc.collect()
#        gc.collect()
#
#        print(f"Found {len(case_empty_containers)} data files.")
#        if len(case_empty_containers) != 0:
#            all_empty_containers[conf_name] = case_empty_containers
#
#    gc.collect()
#    return all_empty_containers



#def compute_relative_kl_coverage(config):
#    nb_bins = config['klc'].get('nb_bins', 15)
#    epsilon = config['klc'].get('epsilon', 1e-20)
#    conf_ranges = config['klc'].get('ranges', None)
#
#    folders = config['klc']["data_dirs"]
#    #scores_names = config['klc'].get('scores_names', None)
#    #if isinstance(scores_names, Sized) and len(scores_names) == 0:
#    #    scores_names = None
#    max_data_files = config['klc'].get('max_data_files', None)
#
#    # Compute klc between each all cases
#    mean_klc_mat = np.zeros((len(folders), len(folders)))
#    std_klc_mat = np.zeros((len(folders), len(folders)))
#    for i, ref_case_name in enumerate(folders.keys()):
#        print(f"Computing KL coverages using case '{ref_case_name}' as reference.")
#        density_refs, refs_range = compute_ref_density(config, folders[ref_case_name])
#        for j, comp_case_name in enumerate(folders.keys()):
#            klcs = relative_kl_coverage_btw_two_cases(config, ref_case_name, comp_case_name, density_refs, refs_range)
#            mean_klc_mat[i,j] = np.mean(klcs)
#            std_klc_mat[i,j] = np.std(klcs)
#        gc.collect()
#    gc.collect()
#    return {"mean": mean_klc_mat, "std": std_klc_mat}

#
#def gather_added_inds_and_conts(config):
#    folders = config['klc'].get("data_dirs", {})
#    scores_names = config['klc'].get('scores_names', None)
#    if isinstance(scores_names, Sized) and len(scores_names) == 0:
#        scores_names = None
#    max_data_files = config['klc'].get('max_data_files', None)
#
#    # Gather all added inds for all data files
#    all_added_inds = {}
#    all_empty_containers = {}
#    for conf_name, base_folder in folders.items():
#        folder = os.path.join(config['resultsBaseDir'], base_folder)
#        print(f"Gathering added inds for case '{conf_name}' in folder '{folder}'...", end = " ")
#        loader = data_files_loader(config, folder, max_data_files)
#        case_added_inds = []
#        case_empty_containers = []
#
#        for data in loader:
#            added_inds = sortedcollections.IndexableSet()
#            containers = []
#            for a in data['algorithms']:
#                added_inds |= a.container
#
#                try:
#                    a.container.clear()
#                except Exception as e:
#                    # XXX hack... and ignore "also_from_parents"
#                    a.container.items.clear()
#                    a.container._init_grid()
#                    a.recentness = []
#                    a._size = 0
#                    a._best = None
#                    a._best_fitness = None
#                    a._best_features = None
#
#                containers.append(a.container)
#            case_added_inds.append(added_inds)
#            case_empty_containers.append(containers)
#            gc.collect()
#        gc.collect()
#
#        print(f"Found {len(case_added_inds)} data files.")
#        if len(case_added_inds) != 0:
#            all_added_inds[conf_name] = case_added_inds
#            all_empty_containers[conf_name] = case_empty_containers
#
#    gc.collect()
#    return all_added_inds, all_empty_containers



#
#def gather_score_mats(config):
#    folders = config['klc'].get("data_dirs", {})
#    scores_names = config['klc'].get('scores_names', None)
#    if isinstance(scores_names, Sized) and len(scores_names) == 0:
#        scores_names = None
#    max_data_files = config['klc'].get('max_data_files', None)
#
#    # Gather scores matrices for all data files
#    scores_mats = {}
#    for conf_name, base_folder in folders.items():
#        folder = os.path.join(config['resultsBaseDir'], base_folder)
#        print(f"Gathering scores for case '{conf_name}' in folder '{folder}'...", end = " ")
#        loader = data_files_loader(config, folder, max_data_files)
#        sc_mats = []
#
#        for data in loader:
#            all_added_inds = sortedcollections.IndexableSet()
#            for a in data['algorithms']:
#                all_added_inds |= a.container
#            mat_inds = metrics.inds_to_scores_mat(all_added_inds, scores_names)
#            sc_mats.append(mat_inds)
#            gc.collect()
#
#        print(f"Found {len(sc_mats)} data files.")
#        if len(sc_mats) != 0:
#            scores_mats[conf_name] = sc_mats
#
#    gc.collect()
#    return scores_mats
#






## TODO
#def compute_relative_kl_coverage(config, scores_mats):
#    if len(scores_mats) == 0:
#        return
#
#    nb_bins = config['klc'].get('nb_bins', 15)
#    epsilon = config['klc'].get('epsilon', 1e-20)
#    conf_ranges = config['klc'].get('ranges', None)
#
#    # Compute klc between each all cases
#    mean_klc_mat = np.zeros((len(scores_mats), len(scores_mats)))
#    std_klc_mat = np.zeros((len(scores_mats), len(scores_mats)))
#    #for ref_conf_name, ref_sc_lst in scores_mats.items():
#    for i, (ref_conf_name, ref_sc_lst) in enumerate(scores_mats.items()):
#        print(f"Computing KL coverages using case '{ref_conf_name}' as reference.")
#        futures = [_compute_klc_density.remote(sc_mat, nb_bins, epsilon, conf_ranges) for sc_mat in ref_sc_lst]
#        density_refs, refs_range = list(zip(*(ray.get(futures))))
#        print(f"DEBUG0: density_refs: {density_refs}")
#        print(f"DEBUG0: refs_range: {refs_range}")
#        for j, (ref_conf_name, ref_sc_lst) in enumerate(scores_mats.items()):
#            futures2 = [_compute_klc.remote(sc_mat, refs_d, refs_r, nb_bins, epsilon) for sc_mat in ref_sc_lst for refs_d, refs_r in zip(density_refs, refs_range)]
#            klcs = list(ray.get(futures2))
#            print(f"DEBUG1 klcs: {len(klcs)} {np.array(klcs).shape}")
#            print(f"DEBUG2 klcs: {klcs}")
#            print(f"DEBUG3 klcs: {np.mean(klcs)} {np.std(klcs)}")
#            mean_klc_mat[i,j] = np.mean(klcs)
#            std_klc_mat[i,j] = np.std(klcs)
#        gc.collect()
#    gc.collect()
#    return {"mean": mean_klc_mat, "std": std_klc_mat}
#


########## BASE FUNCTIONS ########### {{{1

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--configFilename', type=str, default='conf/stats/all.yaml', help = "Path of configuration file")
    parser.add_argument('-o', '--resultsFilename', type=str, default='results/stats.p', help = "Path of results file")
    parser.add_argument('-v', '--verbose', default=False, action='store_true', help = "Enable verbose mode")
    parser.add_argument('-r', '--recompute', default=False, action='store_true', help = "Recompute everything, don't use cached data")
    return parser.parse_args()

def create_config(args):
    config = yaml.safe_load(open(args.configFilename))
    if len(args.resultsFilename) > 0:
        config['resultsFilename'] = args.resultsFilename
    config['verbose'] = args.verbose
    config['recompute'] = args.recompute
    return config

def create_or_open_stats_file(config):
    stats_filename = config['resultsFilename']
    if not os.path.exists(stats_filename):
        data = {}
        with open(stats_filename, "wb") as f:
            pickle.dump(data, f)
    with open(stats_filename, "rb") as f:
        stats_data = pickle.load(f)
    return stats_data, stats_filename

def save_stats_file(config, stats_filename, stats_data):
    with open(stats_filename, "wb") as f:
        pickle.dump(stats_data, f)



########## MAIN ########### {{{1
if __name__ == "__main__":
    import traceback
    args = parse_args()
    config = create_config(args)
    recompute = config.get('recompute', False)

#    # Init ray
#    ray.init()

    # Create or retrieve stats
    stats_data, stats_filename = create_or_open_stats_file(config)

#    if recompute or not 'container' in stats_data:
#        stats_data['containers'] = gather_conts(config)
#    save_stats_file(config, stats_filename, stats_data)

#    if recompute or not 'added_inds' in stats_data:
#        stats_data['added_inds'], stats_data['containers'] = gather_added_inds_and_conts(config)
#    save_stats_file(config, stats_filename, stats_data)

#    if recompute or not 'scores_mats' in stats_data:
#        stats_data['scores_mats'] = gather_score_mats(config)
#    save_stats_file(config, stats_filename, stats_data)
#
#    if recompute or not 'klc' in stats_data:
#        stats_data['klc'] = compute_relative_kl_coverage(config, stats_data['scores_mats'])
#    save_stats_file(config, stats_filename, stats_data)

    if recompute or not 'ref' in stats_data:
        stats_data['ref'] = compute_ref(config)
    save_stats_file(config, stats_filename, stats_data)
    gc.collect()

    if recompute or not 'klc' in stats_data:
        stats_data['klc'] = compute_relative_kl_coverage(config, stats_data['ref'])
    save_stats_file(config, stats_filename, stats_data)
    gc.collect()

    #exp = create_experiment(args, base_config)
    #launch_experiment(exp)
    ##model = train_AE(exp)
    #make_plots(exp)




# MODELINE  "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
