
########## IMPORTS ########### {{{1
#import copy
#import pickle
import numpy as np
#import warnings
import random
from typing import Optional, Tuple, List, Iterable, Iterator, Any, TypeVar, Generic, Union, Sequence, MutableSet, MutableSequence, Type, Callable, Generator, MutableMapping, Mapping, overload

## QDpy
#import qdpy
#from qdpy.base import *
from qdpy.phenotype import IndividualLike
#from qdpy.containers import *
#from qdpy.algorithms import *
#from qdpy import tools


########## METRICS ########### {{{1


def originality(c, conts):
    sets = [set(x) for x in conts]
    c_s = set(c)
    inter = set()
    for s, x in zip(sets, conts):
        if id(x) == id(c):
            continue
        inter |= c_s & s
    originality = 1. - len(inter)/len(c_s)
    return originality

def mean_originality(conts):
    sets = [set(c) for c in conts]
    orig_lst = []
    for i, s1 in enumerate(sets):
        inter = set()
        for j, s2 in enumerate(sets):
            if i == j:
                continue
            inter |= s1 & s2
        orig_lst.append(1. - len(inter)/len(s1))
    return np.mean([orig_lst])


def inds_to_scores_mat(inds: Sequence[IndividualLike], scores_names: Optional[Sequence] = None, default_val: float = 0.):
    assert(len(inds) > 0), "At least one individual must be provided in Sequence ``inds``."
    if scores_names == None:
        scores_names = [x for x in inds[0].scores.keys() if x.startswith("extracted")]
    assert(len(scores_names) > 0), "At least one score name must be provided in Sequence ``scores_names``."

    data = np.empty((len(inds), len(scores_names)), dtype=np.float32)
    for i, ind in enumerate(inds):
        for j, s in enumerate(scores_names):
            if s in ind.scores:
                data[i,j] = ind.scores[s]
            else:
                data[i,j] = 0.
    return data


def cov_scores(inds: Sequence[IndividualLike], scores_names: Optional[Sequence] = None):
    data = inds_to_scores_mat(inds, scores_names)
    cov_mat = np.cov(data, rowvar=False)
    mean_cov = 0.
    for i in range(cov_mat.shape[0]):
        for j in range(cov_mat.shape[1]):
            if i != j and not np.isnan(cov_mat[i, j]):
                mean_cov += cov_mat[i, j]
    mean_cov /= cov_mat.shape[0] * cov_mat.shape[1] - cov_mat.shape[0]
    return cov_mat, mean_cov

def abs_cov_scores(inds: Sequence[IndividualLike], scores_names: Optional[Sequence] = None):
    data = inds_to_scores_mat(inds, scores_names)
    cov_mat = np.abs(np.cov(data, rowvar=False))
    mean_cov = 0.
    for i in range(cov_mat.shape[0]):
        for j in range(cov_mat.shape[1]):
            if i != j and not np.isnan(cov_mat[i, j]):
                mean_cov += cov_mat[i, j]
    mean_cov /= cov_mat.shape[0] * cov_mat.shape[1] - cov_mat.shape[0]
    return cov_mat, mean_cov

def corr_scores(inds: Sequence[IndividualLike], scores_names: Optional[Sequence] = None):
    data = inds_to_scores_mat(inds, scores_names)
    corr_mat = np.corrcoef(data, rowvar=False)
    mean_corr = 0.
    for i in range(corr_mat.shape[0]):
        for j in range(corr_mat.shape[1]):
            if i != j and not np.isnan(corr_mat[i, j]):
                mean_corr += corr_mat[i, j]
    mean_corr /= corr_mat.shape[0] * corr_mat.shape[1] - corr_mat.shape[0]
    return corr_mat, mean_corr



def kl_coverage(inds: Sequence[IndividualLike], refs: Sequence[IndividualLike], scores_names, nb_bins=15, epsilon=1e-20):
    mat_inds = inds_to_scores_mat(inds, scores_names)
    mat_refs = inds_to_scores_mat(refs, scores_names)
    # Compute refs extrema
    refs_min = np.min(mat_refs, 0)
    refs_max = np.max(mat_refs, 0)
    refs_range = list(zip(refs_min, refs_max))
    # Compute histograms
    density_inds = (np.histogramdd(mat_inds, nb_bins, range=refs_range, density=False)[0] / len(mat_inds)).ravel()
    density_inds[np.where(density_inds == 0.)] = epsilon
    density_refs = (np.histogramdd(mat_refs, nb_bins, range=refs_range, density=False)[0] / len(mat_refs)).ravel()
    density_refs[np.where(density_refs == 0.)] = epsilon
    # Compute KLC
    return np.sum(density_inds * np.log(density_inds / density_refs))
    #return np.sum(density_refs * np.log(density_refs / density_inds))


def kl_coverage_prepare_stored_refs(refs: Sequence[IndividualLike], scores_names, nb_bins=15, epsilon=1e-20):
    mat_refs = inds_to_scores_mat(refs, scores_names)
    # Compute refs extrema
    refs_min = np.min(mat_refs, 0)
    refs_max = np.max(mat_refs, 0)
    refs_range = list(zip(refs_min, refs_max))
    # Compute histograms
    density_refs = (np.histogramdd(mat_refs, nb_bins, range=refs_range, density=False)[0] / len(mat_refs)).ravel()
    density_refs[np.where(density_refs == 0.)] = epsilon
    return density_refs, refs_range

def kl_coverage_stored_refs(inds: Sequence[IndividualLike], density_refs, refs_range, scores_names, nb_bins=15, epsilon=1e-20):
    mat_inds = inds_to_scores_mat(inds, scores_names)
    # Compute histograms
    density_inds = (np.histogramdd(mat_inds, nb_bins, range=refs_range, density=False)[0] / len(mat_inds)).ravel()
    density_inds[np.where(density_inds == 0.)] = epsilon
    # Compute KLC
    return np.sum(density_inds * np.log(density_inds / density_refs))
    #return np.sum(density_refs * np.log(density_refs / density_inds))




def kl_coverage2_prepare_stored_refs(refs: Sequence[IndividualLike], scores_names, nb_bins=15, epsilon=1e-20):
    mat_refs = inds_to_scores_mat(refs, scores_names)
    # Compute refs extrema
    refs_min = np.min(mat_refs, 0)
    refs_max = np.max(mat_refs, 0)
    refs_range = list(zip(refs_min, refs_max))
    # Compute histograms
    density_refs = []
    for i in range(mat_refs.shape[1]):
        h = (np.histogram(mat_refs[:,i], nb_bins, range=refs_range[i], density=False)[0] / len(mat_refs)).ravel()
        h[np.where(h == 0.)] = epsilon
        density_refs.append(h)
    return density_refs, refs_range

def kl_coverage2_stored_refs(inds: Sequence[IndividualLike], density_refs, refs_range, scores_names, nb_bins=15, epsilon=1e-20):
    mat_inds = inds_to_scores_mat(inds, scores_names)
    # Compute histograms
    density_inds = []
    klc_lst = []
    # Compute KLC across all dimension
    for i in range(mat_inds.shape[1]):
        h = (np.histogram(mat_inds[:,i], nb_bins, range=refs_range[i], density=False)[0] / len(mat_inds)).ravel()
        h[np.where(h == 0.)] = epsilon
        klc_score = np.sum(h * np.log(h / density_refs[i]))
        klc_lst.append(klc_score)
    # Compute mean KLC
    return np.mean(klc_lst)



# TODO
def kl_originality(c, conts, nb_bins=15, epsilon=1e-20):
    scores_names_lst = []
    for c2 in conts:
        sn = [f"extracted_{id(c2)}_{j}" for j in range(len(c2.features_domain))]
        scores_names_lst.append(sn)
    res = 0.
    # TODO XXX


def corr_dist_scores(inds: Sequence[IndividualLike], scores_names: Optional[Sequence] = None, split_nb=2):
    data = inds_to_scores_mat(inds, scores_names)
    split_data = np.split(data, range(split_nb, data.shape[1], split_nb), 1)
    corr_lst = [np.abs(np.corrcoef(x, rowvar=False)) for x in split_data]
    corr_dist = 0.
    nb = 0.
    for i, c1 in enumerate(corr_lst):
        for _, c2 in enumerate(corr_lst[i+1:]):
            dst = np.linalg.norm(c1 - c2, ord='fro')
            if not np.isnan(dst) and not np.isinf(dst):
                corr_dist += dst
                nb += 1.
    if nb > 0:
        corr_dist /= nb
    return corr_dist


# CMD Measure, as defined in Herdin2005 "Correlation Matrix Distance, a Meaningful Measure for Evaluation of Non-Stationary MIMO Channels"
def numpy_correlation_mat_dist(c1, c2):
    return 1. - np.trace(c1 * c2) / (np.linalg.norm(c1, ord="fro") * np.linalg.norm(c2, ord="fro"))

def cmd_scores(inds: Sequence[IndividualLike], scores_names: Optional[Sequence] = None, split_nb=2):
    data = inds_to_scores_mat(inds, scores_names)
    split_data = np.split(data, range(split_nb, data.shape[1], split_nb), 1)
    corr_lst = [np.abs(np.corrcoef(x, rowvar=False)) for x in split_data]
    corr_dist = 0.
    nb = 0.
    for i, c1 in enumerate(corr_lst):
        for _, c2 in enumerate(corr_lst[i+1:]):
            dst = numpy_correlation_mat_dist(c1, c2)
            if not np.isnan(dst) and not np.isinf(dst):
                corr_dist += dst
                nb += 1.
    if nb > 0:
        corr_dist /= nb
    return corr_dist




# MODELINE  "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
