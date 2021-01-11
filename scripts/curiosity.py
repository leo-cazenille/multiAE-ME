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

__all__ = ["KDTreeBackend"]


########## IMPORTS ########### {{{1
import gc
import copy
#import pickle
import numpy as np
import warnings
import random

# QDpy
import qdpy
from qdpy.base import *
from qdpy.phenotype import *
from qdpy.containers import *
#from qdpy.experiment import QDExperiment
#from qdpy.benchmarks import artificial_landscapes
from qdpy.algorithms import *
from qdpy import tools
#from qdpy.containers import TorchAE, TorchFeatureExtractionContainerDecorator, ContainerLike, NoveltyArchive



########## ALGORITHM CLASSES ########### {{{1



def sel_roulette_score_proportionate(collection: Sequence[Any], score_name: str = "curiosity", default_score_val: Any = 0.) -> Sequence[Any]:
    """Select and return one individual at random (using a roulette selection)

    Parameters
    ----------
    :param collection: Container
        The Container containing individuals.
    """
    assert(len(collection))
    sum_fit_val = [i.scores.get(score_name, default_score_val) for i in collection]
    sum_all_fit = sum(sum_fit_val)
    if sum_all_fit == 0:
        probs = [1. / len(collection) for _ in sum_fit_val]
    else:
        probs = [f / sum_all_fit for f in sum_fit_val]
    return random.choices(collection, weights=probs)[0]



@registry.register
class ScoreProportionateRouletteMutPolyBounded(Evolution):
    """TODO"""
    ind_domain: DomainLike
    sel_pb: float
    init_pb: float
    mut_pb: float
    eta: float
    score_name: str
    default_score_val: Any

    def __init__(self, container: Container, budget: int,
            dimension: int, ind_domain: DomainLike = (0., 1.),
            sel_pb: float = 0.5, init_pb: float = 0.5, mut_pb: float = 0.2, eta: float = 20.,
            score_name: str = "curiosity", default_score_val: Any = 0.,
            **kwargs):
        self.ind_domain = ind_domain
        self.sel_pb = sel_pb
        self.init_pb = init_pb
        self.mut_pb = mut_pb
        self.eta = eta
        self.score_name = score_name
        self.default_score_val = default_score_val

        def sel(container):
            ind = sel_roulette_score_proportionate(container, score_name=self.score_name, default_score_val=self.default_score_val)
            res = copy.deepcopy(ind)
            res.parent = ind
            return res

        def init_fn(base_ind):
            return [random.uniform(ind_domain[0], ind_domain[1]) for _ in range(self.dimension)]
        select_or_initialise = partial(tools.sel_or_init,
                #sel_fn = partial(sel_roulette_score_proportionate, score_name=self.score_name, default_score_val=self.default_score_val),
                sel_fn = sel,
                sel_pb = sel_pb,
                init_fn = init_fn,
                init_pb = init_pb)
        def vary(ind):
            res = tools.mut_polynomial_bounded(ind, low=self.ind_domain[0], up=self.ind_domain[1], eta=self.eta, mut_pb=self.mut_pb)
            #res.parent = ind
            return res

        # No deepcopy_on_selection, because we are already copying ``ind`` in the selection function
        super().__init__(container, budget, dimension=dimension, # type: ignore
                select_or_initialise=select_or_initialise, deepcopy_on_selection=False, vary=vary, **kwargs) # type: ignore



# MODELINE  "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
