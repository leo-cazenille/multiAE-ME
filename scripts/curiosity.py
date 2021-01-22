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

class NPArrayIndividual(IndividualLike, Sequence):

    name: str
    _fitness: FitnessLike
    _features: FeaturesLike
    _scores: ScoresDictLike
    elapsed: float = math.nan

    genotype: np.ndarray

    def __init__(self, iterable: Optional[Iterable] = None,
            name: Optional[str] = None,
            fitness: Optional[FitnessLike] = None, features: Optional[FeaturesLike] = None,
            scores: Optional[ScoresDictLike] = None) -> None:
        if iterable is not None:
            self.genotype = np.array(iterable)
        self.name = name if name else ""
        self._scores = scores if scores is not None else ScoresDict({})
        self.fitness = fitness if fitness is not None else Fitness()
        self.features = features if features is not None else Features([])

    def __repr__(self) -> str:
        if not self:
            return "%s()" % (self.__class__.__name__,)
        return "%s(%r)" % (self.__class__.__name__, list(self))

    def __len__(self) -> int:
        return len(self.genotype)

    def __getitem__(self, key):
        return self.genotype[key]

    def __setitem__(self, idx, value):
        self.genotype[idx] = value

    def __contains__(self, key: Any) -> bool:
        return key in self.genotype

    def __iter__(self) -> Iterator:
        return iter(self.genotype)

    def __reversed__(self) -> Iterator:
        return reversed(self.genotype)

    def __str__(self) -> str:
        return str(self.genotype)

    @property
    def fitness(self) -> FitnessLike:
        return self._fitness
#        return self.scores.setdefault("fitness", Fitness())
    @fitness.setter
    def fitness(self, fit: FitnessLike) -> None:
        #self._scores["fitness"] = fit
        self._fitness = fit

    @property
    def features(self) -> FeaturesLike:
        #return self._scores.setdefault("features", Features())
        return self._features
    @features.setter
    def features(self, ft: FeaturesLike) -> None:
        #self._scores["features"] = ft
        self._features = ft

    @property
    def scores(self) -> ScoresDictLike:
        return self._scores
    @scores.setter
    def scores(self, scores: ScoresDictLike) -> None:
        if isinstance(scores, ScoresDictLike):
            self._scores = scores
        else:
            self._scores = ScoresDict(scores)

    def dominates(self, other: Any, score_name: Optional[str] = None) -> bool:
        """Return true if ``self`` dominates ``other``. """
        if score_name is None:
            return self.fitness.dominates(other.fitness)
        else:
            return self._scores[score_name].dominates(other._scores[score_name])

    def reset(self) -> None:
        self._scores.clear()
        self.fitness.reset()
        #self._scores["fitness"] = self._fitness
        self.features.reset()
        #self._scores["features"] = self._features
        self.elapsed = math.nan


    # TODO : improve performance ! (quick and dirty solution !)
    def __hash__(self):
        return hash(tuple(self))

    def __eq__(self, other):
        return (self.__class__ == other.__class__ and tuple(self) == tuple(other))



#def gen_nparray_individuals(size):
#    while(True):
#        yield NPArrayIndividual(np.empty(size))

@registry.register # type: ignore
class GenNPArrayIndividuals(CreatableFromConfig):
    def __init__(self, size, *args, **kwargs):
        self.size = size
    def __iter__(self):
        return self
    def __next__(self):
        return NPArrayIndividual(np.empty(self.size))
    def __call__(self):
        while(True):
            yield self.__next__()



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
                select_or_initialise=select_or_initialise, deepcopy_on_selection=False, vary=vary,
                #base_ind_gen=GenNPArrayIndividuals(dimension),
                **kwargs) # type: ignore



# MODELINE  "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
