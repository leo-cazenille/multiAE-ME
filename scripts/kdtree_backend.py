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

# QDpy
import qdpy
from qdpy.base import *
from qdpy.phenotype import *
from qdpy.containers import *
#from qdpy.experiment import QDExperiment
#from qdpy.benchmarks import artificial_landscapes
#from qdpy.algorithms import LoggerStat, TQDMAlgorithmLogger, AlgorithmLogger
#from qdpy.containers import TorchAE, TorchFeatureExtractionContainerDecorator, ContainerLike, NoveltyArchive


import kdtree




########## CONTAINER CLASS ########### {{{1

class KDTreePoint(tuple):
    def __new__(self, ind, coords):
        res = tuple.__new__(KDTreePoint, coords)
        res.ind = ind
        return res

@registry.register
class KDTreeBackend(MutableSequence):

    def __init__(self, features_dimensions: int, base_backend: BackendLike = [], features_score_names: Sequence[str] = [], iterable: Optional[Iterable] = None) -> None:
        self.base_backend = base_backend
        self.features_score_names = features_score_names
        self.features_dimensions = features_dimensions
        self.tree = kdtree.create(dimensions=features_dimensions)
        if iterable != None:
            self.base_backend.update(iterable)

    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict['tree']
        return odict

    def __setstate__(self, state):
        if hasattr(state, 'tree'):
            del state.tree
        self.__dict__.update(state)
        self.tree = kdtree.create(dimensions=self.features_dimensions)
        for i in self.base_backend:
            p = self.ind_to_point(i)
            self.tree.add(p)

    def get_ind_features(self, individual: IndividualLike, *args, **kwargs) -> FeaturesLike:
        if len(self.features_score_names) == 0:
            return individual.features
        else:
            return individual.scores.to_features(self.features_score_names, *args, **kwargs)

    def ind_to_point(self, ind: IndividualLike) -> KDTreePoint:
        features = self.get_ind_features(ind)
        p = KDTreePoint(ind, features)
        return p

    def __len__(self) -> int:
        return self.base_backend.__len__()

    @overload
    def __getitem__(self, index: int) -> Any: ...

    @overload
    def __getitem__(self, s: slice) -> Sequence[Any]: ...

    def __getitem__(self, key):
        return self.base_backend.__getitem__(key)

    def __contains__(self, key: Any) -> bool:
        return self.base_backend.__contains__(key)

    def __iter__(self) -> Iterator:
        return self.base_backend.__iter__()

    def __reversed__(self) -> Iterator:
        return self.base_backend.__reversed__()

    def __repr__(self) -> str:
        if not self:
            return "%s()" % (self.__class__.__name__,)
        return "%s(%r)" % (self.__class__.__name__, list(self))

    def __setitem__(self, index, value):
        p = self.ind_to_point(self.base_backend[index])
        self.tree.remove(p)
        self.base_backend.__setitem__(index, value)

    def __delitem__(self, idx) -> None:
        ind = self[idx]
        p = self.ind_to_point(ind)
        self.tree.remove(p)
        self.base_backend.__delitem__(idx)

    def insert(self, index, value):
        self.base_backend.insert(index, value)
        p = self.ind_to_point(value)
        self.tree.add(p)

    def reverse(self):
        return self.base_backend.reverse()

    def count(self, key) -> int:
        return self.base_backend.count(key)

    def index(self, key, start: int = 0, stop: int = sys.maxsize) -> int:
        return self.base_backend.index(key, start, stop)

    def add(self, key) -> None:
        if hasattr(self.base_backend, 'add'):
            self.base_backend.add(key)
        elif hasattr(self.base_backend, 'append'):
            self.base_backend.append(key)
        p = self.ind_to_point(key)
        self.tree.add(p)

#    def discard(self, key) -> None:
#        p = self.ind_to_point(key)
#        self.tree.remove(p)
#        self.base_backend.__discard__(key)

    def update(self, iterable: Iterable) -> None:
        try:
            for item in iterable:
                self.add(item)
        except TypeError:
            raise ValueError(f"Argument needs to be an iterable, got {type(iterable)}")



    def nearest(self, val) -> Tuple[Any, float]:
        if isinstance(val, IndividualLike):
            p = self.ind_to_point(val)
        elif isinstance(val, Iterable):
            p = val
        else:
            raise ValueError(f"``val`` must be either an IndividualLike or a features descriptor values (Iterable).")
        nn, d = self.tree.search_nn(p)
        return nn.data.ind, d

    def knn(self, val: Any, k: int):
        if isinstance(val, IndividualLike):
            p = self.ind_to_point(val)
        elif isinstance(val, Iterable):
            p = val
        else:
            raise ValueError(f"``val`` must be either an IndividualLike or a features descriptor values (Iterable).")
        s = self.tree.search_knn(p, k)
        return [(n.data.ind, d) for n, d in s]

    def rebalance(self):
        if len(self) != 0:
            self.tree = self.tree.rebalance()


# MODELINE  "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
