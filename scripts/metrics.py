
########## IMPORTS ########### {{{1
#import copy
#import pickle
import numpy as np
#import warnings
#import random

## QDpy
#import qdpy
#from qdpy.base import *
#from qdpy.phenotype import *
#from qdpy.containers import *
#from qdpy.algorithms import *
#from qdpy import tools


########## METRICS ########### {{{1


def originality(c, conts):
    originality = 0.
    for ind in c:
        present = np.any([ind in other for other in conts if id(other) != id(c)])
        originality += not present
    originality /= len(c)
    return originality

def mean_originality(conts):
    return np.mean([originality(c, conts) for c in conts])

# MODELINE  "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
