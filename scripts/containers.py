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

#__all__ = ["KDTreeBackend"]


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

from kdtree_backend import *
import nets
from nets import *


########## CONTAINER CLASSES ########### {{{1

clear_all_containers = None
recompute_features_all_ind = None


@registry.register
class SelfAdaptiveNoveltyArchive(Container):
    """TODO""" # TODO

    def __init__(self, iterable: Optional[Iterable] = None,
            rebalancing_period: int = 100,
            k: int = 15, k_resolution: int = 60000,  # k_resolution: increase to have more capacity
            threshold_novelty: float = 0.01, novelty_distance: Union[str, Callable] = "euclidean",
            epsilon_dominance: bool = False, epsilon: float = 0.1,
            parents: Sequence[ContainerLike] = [], **kwargs: Any) -> None:
        self.rebalancing_period = rebalancing_period
        self.k = k
        self.k_resolution = k_resolution
        self.threshold_novelty = threshold_novelty
        self.novelty_distance = novelty_distance
        self.epsilon_dominance = epsilon_dominance
        self.epsilon = epsilon
        if len(parents) == 0:
            raise ValueError("``parents`` must contain at least one parent container to create an archive container.")

        # XXX hack: set backend directly
        features_score_names = kwargs.get('features_score_names', [])
        features_domain = kwargs['features_domain']
        storage_type = kwargs.get('storage_type', [])
        storage = self._create_storage(storage_type)
        backend = KDTreeBackend(len(features_domain), base_backend=storage, features_score_names=features_score_names)
        kwargs['storage_type'] = backend

        super().__init__(iterable, parents=parents, **kwargs)


    def _add_internal(self, individual: IndividualLike, raise_if_not_added_to_parents: bool, only_to_parents: bool) -> Optional[int]:
        if self.nb_operations + self.nb_rejected % self.rebalancing_period == 0:
            self.items.rebalance()
        all_parents = self.all_parents_inds()

        # Compute novelty
        if len(self) == 0:
            novelty = np.inf
        else:
            knn_res = self.items.knn(individual, self.k)
            nn, dists = tuple(zip(*knn_res))
            novelty = np.mean(dists)
        #print(f"DEBUG _add_internal {len(self)} novelty={novelty}")

        # Check if individual should be added
        if self.epsilon_dominance:
            dist_fst_nn = dists[0] if len(self) > 0 else np.inf
            if dist_fst_nn > self.threshold_novelty:
                # Add individual
                return super()._add_internal(individual, raise_if_not_added_to_parents, only_to_parents)
            else:
                # Check if this individual dominates its nearest neighbour according to the 3 epsilon-dominance criteria:
                ind_nn = nn[0]
                ind_nn_fit = self.get_ind_fitness(ind_nn)
                ind_fit = self.get_ind_fitness(individual)
                ind_nn_knn = self.items.knn(ind_nn, self.k)
                ind_nn_nn, ind_nn_dists = tuple(zip(*ind_nn_knn))
                ind_nn_novelty = np.mean(ind_nn_dists)
                cond_novelty = novelty >= (1. - self.epsilon) * ind_nn_novelty
                cond_quality = np.all([ind_fit[i] >= (1. - self.epsilon) * ind_nn_fit[i] for i in range(len(ind_fit))])
                cond_both = np.all([(novelty - ind_nn_novelty) * ind_nn_fit[i] > -(ind_fit[i] - ind_nn_fit[i]) * ind_nn_novelty for i in range(len(ind_fit))])
                #print(f"DEBUG epsilon_dominance: {cond_novelty} {cond_quality} {cond_both}")
                if cond_novelty and cond_quality and cond_both:
                    self.discard(ind_nn)
                    return super()._add_internal(individual, raise_if_not_added_to_parents, only_to_parents)
                else:
                    return super()._add_internal(individual, raise_if_not_added_to_parents, True)

        else:
            # Check if individual is sufficiently novel
            if novelty > self.threshold_novelty:
                # Add individual
                return super()._add_internal(individual, raise_if_not_added_to_parents, only_to_parents)

            else:
                ind_nn = nn[0]
                ind_nn_fit = self.get_ind_fitness(ind_nn)
                ind_fit = self.get_ind_fitness(individual)
                if ind_fit.dominates(ind_nn_fit):
                    self.discard(ind_nn)
                    return super()._add_internal(individual, raise_if_not_added_to_parents, only_to_parents)
                else:
                    return super()._add_internal(individual, raise_if_not_added_to_parents, True)


    # Inspired from original code of Cully2019 "Autonomous skill discovery with Quality-Diversity and Unsupervised Descriptors"
    def compute_new_threshold(self, max_nb_inds = 50000) -> None:
        # Randomly selects individual and regroup their features descriptors into a matrix
        parents_inds = self.all_parents_inds()
        if len(parents_inds) > max_nb_inds:
            inds = random.sample(parents_inds, k=max_nb_inds)
            fts = np.array([ind.features for ind in inds], dtype=np.float16)
        else:
            fts = np.array([ind.features for ind in parents_inds], dtype=np.float16)
        xx = np.sum(fts**2., 1)
        xx = xx.reshape(xx.shape + (1,))
        xy = (2*fts) @ fts.T
        dist = xx @ np.ones((1, fts.shape[0]), dtype=np.float16)
        dist += np.ones((fts.shape[0], 1), dtype=np.float16) @ xx.T
        dist -= xy
        maxdist = np.sqrt(np.max(dist))
        self.threshold_novelty = maxdist / np.sqrt(self.k_resolution)
        print(f"DEBUG compute_new_threshold: {self.threshold_novelty}")


#    def get_ind_features(self, individual: IndividualLike, *args, **kwargs) -> FeaturesLike:
#        print("DEBUG SelfAdaptiveNoveltyArchive.get_ind_features")
#        return super().get_ind_features(individual, *args, **kwargs)



# Unbound get_ind_features method of the ``DebugTorchFeatureExtractionContainerDecorator`` class
def _TorchMultiFeatureExtractionContainerDecorator_get_ind_features(self, individual: IndividualLike, *args, **kwargs) -> FeaturesLike:
    #print("DEBUG _TorchMultiFeatureExtractionContainerDecorator_get_ind_features")
    # Extracted features are already computed. Use the stored values.
    if f"extracted_{id(self)}_0" in individual.scores:
        latent_scores = []
        i = 0
        while f"extracted_{id(self)}_{i}" in individual.scores:
            latent_scores.append(individual.scores[f"extracted_{id(self)}_{i}"])
            i += 1
        return Features(latent_scores)
    res = self.compute_latent([individual])[0]
    #print(f"DEBUG get_ind_features: {res}")
    return res




@registry.register
class TorchMultiFeatureExtractionContainerDecorator(TorchFeatureExtractionContainerDecorator):
    """TODO""" # TODO

    def __init__(self, container: ContainerLike,
            div_coeff: float = 0.5,
            diversity_loss_computation: str = "outputs",
            reset_model_every_training: bool = False,
            train_only_on_last_inds: bool = False,
            training_budget: Optional[int] = None,
            training_period_type: str = "linear",
            tanh_encoder: bool = False,
            model_type: str = "AE",
            epochs_avg_loss: int = 50,
            validation_split: float = 0.25,
            nb_training_sessions: int = 5,
            nb_filters: int = 4,
            batch_norm_before_latent: bool = True,
            trainer_type: str = "NNTrainer",
            **kwargs: Any) -> None:
        self.div_coeff = div_coeff
        self.diversity_loss_computation = diversity_loss_computation
        self.reset_model_every_training = reset_model_every_training
        self.train_only_on_last_inds = train_only_on_last_inds
        self.training_budget = training_budget
        self.training_period_type = training_period_type
        self.tanh_encoder = tanh_encoder
        self.model_type = model_type
        self.epochs_avg_loss = epochs_avg_loss
        self.validation_split = validation_split
        self.nb_training_sessions = nb_training_sessions
        self.nb_filters = nb_filters
        self.batch_norm_before_latent = batch_norm_before_latent
        super().__init__(container, **kwargs)

        self.last_recomputed = 0
        self._training_id = 0
        self._last_training_nb_inds2 = 0
        #self.current_loss = 0.
        #self.current_loss_reconstruction = np.nan
        #self.current_loss_diversity = np.nan

        self.trainer_type = trainer_type
        if self.trainer_type == "NNTrainer":
            trainer_class = NNTrainer
        elif self.trainer_type == "IterativeNNTrainer":
            trainer_class = IterativeNNTrainer
        else:
            raise ValueError("``trainer_type`` must be either 'NNTrainer' or 'IterativeNNTrainer'.")
        self.trainer = trainer_class(nb_training_sessions=self.nb_training_sessions, nb_epochs=self.nb_epochs,
                learning_rate=self.learning_rate, batch_size=self.batch_size, epochs_avg_loss=self.epochs_avg_loss,
                validation_split=self.validation_split, reset_model_every_training=self.reset_model_every_training,
                diversity_loss_computation=self.diversity_loss_computation, div_coeff=self.div_coeff)


    def _init_methods_hooks(self):
        self._orig_get_ind_features = self.container.get_ind_features
        self.container.get_ind_features = partial(_TorchMultiFeatureExtractionContainerDecorator_get_ind_features, self) # type: ignore
        self.get_ind_features = partial(_TorchMultiFeatureExtractionContainerDecorator_get_ind_features, self) # type: ignore


    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict['_orig_get_ind_features']
#        if 'get_ind_features' in odict['container'].__dict__:
#            del odict['container'].__dict__['get_ind_features']
        del odict['get_ind_features']
        #del odict['trainer'] # XXX
        return odict

    # Note: we change the ``get_ind_features`` and ``add`` methods of ``self.container``. So it's necessary to update them here when objects of this class are unpickled
    def __setstate__(self, state):
        if '_orig_get_ind_features' in state:
            del state['_orig_get_ind_features']
        if 'get_ind_features' in state['container'].__dict__:
            del state['container'].__dict__['get_ind_features']
        #del state['_orig_add']
        #del state['container'].__dict__['add']
        self.__dict__.update(state)
        self._init_methods_hooks()


    def compute_latent(self, inds: Sequence[IndividualLike]) -> Sequence[FeaturesLike]:
        assert(len(inds) > 0)
        if self.model == None:
            self._create_default_model(inds[0])

        # Evaluate using trainer (get latent representation of the observations)
        latent_scores = self.trainer.eval(inds, self.model)[0]
        #print(f"DEBUG compute_latent {len(inds)} {latent_scores} {self.model}")

        # Store latent scores into each individual
        for ls, ind in zip(latent_scores, inds):
            for j, s in enumerate(ls):
                ind.scores[f"extracted_{id(self)}_{j}"] = s

        #print("DEBUG scores:", latent_scores)
        # Return final values
        return [Features(ls) for ls in latent_scores]



    def _create_default_model(self, example_ind: IndividualLike) -> None:
        print(f"DEBUG {self.name} ############# CREATE DEFAULT MODEL ###############")
#        # Identify base scores
#        if self.base_scores == None: # Not base scores specified, use all scores (except those created through feature extraction)
#            base_scores = [str(x) for x in example_ind.scores.keys() if not x.startswith("extracted_") ]
#        else:
#            base_scores = self.base_scores # type: ignore
        # Find default model parameters
        input_size = example_ind.scores['observations'].shape[-1] #len(base_scores)
        assert(self.container.features_domain != None)
        latent_size = len(self.container.features_domain) # type: ignore
        # Set extracted scores names as the default features of the container
        self.container.features_score_names = [f"extracted_{id(self)}_{j}" for j in range(latent_size)]
        # Create simple auto-encoder as default model
        #self.model = AE(input_size, latent_size, self.tanh_encoder)
        if self.model_type == "AE":
            self.model = AE(input_size, latent_size, self.tanh_encoder)
        elif self.model_type == "ConvAE":
            self.model = ConvAE(input_size, latent_size, 2, self.nb_filters, self.batch_norm_before_latent)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}.")
        # Register model in global dict
        global nn_models
        nn_models[self.name] = self.model
        # Create ensemble model
        self.trainer.create_ensemble_model(nn_models)

    def _train_and_recompute_if_needed(self, update_params=()):
        global last_training_nb_inds

        # Train and recomputed all features if necessary
        training_inds = self._get_training_inds()
        nb_training_inds = len(training_inds)

        # Create a model, if none exist
        if self.model == None and nb_training_inds > 0:
            self._create_default_model(training_inds[0])

        if self.training_period_type == "linear":
            do_recomputation = nb_training_inds >= self.training_period and nb_training_inds % self.training_period == 0 and nb_training_inds != self._last_training_nb_inds
        elif self.training_period_type == "exp_decay":
            do_recomputation = nb_training_inds >= self._last_training_nb_inds2 + self.training_period * 2**(self._training_id)
            if do_recomputation:
                print(f"DEBUG _train_and_recompute_if_needed exp_decay {nb_training_inds} {self._last_training_nb_inds2} {self.training_period} {self._training_id} {self._last_training_nb_inds2 + self.training_period * 2**(self._training_id)}")
                self._last_training_nb_inds2 = self._last_training_nb_inds2 + self.training_period * 2**(self._training_id)
        else:
            raise ValueError(f"Unknown training_period_type: {self.training_period_type}.")

        #print("DEBUG add !", nb_training_inds, self.training_period, self._last_training_nb_inds)
        if do_recomputation:
            self._training_id += 1
            do_training = nb_training_inds - last_training_nb_inds >= self.training_period
            print(f"DEBUG {self.name} _train_and_recompute_if_needed: {nb_training_inds} {last_training_nb_inds} {self.training_period}")
            self._last_training_nb_inds = nb_training_inds
            try:
                if do_training:
                    #self.clear() # type: ignore
                    global clear_all_containers
                    clear_all_containers() # type: ignore # XXX HACK
                    print(f"DEBUG {self.name} DO TRAINING !")
                    #self.train(self.nb_epochs if nb_training_inds > self.training_period else self.initial_nb_epochs)
                    self.train()
                    last_training_nb_inds = nb_training_inds
                    print(f"DEBUG {self.name} RECOMPUTE FEATURES")
                    #self.recompute_features_all_ind(update_params)
                    global recompute_features_all_ind
                    recompute_features_all_ind()
                self.last_recomputed = nb_training_inds
            except Exception as e:
                print("Training failed !")
                traceback.print_exc()
                raise e

#        elif self.last_recomputed < last_training_nb_inds:
##            self.clear() # type: ignore
##            print(f"DEBUG {self.name} RECOMPUTE FEATURES")
##            self.recompute_features_all_ind(update_params)
#            self.last_recomputed = nb_training_inds
#            if hasattr(self.container, 'compute_new_threshold'): # XXX
#                self.container.compute_new_threshold() # XXX
#            if hasattr(self.container, '_add_rescaling'): # XXX HACK
#                print(f"DEBUG {self.name} call _add_rescaling")
#                self.container._add_rescaling(training_inds) # XXX HACK


    def recompute_features_all_ind(self, update_params={}) -> None:
        #print("DEBUG: features recomputed for all inds..")
        #start_time = timer() # XXX

        training_inds = self._get_training_inds()
        self.compute_latent(training_inds)
        #elapsed = timer() - start_time # XXX
        #print(f"# Features recomputed for {len(training_inds)} inds.. Elapsed={elapsed}") # XXX
        #start_time = timer() # XXX
        self.container.update(training_inds, **update_params)
        #for i in training_inds:
        #    try:
        #        #self.container.add(i)
        #        self._orig_add(i)
        #    except Exception:
        #        pass

        if hasattr(self.container, 'compute_new_threshold'): # XXX
            self.container.compute_new_threshold() # XXX
        if hasattr(self.container, '_add_rescaling'): # XXX HACK
            print(f"DEBUG {self.name} call _add_rescaling")
            self.container._add_rescaling(training_inds) # XXX HACK

        #elapsed = timer() - start_time # XXX
        #print(f"# Tried adding {len(training_inds)} inds back to the container: {len(self)} added.. Elapsed={elapsed}") # XXX



    def train(self) -> None:
        global nn_models
        global last_training_size
        #print("###########  DEBUG: training.. ###########")
        #start_time = timer() # XXX

        training_inds = self._get_training_inds()
        assert(len(training_inds) > 0)
        assert(self.model != None)

        # Skip training if we already exceed the training budget
        if self.training_budget != None and len(training_inds) > self.training_budget:
            return

        # If needed, only use the last inds of the training set
        if self.train_only_on_last_inds:
            nb_new_inds = len(training_inds) - last_training_size
            #print(f"DEBUG train_only_on_last_inds: {nb_new_inds} {last_training_size}")
            last_training_size = len(training_inds)
            training_inds = training_inds[-nb_new_inds:]
        else:
            last_training_size = len(training_inds)
        print(f" training size: {len(training_inds)}")

        # Train !
        self.trainer.create_ensemble_model(nn_models)
        self.trainer.train(training_inds)

#        # Identify base scores
#        if self.base_scores == None: # Not base scores specified, use all scores (except those created through feature extraction)
#            base_scores: List[Any] = [x for x in training_inds[0].scores.keys() if not x.startswith("extracted_") ]
#        else:
#            base_scores = self.base_scores # type: ignore
#
#        # Build dataset
#        data = torch.empty(len(training_inds), len(base_scores))
#        for i, ind in enumerate(training_inds):
#            for j, s in enumerate(base_scores):
#                data[i,j] = ind.scores[s]
#        #dataset = torch.utils.data.TensorDataset(data)
#
#        # Normalize dataset
#        data = (data - data.min()) / (data.max() - data.min())
#
#        # Create an ensemble model
#        model = EnsembleAE(list(nn_models.values()))
#
#        # Reset model, if needed
#        if self.reset_model_every_training:
#            model.reset()
#
#        # Train !
#        loss_lst = []
#        loss_reconstruction_lst = []
#        loss_diversity_lst = []
#        for _ in range(self.nb_training_sessions):
#            l, r, d = self._training_session(data, model, self.nb_epochs, self.epochs_avg_loss, self.validation_split)
#            loss_lst.append(l)
#            loss_reconstruction_lst.append(r)
#            loss_diversity_lst.append(d)

        # Compute and save mean losses
#        self.current_loss = np.mean(loss_lst)
#        self.current_loss_reconstruction = np.mean(loss_reconstruction_lst)
#        self.current_loss_diversity = np.mean(loss_diversity_lst)
#        global current_loss, current_loss_reconstruction, current_loss_diversity
#        current_loss = self.current_loss
#        current_loss_reconstruction = self.current_loss_reconstruction
#        current_loss_diversity = self.current_loss_diversity

        nets.current_loss = self.trainer.current_loss
        nets.current_loss_reconstruction = self.trainer.current_loss_reconstruction
        nets.current_loss_diversity = self.trainer.current_loss_diversity

        #elapsed = timer() - start_time # XXX
        #print(f"# Training: final loss: {loss}  elapsed: {elapsed}")



# MODELINE  "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
