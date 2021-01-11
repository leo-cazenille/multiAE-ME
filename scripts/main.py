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

# Pytorch
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.autograd import Variable
from torch.utils.data import DataLoader


import matplotlib.pyplot as plt

from kdtree_backend import *
import curiosity
from nets import *
from metrics import *




########## CONTAINER CLASS ########### {{{1


@registry.register
class SelfAdaptiveNoveltyArchive(Container):
    """TODO""" # TODO

    def __init__(self, iterable: Optional[Iterable] = None,
            rebalancing_period: int = 100,
            k: int = 15, k_resolution: int = 60000,
            threshold_novelty: float = 0.01, novelty_distance: Union[str, Callable] = "euclidean",
            parents: Sequence[ContainerLike] = [], **kwargs: Any) -> None:
        self.rebalancing_period = rebalancing_period
        self.k = k
        self.k_resolution = k_resolution
        self.threshold_novelty = threshold_novelty
        self.novelty_distance = novelty_distance
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
    def compute_new_threshold(self) -> None:
        fts = np.array([ind.features for ind in self.all_parents_inds()])
        xx = np.sum(fts**2., 1)
        xx = xx.reshape(xx.shape + (1,))
        xy = (2*fts) @ fts.T
        dist = xx @ np.ones((1, fts.shape[0]))
        dist += np.ones((fts.shape[0], 1)) @ xx.T
        dist -= xy
        maxdist = np.sqrt(np.max(dist))
        self.threshold_novelty = maxdist / np.sqrt(self.k_resolution)
        print(f"DEBUG compute_new_threshold: {self.threshold_novelty}")




# Unbound get_ind_features method of the ``DebugTorchFeatureExtractionContainerDecorator`` class
def _TorchMultiFeatureExtractionContainerDecorator_get_ind_features(self, individual: IndividualLike, *args, **kwargs) -> FeaturesLike:
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
        super().__init__(container, **kwargs)

        self.last_recomputed = 0
        self._training_id = 0
        self._last_training_nb_inds2 = 0
        #self.current_loss = 0.
        #self.current_loss_reconstruction = np.nan
        #self.current_loss_diversity = np.nan

        self.trainer = NNTrainer(nb_training_sessions=self.nb_training_sessions, nb_epochs=self.nb_epochs,
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
        if 'get_ind_features' in odict['container'].__dict__:
            del odict['container'].__dict__['get_ind_features']
        del odict['get_ind_features']
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
        input_size = example_ind.observations.shape[-1] #len(base_scores)
        assert(self.container.features_domain != None)
        latent_size = len(self.container.features_domain) # type: ignore
        # Set extracted scores names as the default features of the container
        self.container.features_score_names = [f"extracted_{id(self)}_{j}" for j in range(latent_size)]
        # Create simple auto-encoder as default model
        #self.model = AE(input_size, latent_size, self.tanh_encoder)
        if self.model_type == "AE":
            self.model = AE(input_size, latent_size, self.tanh_encoder)
        elif self.model_type == "ConvAE":
            self.model = ConvAE(input_size, latent_size, self.nb_filters)
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
                self.clear() # type: ignore
                if do_training:
                    print(f"DEBUG {self.name} DO TRAINING !")
                    #self.train(self.nb_epochs if nb_training_inds > self.training_period else self.initial_nb_epochs)
                    self.train()
                    last_training_nb_inds = nb_training_inds
                print(f"DEBUG {self.name} RECOMPUTE FEATURES")
                self.recompute_features_all_ind(update_params)
                if hasattr(self.container, 'compute_new_threshold'): # XXX
                    self.container.compute_new_threshold() # XXX
                self.last_recomputed = nb_training_inds
            except Exception as e:
                print("Training failed !")
                traceback.print_exc()
                raise e

        elif self.last_recomputed < last_training_nb_inds:
            self.clear() # type: ignore
            print(f"DEBUG {self.name} RECOMPUTE FEATURES")
            self.recompute_features_all_ind(update_params)
            self.last_recomputed = nb_training_inds


#    def _compute_loss(self, data, model):
#        criterion_reconstruction = nn.MSELoss()
#        criterion_diversity = nn.MSELoss()
#
#        d = Variable(data)
#        #print(f"DEBUG training2: {d} {d.shape}")
#        output = model(d) # type: ignore
#
#        # Compute reconstruction loss
#        loss_reconstruction = 0.
#        for r in output:
#            loss_reconstruction += criterion_reconstruction(r, d)
#        loss_reconstruction /= len(output)
#
#        # Compute diversity loss
#        loss_diversity = 0.
#        if self.diversity_loss_computation == "outputs":
#            #mean_output = 0.
#            #for r in output:
#            #    mean_output += r
#            #mean_output /= len(output)
#            mean_output = [torch.mean(o, 0) for o in output]
#            mean_output = sum(mean_output) / len(mean_output)
#            with warnings.catch_warnings():
#                warnings.simplefilter("ignore")
#                for r in output:
#                    loss_diversity += criterion_diversity(r, mean_output)
#                    #print(f"DEBUG outputs: {r.shape} {mean_output.shape} {criterion_diversity(r, mean_output)}")
#        elif self.diversity_loss_computation == "latent":
#            latent = model.encoders(d)
#            #mean_latent = 0.
#            #for l in latent:
#            #    mean_latent += l
#            #mean_latent /= len(latent)
#            mean_latent = [torch.mean(l, 0) for l in latent]
#            mean_latent = sum(mean_latent) / len(mean_latent)
#            with warnings.catch_warnings():
#                warnings.simplefilter("ignore")
#                for r in latent:
#                    loss_diversity += criterion_diversity(r, mean_latent)
#                    #print(f"DEBUG latent: {r.shape} {mean_latent.shape} {criterion_diversity(r, mean_latent)}")
#        else:
#            raise ValueError(f"Unknown diversity_loss_computation type: {self.diversity_loss_computation}.")
#
#        loss = loss_reconstruction - self.div_coeff * loss_diversity
##                loss = loss_reconstruction #- self.div_coeff * loss_diversity
#        return loss, loss_reconstruction, loss_diversity
#
#
#    def _training_session(self, data, model, nb_epochs: int = 500, epochs_avg_loss: int = 50, validation_split: float = 0.25):
#        # Create optimizer
#        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5) # type: ignore
#        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.9 ** (epoch / nb_epochs))
#
#        dataloader: Any = DataLoader(data, batch_size=self.batch_size, shuffle=True) # type: ignore
#        train_size = int(np.floor(len(dataloader) * (1. - validation_split)))
#        validation_size = int(np.ceil(len(dataloader) * validation_split))
#        train_dataset, validation_dataset = torch.utils.data.random_split(list(dataloader), [train_size, validation_size])
#
#        rv_loss_lst = []
#        last_mean_rv_loss = np.inf
#        for epoch in range(nb_epochs):
#            # Training
#            rt_loss = 0.
#            rt_loss_reconstruction = 0.
#            rt_loss_diversity = 0.
#            model.train()
#            for data in train_dataset:
#                t_loss, t_loss_reconstruction, t_loss_diversity = self._compute_loss(data, model)
#                optimizer.zero_grad()
#                t_loss.backward()
#                optimizer.step()
#                scheduler.step()
#
#                rt_loss += t_loss
#                rt_loss_reconstruction += t_loss_reconstruction
#                rt_loss_diversity += t_loss_diversity
#            #print(f"loss: {loss}")
#
#            # Validation
#            rv_loss = 0.
#            rv_loss_reconstruction = 0.
#            rv_loss_diversity = 0.
#            model.eval()
#            with torch.no_grad():
#                for data in validation_dataset:
#                    v_loss, v_loss_reconstruction, v_loss_diversity = self._compute_loss(data, model)
#                    rv_loss += v_loss
#                    rv_loss_reconstruction += v_loss_reconstruction
#                    rv_loss_diversity += v_loss_diversity
#
##            # Check stopping criterion
##            if len(rv_loss_lst) >= epochs_avg_loss:
##                del rv_loss_lst[0]
##            rv_loss_lst.append(rv_loss)
##            mean_rv_loss = np.mean(rv_loss_lst)
##            if epoch > epochs_avg_loss and mean_rv_loss > last_mean_rv_loss:
##                break
##            last_mean_rv_loss = mean_rv_loss
##
##        return rv_loss.item(), rv_loss_reconstruction.item(), rv_loss_diversity.item()
#
#            # Check stopping criterion
#            if len(rv_loss_lst) >= epochs_avg_loss:
#                del rv_loss_lst[0]
#            rv_loss_lst.append(v_loss)
#            mean_rv_loss = np.mean(rv_loss_lst)
#            if epoch > epochs_avg_loss and mean_rv_loss > last_mean_rv_loss:
#                print("Training: stop early")
#                break
#            last_mean_rv_loss = mean_rv_loss
#
#        return v_loss.item(), v_loss_reconstruction.item(), v_loss_diversity.item()
#

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

        global current_loss, current_loss_reconstruction, current_loss_diversity
        current_loss = self.trainer.current_loss
        current_loss_reconstruction = self.trainer.current_loss_reconstruction
        current_loss_diversity = self.trainer.current_loss_diversity

        #elapsed = timer() - start_time # XXX
        #print(f"# Training: final loss: {loss}  elapsed: {elapsed}")



########## EXPERIMENT CLASSES ########### {{{1

class MultiAEExperiment(QDExperiment):

    def __init__(self, config_filename, parallelism_type = "concurrent", seed = None, base_config = None):
        super().__init__(config_filename, parallelism_type, seed, base_config)
        self.default_curiosity = self.config.get('default_curiosity', 0.01)
        self.curiosity_reward = self.config.get('curiosity_reward', 1.0)
        self.curiosity_penalty = self.config.get('curiosity_penalty', 0.5)

    def _tell_curiosity(self, algo: QDAlgorithmLike, ind: IndividualLike, added: bool, xattr: Mapping[str, Any] = {}) -> None:
        if not hasattr(ind, 'parent'):
            ind.parent = None
            return
        parent = ind.parent
        if parent == None:
            return
        curiosity = parent.scores.get('curiosity', self.default_curiosity)
        if added:
            curiosity += self.curiosity_reward
        else:
            curiosity -= self.curiosity_penalty
        parent.scores['curiosity'] = curiosity
        #print(f"_tell_curiosity {parent.scores['curiosity']}")


    def reinit_curiosity(self):
        # Create `tell` callbacks to all algos, to compute curiosity
        #for alg in self.algo.algorithms:
        #    alg.add_callback("tell", self._tell_curiosity)
        self.algo.add_callback("tell", self._tell_curiosity)


    def reinit_loggers(self):
        # Update stats
        #stat_loss = LoggerStat("loss", lambda algo: f"{algo.container.current_loss:.4f}", True)
        global current_loss
        stat_loss = LoggerStat("loss", lambda algo: f"{current_loss:.4f}", True)
        stat_loss_reconstruction = LoggerStat("loss_reconstruction", lambda algo: f"{current_loss_reconstruction:.4f}", True)
        stat_loss_diversity = LoggerStat("loss_diversity", lambda algo: f"{current_loss_diversity:.4f}", True)
        stat_training = LoggerStat("training_size", lambda algo: f"{len(algo.container._get_training_inds())}", True)
        self.logger.register_stat(stat_loss, stat_loss_reconstruction, stat_loss_diversity, stat_training)
        self.logger._tabs_size = 5
        self.logger._min_cols_size = 9

        # Create additional loggers
        algos = self.algo.algorithms
        self.algs_loggers = []
        for algo in algos:
            iteration_filenames = os.path.join(self.log_base_path, f"iteration-{algo.name}-%i_" + self.instance_name + ".p")
            final_filename = os.path.join(self.log_base_path, f"final-{algo.name}_" + self.instance_name + ".p")
            logger = AlgorithmLogger(algo, verbose=False,
                    iteration_filenames=iteration_filenames, final_filename=final_filename, save_period=self.save_period)
            self.algs_loggers.append(logger)

        def fn_qd_score(i_alg, algo):
            return f"{algo.algorithms[i_alg].container.qd_score(True):.2f}"
        def fn_originality(i_alg, algo):
            return f"{originality(algo.algorithms[i_alg].container, [a.container for a in algo.algorithms])}"
        def fn_mean_originality(algo):
            return f"{mean_originality([a.container for a in algo.algorithms])}"
        if hasattr(self.algo, 'algorithms'):
            algos = self.algo.algorithms
            for i_alg, alg in enumerate(algos):
                stat_qd_score = LoggerStat(f"qd_score-{alg.name}", partial(fn_qd_score, i_alg), True)
                self.logger.register_stat(stat_qd_score)
            for i_alg, alg in enumerate(algos):
                stat_originality = LoggerStat(f"originality-{alg.name}", partial(fn_originality, i_alg), True)
                self.logger.register_stat(stat_originality)
            stat_mean_originality = LoggerStat(f"mean_originality", fn_mean_originality, True)
            self.logger.register_stat(stat_mean_originality)



class RastriginExperiment(MultiAEExperiment):
    def reinit(self):
        self.nb_features = self.config.get('nb_features', 4)
        self.bench = artificial_landscapes.NormalisedRastriginBenchmark(nb_features = self.nb_features)
        #self.config['fitness_type'] = "perf"
        ##self.config['perfDomain'] = (0., artificial_landscapes.rastrigin([4.5]*2, 10.)[0])
        #self.config['perfDomain'] = (0., math.inf)
        #self.config['features_list'] = ["0", "1"]
        #self.config['0Domain'] = self.bench.features_domain[0]
        #self.config['1Domain'] = self.bench.features_domain[1]
        self.config['algorithms']['ind_domain'] = self.bench.ind_domain
        #self.features_domain = self.bench.features_domain
        super().reinit()
        self.eval_fn = self.bench.fn
        self.optimisation_task = self.bench.default_task
        super().reinit_curiosity()
        super().reinit_loggers()



class BallisticEnv(object):
    M: float = 1.0
    nb_steps: int = 50
    dt: float = 1.0
    Fmax: float = 200.

    def __init__(self):
        self._gt = []
        self.cart_traj = np.empty((self.nb_steps, 2))
        self.polar_traj = np.empty_like(self.cart_traj)
        self.theta = 0.
        self.F = 0.
        self._dead = False

    @property
    def observations(self):
        return self.cart_traj

    def get_flat_observations(self):
        obs = self.observations
        obs_s = len(obs)
        data = np.empty(self.get_flat_obs_size())
        for t in range(obs_s):
            data[t] = obs[t][0]
            data[t+obs_s] = obs[t][1]
        return data

    def get_flat_obs_size(self):
        return np.prod(self.cart_traj.shape)

    @property
    def gt(self):
        return self._gt

    def get_observations_scores_names(self):
        return [f"o{i}" for i in range(self.get_flat_obs_size()) ]

    @property
    def dimensions(self):
        return 2

    @property
    def ind_domain(self):
        return [0.1, 0.9]

    def eval(self, ind):
        self.theta = ind[0] * np.pi / 2.
        self.F = ind[1] * self.Fmax
        self.simulate()
        fitness = 0.5,
        self._gt = self.ft_hardcoded()
        gen0, gen1 = self.ft_genotype()
        hardcoded0, hardcoded1 = self.ft_hardcoded()
        features = [hardcoded0, hardcoded1]
        #scores = {'observations': self.get_flat_observations(), 'gen0': gen0, 'gen1': gen1, 'hardcoded0': hardcoded0, 'hardcoded1': hardcoded1}
        obs = {f"o{i}": o for i, o in enumerate(self.get_flat_observations())}
        scores = {**obs, 'gen0': gen0, 'gen1': gen1, 'hardcoded0': hardcoded0, 'hardcoded1': hardcoded1} # XXX One key-val for each observation !!!
        ind.observations = self.cart_traj.T
        #if self._dead:
        #    fitness = -1.,
        #print(f"# DEBUG eval: {fitness} {features} {scores}")
        return fitness, features, scores


    def simulate(self):
        a = np.array([self.F * np.cos(self.theta) / self.M, (self.F * np.sin(self.theta)-9.81)/ self.M])
        if self.F * np.sin(self.theta) <= 9.81*3.:
            self.cart_traj[:] = 0.
            self.polar_traj[:] = 0.
            self._dead = True

        v = np.zeros(2)
        p = np.zeros(2)
        polar = np.zeros(2)
        self.cart_traj[0] = p
        self.polar_traj[0] = polar

        for t in range(1, self.nb_steps):
            v += a * self.dt
            p += v * self.dt
            a[:] = 0., -9.81
            if p[1] <= 0: # Contact with the ground
                p[1] = 0.
                a[1] = -0.6 * v[1] # Dumping factor
                v[1] = 0.

            polar[:] = np.linalg.norm(p), np.arctan2(p[1], p[0])
            self.cart_traj[t] = p
            self.polar_traj[t] = polar


    def ft_genotype(self):
        return [self.theta / (np.pi / 2.) * 2. - 1.,
                self.F / 200. * 2. - 1.]

    def ft_hardcoded(self):
        Vx = np.cos(self.theta) * self.F
        Vy = np.sin(self.theta) * self.F - 9.81
        Px = Vx / 2.
        Py = Vy / 2.
        tmax = (np.sin(self.theta) * self.F) / 9.81 - 1
        return [ (Vx * tmax + Px) / 2000.*2. - 1., 
                 (-9.81 * 0.5 * tmax*tmax + Vy * tmax + Py) / 2000.*2. - 1.]



class BallisticExperiment(MultiAEExperiment):

#    def __init__(self, config_filename, parallelism_type = "concurrent", seed = None, base_config = None):
#        super().__init__(config_filename, parallelism_type, seed, base_config)

    def _eval(self, ind):
        env = BallisticEnv()
        res = env.eval(ind)
        #print(f"DEBUG _eval: {res}")
        return res

    def reinit(self):
        # Base configs
        env = BallisticEnv()
        self.set_defaultconfig_entry(['algorithms', 'ind_domain'], env.ind_domain)
        self.set_defaultconfig_entry(['algorithms', 'dimension'], env.dimensions)
        self.set_defaultconfig_entry(['containers', 'base_scores'], env.get_observations_scores_names())
        # Reinit
        super().reinit()
        self.eval_fn = self._eval
        self.optimisation_task = "minimisation"
        super().reinit_curiosity()
        super().reinit_loggers()



########## BASE FUNCTIONS ########### {{{1

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--configFilename', type=str, default='conf/test.yaml', help = "Path of configuration file")
    parser.add_argument('-o', '--resultsBaseDir', type=str, default='results/', help = "Path of results files")
    parser.add_argument('-p', '--parallelismType', type=str, default='concurrent', help = "Type of parallelism to use")
#    parser.add_argument('--replayBestFrom', type=str, default='', help = "Path of results data file -- used to replay the best individual")
    parser.add_argument('--seed', type=int, default=None, help="Numpy random seed")
    parser.add_argument('-v', '--verbose', default=False, action='store_true', help = "Enable verbose mode")
    return parser.parse_args()

def create_base_config(args):
    base_config = {}
    if len(args.resultsBaseDir) > 0:
        base_config['resultsBaseDir'] = args.resultsBaseDir
    base_config['verbose'] = args.verbose
    return base_config

def create_experiment(args, base_config):
    config = yaml.safe_load(open(args.configFilename))
    exp_type = config.get('experiment_type', 'rastrigin')
    if exp_type == 'rastrigin':
        exp = RastriginExperiment(args.configFilename, args.parallelismType, seed=args.seed, base_config=base_config)
    elif exp_type == 'ballistic':
        exp = BallisticExperiment(args.configFilename, args.parallelismType, seed=args.seed, base_config=base_config)
    else:
        raise ValueError(f"Unknown experiment type: {exp_type}.")
    print("Using configuration file '%s'. Experiment type '%s'. Instance name: '%s'" % (args.configFilename, exp_type, exp.instance_name))
    return exp

def launch_experiment(exp):
    exp.run()


def make_plots(exp):
    for logger in exp.algs_loggers:
        logger.save(os.path.join(logger.log_base_path, logger.final_filename))
        qdpy.plots.default_plots_grid(logger, exp.log_base_path, suffix=f"-{exp.instance_name}-{logger.algorithms[0].name}", to_grid_parameters={'features_domain': logger.algorithms[0].container.features_extrema})

    qdpy.plots.plot_iterations(exp.logger, os.path.join(exp.log_base_path, f"./iterations_loss-{exp.instance_name}.pdf"), "loss", ylabel="Loss")

    if hasattr(exp.container, "container"):
        parent_archive = exp.container.container.parents[0]
        ylim_contsize = (0, len(parent_archive)) if np.isinf(parent_archive.capacity) else (0, parent_archive.capacity)
        qdpy.plots.plot_iterations(exp.logger, os.path.join(exp.log_base_path, f"./iterations_trainingsize{exp.instance_name}.pdf"), "training_size", ylim=ylim_contsize, ylabel="Training size")

#
#        output_dir = exp.log_base_path
#        suffix=f"-{exp.instance_name}-{logger.algorithms[0].name}"
#        container = logger.algorithms[0].container # XXX
#        grid = container
#
#        plot_path = os.path.join(output_dir, f"performancesGrid{suffix}.pdf")
#        cmap_perf = "inferno" if logger.algorithms[0].optimisation_task == "maximisation" else "inferno_r"
#        fitness_domain = grid.fitness_domain
#        print(f"$$$$$$$$$$$$$ DEBUG1 {plot_path}")
#        qdpy.plots.plotGridSubplots(grid.quality_array[... ,0], plot_path, plt.get_cmap(cmap_perf), grid.features_domain, fitness_domain[0], nbTicks=None)
#        print("$$$$$$$$$$$$$ DEBUG2")




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
