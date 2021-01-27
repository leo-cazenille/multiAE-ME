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

# Pytorch
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.autograd import Variable
from torch.utils.data import DataLoader


# QDpy
import qdpy
from qdpy.base import *
from qdpy.experiment import QDExperiment
from qdpy.benchmarks import artificial_landscapes
from qdpy.algorithms import LoggerStat, TQDMAlgorithmLogger, AlgorithmLogger, QDAlgorithmLike
from qdpy.containers import TorchAE, TorchFeatureExtractionContainerDecorator, ContainerLike, NoveltyArchive, Container
import qdpy.plots


import matplotlib.pyplot as plt

import curiosity
import nets
from nets import *
from metrics import *
import containers
from containers import *
import sim


#import warnings
#warnings.simplefilter('always', UserWarning)




########## EXPERIMENT CLASSES ########### {{{1

class MultiAEExperiment(QDExperiment):

    def __init__(self, config_filename, parallelism_type = "concurrent", seed = None, base_config = None):
        super().__init__(config_filename, parallelism_type, seed, base_config)
        self.default_curiosity = self.config.get('default_curiosity', 0.01)
        self.curiosity_reward = self.config.get('curiosity_reward', 1.0)
        self.curiosity_penalty = self.config.get('curiosity_penalty', 0.5)

    def __getstate__(self):
        odict = self.__dict__.copy()
        if 'algo' in odict:
            del odict['algo']
        if 'container' in odict:
            del odict['container']
        if 'parent_container' in odict:
            del odict['parent_container']
        if 'logger' in odict:
            del odict['logger']
        if 'algs_loggers' in odict:
            del odict['algs_loggers']
        if 'klc_density_refs' in odict:
            del odict['klc_density_refs']
        return odict


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


    def reinit_globals(self):
        containers.clear_all_containers = partial(self.clear_all_containers)
        containers.recompute_features_all_ind = partial(self.recompute_features_all_ind)


    def reinit_curiosity(self):
        # Create `tell` callbacks to all algos, to compute curiosity
        #for alg in self.algo.algorithms:
        #    alg.add_callback("tell", self._tell_curiosity)
        self.algo.add_callback("tell", self._tell_curiosity)


    def _fn_loss(self, algo):
        return f"{nets.current_loss:.4f}"
    def _fn_loss_reconstruction(self, algo):
        return f"{nets.current_loss_reconstruction:.4f}"
    def _fn_loss_diversity(self, algo):
        return f"{nets.current_loss_diversity:.4f}"
    def _fn_train_size1(self, algo):
        return f"{len(algo.container._get_training_inds())}"
    def _fn_train_size2(self, algo):
        return f"{len(algo.container.all_parents_inds())}"
    def _fn_qd_score(self, i_alg, algo):
        return f"{algo.algorithms[i_alg].container.qd_score(True):.3f}"
    def _fn_originality(self, i_alg, algo):
        return f"{originality(algo.algorithms[i_alg].container, [a.container for a in algo.algorithms]):.3f}"
    def _fn_mean_originality(self, algo):
        return f"{mean_originality([a.container for a in algo.algorithms]):.3f}"
    def _fn_mean_corr(self, scores_names, algo):
        return f"{corr_scores(algo.algorithms[0].container.container.parents[0], scores_names)[1]:.4f}"
    def _fn_mean_cov(self, scores_names, algo):
        return f"{cov_scores(algo.algorithms[0].container.container.parents[0], scores_names)[1]:.4f}"
    def _fn_mean_abs_cov(self, scores_names, algo):
        return f"{abs_cov_scores(algo.algorithms[0].container.container.parents[0], scores_names)[1]:.4f}"


    def reinit_loggers(self):
        # Update stats
        #stat_loss = LoggerStat("loss", lambda algo: f"{algo.container.current_loss:.4f}", True)
        #global current_loss
        stat_loss = LoggerStat("loss", self._fn_loss , True)
        stat_loss_reconstruction = LoggerStat("loss_recon", self._fn_loss_reconstruction, True)
        stat_loss_diversity = LoggerStat("loss_div", self._fn_loss_diversity, True)
        if hasattr(self.algo.container, '_get_training_inds()'):
            stat_training = LoggerStat("train_size", self._fn_train_size1, True)
        else:
            stat_training = LoggerStat("train_size", self._fn_train_size2, True)
        self.logger.register_stat(stat_loss, stat_loss_reconstruction, stat_loss_diversity, stat_training)
        self.logger._tabs_size = 5
        self.logger._min_cols_size = 9

        # Create additional loggers
        if not hasattr(self.algo, 'algorithms'):
            return
        algos = self.algo.algorithms
        self.algs_loggers = []
        for algo in algos:
            #iteration_filenames = os.path.join(self.log_base_path, f"iteration-{algo.name}-%i_" + self.instance_name + ".p")
            #final_filename = os.path.join(self.log_base_path, f"final-{algo.name}_" + self.instance_name + ".p")
            iteration_filenames = None
            final_filename = None
            logger = AlgorithmLogger(algo, verbose=False,
                    iteration_filenames=iteration_filenames, final_filename=final_filename, save_period=self.save_period)
            self.algs_loggers.append(logger)

        if hasattr(self.algo, 'algorithms'):
            algos = self.algo.algorithms
            for i_alg, alg in enumerate(algos):
                stat_qd_score = LoggerStat(f"qd_score-{alg.name}", partial(self._fn_qd_score, i_alg), True)
                self.logger.register_stat(stat_qd_score)
            for i_alg, alg in enumerate(algos):
                stat_originality = LoggerStat(f"orig-{alg.name}", partial(self._fn_originality, i_alg), True)
                self.logger.register_stat(stat_originality)
            stat_mean_originality = LoggerStat(f"mean_orig", self._fn_mean_originality, True)
            self.logger.register_stat(stat_mean_originality)
            mean_corr_stat_scores_names = self.config.get("mean_corr_stat_scores_names", None)
            #stat_mean_corr = LoggerStat(f"mean_corr", partial(self._fn_mean_corr, mean_corr_stat_scores_names), True)
            #self.logger.register_stat(stat_mean_corr)
            #stat_mean_cov = LoggerStat(f"mean_cov", partial(self._fn_mean_cov, mean_corr_stat_scores_names), True)
            #self.logger.register_stat(stat_mean_cov)
            stat_mean_abs_cov = LoggerStat(f"mean_abs_cov", partial(self._fn_mean_abs_cov, mean_corr_stat_scores_names), True)
            self.logger.register_stat(stat_mean_abs_cov)

        # Save parent container in the 'container' entry of the data pickle file
        try:
            self.parent_container = self.algo.algorithms[0].container.container.parents[0]
        except Exception as e:
            self.parent_container = None
        if self.config.get("save_parent", True) == True:
            self.logger.saved_dict['container'] = self.parent_container



    def clear_all_containers(self):
        for alg in self.algo.algorithms:
            alg.container.clear()

    def recompute_features_all_ind(self, update_params={}) -> None:
        for alg in self.algo.algorithms:
            alg.container.recompute_features_all_ind(update_params)



    def _fn_klc(self, algo):
        #return f"{kl_coverage(algo.container, self.klc_reference_container, self.klc_scores_names, self.klc_nb_bins, self.klc_epsilon):.3f}"
        return f"{kl_coverage_stored_refs(algo.container, self.klc_density_refs, self.klc_refs_range, self.klc_scores_names, self.klc_nb_bins, self.klc_epsilon):.3f}"
        #return f"{kl_coverage2_stored_refs(algo.container, self.klc_density_refs, self.klc_refs_range, self.klc_scores_names, self.klc_nb_bins, self.klc_epsilon):.3f}"

    def load_ref_data(self):
        ref_file = self.config.get("reference_data_file", "")
        if len(ref_file) > 0:
            with open(ref_file, "rb") as f:
                ref_data = pickle.load(f)
            ref_cont = ref_data['container']
            self.parent_container.update(ref_cont)

        klc_reference_data_file = self.config.get("klc_reference_data_file", "")
        if len(klc_reference_data_file) > 0:
            self.klc_scores_names = self.config['klc_scores_names']
            self.klc_nb_bins = self.config['klc_nb_bins']
            self.klc_epsilon = self.config.get('klc_epsilon', 1e-20)
            with open(klc_reference_data_file, "rb") as f:
                ref_data = pickle.load(f)
            klc_reference_container = ref_data['container']
            self.klc_density_refs, self.klc_refs_range = kl_coverage_prepare_stored_refs(klc_reference_container, self.klc_scores_names, self.klc_nb_bins, self.klc_epsilon)
            #self.klc_density_refs, self.klc_refs_range = kl_coverage2_prepare_stored_refs(klc_reference_container, self.klc_scores_names, self.klc_nb_bins, self.klc_epsilon)
            stat_klc = LoggerStat(f"klc", self._fn_klc, True)
            self.logger.register_stat(stat_klc)



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
        super().reinit_globals()
        super().reinit_curiosity()
        super().reinit_loggers()
        super().load_ref_data()



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
        #obs = {f"o{i}": o for i, o in enumerate(self.get_flat_observations())}
        #scores = {**obs, 'gen0': gen0, 'gen1': gen1, 'hardcoded0': hardcoded0, 'hardcoded1': hardcoded1}
        scores = {'gen0': gen0, 'gen1': gen1, 'hardcoded0': hardcoded0, 'hardcoded1': hardcoded1}
        #ind.observations = self.cart_traj.T
        scores['observations'] = self.cart_traj.T
        #if self._dead: # XXX
        #    fitness = -1., # XXX
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
        super().reinit_globals()
        super().reinit_curiosity()
        super().reinit_loggers()
        super().load_ref_data()




class BipedalWalkerEval(object):

    def __init__(self, env_name, sim_model, indv_eps, max_episode_length, fitness_type, features_list, render_mode=False):
        self.env_name = env_name
        self.sim_model = sim_model
        self.indv_eps = indv_eps
        self.max_episode_length = max_episode_length
        self.fitness_type = fitness_type
        self.features_list = features_list
        self.render_mode = render_mode

    def eval_fn(self, ind, render_mode = False):
        #print(f"DEBUG ind len={len(ind)}")
        render_mode = self.render_mode if self.render_mode == True else render_mode
        env = sim.make_env(self.env_name)
        self.sim_model.set_model_params(ind)
        scores = sim.simulate(self.sim_model,
                env,
                render_mode=render_mode,
                num_episode=self.indv_eps, 
                max_episode_length=self.max_episode_length)
        ind.fitness.values = scores[self.fitness_type],
        ind.features.values = [scores[x] for x in self.features_list]
        ind.scores.update(scores)
        return ind

    def several_eval_fn(self, inds):
        res = []
        for ind in inds:
            res.append(self.eval_fn(ind))
        return res



class BipedalWalkerExperiment(MultiAEExperiment):

#    def __init__(self, config_filename, parallelism_type = "concurrent", seed = None, base_config = None):
#        super().__init__(config_filename, parallelism_type, seed, base_config)

    def reinit(self):
        # Init simulation model
        self.env_name = self.config['game']['env_name']
        self.sim_model = sim.Model(self.config['game'])

        # Base configs
        self.set_defaultconfig_entry(['algorithms', 'ind_domain'], [-1., 1.])
        self.set_defaultconfig_entry(['algorithms', 'dimension'], self.sim_model.param_count)
        print(f"CONFIG param_count: {self.sim_model.param_count}")
        self.set_defaultconfig_entry(['containers', 'base_scores'], ["meanAvgReward", "meanDistance", "meanHeadStability", "meanTorquePerStep", "meanJump", "meanLeg0HipAngle", "meanLeg0HipSpeed", "meanLeg0KneeAngle", "meanLeg0KneeSpeed", "meanLeg1HipAngle", "meanLeg1HipSpeed", "meanLeg1KneeAngle", "meanLeg1KneeSpeed"])
        # Reinit
        super().reinit()

        self.evalobj = BipedalWalkerEval(
                self.env_name,
                self.sim_model,
                self.config['indv_eps'],
                self.config.get('max_episode_length', 3000),
                self.fitness_type,
                self.features_list,
                render_mode = self.config.get('render_mode', False)
        )
        self.eval_fn = self.evalobj.eval_fn
        self.several_eval_fn = self.evalobj.several_eval_fn

        #self.eval_fn = self._eval
        self.optimisation_task = "minimisation"
        super().reinit_globals()
        super().reinit_curiosity()
        super().reinit_loggers()
        super().load_ref_data()


#    def eval_fn(self, ind, render_mode = False):
#        #print(f"DEBUG ind len={len(ind)}")
#        env = sim.make_env(self.env_name)
#        self.sim_model.set_model_params(ind)
#        scores = sim.simulate(self.sim_model,
#                env,
#                render_mode=render_mode,
#                num_episode=self.config['indv_eps'], 
#                max_episode_length=self.config.get('max_episode_length', 3000))
#        ind.fitness.values = scores[self.fitness_type],
#        ind.features.values = [scores[x] for x in self.features_list]
#        ind.scores.update(scores)
#        #obs = np.array(list(scores.values())) # TODO use real observations instead
#        #ind.scores['observations'] = obs
#        return ind





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
    parser.add_argument('--renderMode', default=False, action='store_true', help = "Enable render mode")
    parser.add_argument('-P', '--noProgressBar', default=False, action='store_true', help = "Disable TQDM progress bar")
    return parser.parse_args()

def create_base_config(args):
    base_config = {}
    if len(args.resultsBaseDir) > 0:
        base_config['resultsBaseDir'] = args.resultsBaseDir
    base_config['verbose'] = args.verbose
    base_config['render_mode'] = args.renderMode
    base_config['logger_type'] = "basic" if args.noProgressBar else "tqdm"
    return base_config

def create_experiment(args, base_config):
    config = yaml.safe_load(open(args.configFilename))
    exp_type = config.get('experiment_type', 'rastrigin')
    if exp_type == 'rastrigin':
        exp = RastriginExperiment(args.configFilename, args.parallelismType, seed=args.seed, base_config=base_config)
    elif exp_type == 'ballistic':
        exp = BallisticExperiment(args.configFilename, args.parallelismType, seed=args.seed, base_config=base_config)
    elif exp_type == 'bipedal_walker':
        exp = BipedalWalkerExperiment(args.configFilename, args.parallelismType, seed=args.seed, base_config=base_config)
    else:
        raise ValueError(f"Unknown experiment type: {exp_type}.")
    print("Using configuration file '%s'. Experiment type '%s'. Instance name: '%s'" % (args.configFilename, exp_type, exp.instance_name))
    return exp

def launch_experiment(exp):
    exp.run()


def make_plots(exp):
    if hasattr(exp.algo, 'algorithms') and hasattr(exp, 'algs_loggers'):
        for logger in exp.algs_loggers:
            logger.save(os.path.join(logger.log_base_path, logger.final_filename))
            qdpy.plots.default_plots_grid(logger, exp.log_base_path, suffix=f"-{exp.instance_name}-{logger.algorithms[0].name}", to_grid_parameters={'features_domain': logger.algorithms[0].container.features_extrema})

    qdpy.plots.plot_iterations(exp.logger, os.path.join(exp.log_base_path, f"./iterations_loss-{exp.instance_name}.pdf"), "loss", ylabel="Loss")

    if hasattr(exp.container, "container"):
        parent_archive = exp.container.container.parents[0]
        ylim_contsize = (0, len(parent_archive)) if np.isinf(parent_archive.capacity) else (0, parent_archive.capacity)
        qdpy.plots.plot_iterations(exp.logger, os.path.join(exp.log_base_path, f"./iterations_trainingsize{exp.instance_name}.pdf"), "train_size", ylim=ylim_contsize, ylabel="Training size")

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
