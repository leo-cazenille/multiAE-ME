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

# QDpy
import qdpy
from qdpy.base import *
from qdpy.experiment import QDExperiment
from qdpy.benchmarks import artificial_landscapes
from qdpy.algorithms import LoggerStat, TQDMAlgorithmLogger, AlgorithmLogger
from qdpy.containers import TorchAE, TorchFeatureExtractionContainerDecorator, ContainerLike
import qdpy.plots

# Pytorch
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.autograd import Variable
from torch.utils.data import DataLoader


import matplotlib.pyplot as plt


########## NN ########### {{{1

#num_epochs = 100
#batch_size = 128
#learning_rate = 1e-3
#nb_modules = 4
#div_coeff = 0.5

#class AE(nn.Module):
#    def __init__(self, input_size):
#        super().__init__()
##        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
##        self.tanh = nn.Tanh()
##        self.hidden = []
##        for i in range(1, len(hidden_sizes)):
##            self.hidden.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
##        self.fc2 = nn.Linear(hidden_sizes[-1], output_size)
#
#        self.encoder = nn.Sequential(
#            nn.Linear(input_size, 8),
#            #nn.Dropout(0.2),
#            nn.ReLU(True),
#            nn.Linear(8, 4),
#            nn.ReLU(True),
#            nn.Linear(4, 3))
#        self.decoder = nn.Sequential(
#            nn.Linear(3, 4),
#            nn.ReLU(True),
#            nn.Linear(4, 8),
#            nn.ReLU(True),
#            nn.Linear(8, input_size), nn.Tanh())
#
##    def forward(self, x):
##        out = self.fc1(x)
##        out = self.tanh(out)
##        for hidden in self.hidden:
##            out = hidden(out)
##            out = self.tanh(out)
##        out = self.fc2(out)
##        return out
#
#    def forward(self, x):
#        x = self.encoder(x)
#        x = self.decoder(x)
#        return x
#


#class EnsembleAE(nn.Module):
#    def __init__(self, input_size, latent_size=2, nb_modules = 4):
#        super().__init__()
#        self.nb_modules = nb_modules
#        self.ae_list = nn.ModuleList([TorchAE(input_size, latent_size) for _ in range(self.nb_modules)])
#
#    def forward(self, x):
#        res = [nn(x) for nn in self.ae_list]
#        return torch.Tensor(res)

class EnsembleAE(nn.Module):
    def __init__(self, modules):
        super().__init__()
        self.ae_list = nn.ModuleList(modules)

    def forward(self, x):
        res = [nn(x) for nn in self.ae_list]
        return res

    def encoders(self, x):
        res = [nn.encoder(x) for nn in self.ae_list]
        return res

    def reset(self):
        # Initialize weights
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.3)
                torch.nn.init.uniform_(m.bias, 0.0, 1.0)
        for ae in self.ae_list:
            ae.encoder.apply(init_weights)
            ae.decoder.apply(init_weights)


#        res = [nn(x) for nn in self.ae_list]
#        return torch.Tensor(res)

#        res = torch.empty(len(self.ae_list), x.shape[0], x.shape[1])
#        for i, nn in enumerate(self.ae_list):
#            res[i] = nn(x)
#        return res


#data_transform = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize([0.5], [0.5])
#])


#def train_AE(exp):
#    container = copy.deepcopy(exp.container)
#    dataset = np.array([np.array(ind) for ind in container])
#    min_val, max_val = exp.config['algorithms']['ind_domain']
#    dataset = (dataset + min_val) / (max_val - min_val)
#    dataset = torch.Tensor(dataset)
#    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#    input_size = len(dataset[0])
#
#    #model = AE(input_size)
#    model = EnsembleAE(input_size, nb_modules)
#    #model = nn.DataParallel(model)
#    model = model.cpu()
#    criterion_perf = nn.MSELoss()
#    criterion_diversity = nn.MSELoss()
#
#    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
#    ##optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#
#    for epoch in range(num_epochs):
#        for data in dataloader:
#            #d = data_transform(data)
#            d = Variable(data)
#            #d = data
#            #img, _ = data
#            #img = img.view(img.size(0), -1)
#            #img = Variable(img).cuda()
#            # ===================forward=====================
#            output = model(d)
#            perf = torch.Tensor([criterion_perf(o, d) for o in output])
#            loss_perf = torch.mean(perf)
#            mean_output = torch.mean(output, 0)
#            dist_mean = output - mean_output
#            diversity = torch.Tensor([criterion_diversity(d) for d in dist_mean])
#            loss_diversity = torch.mean(diversity)
#            loss = loss_perf - div_coeff * loss_diversity
#            # ===================backward====================
#            optimizer.zero_grad()
#            loss.backward()
#
##            # Adjust learning rate
##            lr = learning_rate * (0.1 ** (epoch // 500))
##            for param_group in optimizer.param_groups:
##                param_group['lr'] = lr
#
#            optimizer.step()
#        # ===================log========================
#        print('epoch [{}/{}], loss:{:.4f}'
#              #.format(epoch + 1, num_epochs, loss.data[0]))
#              .format(epoch + 1, num_epochs, loss.item()))
#
#
##    with open("state_dict.p", "wb") as f:
##        #pickle.dump(model.state_dict(), f)
##        pickle.dump({32: 23}, f)
#    #print(model.state_dict())
#
#    #print(pickle.dumps(model.state_dict()))
#    torch.save(model.state_dict(), './sim_autoencoder.pth')
#
##    with open("state_dict.p", "wb") as f:
##        #pickle.dump(model.state_dict(), f)
##        o = pickle.dumps(model.state_dict())
##        f.write(o)
##        #pickle.dump({32: 23}, f)
#


nn_models = {}
last_training_nb_inds = 0
current_loss = np.nan



########## CONTAINER CLASS ########### {{{1
@registry.register
class TorchMultiFeatureExtractionContainerDecorator(TorchFeatureExtractionContainerDecorator):
    """TODO""" # TODO

    def __init__(self, container: ContainerLike,
            div_coeff: float = 0.5,
            diversity_loss_computation: str = "outputs",
            reset_model_every_training: bool = False,
            **kwargs: Any) -> None:
        self.div_coeff = div_coeff
        self.diversity_loss_computation = diversity_loss_computation
        self.reset_model_every_training = reset_model_every_training
        if not self.diversity_loss_computation in ['outputs', 'latent']:
            raise ValueError(f"Unknown diversity_loss_computation type: {self.diversity_loss_computation}.")
        super().__init__(container, **kwargs)
        self.last_recomputed = 0

    def _create_default_model(self, example_ind: IndividualLike) -> None:
        print(f"DEBUG {self.name} ############# CREATE DEFAULT MODEL ###############")
        # Identify base scores
        if self.base_scores == None: # Not base scores specified, use all scores (except those created through feature extraction)
            base_scores = [str(x) for x in example_ind.scores.keys() if not x.startswith("extracted_") ]
        else:
            base_scores = self.base_scores # type: ignore
        # Find default model parameters
        input_size = len(base_scores)
        assert(self.container.features_domain != None)
        latent_size = len(self.container.features_domain) # type: ignore
        # Set extracted scores names as the default features of the container
        self.container.features_score_names = [f"extracted_{id(self)}_{j}" for j in range(latent_size)]
        # Create simple auto-encoder as default model
        self.model = TorchAE(input_size, latent_size)
        # Register model in global dict
        global nn_models
        nn_models[self.name] = self.model

    def _train_and_recompute_if_needed(self, update_params=()):
        global last_training_nb_inds

        # Train and recomputed all features if necessary
        training_inds = self._get_training_inds()
        nb_training_inds = len(training_inds)

        # Create a model, if none exist
        if self.model == None and nb_training_inds > 0:
            self._create_default_model(training_inds[0])

        #print("DEBUG add !", nb_training_inds, self.training_period, self._last_training_nb_inds)
        if nb_training_inds >= self.training_period and nb_training_inds % self.training_period == 0 and nb_training_inds != self._last_training_nb_inds:
            do_training = nb_training_inds - last_training_nb_inds >= self.training_period
            print(f"DEBUG {self.name} _train_and_recompute_if_needed: {nb_training_inds} {last_training_nb_inds} {self.training_period}")
            self._last_training_nb_inds = nb_training_inds
            try:
                self.clear() # type: ignore
                if do_training:
                    print(f"DEBUG {self.name} DO TRAINING !")
                    self.train(self.nb_epochs if nb_training_inds > self.training_period else self.initial_nb_epochs)
                    last_training_nb_inds = nb_training_inds
                print(f"DEBUG {self.name} RECOMPUTE FEATURES")
                self.recompute_features_all_ind(update_params)
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


    def train(self, nb_epochs: int) -> None:
        global nn_models
        #print("###########  DEBUG: training.. ###########")
        #start_time = timer() # XXX

        training_inds = self._get_training_inds()
        assert(len(training_inds) > 0)
        assert(self.model != None)

        # Identify base scores
        if self.base_scores == None: # Not base scores specified, use all scores (except those created through feature extraction)
            base_scores: List[Any] = [x for x in training_inds[0].scores.keys() if not x.startswith("extracted_") ]
        else:
            base_scores = self.base_scores # type: ignore

        # Build dataset
        data = torch.empty(len(training_inds), len(base_scores))
        for i, ind in enumerate(training_inds):
            for j, s in enumerate(base_scores):
                data[i,j] = ind.scores[s]
        #dataset = torch.utils.data.TensorDataset(data)
        dataloader: Any = DataLoader(data, batch_size=self.batch_size, shuffle=True) # type: ignore

        # Create an ensemble model
        model = EnsembleAE(list(nn_models.values()))

        # Create criteria and optimizer
        criterion_perf = nn.MSELoss()
        criterion_diversity = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5) # type: ignore

        # Reset model, if needed
        if self.reset_model_every_training:
            model.reset()

        # Train !
        for epoch in range(nb_epochs):
            for data in dataloader:
                d = Variable(data)
                #print(f"DEBUG training2: {d} {d.shape}")
                output = model(d) # type: ignore

                # Compute performance loss
                loss_perf = 0.
                for r in output:
                    loss_perf += criterion_perf(r, d)
                loss_perf /= len(output)

                # Compute diversity loss
                loss_diversity = 0.
                if self.diversity_loss_computation == "outputs":
                    #mean_output = 0.
                    #for r in output:
                    #    mean_output += r
                    #mean_output /= len(output)
                    mean_output = [torch.mean(o, 0) for o in output]
                    mean_output = sum(mean_output) / len(mean_output)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        for r in output:
                            loss_diversity += criterion_diversity(r, mean_output)
                            #print(f"DEBUG outputs: {r.shape} {mean_output.shape} {criterion_diversity(r, mean_output)}")
                elif self.diversity_loss_computation == "latent":
                    latent = model.encoders(d)
                    #mean_latent = 0.
                    #for l in latent:
                    #    mean_latent += l
                    #mean_latent /= len(latent)
                    mean_latent = [torch.mean(l, 0) for l in latent]
                    mean_latent = sum(mean_latent) / len(mean_latent)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        for r in latent:
                            loss_diversity += criterion_diversity(r, mean_latent)
                            #print(f"DEBUG latent: {r.shape} {mean_latent.shape} {criterion_diversity(r, mean_latent)}")
                else:
                    raise ValueError(f"Unknown diversity_loss_computation type: {self.diversity_loss_computation}.")

                loss = loss_perf - self.div_coeff * loss_diversity
#                loss = loss_perf #- self.div_coeff * loss_diversity

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            #print(f"loss: {loss}")
        self.current_loss = loss.item()
        global current_loss
        current_loss = self.current_loss

        #elapsed = timer() - start_time # XXX
        #print(f"# Training: final loss: {loss}  elapsed: {elapsed}")



########## EXPERIMENT CLASSES ########### {{{1

class RastriginExperiment(QDExperiment):

    def eval(self, ind):
        res = self.bench.fn(ind)
        print("DEBUG:", res)
        return res

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
        #self.eval_fn = self.eval
        self.optimisation_task = self.bench.default_task

        # Update stats
        #stat_loss = LoggerStat("loss", lambda algo: f"{algo.container.current_loss:.4f}", True)
        global current_loss
        stat_loss = LoggerStat("loss", lambda algo: f"{current_loss:.4f}", True)
        stat_training = LoggerStat("training_size", lambda algo: f"{len(algo.container._get_training_inds())}", True)
        self.logger.register_stat(stat_loss, stat_training)
        self.logger._tabs_size = 5
        self.logger._min_cols_size = 10

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
        for i_alg, alg in enumerate(algos):
            stat_qd_score = LoggerStat(f"qd_score-{alg.name}", partial(fn_qd_score, i_alg), True)
            self.logger.register_stat(stat_qd_score)



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
    exp_type = base_config.get('experiment_type', 'rastrigin')
    if exp_type == 'rastrigin':
        exp = RastriginExperiment(args.configFilename, args.parallelismType, seed=args.seed, base_config=base_config)
    else:
        raise ValueError(f"Unknown experiment type: {exp_type}.")
    print("Using configuration file '%s'. Instance name: '%s'" % (args.configFilename, exp.instance_name))
    return exp

def launch_experiment(exp):
    exp.run()


def make_plots(exp):
    for logger in exp.algs_loggers:
        logger.save(os.path.join(logger.log_base_path, logger.final_filename))
        qdpy.plots.default_plots_grid(logger, exp.log_base_path, suffix=f"-{exp.instance_name}-{logger.algorithms[0].name}")

    qdpy.plots.plot_iterations(exp.logger, os.path.join(exp.log_base_path, f"./iterations_loss-{exp.instance_name}.pdf"), "loss", ylabel="Loss")

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
