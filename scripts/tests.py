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

# QDpy
import qdpy
from qdpy.base import *
from qdpy.experiment import QDExperiment
from qdpy.benchmarks import artificial_landscapes

# Pytorch
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.autograd import Variable
from torch.utils.data import DataLoader



########## EXPERIMENT CLASS ########### {{{1

class RastriginExperiment(QDExperiment):
    def reinit(self):
        self.bench = artificial_landscapes.RastriginBenchmark(nb_features = 2)
        self.config['fitness_type'] = "perf"
        #self.config['perfDomain'] = (0., artificial_landscapes.rastrigin([4.5]*2, 10.)[0])
        self.config['perfDomain'] = (0., math.inf)
        self.config['features_list'] = ["f0", "f1"]
        self.config['f0Domain'] = self.bench.features_domain[0]
        self.config['f1Domain'] = self.bench.features_domain[1]
        self.config['algorithms']['ind_domain'] = self.bench.ind_domain
        #self.features_domain = self.bench.features_domain
        super().reinit()
        self.eval_fn = self.bench.fn
        self.optimisation_task = self.bench.default_task


########## BASE FUNCTIONS ########### {{{1

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--configFilename', type=str, default='conf/test-simple.yaml', help = "Path of configuration file")
    parser.add_argument('-o', '--resultsBaseDir', type=str, default='results/', help = "Path of results files")
    parser.add_argument('-p', '--parallelismType', type=str, default='concurrent', help = "Type of parallelism to use")
    parser.add_argument('--replayBestFrom', type=str, default='', help = "Path of results data file -- used to replay the best individual")
    parser.add_argument('--seed', type=int, default=None, help="Numpy random seed")
    return parser.parse_args()

def create_base_config(args):
    base_config = {}
    if len(args.resultsBaseDir) > 0:
        base_config['resultsBaseDir'] = args.resultsBaseDir
    return base_config

def create_experiment(args, base_config):
    exp = RastriginExperiment(args.configFilename, args.parallelismType, seed=args.seed, base_config=base_config)
    print("Using configuration file '%s'. Instance name: '%s'" % (args.configFilename, exp.instance_name))
    return exp

def launch_experiment(exp):
    exp.run()


########## NN ########### {{{1

num_epochs = 100
batch_size = 128
learning_rate = 1e-3
nb_modules = 4
div_coeff = 0.5

class AE(nn.Module):
    def __init__(self, input_size):
        super().__init__()
#        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
#        self.tanh = nn.Tanh()
#        self.hidden = []
#        for i in range(1, len(hidden_sizes)):
#            self.hidden.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
#        self.fc2 = nn.Linear(hidden_sizes[-1], output_size)

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 8),
            #nn.Dropout(0.2),
            nn.ReLU(True),
            nn.Linear(8, 4),
            nn.ReLU(True),
            nn.Linear(4, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 4),
            nn.ReLU(True),
            nn.Linear(4, 8),
            nn.ReLU(True),
            nn.Linear(8, input_size), nn.Tanh())

#    def forward(self, x):
#        out = self.fc1(x)
#        out = self.tanh(out)
#        for hidden in self.hidden:
#            out = hidden(out)
#            out = self.tanh(out)
#        out = self.fc2(out)
#        return out

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



# XXX
class EnsembleAE(nn.Module):
    def __init__(self, input_size, nb_modules = 4):
        super().__init__()
        self.nb_modules = nb_modules
        self.ae_list = nn.ModuleList([AE(input_size) for _ in range(self.nb_modules)])

    def forward(self, x):
        res = [nn(x) for nn in self.ae_list]
        return torch.Tensor(res)


#data_transform = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize([0.5], [0.5])
#])

def train_AE(exp):
    container = copy.deepcopy(exp.container)
    dataset = np.array([np.array(ind) for ind in container])
    min_val, max_val = exp.config['algorithms']['ind_domain']
    dataset = (dataset + min_val) / (max_val - min_val)
    dataset = torch.Tensor(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    input_size = len(dataset[0])

    #model = AE(input_size)
    model = EnsembleAE(input_size, nb_modules)
    #model = nn.DataParallel(model)
    model = model.cpu()
    criterion_perf = nn.MSELoss()
    criterion_diversity = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    ##optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for data in dataloader:
            #d = data_transform(data)
            d = Variable(data)
            #d = data
            #img, _ = data
            #img = img.view(img.size(0), -1)
            #img = Variable(img).cuda()
            # ===================forward=====================
            output = model(d)
            perf = torch.Tensor([criterion_perf(o, d) for o in output])
            loss_perf = torch.mean(perf)
            mean_output = torch.mean(output, 0)
            dist_mean = output - mean_output
            diversity = torch.Tensor([criterion_diversity(d) for d in dist_mean])
            loss_diversity = torch.mean(diversity)
            loss = loss_perf - div_coeff * loss_diversity
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()

#            # Adjust learning rate
#            lr = learning_rate * (0.1 ** (epoch // 500))
#            for param_group in optimizer.param_groups:
#                param_group['lr'] = lr

            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              #.format(epoch + 1, num_epochs, loss.data[0]))
              .format(epoch + 1, num_epochs, loss.item()))


#    with open("state_dict.p", "wb") as f:
#        #pickle.dump(model.state_dict(), f)
#        pickle.dump({32: 23}, f)
    #print(model.state_dict())

    #print(pickle.dumps(model.state_dict()))
    torch.save(model.state_dict(), './sim_autoencoder.pth')

#    with open("state_dict.p", "wb") as f:
#        #pickle.dump(model.state_dict(), f)
#        o = pickle.dumps(model.state_dict())
#        f.write(o)
#        #pickle.dump({32: 23}, f)






########## MAIN ########### {{{1
if __name__ == "__main__":
    import traceback
    args = parse_args()
    base_config = create_base_config(args)
    exp = create_experiment(args, base_config)
    launch_experiment(exp)
    model = train_AE(exp)




# MODELINE  "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
