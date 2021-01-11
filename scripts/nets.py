
########## IMPORTS ########### {{{1
import gc
import copy
#import pickle
import numpy as np
import warnings
from typing import Optional, Tuple, List, Iterable, Iterator, Any, TypeVar, Generic, Union, Sequence, MutableSet, MutableSequence, Type, Callable, Generator, MutableMapping, Mapping, overload

# Pytorch
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.autograd import Variable
from torch.utils.data import DataLoader



# FROM https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/5
def cov(m, rowvar=True, inplace=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    if inplace:
        m -= torch.mean(m, dim=1, keepdim=True)
    else:
        m = m - torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()




########## NN ########### {{{1

nn_models = {}
last_training_nb_inds = 0
last_training_size = 0
current_loss = np.nan
current_loss_reconstruction = np.nan
current_loss_diversity = np.nan


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



class AE(nn.Module):
    def __init__(self, input_size, latent_size=2, tanh_encoder=False):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.tanh_encoder = tanh_encoder

        fst_layer_size = input_size//2 if input_size > 7 else 4
        snd_layer_size = input_size//4 if input_size > 7 else 2

        if self.tanh_encoder:
            self.encoder = nn.Sequential(
                nn.Linear(input_size, fst_layer_size),
                nn.ReLU(True),
                nn.Linear(fst_layer_size, snd_layer_size),
                nn.ReLU(True),
                nn.Linear(snd_layer_size, latent_size),
                nn.Tanh())
        else:
            self.encoder = nn.Sequential(
                nn.Linear(input_size, fst_layer_size),
                nn.ReLU(True),
                nn.Linear(fst_layer_size, snd_layer_size),
                nn.ReLU(True),
                nn.Linear(snd_layer_size, latent_size))

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, snd_layer_size),
            nn.ReLU(True),
            nn.Linear(snd_layer_size, fst_layer_size),
            nn.ReLU(True),
            nn.Linear(fst_layer_size, input_size), nn.Tanh())

        # Initialize weights
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.3)
                torch.nn.init.ones_(m.bias)
        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



#
#class ConvNet(nn.Module):
#    def __init__(self):
#        super(ConvNet, self).__init__()
#        self.layer1 = nn.Sequential(
#            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
#            nn.ReLU(),
#            nn.MaxPool2d(kernel_size=2, stride=2))
#        self.layer2 = nn.Sequential(
#            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
#            nn.ReLU(),
#            nn.MaxPool2d(kernel_size=2, stride=2))
#        self.drop_out = nn.Dropout()
#        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
#        self.fc2 = nn.Linear(1000, 10)
#
#    def forward(self, x):
#        out = self.layer1(x)
#        out = self.layer2(out)
#        out = out.reshape(out.size(0), -1)
#        out = self.drop_out(out)
#        out = self.fc1(out)
#        out = self.fc2(out)
#        return out
#



#self.encoder = nn.Sequential(
#    *list(pretrained_model.layer1.children()),
#    *list(pretrained_model.layer2.children()),
#    nn.Conv2d(in_channels=32, out_channels=4,
#              kernel_size=3, padding=1),
#    nn.BatchNorm2d(num_features=4),
#    nn.ReLU()
#)
#
#self.decoder = nn.Sequential(
#    nn.Conv2d(in_channels=4, out_channels=32,
#              kernel_size=3, padding=1),
#    nn.BatchNorm2d(num_features=32),
#    nn.ReLU(),
#    nn.Upsample(scale_factor=2, mode='nearest'),
#    nn.Conv2d(in_channels=32, out_channels=16,
#              kernel_size=5, padding=2),
#    nn.BatchNorm2d(num_features=16),
#    nn.ReLU(),
#    nn.Upsample(scale_factor=2, mode='nearest'),
#    nn.Conv2d(in_channels=16, out_channels=1,
#              kernel_size=5, padding=2),
#    nn.BatchNorm2d(num_features=1),
#    nn.Sigmoid()
#)
#


class ConvEncoder(nn.Module):
    def __init__(self, input_size, latent_size=2, nb_filters=4):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.nb_filters = nb_filters

        #self.drop_out1 = nn.Dropout(0.2)
        #self.drop_out2 = nn.Dropout(0.2)

        # Encoder
#        self.enc_conv1 = nn.Sequential(
#            #nn.Conv1d(1, 2, kernel_size=3, stride=2, padding=1),
#            nn.Conv1d(1, 2, kernel_size=2, stride=2, padding=0),
#            nn.BatchNorm1d(num_features=2),
#            n.ReLU()
#            #nn.LeakyReLU()
#        )
#        self.enc_fc1 = nn.Sequential(
#            nn.Linear(input_size, latent_size*2+1),
#            nn.Sigmoid()
#            #nn.ReLU()
#        )
#        self.enc_fc2 = nn.Sequential(
#            nn.Linear(latent_size*2+1, latent_size),
#            nn.Sigmoid()
#            #nn.ReLU()
#        )

        self.enc_conv1 = nn.Sequential(
            nn.Conv1d(2, nb_filters, kernel_size=3),
            nn.BatchNorm1d(num_features=nb_filters),
            nn.ReLU()
            #nn.LeakyReLU()
        )
        self.enc_pool1 = nn.MaxPool1d(2, return_indices=True)
        self.enc_conv2 = nn.Sequential(
            nn.Conv1d(nb_filters, nb_filters, kernel_size=3),
            nn.BatchNorm1d(num_features=nb_filters),
            nn.ReLU()
            #nn.LeakyReLU()
        )
        self.enc_fc1 = nn.Sequential(
            nn.Linear(nb_filters * (input_size//2-3), latent_size*2+1),
            #nn.Sigmoid()
            nn.ReLU()
        )
        self.enc_fc2 = nn.Sequential(
            nn.Linear(latent_size*2+1, latent_size),
            nn.Sigmoid(),
            #nn.ReLU(),
            nn.BatchNorm1d(num_features=2, affine=False)
        )


        # Initialize weights
        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv1d:
                torch.nn.init.xavier_uniform_(m.weight)
                #torch.nn.init.normal_(m.weight, mean=0.0, std=0.3)
                #torch.nn.init.ones_(m.bias)
        self.enc_conv1.apply(init_weights)
        self.enc_conv2.apply(init_weights)
        self.enc_fc1.apply(init_weights)
        self.enc_fc2.apply(init_weights)

    def forward(self, x):

#        # XXX TODO
#        if len(x.shape) == 1:
#            x = x.reshape(1, 1, x.size(0))
#        elif len(x.shape) == 2:
#            x = x.reshape(x.size(0), 1, x.size(1))

#        x = self.enc_conv1(x)
#        x = x.reshape(x.size(0), -1)
##        x = self.drop_out(x)
#        x = self.enc_fc1(x)
##        x = self.drop_out(x)
#        x = self.enc_fc2(x)
#        #print(f"DEBUG forward {x}")

        x = self.enc_conv1(x)
        m1, self.i1 = self.enc_pool1(x)
        x = self.enc_conv2(m1)
        x = x.reshape(x.size(0), -1)
        #x = self.drop_out1(x)
        x = self.enc_fc1(x)
        #x = self.drop_out2(x)
        x = self.enc_fc2(x)
        #print(f"DEBUG forward {x}")

        return x


class ConvDecoder(nn.Module):
    def __init__(self, encoder, input_size, latent_size=2, nb_filters=4):
        super().__init__()
        self.encoder = encoder
        self.input_size = input_size
        self.latent_size = latent_size
        self.nb_filters = nb_filters

        #self.drop_out1 = nn.Dropout(0.2)
        #self.drop_out2 = nn.Dropout(0.2)

        # Decoder
        self.dec_fc2 = nn.Sequential(
            nn.Linear(latent_size, latent_size*2+1),
            nn.Sigmoid()
            #nn.ReLU()
        )
        self.dec_fc1 = nn.Sequential(
            #nn.Linear(latent_size*2+1, input_size),
            #nn.Linear(latent_size*2+1, 4 * (input_size//2-1)),
            nn.Linear(latent_size*2+1, nb_filters*(input_size//2-3)),
            #nn.Sigmoid()
            nn.ReLU()
        )
        self.dec_conv1 = nn.Sequential(
            #nn.Conv1d(4, 2, kernel_size=3, stride=1, padding=0),
            #nn.ConvTranspose1d(4, 2, kernel_size=3, stride=2, padding=0),
            nn.ConvTranspose1d(nb_filters, nb_filters, 3),
            nn.BatchNorm1d(num_features=nb_filters),
            nn.ReLU()
            #nn.LeakyReLU()
        )
        self.dec_unpool1 = nn.MaxUnpool1d(2)
        self.dec_conv2 = nn.Sequential(
            nn.ConvTranspose1d(nb_filters, 2, 3),
            nn.BatchNorm1d(num_features=2),
            #nn.ReLU()
            #nn.LeakyReLU()
            nn.Sigmoid()
        )

#        self.dec_out = nn.Sequential(
#            nn.Linear(input_size, input_size),
#            nn.Sigmoid()
#            #nn.ReLU()
#        )

        # Initialize weights
        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv1d:
                torch.nn.init.xavier_uniform_(m.weight)
                #torch.nn.init.normal_(m.weight, mean=0.0, std=0.3)
                #torch.nn.init.ones_(m.bias)
        self.dec_fc2.apply(init_weights)
        self.dec_fc1.apply(init_weights)
        self.dec_conv1.apply(init_weights)
        self.dec_conv2.apply(init_weights)
#        self.dec_out.apply(init_weights)


    def forward(self, x):
#        # Decoder
#        x = self.dec_fc2(x)
##        x = self.drop_out(x)
#        x = self.dec_fc1(x)
#        x = x.reshape(x.size(0), 4, self.input_size//2-1)
#        x = self.dec_conv1(x)
##        x = self.drop_out(x)
#        x = self.dec_out(x)

        # Decoder
        #x = self.drop_out1(x)
        x = self.dec_fc2(x)
        #x = self.drop_out2(x)
        x = self.dec_fc1(x)
        x = x.reshape(x.size(0), self.nb_filters, self.input_size//2-3)
        x = self.dec_conv1(x)
        x = self.dec_unpool1(x, self.encoder.i1)
        x = self.dec_conv2(x)
#        x = self.dec_out(x)
        return x

class ConvAE(nn.Module):
    def __init__(self, input_size, latent_size=2, nb_filters=4):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.encoder = ConvEncoder(input_size, latent_size, nb_filters=nb_filters)
        self.decoder = ConvDecoder(self.encoder, input_size, latent_size, nb_filters=nb_filters)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x




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
            if type(m) == nn.Linear or type(m) == nn.Conv1d:
                #torch.nn.init.normal_(m.weight, mean=0.0, std=0.3)
                #torch.nn.init.uniform_(m.bias, 0.0, 1.0)
                torch.nn.init.xavier_uniform_(m.weight)
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


class NNTrainer(object):
    def __init__(self,
#            base_scores,
            nn_models: Optional[Any] = None,
            nb_training_sessions: int = 5,
            nb_epochs: int = 500,
            learning_rate: float = 0.1,
            batch_size: int = 128,
            epochs_avg_loss: int = 50,
            validation_split: float = 0.25,
            #train_only_on_last_inds: bool = False,
            reset_model_every_training: bool = True,
            diversity_loss_computation: str = "outputs",
            div_coeff: float = 0.3
            ):
        self.nn_models = nn_models
#        self.base_scores = base_scores
        self.nb_training_sessions = nb_training_sessions
        self.nb_epochs = nb_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs_avg_loss = epochs_avg_loss
        self.validation_split = validation_split
        #self.train_only_on_last_inds = train_only_on_last_inds
        self.reset_model_every_training = reset_model_every_training
        self.diversity_loss_computation = diversity_loss_computation
        self.div_coeff = div_coeff
#        self.last_training_size = 0
        self.current_loss = 0.
        self.current_loss_reconstruction = 0.
        self.current_loss_diversity = 0.
        self.min = 0.
        self.max = 1.

        if not self.diversity_loss_computation in ['outputs', 'pwoutputs', 'latent', 'covlatent']:
            raise ValueError(f"Unknown diversity_loss_computation type: {self.diversity_loss_computation}.")

        if nn_models != None:
            self.create_ensemble_model(self.nn_models)


    def create_ensemble_model(self, nn_models):
        # Create an ensemble model
        self.model = EnsembleAE(list(nn_models.values()))


    def compute_loss(self, data, model):
        criterion_reconstruction = nn.MSELoss()
        criterion_diversity = nn.MSELoss()

        d = Variable(data)
        #print(f"DEBUG training2: {d} {d.shape}")
        output = model(d) # type: ignore

        # Compute reconstruction loss
        loss_reconstruction = torch.Tensor([0.])
        for r in output:
            loss_reconstruction += criterion_reconstruction(r, d)
        loss_reconstruction /= len(output)

        # Compute diversity loss
        loss_diversity = torch.Tensor([0.])
        if self.diversity_loss_computation == "outputs":
            #mean_output = 0.
            #for r in output:
            #    mean_output += r
            #mean_output /= len(output)
            mean_output = [torch.mean(o, 0) for o in output]
            mean_output = sum(mean_output) / len(mean_output)
            for r in output:
                loss_diversity += criterion_diversity(r, mean_output)
#            with warnings.catch_warnings():
#                warnings.simplefilter("ignore")
#                for r in output:
#                    loss_diversity += criterion_diversity(r, mean_output)
#                    #print(f"DEBUG outputs: {r.shape} {mean_output.shape} {criterion_diversity(r, mean_output)}")
        elif self.diversity_loss_computation == "pwoutputs":
            for r1 in output:
                for r2 in output:
                    loss_diversity += criterion_diversity(r1, r2)
        elif self.diversity_loss_computation == "latent":
            latent = model.encoders(d)
            #mean_latent = 0.
            #for l in latent:
            #    mean_latent += l
            #mean_latent /= len(latent)
            n_latent = [(l - l.min(0)[0]) / (l.max(0)[0] - l.min(0)[0]) for l in latent]
            mean_n_latent = torch.Tensor([(torch.nansum(l, 0) / len(l)).tolist() for l in n_latent])
            mean_n_latent = torch.nansum(mean_n_latent, 0) / len(mean_n_latent)
            for r in latent:
                loss_diversity += criterion_diversity(r, mean_n_latent)
#            with warnings.catch_warnings():
#                warnings.simplefilter("ignore")
#                for r in latent:
#                    loss_diversity += criterion_diversity(r, mean_latent)
#                    #print(f"DEBUG latent: {r.shape} {mean_latent.shape} {criterion_diversity(r, mean_latent)}")

        elif self.diversity_loss_computation == "covlatent":
            latent = model.encoders(d)
            latent_flat = torch.cat([l for l in latent], 1)

            #cov_mag = torch.zeros(1)
            #for d in latent_flat:
            #    for i in range(len(d)):
            #        for j in range(len(d)):
            #            if i == j:
            #                continue
            #            cov_mag += d[i] * d[j]
            #cov_mag /= latent_flat.size(0)
            #loss_diversity = cov_mag

            c = torch.abs(cov(latent_flat, rowvar=False))
            #print(f"DEBUG covlatent {c}")
            loss_diversity = torch.zeros(1)
            for i in range(c.size(0)):
                for j in range(c.size(1)):
                    if i != j:
                        loss_diversity -= c[i,j]
                    #else: # XXX ?
                    #    loss_diversity += c[i,j] # XXX ?

        else:
            raise ValueError(f"Unknown diversity_loss_computation type: {self.diversity_loss_computation}.")

        loss = loss_reconstruction - self.div_coeff * loss_diversity
        return loss, loss_reconstruction, loss_diversity


    def training_session(self, data):
        # Create optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5) # type: ignore
        #self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.9 ** (epoch / self.nb_epochs))
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', verbose=True)

        dataloader: Any = DataLoader(data, batch_size=self.batch_size, shuffle=True) # type: ignore
        train_size = int(np.floor(len(dataloader) * (1. - self.validation_split)))
        validation_size = int(np.ceil(len(dataloader) * self.validation_split))
        train_dataset, validation_dataset = torch.utils.data.random_split(list(dataloader), [train_size, validation_size])

        rv_loss_lst = []
        last_mean_rv_loss = np.inf
        for epoch in range(self.nb_epochs):
            # Training
            rt_loss = 0.
            rt_loss_reconstruction = 0.
            rt_loss_diversity = 0.
            self.model.train()
            for data in train_dataset:
                t_loss, t_loss_reconstruction, t_loss_diversity = self.compute_loss(data, self.model)
                self.optimizer.zero_grad()
                t_loss.backward()
                self.optimizer.step()

                rt_loss += t_loss
                rt_loss_reconstruction += t_loss_reconstruction
                rt_loss_diversity += t_loss_diversity

            # Validation
            rv_loss = 0.
            rv_loss_reconstruction = 0.
            rv_loss_diversity = 0.
            self.model.eval()
            with torch.no_grad():
                for data in validation_dataset:
                    v_loss, v_loss_reconstruction, v_loss_diversity = self.compute_loss(data, self.model)
                    rv_loss += v_loss
                    rv_loss_reconstruction += v_loss_reconstruction
                    rv_loss_diversity += v_loss_diversity

    #            # Check stopping criterion
    #            if len(rv_loss_lst) >= self.epochs_avg_loss:
    #                del rv_loss_lst[0]
    #            rv_loss_lst.append(rv_loss)
    #            mean_rv_loss = np.mean(rv_loss_lst)
    #            if epoch > self.epochs_avg_loss and mean_rv_loss > last_mean_rv_loss:
    #                break
    #            last_mean_rv_loss = mean_rv_loss
    #
    #        return rv_loss.item(), rv_loss_reconstruction.item(), rv_loss_diversity.item()

            # Check stopping criterion
            if len(rv_loss_lst) >= self.epochs_avg_loss:
                del rv_loss_lst[0]
            rv_loss_lst.append(v_loss)
            mean_rv_loss = np.mean(rv_loss_lst)
            if epoch > self.epochs_avg_loss and mean_rv_loss > last_mean_rv_loss:
                print(f"Training: stop early: mean_rv_loss={mean_rv_loss} last_mean_rv_loss{last_mean_rv_loss}")
                break
            last_mean_rv_loss = mean_rv_loss

            print(f"# Epoch {epoch}/{self.nb_epochs}  Validation loss:{v_loss} loss_reconstruction:{v_loss_reconstruction} loss_diversity:{v_loss_diversity} ")
            self.scheduler.step(v_loss)
            #self.scheduler.step()

        return v_loss.item(), v_loss_reconstruction.item(), v_loss_diversity.item()


    def train(self, training_inds) -> None:
        #print("###########  DEBUG: training.. ###########")
        #start_time = timer() # XXX

        assert(len(training_inds) > 0)

#        # Skip training if we already exceed the training budget
#        if self.training_budget != None and len(training_inds) > self.training_budget:
#            return

#        # If needed, only use the last inds of the training set
#        if self.train_only_on_last_inds:
#            nb_new_inds = len(training_inds) - self.last_training_size
#            #print(f"DEBUG train_only_on_last_inds: {nb_new_inds} {self.last_training_size}")
#            self.last_training_size = len(training_inds)
#            training_inds = training_inds[-nb_new_inds:]
#        else:
#            self.last_training_size = len(training_inds)
#        print(f" training size: {len(training_inds)}")

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

        # Build dataset
        data = torch.empty(len(training_inds), *training_inds[0].observations.shape)
        for i, ind in enumerate(training_inds):
            data[i] = torch.Tensor(ind.observations)
            #for j, s in enumerate(base_scores):
            #    data[i,j] = ind.scores[s]

        # Normalize dataset
        self.min = data.min()
        self.max = data.max()
        data = (data - self.min) / (self.max - self.min)

        # Reset model, if needed
        if self.reset_model_every_training:
            self.model.reset()

        # Train !
        loss_lst = []
        loss_reconstruction_lst = []
        loss_diversity_lst = []
        for _ in range(self.nb_training_sessions):
            l, r, d = self.training_session(data)
            loss_lst.append(l)
            loss_reconstruction_lst.append(r)
            loss_diversity_lst.append(d)
            print(f"Finished session: loss={l} loss_reconstruction={r} loss_diversity={d}")

        # Compute and save mean losses
        self.current_loss = np.mean(loss_lst)
        self.current_loss_reconstruction = np.mean(loss_reconstruction_lst)
        self.current_loss_diversity = np.mean(loss_diversity_lst)
        #global current_loss, current_loss_reconstruction, current_loss_diversity
        #current_loss = self.current_loss
        #current_loss_reconstruction = self.current_loss_reconstruction
        #current_loss_diversity = self.current_loss_diversity

        #elapsed = timer() - start_time # XXX
        #print(f"# Training: final loss: {loss}  elapsed: {elapsed}")

        return self.current_loss, self.current_loss_reconstruction, self.current_loss_diversity


#    def eval(self, ind):
#        obs = torch.Tensor(ind.observations)
#        obs = (obs - self.min) / (self.max - self.min)
#        obs = obs.reshape(1, *obs.shape)
#        self.model.eval()
#        res = self.model.encoders(obs)
#        res = [r[0].tolist() for r in res]
#        return res

    def eval(self, inds, model = None):
        assert(len(inds) > 0)
        _model = self.model if model == None else model
        obs = torch.empty(len(inds), *inds[0].observations.shape)
        for i, ind in enumerate(inds):
            obs[i] = torch.Tensor(ind.observations)
        obs = (obs - self.min) / (self.max - self.min)
        _model.eval()
        if hasattr(_model, "encoders"):
            res = _model.encoders(obs)
        else:
            res = [_model.encoder(obs)]
        res = [r.tolist() for r in res]
        return res



# MODELINE  "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker