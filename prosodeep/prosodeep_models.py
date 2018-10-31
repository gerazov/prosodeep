#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ProsoDeep - Pytorch model implementations.

@authors:
    Branislav Gerazov Apr 2018

Copyright 2017, 2018 by GIPSA-lab, Grenoble INP, Grenoble, France.

See the file LICENSE for the licence associated with this software.
"""
import numpy as np
import torch
from torch.utils.data.sampler import SequentialSampler, RandomSampler, BatchSampler
import copy


class deep_baseline_model():
    """
    Deep NN model trains on all inputs together. It is a Merlin inspired
    baseline.
    """
    def __init__(self,
                 n_feature=None,
                 n_hidden=17,
                 activation='relu',
                 n_output=6,
                 batch_size='auto',
                 max_iter=50,
                 weight_decay=1e-4,
                 shuffle=True,
                 random_state=42,
                 verbose=True,
                 solver='adam',
                 learning_rate_init=1e-3,
                 use_cuda=True,
                 ):

        self.n_feature = n_feature
        self.n_hidden = n_hidden
        self.activation = activation
        self.n_output = n_output
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.shuffle = shuffle
        self.random_state = random_state
        if random_state is not None:
            torch.manual_seed(random_state)
        self.verbose = verbose
        self.optimizer_type = solver
        self.learning_rate = learning_rate_init
        self.use_cuda = use_cuda

        self.init_model()

    def init_model(self):
        # init contour generators
        if self.verbose:
            print('Initialising baseline DNN ...')
        self.model = ContourGeneratorMLP(
                        n_feature=self.n_feature,
                        n_hiddens=self.n_hidden,
                        n_output=self.n_output,
                        activation=self.activation)

        if self.verbose:
            print(self.model)

        self.criterion = torch.nn.MSELoss()
        if self.use_cuda:
            self.model.cuda()
            self.criterion.cuda()
        self.init_optimizer()

    def reset_model(self):
        self.model.reset_parameters()
        self.init_optimizer()

    def init_optimizer(self):
        if self.optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=self.learning_rate,
                                              weight_decay=self.weight_decay)
        elif self.optimizer_type == 'sgdr':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                              lr=self.learning_rate,
                                              weight_decay=self.weight_decay)
            self.T_max = 10
            self.scheduler = torch.optim.CosineAnnealingLR(self.optimizer,
                                                           self.T_max, eta_min=0)
        elif self.optimizer_type == 'adadelta':
            self.optimizer = torch.optim.Adadelta(self.model.parameters())
        elif self.optimizer_type == 'adagrad':
            self.optimizer = torch.optim.Adagrad(self.model.parameters())
        elif self.optimizer_type == 'sparseAdam':
            self.optimizer = torch.optim.SparseAdam(self.model.parameters())
        elif self.optimizer_type == 'asgd':
            self.optimizer = torch.optim.ASGD(self.model.parameters())
        elif self.optimizer_type == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.model.parameters())
        elif self.optimizer_type == 'rprop':
            self.optimizer = torch.optim.Rprop(self.model.parameters())
        elif self.optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=self.learning_rate,
                                             momentum=0.9)
        else:
            raise ValueError('Optimizer not recognized')

    def fit(self, X, y,
            X_val=None, y_val=None):
        """
        Train model.
        """
        # set to training - should work for vae and others as well?
        self.model.train()

        if X.ndim == 1:  # if it's a single sample
            X = X[np.newaxis, :]
        n_samples = X.shape[0]

        # cast to Tensor
        dtype = torch.float32
        if self.use_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        X = torch.from_numpy(X).type(dtype).to(device)
        y = torch.from_numpy(y).type(dtype).to(device)
        if X_val is not None:
            n_samples_val = X_val.shape[0]
            X_val = torch.from_numpy(X_val).type(dtype).to(device)
            y_val = torch.from_numpy(y_val).type(dtype).to(device)

        if self.batch_size == 'auto':
            batch_size = np.min((64, n_samples))
        elif self.batch_size == 'all':
            batch_size = n_samples
        else:
            batch_size = int(np.min((int(self.batch_size), n_samples)))
        sampler = SequentialSampler(range(n_samples))
        if self.shuffle:
            if self.random_state is not None:
                torch.manual_seed(self.random_state)
            sampler = RandomSampler(range(n_samples))

        # batch loop
        batch_iter = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
        n_batches = len(batch_iter)

        loss_in_batch = np.zeros(n_batches)
        mse_in_batch = np.zeros(n_batches)

        self.model.train()  # set to training
        for i, batch_ind in enumerate(batch_iter):
            if self.verbose:
                print('\rtraining batch {}/{} '.format(i, n_batches), end='')
            X_batch = X[batch_ind]
            y_batch = y[batch_ind]

            prediction_batch = self.model(X_batch)

            mse = self.criterion(prediction_batch, y_batch)
            # must be (1. nn output, 2. target)
            loss = torch.zeros_like(mse)
            loss += mse  # otherwise it makes it the same reference

            self.optimizer.zero_grad()  # clear gradients for next train
            loss.backward(retain_graph=True)  # backpropagation, compute gradients
            self.optimizer.step()  # apply gradients

            loss_in_batch[i] = loss.item()
            mse_in_batch[i] = mse.item()

        self.losses_ = loss_in_batch
        if any(np.isnan(self.losses_)):
            print('Loss is NaN!')
        self.mses_ = mse_in_batch

        # evaluate on validation data
        if X_val is not None:
            self.model.eval()
            with torch.no_grad():
                self.losses_train_ = self.losses_
                self.mses_train_ = self.mses_
                # run batches
                sampler_val = SequentialSampler(range(n_samples_val))
                batch_iter = BatchSampler(sampler_val, batch_size=batch_size,
                                          drop_last=False)
                n_batches = len(batch_iter)

                loss_in_batch = np.zeros(n_batches)
                mse_in_batch = np.zeros(n_batches)

                for i, batch_ind in enumerate(batch_iter):
                    if self.verbose:
                        print('\rvalidation batch {}/{} '.format(
                                i, n_batches), end='')
                    X_batch = X_val[batch_ind]
                    y_batch = y_val[batch_ind]

                    prediction_batch = self.model(X_batch)

                    mse = self.criterion(prediction_batch, y_batch)
                    # must be (1. nn output, 2. target)
                    loss = torch.zeros_like(mse)
                    loss += mse  # otherwise it makes it the same reference
                    loss_in_batch[i] = loss.item()
                    mse_in_batch[i] = mse.item()

                self.losses_ = loss_in_batch
                if any(np.isnan(self.losses_)):
                    print('Loss is NaN!')
                self.mses_ = mse_in_batch

        if self.verbose:
            print('\r', end='')


    def predict(self, X):  #
        """
        X is a np array (n_sample x n_feats)
        """
        if X.ndim == 1:  # if it's a single sample
            X = X[np.newaxis, :]
        n_samples = X.shape[0]
        y = np.zeros((n_samples, self.n_output))

        # cast to Tensor
        dtype = torch.float32
        if self.use_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        X = torch.from_numpy(X).type(dtype).to(device)
        y = torch.from_numpy(y).type(dtype).to(device)
        # should batch size be the same with fit, maybe should be bigger for speed?
        batch_size = int(np.min((256, n_samples)))
        sampler = SequentialSampler(range(n_samples))

        self.model.eval()
        with torch.no_grad():
            for batch_ind in BatchSampler(sampler, batch_size=batch_size,
                                          drop_last=False):
                X_batch = X[batch_ind]
                prediction_batch = self.model(X_batch)
                y[batch_ind] = prediction_batch
        # output
        return y.detach().cpu().numpy()


class deep_model():
    """
    Deep prosody model incorporates all contour generators into one model.
    This version uses a static graph with masking.
    """
    def __init__(self,
                 contour_types=None,  # init of nncgs
                 contour_generators=None,
                 n_feature=4,
                 n_in_contour=15,
                 n_hidden=15,
                 activation='tanh',
                 n_output=4,
                 n_context=None,
                 n_hidden_context=None,
                 batch_size='auto',
                 max_iter=50,
                 weight_decay=1e-4,
                 shuffle=True,
                 random_state=42,
                 verbose=True,
                 solver='adam',
                 learning_rate_init=1e-3,
                 use_strength=False,
                 strength_method='context',
                 reg_strengths=None,
                 reg_strengths_mean=None,
                 use_cuda=True,
                 contours_in_graph=None,  # list of nncgs in static graph
                 vae=False,
                 vae_input='',
                 n_latent=2,
                 reg_vae=1):

        self.contour_types = contour_types
        self.contour_generators = contour_generators  # if None init

        self.n_feature = n_feature
        self.n_in_contour = n_in_contour

        self.n_hidden = n_hidden
        self.activation = activation
        self.n_output = n_output
        self.n_context = n_context
        self.n_hidden_context = n_hidden_context

        self.max_iter = max_iter
        self.batch_size = batch_size
        self.weight_decay = weight_decay

        self.shuffle = shuffle
        self.random_state = random_state
        if random_state is not None:
            torch.manual_seed(random_state)

        self.verbose = verbose
        self.optimizer_type = solver
        self.learning_rate = learning_rate_init
        self.activation = activation

        self.use_strength = use_strength
        self.strength_method = strength_method
        self.reg_strengths = reg_strengths
        self.reg_strengths_mean = reg_strengths_mean

        self.contours_in_graph = contours_in_graph

        self.use_cuda = use_cuda

        self.vae = vae
        self.vae_input = vae_input
        self.n_latent = n_latent
        self.reg_vae = reg_vae
        if 'context' in self.vae_input:
            self.vae_n_input = n_feature + n_context
        else:
            self.vae_n_input = n_feature

        self.init_model()

    def init_model(self):
        # init contour generators
        if self.verbose:
            print('Initialising contour generators ...')
        contour_gen_list = self.contour_types
        if self.contour_generators is None:
            self.contour_generators = {}  # init dictinary
        else:
            for pre_contour_type in self.contour_generators.keys():
                # go through pretrained contour generators
                if pre_contour_type not in contour_gen_list:  # delete it from dict
                    del self.contour_generators[pre_contour_type]

                if pre_contour_type in contour_gen_list:  # remove it from list
                    contour_gen_list.remove(pre_contour_type)

        # add missing ones
        for contour_type in contour_gen_list:
            if self.vae:
                self.contour_generators[contour_type] = ContourGeneratorVar(
                        n_feature=self.vae_n_input,
                        enc_hidden=self.n_hidden[0],  # from list to int
                        n_latent=self.n_latent,
                        dec_hidden=self.n_hidden[0],
                        n_output=self.n_output)

            elif self.use_strength:
                if 'context' in self.strength_method:
                    self.contour_generators[contour_type] = ContourGeneratorStrengthMLP(
                                    self.n_feature,
                                    self.n_hidden,
                                    self.n_output,
                                    self.n_context,
                                    self.n_hidden_context,
                                    self.activation)
            else:
                self.contour_generators[contour_type] = ContourGeneratorMLP(
                        n_feature=self.n_feature,
                        n_hiddens=self.n_hidden,
                        n_output=self.n_output,
                        activation=self.activation)

        self.criterion = torch.nn.MSELoss()

        self.static_graph = StaticGraph(self.contour_generators, self.contours_in_graph,
                                        self.n_in_contour, self.n_output,
                                        self.use_strength, self.strength_method,
                                        # self.regularise_compensation,
                                        self.vae, self.vae_input)
        if self.verbose:
            print(self.static_graph)

        if self.use_cuda:
            self.static_graph.cuda()
            self.criterion.cuda()

        self.init_optimizer()

    def reset_model(self):
        self.static_graph.reset_parameters()
        self.init_optimizer()

    def init_optimizer(self):
        if self.optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(self.static_graph.parameters(),
                                              lr=self.learning_rate,
                                              weight_decay=self.weight_decay)
        elif self.optimizer_type == 'sgdr':
            self.optimizer = torch.optim.SGD(self.static_graph.parameters(),
                                              lr=self.learning_rate,
                                              weight_decay=self.weight_decay)
            self.T_max = 10
            self.scheduler = torch.optim.CosineAnnealingLR(self.optimizer, self.T_max, eta_min=0)
        elif self.optimizer_type == 'adadelta':
            self.optimizer = torch.optim.Adadelta(self.static_graph.parameters())
        elif self.optimizer_type == 'adagrad':
            self.optimizer = torch.optim.Adagrad(self.static_graph.parameters())
        elif self.optimizer_type == 'sparseAdam':
            self.optimizer = torch.optim.SparseAdam(self.static_graph.parameters())
        elif self.optimizer_type == 'asgd':
            self.optimizer = torch.optim.ASGD(self.static_graph.parameters())
        elif self.optimizer_type == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.static_graph.parameters())
        elif self.optimizer_type == 'rprop':
            self.optimizer = torch.optim.Rprop(self.static_graph.parameters())
        elif self.optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(self.static_graph.parameters(),
                                             lr=self.learning_rate,
                                             momentum=0.9)
        else:
            raise ValueError('Optimizer not recognized')

    def fit(self, X, masks, y,
            X_val=None, masks_val=None, y_val=None):
        """
        Train model.
        """
        # set to training - should work for vae and others as well?
        self.static_graph.train()

        if X.ndim == 1:  # if it's a single sample
            X = X[np.newaxis, :]
        n_samples = X.shape[0]
        # cast to Tensor
        dtype = torch.float32
        if self.use_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        X = torch.from_numpy(X).type(dtype).to(device)
        y = torch.from_numpy(y).type(dtype).to(device)
        masks = torch.from_numpy(masks).type(dtype).to(device)

        if X_val is not None:
            n_samples_val = X_val.shape[0]
            X_val = torch.from_numpy(X_val).type(dtype).to(device)
            y_val = torch.from_numpy(y_val).type(dtype).to(device)
            masks_val = torch.from_numpy(masks_val).type(dtype).to(device)

        if self.batch_size == 'auto':
            batch_size = np.min((64, n_samples))
        elif self.batch_size == 'all':
            batch_size = n_samples
        else:
            batch_size = int(np.min((int(self.batch_size), n_samples)))
        sampler = SequentialSampler(range(n_samples))
        if self.shuffle:
            if self.random_state is not None:
                torch.manual_seed(self.random_state)
            sampler = RandomSampler(range(n_samples))

        # batch loop
        batch_iter = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
        n_batches = len(batch_iter)

        loss_in_batch = np.zeros(n_batches)
        mse_in_batch = np.zeros(n_batches)
        if self.vae and 'mmd' in self.vae_input:
            mmd_in_batch = np.zeros(n_batches)

        self.static_graph.train()  # set to training
        for i, batch_ind in enumerate(batch_iter):
            if self.verbose:
                print('\rtraining batch {}/{} '.format(i, n_batches), end='')
            X_batch = X[batch_ind]
            masks_batch = masks[batch_ind]
            y_batch = y[batch_ind]

            if self.vae:
                prediction_batch, zs_batch, mus_batch, logvars_batch = \
                        self.static_graph(X_batch, masks_batch)

            elif self.use_strength and 'context' in self.strength_method:
                prediction_batch, strengths_batch = self.static_graph(
                        X_batch, masks_batch)
            else:
                prediction_batch = self.static_graph(X_batch, masks_batch)

            mse = self.criterion(prediction_batch, y_batch)
            # must be (1. nn output, 2. target)
            loss = torch.zeros_like(mse)
            loss += mse  # otherwise it makes it the same reference

            if self.use_strength and 'context' in self.strength_method:
                # regularise all the strengths
                reg_strengths = self.reg_strengths * torch.sum(
                        (strengths_batch - 1)**2)
                # regularise the mean of the strengths
                reg_strengths += self.reg_strengths_mean * \
                        (strengths_batch.mean() - 1)**2
                loss += reg_strengths

            if self.vae:
                if 'mmd' in self.vae_input:
                    mmd = loss_mmd(zs_batch, masks_batch)
                    loss += self.reg_vae * mmd
                else:  # KLD
                    loss += self.reg_vae * loss_kld(
                            mus_batch, logvars_batch, masks_batch)

            self.optimizer.zero_grad()   # clear gradients for next train
            loss.backward(retain_graph=True)         # backpropagation, compute gradients
            self.optimizer.step()        # apply gradients

            loss_in_batch[i] = loss.item()
            mse_in_batch[i] = mse.item()
            if self.vae and 'mmd' in self.vae_input:
                # mmd_in_batch[i] = mmd.item()
                mmd_in_batch[i] = mmd.item()

        self.losses_ = loss_in_batch
        if any(np.isnan(self.losses_)):
            print('Loss is NaN!')
        self.mses_ = mse_in_batch
        if self.vae and 'mmd' in self.vae_input:
            self.mmds_ = mmd_in_batch

        # evaluate on validation data
        if X_val is not None:
            with torch.no_grad():
                self.losses_train_ = self.losses_
                self.mses_train_ = self.mses_
                if self.vae and 'mmd' in self.vae_input:
                    self.mmds_train_ = self.mmds_

                # run batches
                sampler_val = SequentialSampler(range(n_samples_val))
                batch_iter = BatchSampler(sampler_val, batch_size=batch_size,
                                          drop_last=False)
                n_batches = len(batch_iter)

                loss_in_batch = np.zeros(n_batches)
                mse_in_batch = np.zeros(n_batches)
                if self.vae and 'mmd' in self.vae_input:
                    mmd_in_batch = np.zeros(n_batches)

                self.static_graph.eval()
                for i, batch_ind in enumerate(batch_iter):
                    if self.verbose:
                        print('\rvalidation batch {}/{} '.format(i, n_batches),
                              end='')
                    X_batch = X_val[batch_ind]
                    masks_batch = masks_val[batch_ind]
                    y_batch = y_val[batch_ind]

                    if self.vae:
                        prediction_batch, zs_batch, mus_batch, logvars_batch = \
                                self.static_graph(X_batch, masks_batch)

                    elif self.use_strength and 'context' in self.strength_method:
                        prediction_batch, strengths_batch = self.static_graph(
                                X_batch, masks_batch)
                    else:
                        prediction_batch = self.static_graph(X_batch, masks_batch)

                    mse = self.criterion(prediction_batch, y_batch)
                    # must be (1. nn output, 2. target)
                    loss = torch.zeros_like(mse)
                    loss += mse  # otherwise it makes it the same reference

                    if self.use_strength and 'context' in self.strength_method:
                        # regularise all the strengths
                        reg_strengths = self.reg_strengths * torch.sum(
                                (strengths_batch - 1)**2)
                        # regularise the mean of the strengths
                        reg_strengths += self.reg_strengths_mean * \
                                (strengths_batch.mean() - 1)**2
                        loss += reg_strengths

                    if self.vae:
                        if 'mmd' in self.vae_input:
                            mmd = loss_mmd(zs_batch, masks_batch)
                            loss += self.reg_vae * mmd
                        else:  # KLD
                            loss += self.reg_vae * loss_kld(
                                    mus_batch, logvars_batch, masks_batch)

                    loss_in_batch[i] = loss.item()
                    mse_in_batch[i] = mse.item()
                    if self.vae and 'mmd' in self.vae_input:
                        mmd_in_batch[i] = mmd.item()

                self.losses_ = loss_in_batch
                if any(np.isnan(self.losses_)):
                    print('Loss is NaN!')
                self.mses_ = mse_in_batch
                if self.vae and 'mmd' in self.vae_input:
                    self.mmds_ = mmd_in_batch

        if self.verbose:
            print('\r', end='')


    def predict(self, X, masks, sample_vae=False):  #
        """
        X and masks are np arrays. sample_vae enables sampling for VAE.
        """
        # set to predict - should work for vae and others as well?
        if X.ndim == 1:  # if it's a single sample
            X = X[np.newaxis, :]

        n_samples = X.shape[0]
        n_contours = X.shape[2]
        y = np.zeros((n_samples, self.n_output))

        # cast to Tensor
        dtype = torch.float32
        if self.use_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        X = torch.from_numpy(X).type(dtype).to(device)
        y = torch.from_numpy(y).type(dtype).to(device)
        masks = torch.from_numpy(masks).type(dtype).to(device)
        if self.use_strength and 'context' in self.strength_method:
            strengths = X.new_zeros((n_samples, n_contours))

        # should batch size be the same with fit, maybe should be bigger for speed?
        batch_size = int(np.min((256, n_samples)))
        sampler = SequentialSampler(range(n_samples))

        if not sample_vae:
            self.static_graph.eval()
        else:
            self.static_graph.train()
        with torch.no_grad():
            for batch_ind in BatchSampler(sampler, batch_size=batch_size, drop_last=False):
                X_batch = X[batch_ind]
                masks_batch = masks[batch_ind]
                if self.vae:
                    prediction_batch, zs_batch, mus_batch, sigmas_batch = \
                            self.static_graph(X_batch, masks_batch)
                elif self.use_strength and 'context' in self.strength_method:
                    prediction_batch, strengths_batch = self.static_graph(
                            X_batch, masks_batch)
                    strengths[batch_ind] = strengths_batch
                else:
                    prediction_batch = self.static_graph(X_batch, masks_batch)

                y[batch_ind] = prediction_batch

        # output
        if self.use_strength and 'context' in self.strength_method:
            return y.detach().cpu().numpy(), strengths.detach().cpu().numpy()
        else:
            return y.detach().cpu().numpy()


class deep_rnn_baseline_model():
    """
    Deep RNN prosody model baseline - modelled after Merlin LSTM benchmark.
    """
    def __init__(self,
                 n_feature=4,
                 rnn_inputs=None,
                 n_hidden=[17],
                 n_hidden_rnn=[17],
                 rnn_model='lstm',
                 activation='tanh',
                 n_output=4,
                 batch_size='auto',
                 weight_decay=1e-4,
                 shuffle=True,
                 random_state=42,
                 verbose=True,
                 optimizer='adam',
                 learning_rate_init=1e-3,
                 use_cuda=True,
                 ):

        self.n_feature = n_feature
        self.rnn_inputs = rnn_inputs
        self.n_hidden = n_hidden  # last defines the lstm
        self.n_feature_rnn = n_hidden[-1]
        self.n_hidden_rnn = n_hidden_rnn

        self.rnn_model = rnn_model
        self.activation = activation
        self.n_output = n_output

        self.batch_size = batch_size
        self.weight_decay = weight_decay

        self.shuffle = shuffle
        self.random_state = random_state
        if random_state is not None:
            torch.manual_seed(random_state)

        self.verbose = verbose
        self.optimizer_type = optimizer
        self.learning_rate = learning_rate_init

        if use_cuda and not torch.cuda.is_available():
            use_cuda = False
        if self.verbose:
            print('CUDA set to ', use_cuda)
        self.use_cuda = use_cuda

        self.init_model()

    def init_model(self):
        if self.verbose:
            print('Initialising baseline RNN ...')
        self.model = ContourGeneratorMLP(
                    n_feature=self.n_feature,
                    n_hiddens=self.n_hidden,
                    activation=self.activation,
                    output_hidden=True
                    )
        self.model_rnn = ContourGeneratorRNN(
                    n_feature=self.n_feature_rnn,
                    n_hidden_rnn=self.n_hidden_rnn,
                    n_output=self.n_output,
                    rnn_model=self.rnn_model
                    )

        if self.verbose:
            print(self.model)

        self.criterion = torch.nn.MSELoss()
        if self.use_cuda:
            self.model.cuda()
            self.model_rnn.cuda()
            self.criterion.cuda()

        self.init_optimizer()

    def reset_model(self):
        self.model.reset_parameters()
        self.model_rnn.reset_parameters()
        self.init_optimizer()

    def init_optimizer(self):
        if self.optimizer_type == 'adam':  # kiss
            self.optimizer = torch.optim.Adam(
                    list(self.model.parameters()) + list(self.model_rnn.parameters()),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay)
        else:
            raise ValueError('Optimizer not recognized - use adam : )')

    def fit(self,
            X, y,
            X_val=None, y_val=None):
        """
        all of these are np arrays
        x for RNN should be multiple utterances so:
            (n_utterances, seq_len, n_features + n_context, n_cgs)
        """
        # set to training
        self.model.train()
        self.model_rnn.train()

        if X.ndim == 2:  # if it's a single utterance
            X = X[np.newaxis, :]
        n_samples = X.shape[0]

        dtype = torch.float32
        if self.use_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        X = torch.from_numpy(X).type(dtype).to(device)
        y = torch.from_numpy(y).type(dtype).to(device)
        if X_val is not None:
            n_samples_val = X_val.shape[0]
            X_val = torch.from_numpy(X_val).type(dtype).to(device)
            y_val = torch.from_numpy(y_val).type(dtype).to(device)

        if self.batch_size == 'auto':
            batch_size = np.min((64, n_samples))
        elif self.batch_size == 'all':
            batch_size = n_samples
        else:
            batch_size = int(np.min((self.batch_size, n_samples)))  # pytorch insists on int

        if self.shuffle:
            if self.random_state is not None:  # for random sampler
                torch.manual_seed(self.random_state)
            sampler = RandomSampler(range(n_samples))
        else:
            sampler = SequentialSampler(range(n_samples))
        batch_iter = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
        n_batches = len(batch_iter)

        loss_in_batch = np.zeros(n_batches)
        mse_in_batch = np.zeros(n_batches)  # store mean mse for each batch
        for i, batch_ind in enumerate(batch_iter):
            if self.verbose:
                print('\rtraining batch {}/{} '.format(i, n_batches), end='')

            for i_utt, utt_ind in enumerate(batch_ind):
                X_utt = X[utt_ind]
                y_utt = y[utt_ind]
                X_utt = X_utt.unsqueeze(1)  # batch size
                y_utt = y_utt.unsqueeze(1)

                dnn_out_utt = self.model(X_utt)
                prediction_utt = self.model_rnn(dnn_out_utt)
                mse = self.criterion(prediction_utt, y_utt)  # must be (1. nn output, 2. target)
                if i_utt == 0:
                    loss = torch.zeros_like(mse)  # first run init
                    mse_acc = 0
                mse_acc += mse.item()
                loss += mse  # otherwise it makes it the same reference

            self.optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            self.optimizer.step()  # apply gradients
            # accumulate losses in batch
            loss_in_batch[i] = loss.item() / batch_size
            mse_in_batch[i] = mse_acc / batch_size

        self.losses_ = loss_in_batch
        if any(np.isnan(self.losses_)):
            print('Loss is NaN!')
            raise ValueError()
        self.mses_ = mse_in_batch

        # evaluate on validation data
        if X_val is not None:
            self.model.eval()
            self.model_rnn.eval()
            with torch.no_grad():
                self.losses_train_ = self.losses_
                self.mses_train_ = self.mses_
                # run batches
                sampler_val = SequentialSampler(range(n_samples_val))
                batch_iter = BatchSampler(sampler_val,
                                          batch_size=batch_size,
                                          drop_last=False)
                n_batches = len(batch_iter)
                loss_in_batch = np.zeros(n_batches)
                mse_in_batch = np.zeros(n_batches)
                for i, batch_ind in enumerate(batch_iter):
                    if self.verbose:
                        print('\rvalidation batch {}/{} '.format(
                                i, n_batches), end='')
                    for i_utt, utt_ind in enumerate(batch_ind):
                        X_utt = X[utt_ind]
                        y_utt = y[utt_ind]
                        X_utt = X_utt.unsqueeze(1)  # batch size
                        y_utt = y_utt.unsqueeze(1)
                        dnn_out_utt = self.model(X_utt)
                        prediction_utt = self.model_rnn(dnn_out_utt)
                        mse = self.criterion(prediction_utt, y_utt)
                        # must be (1. nn output, 2. target)
                        if i_utt == 0:
                            loss = torch.zeros_like(mse)  # first run init
                            mse_acc = 0
                        loss += mse  # otherwise it makes it the same reference
                        mse_acc += mse.item()
                    # accumulate losses in batch
                    loss_in_batch[i] = loss.item() / batch_size
                    mse_in_batch[i] = mse_acc / batch_size

                self.losses_ = loss_in_batch
                if any(np.isnan(self.losses_)):
                    print('Loss is NaN!')
                self.mses_ = mse_in_batch

        if self.verbose:
            print('\r', end='')


    def predict(self, X):
        """
        all of these are np arrays
        x for RNN should be multiple utterances so:
            (n_utterances, seq_len, n_features)

        Output
        ======
        y shape (n_utterances, seq_len, n_output)
        """
        self.model.eval()
        self.model_rnn.eval()
        if X.ndim == 2:  # if it's a single utterance
            X = X[np.newaxis, :]
        n_samples = X.shape[0]
        seq_len = X.shape[1]
        # cast to Tensor
        dtype = torch.float32
        if self.use_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        X = torch.from_numpy(X).type(dtype).to(device)
        if self.batch_size == 'auto':
            batch_size = np.min((64, n_samples))
        elif self.batch_size == 'all':
            batch_size = n_samples
        else:
            batch_size = int(np.min((int(self.batch_size), n_samples)))
        sampler = SequentialSampler(range(n_samples))
        batch_iter = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
        n_batches = len(batch_iter)
        y = X.new_zeros((n_samples, seq_len, self.n_output), requires_grad=False)
        with torch.no_grad():
            for i, batch_ind in enumerate(batch_iter):
                if self.verbose:
                    print('\rpredicting batch {}/{} '.format(i, n_batches), end='')
                for utt_ind in batch_ind:
                    X_utt = X[utt_ind]
                    X_utt = X_utt.unsqueeze(1)  # batch size
                    dnn_out_utt = self.model(X_utt)
                    prediction_utt = self.model_rnn(dnn_out_utt)
                    y[utt_ind, :, :] = prediction_utt[:,0,:]  # batch index

        if self.verbose:
            print('\r', end='')

        return y.detach().cpu().numpy()

class deep_rnn_model():
    """
    Deep RNN prosody model incorporates all contour generators into one model.
    """
    def __init__(self,
                 contour_types=None,  # init of nncgs
                 contour_generators=None,
                 contours_in_graph=None,  # list of nncgs in static graph
                 n_feature=4,
                 rnn_inputs=None,
                 n_hidden=17,
                 rnn_model='rnn',
                 activation='tanh',
                 n_output=4,
                 batch_size='auto',
                 weight_decay=1e-4,
                 shuffle=False,
                 random_state=42,
                 verbose=True,
                 optimizer='adam',
                 learning_rate_init=1e-3,
                 use_cuda=True,
                 vae=False,
                 n_feature_vae=None,
                 n_hidden_vae=None,  # for the vae
                 n_latent=2,
                 reg_vae=.3,
                 vae_as_input=False,
                 vae_as_input_hidd=False,  # use vae as input and as hidden init
                 ):

        self.contour_types = contour_types
        self.contour_generators = contour_generators  # if None init
        self.contours_in_graph = contours_in_graph

        self.n_feature = n_feature
        self.rnn_inputs = rnn_inputs
        self.n_hidden = n_hidden

        self.rnn_model = rnn_model
        self.activation = activation
        self.n_output = n_output

        self.batch_size = batch_size
        self.weight_decay = weight_decay

        self.shuffle = shuffle
        self.random_state = random_state
        if random_state is not None:
            torch.manual_seed(random_state)

        self.verbose = verbose
        self.optimizer_type = optimizer
        self.learning_rate = learning_rate_init


        if use_cuda and not torch.cuda.is_available():
            use_cuda = False
        if self.verbose:
            print('CUDA set to ', use_cuda)
        self.use_cuda = use_cuda

        self.vae = vae
        self.n_hidden_vae = n_hidden_vae
        self.n_latent = n_latent
        self.reg_vae = reg_vae
        self.n_feature_vae = n_feature_vae
        self.vae_as_input = vae_as_input
        self.vae_as_input_hidd = vae_as_input_hidd

        self.init_model()

    def init_model(self):
        # init contour generators
        contour_gen_list = self.contour_types
        if self.contour_generators is None:
            self.contour_generators = {}  # init dictinary
        else:
            for pre_contour_type in self.contour_generators.keys():
                # go through pretrained contour generators
                if pre_contour_type not in contour_gen_list:  # delete it from dict
                    del self.contour_generators[pre_contour_type]
                if pre_contour_type in contour_gen_list:  # remove it from list
                    contour_gen_list.remove(pre_contour_type)
        # add missing ones
        for contour_type in contour_gen_list:
            if self.vae:
                self.contour_generators[contour_type] = ContourGeneratorVRNN(
                                        n_feature=self.n_feature,
                                        n_hidden_rnn=self.n_hidden,
                                        n_output=self.n_output,
                                        n_feature_vae=self.n_feature_vae,
                                        n_hidden_vae=self.n_hidden_vae,
                                        n_latent=self.n_latent,
                                        rnn_model=self.rnn_model,
                                        vae_as_input=self.vae_as_input,
                                        vae_as_input_hidd=self.vae_as_input_hidd,
                                        )
            else:
                self.contour_generators[contour_type] = ContourGeneratorRNN(
                                        n_feature=self.n_feature,
                                        n_hidden_rnn=self.n_hidden,  # from list to int
                                        n_output=self.n_output,
                                        rnn_model=self.rnn_model,
                                        )
        self.criterion = torch.nn.MSELoss()
        self.rnn_graph = RNNGraph(self.contour_generators,
                                  self.contours_in_graph,
                                  self.vae,
                                  self.rnn_inputs)
        if self.use_cuda:
            self.rnn_graph.cuda()
            self.criterion.cuda()

        if self.optimizer_type == 'adam':  # kiss
            self.optimizer = torch.optim.Adam(self.rnn_graph.parameters(),
                                              lr=self.learning_rate,
                                              weight_decay=self.weight_decay)
        else:
            raise ValueError('Optimizer not recognized - use adam : )')

    def reset_model(self):
        self.rnn_graph.reset_parameters()

        if self.optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(self.parameters,
                                              lr=self.learning_rate,
                                              weight_decay=self.weight_decay)
        else:
            raise ValueError('Optimizer not recognized - use adam : )')

    def fit(self,
            X, cg_masks, contour_starts, y,
            X_val=None, cg_masks_val=None, contour_starts_val=None, y_val=None):
        """
        all of these are np arrays
        x for RNN should be multiple utterances so:
            (n_utterances, seq_len, n_features + n_context, n_cgs)
        cg_masks gives when each cg module is active
            (n_utterances, seq_len, n_cgs)
        contour_starts marks the starts of the contour with 1, for hidden init
            (n_utterances, seq_len, n_cgs)
        """
        # set to training
        self.rnn_graph.train()

        if X.ndim == 3:  # if it's a single utterance
            X = X[np.newaxis, :]
        n_samples = X.shape[0]

        dtype = torch.float32
        if self.use_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        X = torch.from_numpy(X).type(dtype).to(device)
        y = torch.from_numpy(y).type(dtype).to(device)
        cg_masks = torch.from_numpy(cg_masks).type(dtype).to(device)
        contour_starts = torch.from_numpy(contour_starts).type(dtype).to(device)

        if X_val is not None:
            n_samples_val = X_val.shape[0]
            X_val = torch.from_numpy(X_val).type(dtype).to(device)
            y_val = torch.from_numpy(y_val).type(dtype).to(device)
            cg_masks_val = torch.from_numpy(cg_masks_val).type(dtype).to(device)
            contour_starts_val = torch.from_numpy(contour_starts_val).type(dtype).to(device)

        if self.batch_size == 'auto':
            batch_size = np.min((64, n_samples))
        elif self.batch_size == 'all':
            batch_size = n_samples
        else:
            batch_size = int(np.min((self.batch_size, n_samples)))  # pytorch insists on int

        if self.shuffle:
            if self.random_state is not None:  # for random sampler
                torch.manual_seed(self.random_state)
            sampler = RandomSampler(range(n_samples))
        else:
            sampler = SequentialSampler(range(n_samples))
        batch_iter = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
        n_batches = len(batch_iter)

        loss_in_batch = np.zeros(n_batches)
        mse_in_batch = np.zeros(n_batches)  # store mean mse for each batch
        if self.vae:
            mmd_in_batch = np.zeros(n_batches)

        for i, batch_ind in enumerate(batch_iter):
            if self.verbose:
                print('\rtraining batch {}/{} '.format(i, n_batches), end='')

            for i_utt, utt_ind in enumerate(batch_ind):
                X_utt = X[utt_ind]
                cg_masks_utt = cg_masks[utt_ind]
                contour_starts_utt= contour_starts[utt_ind]
                y_utt = y[utt_ind]

                if self.vae:
                    prediction_utt, zs_utt = self.rnn_graph(X_utt,
                                                            cg_masks_utt,
                                                            contour_starts_utt)
                else:
                    prediction_utt = self.rnn_graph(X_utt,
                                                    cg_masks_utt,
                                                    contour_starts_utt)
                mse = self.criterion(prediction_utt, y_utt)  # must be (1. nn output, 2. target)
                if i_utt == 0:
                    loss = torch.zeros_like(mse)  # first run init
                    mse_acc = 0
                mse_acc += mse.item()
                loss += mse  # otherwise it makes it the same reference

                if self.vae:
                    mmd = loss_mmd_rnn(zs_utt)
                    if i_utt == 0:
                        mmd_acc = 0
                    mmd_acc += mmd.item()
                    loss += self.reg_vae * mmd

            self.optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            self.optimizer.step()  # apply gradients

            # accumulate losses in batch
            loss_in_batch[i] = loss.item() / batch_size
            mse_in_batch[i] = mse_acc / batch_size
            if self.vae:
                mmd_in_batch[i] = mmd_acc / batch_size

        self.losses_ = loss_in_batch
        if any(np.isnan(self.losses_)):
            print('Loss is NaN!')
        self.mses_ = mse_in_batch
        if self.vae:
            self.mmds_ = mmd_in_batch

        # evaluate on validation data
        if X_val is not None:
            with torch.no_grad():
                self.rnn_graph.eval()
                self.losses_train_ = self.losses_
                self.mses_train_ = self.mses_
                if self.vae:
                    self.mmds_train_ = self.mmds_

                # run batches
                sampler_val = SequentialSampler(range(n_samples_val))
                batch_iter = BatchSampler(sampler_val, batch_size=batch_size, drop_last=False)
                n_batches = len(batch_iter)

                loss_in_batch = np.zeros(n_batches)
                mse_in_batch = np.zeros(n_batches)
                if self.vae:
                    mmd_in_batch = np.zeros(n_batches)

                for i, batch_ind in enumerate(batch_iter):
                    if self.verbose:
                        print('\rvalidation batch {}/{} '.format(i, n_batches), end='')
                    for i_utt, utt_ind in enumerate(batch_ind):
                        X_utt = X[utt_ind]
                        cg_masks_utt = cg_masks[utt_ind]
                        contour_starts_utt= contour_starts[utt_ind]
                        y_utt = y[utt_ind]

                        if self.vae:
                            prediction_utt, zs_utt = self.rnn_graph(
                                    X_utt, cg_masks_utt, contour_starts_utt)
                        else:
                            prediction_utt = self.rnn_graph(
                                    X_utt, cg_masks_utt, contour_starts_utt)

                        mse = self.criterion(prediction_utt, y_utt)
                        # must be (1. nn output, 2. target)
                        if i_utt == 0:
                            loss = torch.zeros_like(mse)  # first run init
                            mse_acc = 0
                        loss += mse  # otherwise it makes it the same reference
                        mse_acc += mse.item()

                    if self.vae:
                        mmd = loss_mmd_rnn(zs_utt)
                        if i_utt == 0:
                            mmd_acc = 0
                        loss += self.reg_vae * mmd
                        mmd_acc += mmd.item()

                    # accumulate losses in batch
                    loss_in_batch[i] = loss.item() / batch_size
                    mse_in_batch[i] = mse_acc / batch_size
                    if self.vae:
                        mmd_in_batch[i] = mmd_acc / batch_size

                self.losses_ = loss_in_batch
                if any(np.isnan(self.losses_)):
                    print('Loss is NaN!')
                self.mses_ = mse_in_batch
                if self.vae:
                    self.mmds_ = mmd_in_batch

        if self.verbose:
            print('\r', end='')


    def predict(self, X, cg_masks, contour_starts,
                sample_vae=False):  # allow sampling for VAE
        """
        all of these are np arrays
        x for RNN should be multiple utterances so:
            (n_utterances, seq_len, n_features + n_context, n_cgs)
        cg_masks gives when each cg module is active
            (n_utterances, seq_len, n_cgs)
        contour_starts marks the starts of the contour with 1, for hidden init
            (n_utterances, seq_len, n_cgs)

        Output
        ======
        y shape (n_utterances, seq_len, n_output)
        """
        if not sample_vae:
            self.rnn_graph.eval()
        else:
            self.rnn_graph.train()

        if X.ndim == 3:  # if it's a single utterance
            X = X[np.newaxis, :]
        n_samples = X.shape[0]
        seq_len = X.shape[1]

        # cast to Tensor
        dtype = torch.float32
        if self.use_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        X = torch.from_numpy(X).type(dtype).to(device)
        cg_masks = torch.from_numpy(cg_masks).type(dtype).to(device)
        contour_starts = torch.from_numpy(contour_starts).type(dtype).to(device)

        if self.batch_size == 'auto':
            batch_size = np.min((64, n_samples))
        elif self.batch_size == 'all':
            batch_size = n_samples
        else:
            batch_size = int(np.min((int(self.batch_size), n_samples)))

        sampler = SequentialSampler(range(n_samples))
        batch_iter = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
        n_batches = len(batch_iter)
        y = X.new_zeros((n_samples, seq_len, self.n_output), requires_grad=False)

        with torch.no_grad():
            for i, batch_ind in enumerate(batch_iter):
                if self.verbose:
                    print('\rpredicting batch {}/{} '.format(i, n_batches), end='')

                for utt_ind in batch_ind:
                    X_utt = X[utt_ind]
                    cg_masks_utt = cg_masks[utt_ind]
                    contour_starts_utt = contour_starts[utt_ind]
                    if self.vae:
                        y_utt, _ = self.rnn_graph(
                                X_utt, cg_masks_utt, contour_starts_utt)
                    else:
                        y_utt = self.rnn_graph(
                                X_utt, cg_masks_utt, contour_starts_utt)
                    y[utt_ind, :, :] = y_utt

        if self.verbose:
            print('\r', end='')

        return y.detach().cpu().numpy()


class PyTorchRegressor():
    def __init__(self, n_feature=4,
                 hidden_layer_sizes=17,
                 activation='logistic',
                 n_output=3,
                 n_context=None,
                 n_hidden_context=None,
                 batch_size='auto',
                 max_iter=50,
                 adjust_max_iter=False,
                 alpha=1e-4, shuffle=True,  random_state=42,
                 verbose=False, warm_start=True, early_stopping=False,
                 validation_fraction=0.01, solver='adam', learning_rate_init=1e-3,
                 use_strength=False, strength_method='manual',
                 reg_strengths=None, reg_strengths_mean=None,
                 use_cuda=False,
                 early_thresh=None,
                 patience=None):

        self.n_feature = n_feature
        self.n_context = n_context
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_hidden_context = n_hidden_context
        self.n_output = n_output
        self.activation = activation
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.adjust_max_iter = adjust_max_iter
        self.weight_decay = alpha

        self.shuffle = shuffle
        self.random_state = random_state
        if random_state is not None:
            torch.manual_seed(random_state)

        self.verbose = verbose
        self.warm_start = warm_start
        self.early_stopping = early_stopping  # TODO
        self.validation_fraction = validation_fraction
        self.optimizer_type = solver
        self.learning_rate = learning_rate_init

        self.use_strength = use_strength
        self.strength_method = strength_method
        self.reg_strengths = reg_strengths
        self.reg_strengths_mean = reg_strengths_mean

        self.use_cuda = use_cuda

        self.early_thresh = early_thresh
        self.patience = patience

        self.init_model()

    def init_model(self):
        if not self.use_strength:
            self.contour_generator = ContourGeneratorMLP(
                                                      n_feature=self.n_feature,
                                                      n_hiddens=self.hidden_layer_sizes,
                                                      n_output=self.n_output,
                                                      activation=self.activation)
        else:
            if 'context' in self.strength_method:
                self.contour_generator = ContourGeneratorStrengthMLP(
                                                      n_feature=self.n_feature,
                                                      n_hiddens=self.hidden_layer_sizes,
                                                      n_output=self.n_output,
                                                      n_context=self.n_context,
                                                      n_hiddens_context=self.n_hidden_context,
                                                      activation=self.activation)
        self.criterion = torch.nn.MSELoss()
        if self.use_cuda:
            self.contour_generator.cuda()
            self.criterion.cuda()
        self.optimizer = torch.optim.Adam(self.contour_generator.parameters(),
                                          lr=self.learning_rate,
                                          weight_decay=self.weight_decay)

    def reset_model(self):
        self.contour_generator.reset_parameters()
        if self.optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(self.contour_generator.parameters(),
                                              lr=self.learning_rate,
                                              weight_decay=self.weight_decay)
        else:
            raise ValueError('Optimizer not recognized')

    def reset_optimizer(self):
        if self.optimizer_type == 'adam':
            parameters = [p for p in self.contour_generator.parameters() if p.requires_grad]
            self.optimizer = torch.optim.Adam(parameters,
                                              lr=self.learning_rate,
                                              weight_decay=self.weight_decay)
        else:
            raise ValueError('Optimizer not recognized')

    def fit(self, X, y, strengths=None, contexts=None,
            X_val=None, y_val=None,  # validation set
            strengths_val=None, contexts_val=None):

        self.contour_generator.train()
        n_samples = X.shape[0]
        if X_val is not None:
            n_samples_val = X_val.shape[0]

        # cast to Tensor
        dtype = torch.float32
        if self.use_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        X = torch.from_numpy(X).type(dtype).to(device)
        y = torch.from_numpy(y).type(dtype).to(device)

        if self.use_strength:
            if 'context' in self.strength_method:
                contexts = torch.from_numpy(contexts).type(dtype).to(device)

        # validation stuff
        if X_val is not None:
            X_val = torch.from_numpy(X_val).type(dtype).to(device)
            y_val = torch.from_numpy(y_val).type(dtype).to(device)
        if strengths_val is not None:
            strengths_val = torch.from_numpy(strengths_val).type(dtype).to(device)
            strengths_val = torch.unsqueeze(strengths_val, dim=1).type(dtype).to(device)
        if contexts_val is not None:
            contexts_val = torch.from_numpy(contexts_val).type(dtype).to(device)

        if not self.warm_start:
            self.reset_model()

        if self.batch_size == 'auto':
            batch_size = np.min((200, n_samples))
        elif self.batch_size == 'all':
            batch_size = n_samples
        else:
            batch_size = int(np.min((int(self.batch_size), n_samples)))

        if self.shuffle:
            if self.random_state is not None:  # for random sampler
                torch.manual_seed(self.random_state)
            sampler = RandomSampler(range(n_samples))
        else:
            sampler = SequentialSampler(range(n_samples))
        batch_iter = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
        n_batches = len(batch_iter)

        # adjust max_iter based on batch size of 32 (for Interspeech paper)
        if self.adjust_max_iter:
            n_batches_32 = len(BatchSampler(sampler, batch_size=32, drop_last=False))
            max_iter_adjusted = self.max_iter * n_batches_32 // n_batches
            if max_iter_adjusted == 0:  # when using batch_size == 1
                max_iter_adjusted = 1
        else:
            max_iter_adjusted = self.max_iter
        self.max_iter_ = max_iter_adjusted

        losses = np.zeros(max_iter_adjusted)
        mses = np.zeros(max_iter_adjusted)
        if X_val is not None:
            losses_train = np.zeros(max_iter_adjusted)
            mses_train = np.zeros(max_iter_adjusted)
        best_error = np.inf

        for t in range(0, max_iter_adjusted):
            if self.verbose:
                print('pyregressor epoch {}/{}\r'.format(t, max_iter_adjusted), end='')

            batch_iter = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
            n_batches = len(batch_iter)  # should be the same ...
            loss_in_batch = np.zeros(n_batches)
            mse_in_batch = np.zeros(n_batches)
            for i, batch_ind in enumerate(batch_iter):
                X_batch = X[batch_ind]
                y_batch = y[batch_ind]
                if self.use_strength:
                    if self.strength_method == 'manual':
                        strengths_batch = strengths[batch_ind]
                        prediction_batch = self.contour_generator(
                                X_batch, strengths=strengths_batch)

                    elif 'context' in self.strength_method:
                        contexts_batch = contexts[batch_ind]
                        prediction_batch, strengths_batch = \
                                self.contour_generator(X_batch,
                                                       contexts=contexts_batch)

                else:
                    prediction_batch = self.contour_generator(X_batch)

                mse = self.criterion(prediction_batch, y_batch)
                # must be (1. nn output, 2. target)
                loss = torch.zeros_like(mse)  # init
                loss += mse  # otherwise it makes it the same reference
                if self.use_strength and 'context' in self.strength_method:
                    # regularise all the strengths
                    reg_strengths = self.reg_strengths * torch.sum((strengths_batch - 1)**2)
                    # regularise the mean of the strengths
                    reg_strengths += self.reg_strengths_mean * (strengths_batch.mean() - 1)**2
                    loss += reg_strengths

                self.optimizer.zero_grad()   # clear gradients for next train
                loss.backward(retain_graph=True)         # backpropagation, compute gradients
                self.optimizer.step()        # apply gradients

                loss_in_batch[i] = loss.item()
                mse_in_batch[i] = mse.item()

            losses[t] = loss_in_batch.mean()
            mses[t] = mse_in_batch.mean()

            if X_val is not None:  # calculate loss on validation set
                with torch.no_grad():
                    self.contour_generator.eval()
                    losses_train[t] = losses[t]
                    mses_train[t] = mses[t]
                    # run batches
                    sampler_val = SequentialSampler(range(n_samples_val))
                    batch_iter = BatchSampler(sampler_val, batch_size=batch_size,
                                              drop_last=False)
                    n_batches = len(batch_iter)
                    loss_in_batch = np.zeros(n_batches)
                    mse_in_batch = np.zeros(n_batches)
                    for i, batch_ind in enumerate(batch_iter):
                        X_batch = X_val[batch_ind]
                        y_batch = y_val[batch_ind]
                        if self.use_strength:
                            if 'context' in self.strength_method:
                                contexts_batch = contexts_val[batch_ind]
                                prediction_batch, strengths_batch = self.contour_generator(
                                        X_batch, contexts=contexts_batch)
                        else:
                            prediction_batch = self.contour_generator(X_batch)

                        mse = self.criterion(prediction_batch, y_batch)
                        # must be (1. nn output, 2. target)
                        loss = torch.zeros_like(mse)  # init
                        loss += mse  # otherwise it makes it the same reference
                        if self.use_strength and 'context' in self.strength_method:
                            # regularise all the strengths
                            reg_strengths = self.reg_strengths * \
                                    torch.sum((strengths_batch - 1)**2)
                            # regularise the mean of the strengths
                            reg_strengths += self.reg_strengths_mean * \
                                    (strengths_batch.mean() - 1)**2
                            loss += mse + reg_strengths
                        loss_in_batch[i] = loss.item()
                        mse_in_batch[i] = mse.item()

                    ## acumulate validation loss
                    loss_epoch = loss_in_batch.mean()
                    losses[t] = loss_epoch
                    mse_epoch = mse_in_batch.mean()
                    mses[t] = mse_epoch
                    if self.verbose:
                        print('Validation error {:.4f}\r'.format(loss_epoch), end='')

                    # early stopping
                    if best_error - loss_epoch > self.early_thresh:
                        patience = copy.copy(self.patience)
                    else:
                        patience -= 1

                    if best_error > loss_epoch:  # update best error
                        best_error = loss_epoch
                        # presave model
                        best_epoch = t
                        best_model = self.contour_generator.state_dict()
                        best_optim = self.optimizer.state_dict()

                    if not patience:
                        if self.verbose:
                            print(f'Lost patience at epoch {t} with '
                                  'mse {mse_epoch}, best epoch: {best_epoch}\r',
                                  end='')
                        break  # break the epoch loop

        self.losses_ = losses
        self.losses_train_ = losses_train
        self.loss_ = losses[-1]
        self.mses_ = mses
        self.mses_train_ = mses_train
        self.mse_ = mses[-1]

        # early stop
        if X_val is not None:
            # stop
            self.contour_generator.load_state_dict(best_model)
            self.optimizer.load_state_dict(best_optim)
            self.best_epoch_ = best_epoch

            # transfer validation loss
            self.losses_ = losses
            self.loss_ = losses[best_epoch]
            self.mses_ = mses
            self.mse_ = mses[best_epoch]

            # transfer training loss
            self.losses_train_ = losses_train
            self.loss_train_ = losses_train[best_epoch]
            self.mses_train_ = mses_train
            self.mse_train_ = mses_train[best_epoch]

        if self.verbose:
            print('\r', end='')


    def predict(self, X, strengths=None, contexts=None):
        self.contour_generator.eval()
        n_samples = X.shape[0]
        y = np.zeros((n_samples, self.n_output))

        # cast to Tensor
        dtype = torch.float32
        if self.use_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        X = torch.from_numpy(X).type(dtype).to(device)
        y = torch.from_numpy(y).type(dtype).to(device)

        if self.use_strength:
            if 'context' in self.strength_method:
                if contexts is not None:  # for expansion - None to ignore context
                    contexts = torch.from_numpy(contexts).type(dtype).to(device)
                    strengths = X.new_zeros((n_samples, 1))

        batch_size = int(np.min((256, n_samples)))
        sampler = SequentialSampler(range(n_samples))

        with torch.no_grad():
            for batch_ind in BatchSampler(sampler, batch_size=batch_size, drop_last=False):
                X_batch = X[batch_ind]
                if self.use_strength:
                    if 'context' in self.strength_method:
                        if contexts is not None:
                            contexts_batch = contexts[batch_ind]
                        else:
                            contexts_batch = None
                        prediction, strengths_batch = self.contour_generator(
                                X_batch, contexts=contexts_batch)
                        if strengths_batch is not None:
                            strengths[batch_ind] = strengths_batch
                else:
                    prediction = self.contour_generator(X_batch)
                y[batch_ind] = prediction

        if self.use_strength and 'context' in self.strength_method:
            if contexts is not None:  # for expansion - strengths are also None
                return y.detach().cpu().numpy(), strengths.detach().cpu().numpy()
            else:
                return y.detach().cpu().numpy(), None
        else:
            return y.detach().cpu().numpy()


class StaticGraph(torch.nn.Module):
    """
    This is a static graph made out of a set of contour generators according
    to specifications in contours_in_graph.
    """
    def __init__(self,
                 contour_generators,
                 contours_in_graph,
                 n_in_contour,
                 n_output,
                 use_strength,
                 strength_method,
                 vae,
                 vae_input):

        super(StaticGraph, self).__init__()
        self.contour_generators = contour_generators
        self.contours_in_graph = contours_in_graph
        self.n_contours_in_graph = len(contours_in_graph)
        self.n_output = n_output
        self.use_strength = use_strength
        self.strength_method = strength_method
        self.vae = vae
        self.vae_input = vae_input

        # assemble graph
        for i, contour_in_graph in enumerate(self.contours_in_graph):
            # TODO change for longer function types
            self.add_module(contour_in_graph, contour_generators[contour_in_graph[:2]])

    def reset_parameters(self):
        for module in self.modules():
            module.reset_parameters()

    def forward(self, x, masks):
        n_output = self.n_output
        for i, name in enumerate(self.contours_in_graph):
            module = getattr(self, name)
            x_module_all = x[:, :, i]

            strengths_module = x_module_all[:, 0]
            contexts_module = x_module_all[:, 5:]
            x_module = x_module_all[:, 1:5]
            if self.vae:
                if 'context' in self.vae_input:
                    y_module, zs_module, mus_module, logvars_module = module(
                            x_module_all[:, 1:])
                else:
                    y_module, zs_module, mus_module, logvars_module = module(x_module)

            elif self.use_strength:
                if self.strength_method == 'manual':
                    y_module = module(x_module, strengths_module)
                elif 'context' in self.strength_method:
                    y_module, strengths_module = module(x_module, contexts_module)
            else:
                y_module = module(x_module)

            if i == 0:
                y = y_module.unsqueeze(dim=2)  # we're stacking in the 3rd D
                if self.vae:
                    zs = zs_module.unsqueeze(dim=2)
                    mus = mus_module.unsqueeze(dim=2)
                    logvars = logvars_module.unsqueeze(dim=2)

                elif self.use_strength and 'context' in self.strength_method:
                    strengths = strengths_module
            else:
                y = torch.cat((y, y_module.unsqueeze(dim=2)), dim=2)
                if self.vae:
                    zs = torch.cat((zs, zs_module.unsqueeze(dim=2)), dim=2)
                    mus = torch.cat((mus, mus_module.unsqueeze(dim=2)), dim=2)
                    logvars = torch.cat((logvars,
                                         logvars_module.unsqueeze(dim=2)),
                                         dim=2)
                elif self.use_strength and 'context' in self.strength_method:
                    strengths = torch.cat((strengths, strengths_module), dim=1)

        # repeat mask accordingly
        masks_y = masks.unsqueeze(dim=1).repeat(1, n_output, 1)
        # variable doesn't have the resize_ method
        y = y * masks_y
        y_sum = y.sum(dim=2)

        if self.vae:
            return y_sum, zs, mus, logvars
        elif self.use_strength and 'context' in self.strength_method:
            return y_sum, strengths
        else:
            return y_sum

class RNNGraph(torch.nn.Module):
    """
    This is a rnn version of the static graph made out of a set of contour
    generators according to specifications in contours_in_graph.
    """
    def __init__(self,
                 contour_generators,
                 contours_in_graph,
                 vae,
                 rnn_inputs):

        super(RNNGraph, self).__init__()
        self.contour_generators = contour_generators
        self.contours_in_graph = contours_in_graph
        self.n_contours_in_graph = len(contours_in_graph)

        self.vae = vae

        self.rnn_inputs = rnn_inputs

        # assemble graph
        for i, contour_in_graph in enumerate(self.contours_in_graph):
            self.add_module(contour_in_graph, contour_generators[contour_in_graph[:2]])

    def reset_parameters(self):
        for module in self.modules():
            module.reset_parameters()

    def forward(self, x, cg_masks, contour_starts):
        """
        x for RNN should be one utterance so:
            (seq_len, n_features + n_context, n_cgs)
        cg_masks gives when each cg module is active
            (seq_len, n_cgs)
        contour_starts marks the start of contour with 1, for hidden state init
            (seq_len, n_cgs)
        """
        seq_len = x.shape[0]
        n_output = getattr(self, self.contours_in_graph[0]).n_output
        y = x.new_zeros((seq_len, n_output))
        # also accumulate zs for the mmd
        if self.vae:
            n_latent = getattr(self, self.contours_in_graph[0]).n_latent
            zs = x.new_zeros((seq_len*5, n_latent))  # should be enough
            zs_count = 0
        # step through contour generators
        for module_ind, name in enumerate(self.contours_in_graph):
            mask_module = cg_masks[:, module_ind]
            mask = mask_module == 1
            if not mask.any():
                continue
            module = getattr(self, name)
            x_module = x[:, :, module_ind]
            if self.vae:
                contexts_module = x_module[:, 5:]
            if self.rnn_inputs is None:
                x_module = x_module[:, 1:5]
            else:
                x_module = x_module[:, self.rnn_inputs]
                x_module = x_module.unsqueeze(1)  # for the batch size
            # t_seq = torch.arange(seq_len).type(dtype).long()
            t_seq = torch.arange(seq_len).type(x.type()).to(x.device)
            t_seq = t_seq[mask]
            contour_starts_seq = contour_starts[:, module_ind]
            contour_starts_seq = contour_starts_seq[mask]
            mask_starts = contour_starts_seq == 1
            starts = t_seq[mask_starts]
            if len(starts) > 1:
                mask_stops = torch.cat((mask_starts[1:],
                                        torch.tensor([1]).type(x.type()).to(x.device) == 1))
                stops = t_seq[mask_stops] + 1
            else:
                stops = [t_seq[-1] + 1]
            if len(stops) > len(starts):  # can happen if the first start is not 0
                stops = stops[1:]

            for start, stop in zip(starts, stops):
                start, stop = int(start), int(stop)  # pytorch doesn't like numpy
                h_module = None
                if module.rnn_model == 'lstm':
                    c_module = None
                assert start != stop
                x_rnn = x_module[start : stop]
                if np.isnan(x_rnn.detach().cpu().numpy()).any():
                    raise
                x_rnn = x_rnn.unsqueeze(1)  # for batch size
                assert x_rnn.ndimension() == 3
                if self.vae:
                    x_vae = contexts_module[start]
                    x_vae = x_vae.unsqueeze(dim=0)
                    if module.rnn_model == 'lstm':
                        y_rnn, z_vae = module(x_rnn, x_vae, h_module, c_module)
                    else:
                        y_rnn, z_vae = module(x_rnn, x_vae, h_module)
                    assert z_vae is not None  # used for initialisation
                    zs[zs_count] = z_vae
                    zs_count += 1
                else:
                    if module.rnn_model == 'lstm':
                        y_rnn = module(x_rnn, h_module, c_module)
                    else:
                        y_rnn = module(x_rnn, h_module)
                y[start : stop] = y[start : stop] + y_rnn[:, 0, :]

        if self.vae:
            zs = zs[:zs_count]  # trim

        if self.vae:
            return y, zs
        else:
            return y

class ContourGeneratorMLP(torch.nn.Module):
    """
    This contour generator has no strengths but supports multiple layers.
    """
    def __init__(self,
                 n_feature,
                 n_hiddens,
                 n_output=None,
                 activation='relu',
                 output_hidden=False  # for stacking an RNN for the baselineRNN
                 ):
        super(ContourGeneratorMLP, self).__init__()
        # n_hiddens and n_hiddens_context are lists of hidden layer widths.
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        self.output_hidden = output_hidden

        if type(n_hiddens) != list:
            n_hiddens = [n_hiddens]  # in case it is not a list already
        if n_hiddens[0] == 0:
            n_hiddens = None
        if n_hiddens is not None:  # None means no hidden layers
            self.n_hidden_layers = len(n_hiddens)
        else:
            self.n_hidden_layers = 0

        # contour generator network
        n_last = n_feature
        if n_hiddens is not None:
            for i, n_hidden in enumerate(n_hiddens):
                setattr(self, 'hidden{}'.format(i),
                        torch.nn.Linear(n_last, n_hidden))
                n_last = n_hidden

        if not output_hidden:  # add a linear output layer
            self.out_contour = torch.nn.Linear(n_last, n_output)   # output layer
        self.reset_parameters()

    def reset_parameters(self):
        # contour generator
        for i in range(self.n_hidden_layers):
            getattr(self, 'hidden{}'.format(i)).reset_parameters()
        if not self.output_hidden:
            self.out_contour.reset_parameters()

    def forward(self, x):
        # contour generator
        for i in range(self.n_hidden_layers):
            x = getattr(self, 'hidden{}'.format(i))(x)
            x = self.activation(x)
        if self.output_hidden:
            return x
        else:
            y = self.out_contour(x)  # no activation
            return y


class ContourGeneratorStrengthMLP(torch.nn.Module):
    """
    This is contour generator learns strengths for the contours based on context.
    The strength coefficient is obtained using a MLP that takes the
    function types that are active at any time during the contour.
    """
    def __init__(self,
                 n_feature,
                 n_hiddens,
                 n_output,
                 n_context=None,
                 n_hiddens_context=None,
                 activation='tanh'
                 ):
        super(ContourGeneratorStrengthMLP, self).__init__()
        # n_hiddens and n_hiddens_context are lists of hidden layer widths.
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        if type(n_hiddens) != list:
            n_hiddens = [n_hiddens]  # in case it is not a list already
        if n_hiddens[0] == 0:
            n_hiddens = None
            self.n_hidden_layers = 0
        else:
            self.n_hidden_layers = len(n_hiddens)
        if n_hiddens_context is not None:
            if type(n_hiddens_context) != list:
                n_hiddens_context = [n_hiddens_context]
            if n_hiddens_context[0] == 0:
                n_hiddens_context = None
            if n_hiddens_context is not None:  # None means no hidden layers
                self.n_hidden_layers_context = len(n_hiddens_context)
            else:
                self.n_hidden_layers_context = 0

        # contour generator network
        n_last = n_feature
        if n_hiddens is not None:
            for i, n_hidden in enumerate(n_hiddens):
                setattr(self, 'hidden{}'.format(i),
                        torch.nn.Linear(n_last, n_hidden))
                n_last = n_hidden
        self.out_contour = torch.nn.Linear(n_last, n_output)   # output layer

        # context network
        if n_context is not None:
            n_last = n_context
            if n_hiddens_context is not None:
                for i, n_hidden in enumerate(n_hiddens_context):
                    setattr(self, 'hidden_context{}'.format(i),
                            torch.nn.Linear(n_last, n_hidden))
                    n_last = n_hidden
            self.out_strength = torch.nn.Linear(n_last, 1)   # output layer

        self.reset_parameters()

    def reset_parameters(self):
        # contour generator
        for i in range(self.n_hidden_layers):
            getattr(self, 'hidden{}'.format(i)).reset_parameters()
        self.out_contour.reset_parameters()
        # context
        for i in range(self.n_hidden_layers_context):
            getattr(self, 'hidden_context{}'.format(i)).reset_parameters()
            # set weights to 0 so strength is close to 1
            getattr(self, 'hidden_context{}'.format(i)).bias.fill_(0)
        self.out_strength.reset_parameters()

    def forward(self, x, contexts=None):
        # contour generator
        for i in range(self.n_hidden_layers):
            x = getattr(self, 'hidden{}'.format(i))(x)
            x = self.activation(x)
        y = self.out_contour(x)  # no activation
        if contexts is not None:
            for i in range(self.n_hidden_layers_context):
                contexts = getattr(self, 'hidden_context{}'.format(i))(contexts)
                contexts = self.activation(contexts)
            strengths = self.out_strength(contexts)
            strengths = torch.sigmoid(strengths) * 2
            y = y * strengths
            return y, strengths
        else:
            return y


class ContourGeneratorVar(torch.nn.Module):
    """
    This is contour generator that learns a latent space representation
    of the variety in a prosodic contours' realisation.
    """
    def __init__(self, n_feature, enc_hidden,
                 n_latent,
                 n_output,
                 dec_hidden=None
                 ):

        super(ContourGeneratorVar, self).__init__()
        """
        n_hiddens and n_hiddens_context are lists of hidden layer widths.
        """
        self.sigmoid = torch.sigmoid
        self.tanh = torch.tanh
        self.relu = torch.relu

        if dec_hidden is None:
            dec_hidden = enc_hidden

        # encoder
        self.enc_hidden = torch.nn.Linear(n_feature, enc_hidden)
        self.enc_mu = torch.nn.Linear(enc_hidden, n_latent)
        self.enc_logvar = torch.nn.Linear(enc_hidden, n_latent)
        # decoder
        self.dec_hidden = torch.nn.Linear(n_latent, dec_hidden)
        self.dec_out = torch.nn.Linear(dec_hidden, n_output)

        self.reset_parameters()

    def reset_parameters(self):
        self.enc_hidden.reset_parameters()
        self.enc_mu.reset_parameters()
        self.enc_logvar.reset_parameters()
        self.dec_hidden.reset_parameters()
        self.dec_out.reset_parameters()

    def encode(self, x):
        # encoder
        x = self.tanh(self.enc_hidden(x))
#        x = self.relu(self.enc_hidden(x))
        mu = self.enc_mu(x)
        logvar = self.enc_logvar(x)
        return mu, logvar

    def sample(self, mu, logvar):
        # from VAE pytorch examples
        std = torch.exp(.5 * logvar)
        # eps = Variable(logvar.data.new(logvar.size()).normal_())
        eps = torch.zeros_like(logvar).normal_()
        return mu + eps*std

    def decode(self, z):
        # decoder
        y = self.tanh(self.dec_hidden(z))
#        y = self.relu(self.dec_hidden(z))
        y = self.dec_out(y)
        return y

    def forward(self, x):

        mu, logvar = self.encode(x)

        if self.training:  # sample only on training?
            z = self.sample(mu, logvar)
        else:
            z = mu

        y = self.decode(z)

        return y, z, mu, logvar


class ContourGeneratorRNN(torch.nn.Module):
    """
    This is a RNN contour generator, should learn the mean as the MLP CG.

    RNN layers:
        batch_first – If True, then the input and output tensors are provided as
        (batch, seq, feature), otherwise (seq_len, batch, input_size)
        input of shape (seq_len, batch, input_size)
        h_0 of shape (num_layers * num_directions, batch, hidden_size)
        output of shape (seq_len, batch, num_directions * hidden_size)
            For the unpacked case, the directions can be separated using
            output.view(seq_len, batch, num_directions, hidden_size)
        h_n (num_layers * num_directions, batch, hidden_size): tensor
            containing the hidden state for k = seq_len.
            Like output, the layers can be separated using
            h_n.view(num_layers, num_directions, batch, hidden_size).
    RNN cells:
    Inputs: input, hidden
        - **input** (batch, input_size): tensor containing input features
        - **hidden** (batch, hidden_size): tensor containing the initial hidden
          state for each element in the batch.
    Outputs: h'
        - **h'** (batch, hidden_size): tensor containing the next hidden state
          for each element in the batch
    """
    def __init__(self,
                 n_feature,
                 n_hidden_rnn,
                 n_output,
                 rnn_model='rnn',
                 ):
        super(ContourGeneratorRNN, self).__init__()

        self.n_feature = n_feature
        self.layers = len(n_hidden_rnn)
        self.n_hidden_rnn = n_hidden_rnn[0]  # all the layers have the same size
        self.n_output = n_output
        self.rnn_model = rnn_model
        self.sigmoid = torch.sigmoid
        self.tanh = torch.tanh
        self.relu = torch.relu

        if rnn_model == 'rnn':
            self.rnn = torch.nn.RNN(n_feature, self.n_hidden_rnn,
                                    num_layers=self.layers, batch_first=False)
        elif rnn_model == 'gru':
            self.rnn = torch.nn.GRU(n_feature, self.n_hidden_rnn,
                                    num_layers=self.layers, batch_first=False)
        elif rnn_model == 'lstm':
            self.rnn = torch.nn.LSTM(n_feature, self.n_hidden_rnn,
                                     num_layers=self.layers, batch_first=False)
        self.out = torch.nn.Linear(self.n_hidden_rnn, n_output)
        self.reset_parameters()

    def reset_parameters(self):
        self.rnn.reset_parameters()
        self.out.reset_parameters()

    def init_hidden(self, tensor):
        return tensor.new_zeros((1, 1, self.n_hidden_rnn))

    def forward(self, x_rnn, hx=None, cx=None):

        if hx is None and self.rnn_model == 'lstm':
            hx = self.init_hidden(x_rnn)
            cx = self.init_hidden(x_rnn)

        if self.rnn_model == 'lstm':
            ho, _ = self.rnn(x_rnn, (hx, cx))
        else:
            ho, _ = self.rnn(x_rnn, hx)
        y = self.out(ho)
        return y


class ContourGeneratorVRNN(torch.nn.Module):
    """
    This is a variational encoded RNN contour generator, should learn var as VAE.

    vae_as_input uses the latent space sample to generate a SOS input that
    initialises the hidden state for processing the real input. Like in:
    https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning

    RNN cells:
    Inputs: input, hidden
        - **input** (batch, input_size): tensor containing input features
        - **hidden** (batch, hidden_size): tensor containing the initial hidden
          state for each element in the batch.

    Outputs: h'
        - **h'** (batch, hidden_size): tensor containing the next hidden state
          for each element in the batch
    """
    def __init__(self, n_feature, n_hidden_rnn, n_output,
                 n_feature_vae, n_hidden_vae, n_latent,
                 rnn_model='rnn',
                 vae_as_input=False,
                 vae_as_input_hidd=False
                 ):

        super(ContourGeneratorVRNN, self).__init__()

        self.n_feature = n_feature
        self.n_hidden_rnn = n_hidden_rnn[0]
        self.layers = len(n_hidden_rnn)
        self.n_output = n_output
        self.n_feature_vae = n_feature_vae
        self.n_hidden_vae = n_hidden_vae
        self.n_latent = n_latent
        self.rnn_model = rnn_model
        self.vae_as_input = vae_as_input
        self.vae_as_input_hidd = vae_as_input_hidd

        # RNN part
        if rnn_model == 'rnn':
            self.rnn = torch.nn.RNN(n_feature, self.n_hidden_rnn,
                                    num_layers=self.layers, batch_first=False)
        elif rnn_model == 'gru':
            self.rnn = torch.nn.GRU(n_feature, self.n_hidden_rnn,
                                    num_layers=self.layers, batch_first=False)
        elif rnn_model == 'lstm':
            self.rnn = torch.nn.LSTM(n_feature, self.n_hidden_rnn,
                                     num_layers=self.layers, batch_first=False)
        self.out = torch.nn.Linear(self.n_hidden_rnn, n_output)

        # VAE part
        self.tanh = torch.tanh
        # encoder
        self.enc_hidden = torch.nn.Linear(n_feature_vae, n_hidden_vae)
        self.enc_mu = torch.nn.Linear(n_hidden_vae, n_latent)
        self.enc_logvar = torch.nn.Linear(n_hidden_vae, n_latent)
        # decoder
        if vae_as_input:
            self.dec_hid_to_input = torch.nn.Linear(n_latent, n_feature)
            if vae_as_input_hidd:
                self.dec_hidden = torch.nn.Linear(n_latent, self.n_hidden_rnn)
        else:
            self.dec_hidden = torch.nn.Linear(n_latent, self.n_hidden_rnn)

        self.reset_parameters()

    def reset_parameters(self):
        self.rnn.reset_parameters()
        self.out.reset_parameters()
        self.enc_hidden.reset_parameters()
        self.enc_mu.reset_parameters()
        self.enc_logvar.reset_parameters()
        if self.vae_as_input:
            self.dec_hid_to_input.reset_parameters()
            if self.vae_as_input_hidd:
                self.dec_hidden.reset_parameters()
        else:
            self.dec_hidden.reset_parameters()

    def encode(self, x):
        x = self.tanh(self.enc_hidden(x))
        mu = self.enc_mu(x)
        logvar = self.enc_logvar(x)
        return mu, logvar

    def sample(self, mu, logvar):
        std = torch.exp(.5 * logvar)
        eps = torch.zeros_like(logvar).normal_()
        return mu + eps*std

    def decode(self, z):
        return self.tanh(self.dec_hidden(z))  # maybe linear?

    def init_zeros(self, tensor):
        hx = tensor.new_zeros((1, self.n_hidden_rnn))
        return hx

    def forward(self, x_rnn, x_vae,
                hx=None, cx=None):
        if hx is None:  # then init it using the vae
            mu, logvar = self.encode(x_vae)

            if self.training:  # sample only on training
                z = self.sample(mu, logvar)
            else:
                z = mu

            if self.rnn_model == 'lstm':
                cx = self.init_zeros(x_rnn)  # pass atributes of x_rnn
                cx = cx.unsqueeze(dim=0)  # for the seq size

            if self.vae_as_input:
                x = self.dec_hid_to_input(z)
                x = x.unsqueeze(dim=0)  # for the seq size
                # if self.rnn_model == 'lstm':
                if self.vae_as_input_hidd:
                    hx = self.dec_hidden(z)
                else:
                    hx = self.init_zeros(x_rnn)   # pass atributes of x_rnn
                hx = hx.unsqueeze(dim=0)  # for the seq size
                _, (hx, _) = self.rnn(x, (hx, cx))
                # Outputs: output, (h_n, c_n)
                # else:
                #     if self.vae_as_input_hidd:
                #         hx = self.dec_hidden(z)
                #         hx = hx.unsqueeze(dim=0)  # if context is 2-dimensional this
                #     else:
                #         hx = None
                #     _, hx = self.rnn(x, hx)
                #     # Outputs: output, h_n
            else:  # vae as hidden
                hx = self.decode(z)
                hx = hx.unsqueeze(dim=0)  # if context is 2-dimensional this
                # will make it 3-dimensional
        else:
            z = None

        if self.rnn_model == 'lstm':
            ho, _ = self.rnn(x_rnn, (hx, cx))
        else:
            ho, _ = self.rnn(x_rnn, hx)
        y = self.out(ho)
        return y, z

def loss_mse(x, y):
    x = x.view(-1)
    y = y.view(-1)
    return torch.sum((x - y)**2) / x.shape[0]

def loss_kld(mu, logvar, mask):
    r"""
    adapted from https://github.com/pytorch/examples/tree/master/vae
    see Appendix B from VAE paper:
    Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    https://arxiv.org/abs/1312.6114
    :math:`0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)`

    mu = batch_size x n_latent x n_modules
    mask = batch_size x n_modules
    """
    mask = mask.unsqueeze(dim=1).repeat(1, mu.shape[1], 1)
    mu = mu.masked_select(mask == 1)
    logvar = logvar.masked_select(mask == 1)
    if (logvar > 10).any():
        logvar.exp_()
    kld = -0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar),
                           dim=0)
    kld = torch.mean(kld)  #
    return kld

def loss_mmd(zs, mask):
    """
    Maximum Mean Discrepancy loss measure based on InfoVAEs. Code based on
    author's implementation:
    http://szhao.me/2017/06/10/a-tutorial-on-mmd-variational-autoencoders.html

    zs : torch Variable be of dimensions (batch_size, n_latent, n_modules)
    mask : batch_size x n_modules
    """
    def compute_mmd(x, y):
        x_kernel = compute_kernel(x, x)
        y_kernel = compute_kernel(y, y)
        xy_kernel = compute_kernel(x, y)
        return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)

    def compute_kernel(x, y):
        x_size = x.shape[0]  # 200
        y_size = y.shape[0]  # 200
        dim = x.shape[1]  # 2
        tiled_x = x.unsqueeze(dim=1).repeat(1, y_size, 1)
        tiled_y = y.unsqueeze(dim=0).repeat(x_size, 1, 1)
        return torch.exp(-torch.mean((tiled_x - tiled_y)**2, dim=2)
                         / dim)

    mask = mask.unsqueeze(dim=1).repeat(1, zs.shape[1], 1)
    zs = zs.masked_select(mask == 1)  # this flattens it
    zs.unsqueeze_(dim=1)
    true_samples = torch.randn_like(zs)

    return compute_mmd(true_samples, zs)

def loss_mmd_rnn(zs):
    """
    Maximum Mean Discrepancy loss measure based on InfoVAEs. Code based on
    author's implementation:
    http://szhao.me/2017/06/10/a-tutorial-on-mmd-variational-autoencoders.html

    zs : torch Variable be of dimensions (batch_size, n_latent, n_modules)
    mask : batch_size x n_modules
    """
    def compute_mmd(x, y):
        x_kernel = compute_kernel(x, x)
        y_kernel = compute_kernel(y, y)
        xy_kernel = compute_kernel(x, y)
        return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)

    def compute_kernel(x, y):
        x_size = x.shape[0]  # 200
        y_size = y.shape[0]  # 200
        dim = x.shape[1]  # 2
        tiled_x = x.unsqueeze(dim=1).repeat(1, y_size, 1)
        tiled_y = y.unsqueeze(dim=0).repeat(x_size, 1, 1)

        return torch.exp(-torch.mean((tiled_x - tiled_y)**2, dim=2)
                         / dim)

    true_samples = torch.randn_like(zs)
    return compute_mmd(true_samples, zs)
