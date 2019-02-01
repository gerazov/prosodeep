#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ProsoDeep - Learning utils used for model training.

@authors:
    Branislav Gerazov Oct 2017

Copyright 2019 by GIPSA-lab, Grenoble INP, Grenoble, France.

See the file LICENSE for the licence associated with this software.
"""
import logging
import numpy as np
import torch
import os
import pickle
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
import copy
from prosodeep import prosodeep_models, prosodeep_corpus


def analysis_by_synthesis(corpus, mask_all_files, mask_file_dict, mask_contours,
                          n_units_dict, mask_unit_dict, contour_keys,
                          contour_generators, params):
    '''
    Runs prosodeep analysis by synthesis.

    Parameters
    ===========

    corpus : pandas data frame
        Holds all data from corpus.
    orig_columns : list of str
        The original prosodic parameters f0, f1, f2 and coeff_dur,
    target_columns : list of str
        The targets for training the contour generators.
    iterations : int
        Number of iterations of analysis-by-synthesis loop.
    mask_all_files : dataseries bool
        Mask for all good files in the corpus DataFrame.
    mask_file_dict : dict
        Dictionary with masks for each of the files in corpus.
    mask_contours : dict
        Dictinary with masks for each of the contours.
    n_units_dict : dict
        Dictionary with the number of units in each file in corpus.
    mask_unit_dict : dict
        Dictionary of masks for each unit number in corpus.
    contour_keys : list
        Types of functions covered by the contour generators.
    contour_generators : dict
        Dictionary of contour generators.
    '''
    # init
    log = logging.getLogger('an-by-syn')
    orig_columns = params.orig_columns
    target_columns = params.target_columns
    iterations = params.iterations
    use_strength = params.use_strength
    strength_method = params.strength_method
    context_columns = params.context_columns
    # validation stuff
    use_validation = params.use_validation
    validation_size = params.validation_size

    # do the an-by-syn iterations updating the targets
    losses = {key : np.empty(iterations-1) for key in contour_keys}
    losses_DC = np.asarray([])
    for i in range(iterations):
        log.info('='*42)
        log.info('Analysis-by-synthesis iteration {} ...'.format(i))
        log.info('='*42)
        pred_columns = [column + '_it{:03}'.format(i) for column in orig_columns]

        if i > 0:  # or params.use_pretrained_models:  # skip 0th prediction
            for contour_type in contour_keys:
                log.info('Training for contour type : {}'.format(contour_type))
                contour_generator = contour_generators[contour_type]
                mask_row = mask_all_files & mask_contours[contour_type]
                X = corpus.loc[mask_row, 'ramp1':'ramp4'].values.astype('float32')
                y = corpus.loc[mask_row, target_columns].values.astype('float32')

                X_all = copy.copy(X)  # for final prediction

                if use_validation:  # make a train-val split
                    # if random state is fixed it will generate the same
                    # validation set every an-by-syn loop iteration
                    train_ind, val_ind = prosodeep_corpus.split_corpus(
                            corpus.loc[mask_row], validation_size,
                            stratify=False, random_state=42)
                    X_val, y_val = X[val_ind], y[val_ind]
                    X, y = X[train_ind], y[train_ind]
                else:
                    X_val, y_val, strengths_val, contexts_val = [None] * 4

                if not use_strength:  # add inputs
                    contour_generator.fit(X, y,
                                          X_val=X_val, y_val=y_val)
                else:
                    if strength_method == 'manual':
                        strengths = corpus.loc[mask_row, 'strength'].values.astype('float32')
                        strengths_all = copy.copy(strengths)
                        if use_validation:
                            strengths_val = strengths[val_ind]
                            strengths = strengths[train_ind]

                        contour_generator.fit(X, y, strengths=strengths,
                                              X_val=X_val, y_val=y_val,
                                              strengths_val=strengths_val)
                    elif 'context' in strength_method:
                        # add context as input
                        contexts = corpus.loc[mask_row, context_columns].values.astype('float32')
                        contexts[contexts>0] = 1  # if multiple occurences are counted for
                        contexts_all = copy.copy(contexts)
                        if use_validation:
                            contexts_val = contexts[val_ind]
                            contexts = contexts[train_ind]
#                        X = np.c_[X, context]
                        # strengths are not valid as it's training
                        contour_generator.fit(X, y, contexts=contexts,
                                              X_val=X_val, y_val=y_val,
                                              contexts_val=contexts_val)

                losses[contour_type][i-1] = contour_generator.mse_
                if use_validation:
                    log.info('Best epoch: {}/{}, mean squared error: {}'.format(
                            contour_generator.best_epoch_,
                            contour_generator.max_iter_,
                            contour_generator.mse_))
                else:
                    log.info('mean squared error : {}'.format(contour_generator.mse_))
                if contour_type == 'DC':
                    losses_DC = np.r_[losses_DC, contour_generator.mses_]


                # run a forward pass on all the data to get the predictions
                if not use_strength:
                    y_pred = contour_generator.predict(X_all)
                else:
                    if strength_method == 'manual':
                        y_pred = contour_generator.predict(X_all, strengths=strengths_all)

                    elif 'context' in strength_method:
                        y_pred, strengths = contour_generator.predict(X_all,
                                                                      contexts=contexts_all)
                        corpus.loc[mask_row, 'strength'] = strengths

                corpus.loc[mask_row, pred_columns] = y_pred

        else:  # set initial predictions to zero
            log.info('Setting initial predictions to 0 ....')
            corpus.loc[mask_all_files, pred_columns] = 0

        #% sum the predictions for each unit, calculate the error and new targets
        log.info('Summing predictions, calculate the error and new targets ...')
        for file, mask_file in mask_file_dict.items():
            print('\rProcessing file {}'.format(file),end='')
            n_units = n_units_dict[file]
            for n_unit in range(n_units+1):
                mask_unit = mask_unit_dict[n_unit]
                mask_row = mask_file & mask_unit
                y_pred = corpus.loc[mask_row, pred_columns].values
                strengths = corpus.loc[mask_row, 'strength'].values

#                if not use_strength:  # plain sum
                y_pred_sum = np.sum(y_pred, axis=0)
#                else:  # weighted sum
#                    y_pred_sum = np.sum(y_pred * strengths[:,None], axis=0)  # should be normalised?

#               # comment this if you want to keep final targets
                if i == iterations-1:  # last iteration just output the sum of predictions
                    targets = y_pred_sum
                else:
                # up to here and then deindent:
                    n_contours = corpus[mask_row].shape[0]  # coeff to divide the error
                    y_orig = corpus.loc[mask_row, orig_columns].values  # every row should be the same
                    y_error = y_orig - y_pred_sum  # this will automatically tile row to matrix

                    # strength should influence error distribution
                    if not use_strength:  # distribute equally
                        targets = y_pred + y_error/n_contours
                    else:  # weighted distribution
                        norm = np.sum(strengths)  # to normalise the distribution
                        targets = y_pred + y_error * strengths[:,None] / norm
    #                        if file == 'DC_393.TextGrid':
    #                            raise
                # up to here
                corpus.loc[mask_row, target_columns] = targets  # write new targets

        print('\r',end='')  # delete line

    return corpus, contour_generators, losses, losses_DC


def construct_contour_generator(contour_type, params):
    '''
    Construct Neural Network based contour generator.

    Parameters
    ==========
    learn_rate : float
        Learning rate of NN optimizer.
    max_iter : int
        Maximum of training iterations.
    l2 : float
        L2 regulizer value.
    hidden_units : int
        Number of units in the hidden layer.

    '''
    log = logging.getLogger('contour_gen')
    learn_rate = params.learn_rate
    max_iter = params.max_iter
    adjust_max_iter = params.adjust_max_iter
    l2 = params.l2
    vowel_pts = params.vowel_pts
    n_output = vowel_pts + 1
    hidden_units = params.hidden_units
    use_strength = params.use_strength
    strength_method = params.strength_method
    early_thresh = params.early_thresh
    patience = params.patience
    activation = params.activation
    optimization = params.optimization
    # for the pytorch regressor early stopping criteria:

    if params.sreg_phrases_only and contour_type not in params.phrase_types:
        reg_strengths = 0
        reg_strengths_mean = 0
    else:
        reg_strengths = params.reg_strengths
        reg_strengths_mean = params.reg_strengths_mean
    batch_size = params.batch_size
    n_context = params.n_context
    n_hidden_context = params.n_hidden_context
    use_cuda = params.use_cuda
    if use_cuda and not torch.cuda.is_available():
        use_cuda = False
        log.info('GPU is sadly unavailable will use CPU instead ...')

    if params.model_type == 'anbysyn':
        contour_generator = prosodeep_models.PyTorchRegressor(
                n_feature=4,
                n_context=n_context,
                n_output=n_output,
                hidden_layer_sizes=hidden_units,
                n_hidden_context=n_hidden_context,
                activation=activation,
                batch_size=batch_size,  # auto batch_size=min(200, n_samples)
                max_iter=max_iter,  # is this 50 in the original??
                adjust_max_iter=adjust_max_iter,
                alpha=l2,
                shuffle=True,  # shuffle samples in each iteration
                random_state=42,
                verbose=True,
                warm_start=True,  # reuse previous fit
                early_stopping=False, validation_fraction=0.01,
                solver=optimization,
                learning_rate_init=learn_rate,
                use_strength=use_strength,
                strength_method=strength_method,
                reg_strengths=reg_strengths,
                reg_strengths_mean=reg_strengths_mean,
                use_cuda=use_cuda,
                early_thresh=early_thresh,
                patience=patience)

    return contour_generator


def train_model(corpus, contour_generators=None, params=None):
    '''
    Train deep model based on a big static graph with masks for local context.

    Parameters
    ===========

    corpus : pandas data frame
        Holds all data from corpus.
    orig_columns : list of str
        The original prosodic parameters f0, f1, f2 and coeff_dur,
    target_columns : list of str
        The targets for training the contour generators.
    iterations : int
        Number of iterations of analysis-by-synthesis loop.
    mask_all_files : dataseries bool
        Mask for all good files in the corpus DataFrame.
    mask_file_dict : dict
        Dictionary with masks for each of the files in corpus.
    mask_contours : dict
        Dictinary with masks for each of the contours.
    n_units_dict : dict
        Dictionary with the number of units in each file in corpus.
    mask_unit_dict : dict
        Dictionary of masks for each unit number in corpus.
    contour_keys : list
        Types of functions covered by the contour generators.
    contour_generators : dict
        Dictionary of contour generators.
    '''
    # init
    log = logging.getLogger('deep-stat-train')
    orig_columns = params.orig_columns
    iterations = params.iterations
    batch_size = params.batch_size
    use_strength = params.use_strength
    strength_method = params.strength_method
    target_columns = params.target_columns
    contour_types = params.phrase_types + params.function_types
    n_context = params.n_context
    n_hidden_context = params.n_hidden_context
    hidden_units = params.hidden_units
    vowel_pts = params.vowel_pts
    n_output = vowel_pts + 1
    learn_rate = params.learn_rate
    l2 = params.l2
    use_cuda = params.use_cuda
    reg_strengths = params.reg_strengths
    reg_strengths_mean = params.reg_strengths_mean
    optimization = params.optimization
    early_stopping = params.early_stopping
    early_thresh = params.early_thresh
    patience = params.patience
    vae = params.vae
    vae_input = params.vae_input
    n_latent = params.n_latent
    reg_vae = params.reg_vae
    use_validation = params.use_validation
    validation_size = params.validation_size
    model_type = params.model_type
    activation = params.activation
    drop_rate = params.drop_rate

    # transform corpus to have one row per (filename, n_unit) combination
    # check for transformed corpus
    pklname = params.pkl_path + params.corpus_name
    if not params.do_all_phrases:
        pklname += '_allphraseFalse'
    pklname += '_static_reformated'
    if vae:
        pklname += '_vae'
    if params.normalisation_type:
        pklname += '_norm'
    pklname += '.pkl'

    if params.load_corpus and os.path.isfile(pklname):
        logging.info('Loading reformatted corpus ...')
        with open(pklname, 'rb') as f:
            data = pickle.load(f)
    else:
        log.info('Reformating corpus ...')
        data = prosodeep_corpus.reformat_corpus_static(corpus, params)
        #% save reformatted data:
        with open(pklname, 'wb') as f:
            pickle.dump(data, f, -1)

    X, masks, y, contours_in_graph, unique_combinations = data

    # reshape data for baseline
    # X shape is (n_samples x n_input x n_modules_in_graph)
    # where n_input is [strength + n_syll_ramps + n_context_columns]
    # we need: (n_samples x n_feats)
    # where n_feats = (n_syll_ramps + n_context_columns)* n_modules_in_graph
    # since every function can have a different context
    if 'baseline' in model_type:
        if 'context' in model_type:
            X = X[:, 1:, :]
        else:
            X = X[:, 1:5, :]  # just the ramps
        X = np.reshape(X, (X.shape[0], -1))

    # now turn all the nans into something improbable (0s?)
    X[np.isnan(X)] = 0

    # init model
    log.info('Initialising model ...')
    if 'baseline' in model_type:
        n_feature = X.shape[-1]  # 156 for chen
        model = prosodeep_models.deep_baseline_model(
                batch_size=batch_size,
                n_feature=n_feature,
                n_hidden=hidden_units,
                activation=activation,
                n_output=n_output,
                max_iter=iterations,  # this is now iteration
                weight_decay=l2,
                shuffle=True,
                random_state=42,
                verbose=True,
                solver=optimization,
                learning_rate_init=learn_rate,
                use_cuda=use_cuda,
                drop_rate=drop_rate,
                )
    else:
        n_feature = 4
        model = prosodeep_models.deep_model(
                contour_types=contour_types,  # which nncgs to init
                contour_generators=contour_generators,
                batch_size=batch_size,
                n_feature=n_feature,
                n_hidden=hidden_units,
                activation=activation,
                n_output=n_output,
                n_context=n_context,
                n_hidden_context=n_hidden_context,
                max_iter=iterations,  # this is now iteration
                weight_decay=l2,  # L2 penalty 1e-4 default - config says should be 0.1??
                shuffle=True,  # shuffle samples in each iteration
                random_state=42,
                verbose=True,
                solver=optimization,  # adam is newer, I don't think you can use rprop
                learning_rate_init=learn_rate,  # default 0.001, in Config it's 0.1
                use_strength=use_strength,  # to include strength in training
                strength_method=strength_method,  # how to determine strength
                reg_strengths=reg_strengths,
                reg_strengths_mean=reg_strengths_mean,
                use_cuda=use_cuda,
                contours_in_graph=contours_in_graph,
                vae=vae,
                vae_input=vae_input,
                n_latent=n_latent,
                reg_vae=reg_vae
                )

    # do the epochs
    log.info('Training model ...')
    batch_losses = np.asarray([])
    batch_mses = np.asarray([])
    epoch_losses = np.asarray([])
    epoch_mses = np.asarray([])
    epoch_cnt = np.asarray([])
    log.info('='*42)
    # TODO change iterations to iterations+1
    if optimization == 'sgdr':
        T_max = model.T_max  # for the sgdr scheduler
        last_check = 0
    best_error = np.inf

    # train-validation split
    X_all, masks_all = X.copy(), masks.copy()  # for final pred
    if use_validation:  # make a train-val split
        files = [key[0] for key in unique_combinations.keys()]
        gss = GroupShuffleSplit(test_size=validation_size,
                                random_state=42)
        train_ind, val_ind = gss.split(files, files, files).__next__()
        X_val, masks_val, y_val = X[val_ind], masks[val_ind], y[val_ind]
        X, masks, y = X[train_ind], masks[train_ind], y[train_ind]
    else:
        X_val, masks_val, y_val = [None] * 3

    for iteration in range(0, iterations):
        log.info('Epoch {}/{} ...'.format(iteration+1, iterations))

        if optimization == 'sgdr':
            if iteration - last_check == T_max:  # double it
                T_max = 2 * T_max
                model.T_max = T_max
                model.reset_optimizer()
                last_check = iteration
            else:
                model.scheduler.step()

            print('learning rate {:.6f}'.format(
                    model.optimizer.param_groups[0]['lr']))  # all have the same one

        if 'baseline' in model_type:
            model.fit(X, y,
                      X_val, y_val)  # this only does one epoch
        else:
            model.fit(X, masks, y,
                      X_val, masks_val, y_val)  # this only does one epoch
            # X.shape = n_samples x n_input x n_modules_in_graph
            # n_input = strength + n_syll_ramps + n_context_columns

        # accumulate losses for epoch
        mse_epoch = model.mses_.mean()
        loss_epoch = model.losses_.mean()
        if params.vae and 'mmd' in params.vae_input:
            mmd = model.mmds_.mean()
        batch_mses = np.r_[batch_mses, model.mses_]
        batch_losses = np.r_[batch_losses, model.losses_]
        epoch_mses = np.r_[epoch_mses, mse_epoch]
        epoch_losses = np.r_[epoch_losses, loss_epoch]
        if iteration == 0:
            epoch_cnt = np.array([model.mses_.size])
        else:
            epoch_cnt = np.r_[epoch_cnt, epoch_cnt[-1] + model.mses_.size]

        if params.vae and 'mmd' in params.vae_input:
            log.info('mse: {:.5f} \t mmd: {:.5f} \t loss: {:.5f}'.format(
                   mse_epoch, mmd, loss_epoch))
        else:
            log.info('mse: {:.5f} \t loss: {:.5f}'.format(
                   mse_epoch, loss_epoch))

        if early_stopping:  # check if error is not improving
            if best_error - loss_epoch > early_thresh:
                patience = params.patience
            else:
                patience -= 1
                log.info('loosing patience: {}'.format(patience))

        if best_error > loss_epoch:  # update best error
            best_error = loss_epoch
            # run a forward pass on all the data to get the predictions
            if use_strength and 'context' in strength_method:
                y_pred, strengths = model.predict(X_all, masks_all)
            else:
                if 'baseline' in model_type:
                    y_pred = model.predict(X_all)
                else:
                    y_pred = model.predict(X_all, masks_all)
            # presave model
            if use_strength and 'context' in strength_method:
                best_output = y_pred, strengths
            else:
                best_output = y_pred
            best_epoch = iteration
            best_losses = (batch_losses, epoch_cnt, epoch_losses,
                           batch_mses, epoch_mses)
            if 'baseline' in model_type:
                best_model = model.model.state_dict()
            else:
                best_model = model.static_graph.state_dict()
            best_optim = model.optimizer.state_dict()  # for cont training
            best_return = best_model, best_optim, best_losses

        # stop if criteria are met
        if (iteration == iterations - 1) or not patience:
            # if last iteration or we've run out of patience
            if use_strength and 'context' in strength_method:
                y_pred, strengths = best_output  # restore best results
            else:
                y_pred = best_output
            iteration = best_epoch
            if early_stopping:
                log.info('Best error at epoch {} : mse {:.5f} \t loss {:.5f}'.format(
                       best_epoch, best_losses[4][-1], best_losses[2][-1]))

            log.info('Writing predictions in corpus ...')
            pred_columns = [column + '_it{:03}'.format(iteration)
                            for column in orig_columns]
            columns = corpus.columns.tolist()
            columns[-vowel_pts-1:] = pred_columns
            corpus.columns = columns
            for ind_y, indexes in enumerate(unique_combinations.values()):
                corpus.loc[indexes, pred_columns] = y_pred[ind_y, :]
                # also write final predictions in target columns
                corpus.loc[indexes, target_columns] = y_pred[ind_y, :]

                if use_strength and 'context' in strength_method:
                    # strengths are output for every contour in the combination
                    # we can mask the strengths of the contours that don't appear
                    # in each combination using X where there are 1s only for
                    # those that appear and 0s otherwise
                    strengths_mask_row = X_all[ind_y, 0, :]  # 0 is the input strength
                    strengths_mask_row = strengths_mask_row.astype(bool)
                    strengths_in_row = strengths[ind_y, strengths_mask_row]
                    contours_in_row = np.array(contours_in_graph)[strengths_mask_row]  # for sanity check

                    strengths_in_row = strengths_in_row.tolist()
                    contours_in_row = contours_in_row.tolist()
                    for index in indexes:
                        for strength, contourtype in zip(strengths_in_row, contours_in_row):
                            if corpus.loc[index, 'contourtype'] == contourtype[:2]:
                                corpus.loc[index, 'strength'] = strength
                                # now remove it for future loops - this should take
                                # care of the multiple identical contourtypes in
                                # contours_in_graph
                                strengths_in_row.remove(strength)
                                contours_in_row.remove(contourtype)
                    # check if all were taken care of
                    assert not strengths_in_row
                    assert not contours_in_row
            break  # for the patience to work

    # return
    best_model, best_optim, best_losses = best_return
    if 'baseline' in model_type:
        model.model.load_state_dict(best_model)
    else:
        model.static_graph.load_state_dict(best_model)
    model.optimizer.load_state_dict(best_optim)
    batch_losses, epoch_cnt, epoch_losses, batch_mses, epoch_mses = best_losses
    if use_cuda:
        if 'baseline' in model_type:
            model.model.cpu()
        else:
            model.static_graph.cpu()
        model.criterion.cpu()
    return corpus, model, batch_losses, epoch_cnt, epoch_losses


def train_rnn_model(corpus, contour_generators=None, params=None):
    '''
    Train deep RNN model based on a big static graph with masks for local
    context.

    Parameters
    ===========

    corpus : pandas data frame
        Holds all data from corpus.
    contour_generators : dict
        Dictionary of contour generators.
    '''
    # init
    log = logging.getLogger('deep-rnn-train')
    orig_columns = params.orig_columns
#    target_columns = params.target_columns
    epochs = params.iterations
    batch_size = params.batch_size
    target_columns = params.target_columns
    contour_types = params.phrase_types + params.function_types
    n_context = params.n_context
    n_hidden = params.hidden_units
    n_hidden_rnn = params.n_hidden_rnn
    vowel_pts = params.vowel_pts
    n_output = vowel_pts + 1
    learn_rate = params.learn_rate
    l2 = params.l2
    use_cuda = params.use_cuda
    optimization = params.optimization
    # early stopping:
    early_stopping = params.early_stopping
    early_thresh = params.early_thresh
    patience = params.patience
    # vae
    vae = params.vae
    n_hidden_vae = params.n_hidden_vae[0]
    n_latent = params.n_latent
    reg_vae = params.reg_vae
    # rnn
    rnn_model = params.rnn_model
    vae_as_input = params.vae_as_input
    vae_as_input_hidd = params.vae_as_input_hidd

#    use_test_set = params.use_test_set
    use_validation = params.use_validation
    validation_size = params.validation_size

    use_test_set = params.use_test_set

    model_type = params.model_type
    if 'unified' in model_type:
        unified = True
        if 'atts' in model_type:
            unify_atts = True
            phrase_types = params.phrase_types
            syntax_functions = params.syntax_functions
            morpheme_functions = params.morpheme_functions
        else:
            unify_atts = False
            phrase_types = None
            syntax_functions = None
            morpheme_functions = None
    else:
        unified = False
        unify_atts = False
        phrase_types = None
        syntax_functions = None
        morpheme_functions = None
    activation = params.activation

    # check for transformed corpus
    pklname = params.pkl_path + params.corpus_name
    if not params.do_all_phrases:
        pklname += '_allphraseFalse'
    pklname += '_rnn_reformated'
    if vae:
        pklname += '_vae'
    if params.normalisation_type:
        pklname += '_norm'
    if not use_test_set:
        pklname += '_notest'
    pklname += '.pkl'

#    pklname = params.pkl_path + 'yanpin_v7_5pts_rightscope_static_reformated.pkl'
    if params.load_corpus and os.path.isfile(pklname):
        logging.info('Loading reformatted corpus ' + pklname)
        with open(pklname, 'rb') as f:
            data = pickle.load(f)

#% transform corpus to have one row per (filename, n_unit) combination
    else:
        log.info(f'No {pklname} found, reformating corpus ...')
        data = prosodeep_corpus.reformat_corpus_rnn(corpus, params)
        #% save reformatted data:
        with open(pklname, 'wb') as f:
            pickle.dump(data, f, -1)

    X, cg_masks, contour_starts, y, contours_in_graph, unique_combinations = data
    # X shape is (n_files, seq_len, n_input, n_modules_in_graph)
    # debug
    # files = [key[0] for key in unique_combinations.keys()]

    # reshape data for baseline
    # where n_input is [strength + n_syll_ramps + n_context_columns]
    # we need: (n_files, seq_len, n_feats)
    # where n_feats = (n_syll_ramps + n_context_columns) * n_modules_in_graph
    # or n_feats = n_syll_ramps * n_modules_in_graph
    # since every function can have a different context
    if 'baseline' in model_type:
        # if 'context' in model_type:
        #     X = X[:, :, 1:, :]
        # else:
        X = X[:, :, 1:5, :]  # just the ramps - context is implicitly modelled
        X = np.reshape(X, (X.shape[0], X.shape[1], -1))

    # now turn all the nans into something improbable (0s?)
    X[np.isnan(X)] = 0

    # init model
    log.info('Initialising model ...')
    if 'baseline' in model_type:
        n_feature = X.shape[-1]  # 156 for chen
        model = prosodeep_models.deep_rnn_baseline_model(
                batch_size=batch_size,
                n_feature=n_feature,
                n_hidden=n_hidden,
                n_hidden_rnn=n_hidden_rnn,
                activation=activation,  # for the DNN part, the RNN is tanh
                n_output=n_output,
                weight_decay=l2,
                shuffle=True,
                random_state=42,
                verbose=True,
                optimizer=optimization,
                learning_rate_init=learn_rate,
                use_cuda=use_cuda,
                )
    else:
        if params.rnn_inputs is None:
            n_feature = 4
        else:
            n_feature = len(params.rnn_inputs)
        model = prosodeep_models.deep_rnn_model(
                contour_types=contour_types,  # which nncgs to init
                contour_generators=contour_generators,
                contours_in_graph=contours_in_graph,
                n_feature=n_feature,
                rnn_inputs=params.rnn_inputs,
                n_hidden=n_hidden_rnn,  # there's no deep level beneath
                rnn_model=rnn_model,
                activation='tanh',
                n_output=n_output,
                batch_size=batch_size,
                weight_decay=l2,  # L2 penalty 1e-4 default - config says should be 0.1??
                shuffle=True,  # shuffle samples in each iteration
                random_state=42,
                verbose=True,
                optimizer=optimization,  # adam is newer, I don't think you can use rprop
                learning_rate_init=learn_rate,  # default 0.001, in Config it's 0.1
                use_cuda=use_cuda,
                vae=vae,
                n_feature_vae=n_context,
                n_hidden_vae=n_hidden_vae,
                n_latent=n_latent,
                reg_vae=reg_vae,
                vae_as_input=vae_as_input,
                vae_as_input_hidd=vae_as_input_hidd,
                unified=unified,
                unify_atts=unify_atts,
                phrase_types=phrase_types,
                syntax_functions=syntax_functions,
                morpheme_functions=morpheme_functions
                )

    # do the epochs
    log.info('Training model ...')
    batch_losses = np.asarray([])
    batch_mses = np.asarray([])
    epoch_losses = np.asarray([])
    epoch_mses = np.asarray([])
    epoch_cnt = np.asarray([])
    log.info('='*42)

    best_error = np.inf

    # train-validation split
    X_all, cg_masks_all, contour_starts_all = (X.copy(), cg_masks.copy(),
                                               contour_starts.copy()) # for final pred
    if use_validation:  # make a train-val split
#        gss = GroupShuffleSplit(test_size=validation_size,
#                                random_state=42)
#        train_ind, val_ind = gss.split(files, files, files).__next__()
        ss = ShuffleSplit(n_splits=1,
                          test_size=validation_size,
                          random_state=42)
        train_ind, val_ind = next(ss.split(range(X.shape[0])))
        X_val, y_val = X[val_ind], y[val_ind]
        cg_masks_val, contour_starts_val = cg_masks[val_ind], contour_starts[val_ind]
        X, y = X[train_ind], y[train_ind]
        cg_masks, contour_starts = cg_masks[train_ind], contour_starts[train_ind]
    else:
        X_val, cg_masks_val, contour_starts_val, y_val = [None] * 4

    for epoch in range(0, epochs):  # no 0th iteration now
        #%
#        i = 0
        log.info('Epoch {}/{} ...'.format(epoch+1, epochs))

        # do one epoch
        if 'baseline' in model_type:
            model.fit(X, y, X_val, y_val)
        else:
            model.fit(X, cg_masks, contour_starts, y,
                      X_val, cg_masks_val, contour_starts_val, y_val)
        # X.shape = n_files x n_samples x n_input x n_modules_in_graph
        # n_input = strength + n_syll_ramps + n_context_columns

        # accumulate losses for epoch
        mse_epoch = model.mses_.mean()
        loss_epoch = model.losses_.mean()
        if vae:
            mmd = model.mmds_.mean()
        batch_mses = np.r_[batch_mses, model.mses_]
        batch_losses = np.r_[batch_losses, model.losses_]
        epoch_mses = np.r_[epoch_mses, mse_epoch]
        epoch_losses = np.r_[epoch_losses, loss_epoch]
        if epoch == 0:
            epoch_cnt = np.array([model.mses_.size])
        else:
            epoch_cnt = np.r_[epoch_cnt, epoch_cnt[-1] + model.mses_.size]

        if params.vae:
            log.info('mse: {:.5f} \t mmd: {:.5f} \t loss: {:.5f}'.format(
                   mse_epoch, mmd, loss_epoch))
        else:
            log.info('mse loss: {:.5f}'.format(
                   mse_epoch))

        if np.isnan(loss_epoch):  # we have a nan - why??
            log.warning('Loss is NaN! Breaking ...')
            patience = 0

        else:

            if early_stopping:  # check if error is not improving
                if best_error - loss_epoch > early_thresh:
                    patience = params.patience
                else:
                    patience -= 1
                    log.info('loosing patience: {}'.format(patience))

            if best_error > loss_epoch:  # update best error
                best_error = loss_epoch
                # presave model
                # run a forward pass on all the data to get the predictions
                if 'baseline' in model_type:
                    best_output = model.predict(X_all)
                else:
                    best_output = model.predict(X_all,
                                                cg_masks_all,
                                                contour_starts_all)
                best_epoch = epoch
                best_losses = (batch_losses, epoch_cnt, epoch_losses,
                               batch_mses, epoch_mses)
                if 'baseline' in model_type:
                    best_model = [model.model.state_dict(),
                                  model.model_rnn.state_dict()]
                else:
                    best_model = model.rnn_graph.state_dict()
                best_optim = model.optimizer.state_dict()  # for cont training
                best_return = best_model, best_optim, best_losses

        # stop if criteria is met
        if (epoch == epochs - 1) or not patience:
            log.info('Ending training ...')
            if early_stopping:
                log.info('Best error at epoch {} : mse {:.5f} \t loss {:.5f}'.format(
                       best_epoch, best_losses[4][-1], best_losses[2][-1]))
            # if last epoch or we've run out of patience
            y_pred = best_output

#            if not patience:
            epoch = best_epoch

            log.info('Writing predictions in corpus ...')
            pred_columns = [column + '_it{:03}'.format(epoch) for column in orig_columns]
            columns = corpus.columns.tolist()
            columns[-vowel_pts-1:] = pred_columns
            corpus.columns = columns
            files = corpus.file.unique()
            for f_ind, file in enumerate(files):
                mask_file = corpus.file == file
                n_units = corpus.loc[mask_file].n_unit.unique()
                for u_ind, n_unit in enumerate(n_units):
                    indexes = unique_combinations[file, n_unit]
                    corpus.loc[indexes, pred_columns] = y_pred[f_ind, u_ind, :]
                    # also write final predictions in target columns
                    corpus.loc[indexes, target_columns] = y_pred[f_ind, u_ind, :]

            break  # for the patience to work

    # return
#    if early_stopping:
    best_model, best_optim, best_losses = best_return
    if 'baseline' in model_type:
        model.model.load_state_dict(best_model[0])
        model.model_rnn.load_state_dict(best_model[1])
    else:
        model.rnn_graph.load_state_dict(best_model)
    model.optimizer.load_state_dict(best_optim)
    batch_losses, epoch_cnt, epoch_losses, batch_mses, epoch_mses = best_losses
    if model.use_cuda:
        if 'baseline' in model_type:
            model.model.cpu()
            model.model_rnn.cpu()
        else:
            model.rnn_graph.cpu()
        model.criterion.cpu()
    return corpus, model, batch_losses, epoch_cnt, epoch_losses

def synthesise_deep_contours(corpus, model, filename, params):
    '''
    synthesise individual contours for a file and put them in corpus.
    '''
#    filename = 'yanpin_000001.TextGrid'
    # filename = 'DC_328.TextGrid'
    log = logging.getLogger('deep-synth')
    log.info('Synthesizing contours for file {} ...'.format(filename))
#    target_columns = params.target_columns
    pred_columns = corpus.columns[-params.vowel_pts-1:]  # write here the synthesis
#    corpus.loc[:, pred_columns] = np.nan
    print()
    for ind in corpus[corpus.file == filename].index:
        # ind = 3401  # DC start
        # ind = 3409  # DG start
        print('\rSynthesizing row : {}/{}'.format(
                ind, corpus[corpus.file == filename].index.size), end='')
        X = corpus.loc[ind, 'ramp1':'ramp4'].values.astype('float32')
        X = torch.tensor(X, dtype=torch.float32).unsqueeze_(0)

        if params.use_strength or (params.vae and 'context' in params.vae_input):
            contexts = corpus.loc[ind, params.context_columns].values.astype('float32')
    #                y_pred, strengths = model.predict(X, contexts=contexts,
    #                                                  contours_in_pass=contours_in_pass)
            contexts = torch.tensor(contexts, dtype=torch.float32).unsqueeze_(0)

        contour_type = corpus.loc[ind, 'contourtype']
#        contours_in_pass = [contour_type]
        contour_generator = model.contour_generators[contour_type].cpu()
        if params.vae:
            if 'context' in params.vae_input:
                X = torch.cat((X, contexts), dim=1)
            y_pred, zs, mus, sigmas = contour_generator(X)

        elif params.use_strength:
            if params.strength_method == 'manual':
                strengths = np.array(corpus.loc[ind, 'strength']).astype('float32')
#                y_pred = model.predict(X, strengths=strengths,
#                                       contours_in_pass=contours_in_pass)
                strengths = torch.tensor(strengths, dtype=torch.float32).unsqueeze_(0)
                y_pred = contour_generator(X, strengths=strengths)

            elif 'context' in params.strength_method:
                y_pred, strengths = contour_generator(X, contexts=contexts)
        else:
#            y_pred = model.predict(X, contours_in_pass=contours_in_pass)
            y_pred = contour_generator(X)

        corpus.loc[ind, pred_columns] = y_pred.data.numpy().ravel()  # make 1D
        print('\r', end='')

    return corpus

def synthesise_rnn_contours(corpus, model, filename, params):
    '''
    synthesise individual contours for a file using RNN and put them in corpus.
    '''
#    filename = 'yanpin_000001.TextGrid'
#    filename = 'DC_328.TextGrid'
    log = logging.getLogger('rnn-synth')
    log.info('Synthesizing contours for file {} ...'.format(filename))
#    target_columns = params.target_columns
    pred_columns = corpus.columns[-params.vowel_pts-1:]  # write here the synthesis
#    corpus.loc[:, pred_columns] = np.nan
    mask_file = corpus.file == filename
    marks = corpus[mask_file].marks
    start_inds = marks.index[marks.str.contains('start')].values
    end_inds = marks.index[marks.str.contains('end')].values
    for c_i, (start, end) in enumerate(zip(start_inds, end_inds)):
        contour = corpus.loc[start, 'contourtype']
#        print('\rSynthesizing contour {}  {}/{}'.format(
#                contour, c_i, len(start_inds), end=''))

        module = model.contour_generators[contour].cpu()

        h_module = None  # init
        if module.rnn_model == 'lstm':
            c_module = None


        x = corpus.loc[start : end, 'ramp1':'ramp4'].values.astype('float32')
        x = torch.tensor(x, dtype=torch.float32).unsqueeze_(1)  # for the batch size
        if params.rnn_inputs is not None:
            x = x[:,:,params.rnn_inputs]
            x.unsqueeze_(2)
        if params.vae:
            contexts = corpus.loc[start, params.context_columns].values.astype('float32')
            contexts = torch.tensor(contexts, dtype=torch.float32).unsqueeze_(dim=0)
#                contexts = Variable(torch.FloatTensor(contexts))
            if module.rnn_model == 'lstm':
                y, _ = module(x, contexts, h_module, c_module)
            else:
                y, _ = module(x, contexts, h_module)
        else:
            if module.rnn_model == 'lstm':
                y = module(x, h_module, c_module)
            else:
                y = module(x, h_module)
        y.squeeze_()
        y_pred = y.data.numpy()
#            assert y_pred.shape[0] == end - start + 1
#            assert y_pred.shape[1] == 6
        corpus.loc[start:end, pred_columns] = y_pred  # make 1D

    print('\r', end='')

    return corpus

def synthesise_deep_testset(corpus, model, params):
    '''
    synthesise sum contours for test set for evaluation. presaves reformatted
    data but only works for fixed random seed!
    '''
    log = logging.getLogger('test-deepsynth')
    target_columns = params.target_columns
    pred_columns = corpus.columns[-params.vowel_pts-1:]  # write here the synthesis
    model_type = params.model_type
    # check for transformed corpus
    pklname = params.pkl_path + params.corpus_name
    if not params.do_all_phrases:
        pklname += '_allphraseFalse'
    pklname += '_static_reformated'
    if params.vae:
        pklname += '_vae'
    if params.normalisation_type:
        pklname += '_norm'

    if 'baseline' in model_type:
        # reshape data for baseline - but keep same contour ordering as for
        # training!!!
        # X shape is (n_samples x n_input x n_modules_in_graph)
        # where n_input is [strength + n_syll_ramps + n_context_columns]
        # we need: (n_samples x n_feats)
        # where n_feats = (n_syll_ramps + n_context_columns)* n_modules_in_graph
        # since every function can have a different context
        pklname_test = pklname + '_testset.pkl'
        pklname_test = pklname_test.replace('static','baseline')
        if params.load_corpus and os.path.isfile(pklname_test):
            logging.info('Loading reformatted corpus ...')
            with open(pklname_test, 'rb') as f:
                data = pickle.load(f)
            X, masks, y, contours_in_graph, unique_combinations = data
        else:
            pklname_train = pklname + '.pkl'
            # load train set and get contours to order data for test set
            with open(pklname_train, 'rb') as f:
                data = pickle.load(f)
            _, _, _, contours_in_graph, _ = data
            log.info('Reformating corpus ...')
            data = prosodeep_corpus.reformat_corpus_static(
                    corpus, params,
                    contours_in_graph=contours_in_graph)
            X, masks, y, contours_in_graph, unique_combinations = data
            #% save reformatted data:
            with open(pklname_test, 'wb') as f:
                pickle.dump(data, f, -1)

        if 'context' in model_type:
            X = X[:, 1:, :]
        else:
            X = X[:, 1:5, :]  # just the ramps
        X = np.reshape(X, (X.shape[0], -1))
        X[np.isnan(X)] = 0

    else:  # deep static models
        pklname += '_testset.pkl'
        if params.load_corpus and os.path.isfile(pklname):
            logging.info('Loading reformatted corpus ...')
            with open(pklname, 'rb') as f:
                data = pickle.load(f)
        else:
            log.info('Reformating test corpus ...')
            data = prosodeep_corpus.reformat_corpus_static(corpus, params)
            #% save reformatted data:
            with open(pklname, 'wb') as f:
                pickle.dump(data, f, -1)

        X, masks, y, contours_in_graph, unique_combinations = data
        contour_types = set([x[:2] for x in contours_in_graph])
        X[np.isnan(X)] = 0

#%% init model
    if 'baseline' not in model_type:
        model_test = prosodeep_models.deep_model(
            contours_in_graph=contours_in_graph,
            contour_generators=model.contour_generators,
            # contour_types=model.contour_types,  # prev it didn't remember them
            contour_types=contour_types,
            batch_size=model.batch_size,
            n_feature=model.n_feature,
            n_hidden=model.n_hidden,
            activation=model.activation,
            n_output=model.n_output,
            n_context=model.n_context,
            n_hidden_context=model.n_hidden_context,
            max_iter=model.max_iter,
            weight_decay=model.weight_decay,
            shuffle=False,
            random_state=model.random_state,
            verbose=False,
            solver=model.optimizer_type,
            learning_rate_init=model.learning_rate,
            use_strength=model.use_strength,
            strength_method=model.strength_method,
            reg_strengths=model.reg_strengths,
            reg_strengths_mean=model.reg_strengths_mean,
            use_cuda=model.use_cuda,
            vae=model.vae,
            vae_input=model.vae_input,
            n_latent=model.n_latent,
            reg_vae=model.reg_vae
            )
    #% run forward pass
    log.info('Running forward pass ...')
    # run a forward pass on all the data to get the predictions
    if params.use_strength and 'context' in params.strength_method:
        y_pred, strengths = model_test.predict(X, masks)
    elif 'baseline' in model_type:
        model.model.cuda()  # because models are sent to cpu after training
        y_pred = model.predict(X)
        model.model.cpu()
    else:
        y_pred = model_test.predict(X, masks)
    #% write predictions in corpus
    log.info('Writing predictions in corpus ...')
    for ind_y, indexes in enumerate(unique_combinations.values()):
        corpus.loc[indexes, pred_columns] = y_pred[ind_y, :]
        # also write final predictions in target columns
        corpus.loc[indexes, target_columns] = y_pred[ind_y, :]

        if params.use_strength and 'context' in params.strength_method:
            # strengths are output for every contour in the combination
            # we can mask the strengths of the contours that don't appear
            # in each combination using X where there are 1s only for
            # those that appear and 0s otherwise
            strengths_mask_row = X[ind_y, 0, :]  # 0 is the input strength
            strengths_mask_row = strengths_mask_row.astype(bool)
            strengths_in_row = strengths[ind_y, strengths_mask_row]
            contours_in_row = np.array(contours_in_graph)[strengths_mask_row]  # for sanity check
            strengths_in_row = strengths_in_row.tolist()
            contours_in_row = contours_in_row.tolist()
            for index in indexes:
                for strength, contourtype in zip(strengths_in_row, contours_in_row):
                    if corpus.loc[index, 'contourtype'] == contourtype[:2]:
                        corpus.loc[index, 'strength'] = strength
                        # now remove it for future loops - this should take
                        # care of the multiple identical contourtypes in
                        # contours_in_graph
                        strengths_in_row.remove(strength)
                        contours_in_row.remove(contourtype)

    return corpus


def synthesise_rnn_testset(corpus, model, params):
    '''
    synthesise sum contours for test set for evaluation. presaves reformatted
    data but only works for fixed random seed!
    '''
#    filename = 'yanpin_000001.TextGrid'
#    filename = 'DC_328.TextGrid'
    log = logging.getLogger('test-deepsynth')
    target_columns = params.target_columns
    model_type = params.model_type
    pred_columns = corpus.columns[-params.vowel_pts-1:]  # write here the synthesis

    #% check for transformed corpus
    pklname = params.pkl_path + params.corpus_name
    if not params.do_all_phrases:
        pklname += '_allphraseFalse'
    pklname += '_rnn_reformated'
    if params.vae:
        pklname += '_vae'
    if params.normalisation_type:
        pklname += '_norm'
    if 'baseline' in model_type:
        # reshape data for baseline - but keep same contour ordering as for
        # training!!!
        # X shape is (n_samples x len_seq x n_input x n_modules_in_graph)
        # where n_input is [strength + n_syll_ramps + n_context_columns]
        # we need: (n_samples x n_feats)
        # where n_feats = (n_syll_ramps + n_context_columns)* n_modules_in_graph
        # since every function can have a different context
        pklname_test = pklname + '_testset.pkl'
        pklname_test = pklname_test.replace('_rnn','_baseline_rnn')
        if params.load_corpus and os.path.isfile(pklname_test):
            logging.info('Loading reformatted corpus ...')
            with open(pklname_test, 'rb') as f:
                data = pickle.load(f)
            (X, cg_masks, contour_starts,
             y, contours_in_graph, unique_combinations) = data
        else:
            pklname_train = pklname + '.pkl'
            # load train set and get contours to order data for test set
            with open(pklname_train, 'rb') as f:
                data = pickle.load(f)
            contours_in_graph = data[4]
            log.info('Reformating corpus ...')
            data = prosodeep_corpus.reformat_corpus_rnn(
                    corpus, params,
                    contours_in_graph=contours_in_graph)
            (X, cg_masks, contour_starts,
             y, contours_in_graph, unique_combinations) = data
            #% save reformatted data:
            with open(pklname_test, 'wb') as f:
                pickle.dump(data, f, -1)

        if 'context' in model_type:
            X = X[:, :, 1:, :]
        else:
            X = X[:, :, 1:5, :]  # just the ramps
        X = np.reshape(X, (X.shape[0], X.shape[1], -1))
        X[np.isnan(X)] = 0

    else:  # deep static models
        pklname += '_testset.pkl'
        if params.load_corpus and os.path.isfile(pklname):
            logging.info('Loading reformatted corpus ...')
            with open(pklname, 'rb') as f:
                data = pickle.load(f)
        else:
            log.info('Reformating test corpus ...')
            data = prosodeep_corpus.reformat_corpus_rnn(corpus, params)
            #% save reformatted data:
            with open(pklname, 'wb') as f:
                pickle.dump(data, f, -1)

        (X, cg_masks, contour_starts,
         y, contours_in_graph, unique_combinations) = data

    #% init model
    if 'baseline' not in model_type:
        model_test = prosodeep_models.deep_rnn_model(
                contour_types=model.contour_types,  # which nncgs to init
                contour_generators=model.contour_generators,
                contours_in_graph=contours_in_graph,
                n_feature=model.n_feature,
                rnn_inputs=model.rnn_inputs,
                n_hidden=model.n_hidden,
                rnn_model=model.rnn_model,
                activation=model.activation,
                n_output=model.n_output,
                batch_size=model.batch_size,
                shuffle=False,  # shuffle samples in each iteration
                random_state=42,
                verbose=False,
                use_cuda=model.use_cuda,
                vae=model.vae,
                n_hidden_vae=model.n_hidden_vae,
                n_latent=model.n_latent,
                reg_vae=model.reg_vae,
                vae_as_input=model.vae_as_input,
                )

    #% run forward pass
    log.info('Running forward pass ...')
    # run a forward pass on all the data to get the predictions
    if 'baseline' in model_type:
        model.model.cuda()  # because models are sent to cpu after training
        model.model_rnn.cuda()
        y_pred = model.predict(X)
        model.model.cpu()
        model.model_rnn.cpu()
    else:
        y_pred = model_test.predict(X, cg_masks, contour_starts)
    #% write predictions in corpus
    log.info('Writing predictions in corpus ...')
    files = corpus.file.unique()
    for f_ind, file in enumerate(files):
        mask_file = corpus.file == file
        n_units = corpus.loc[mask_file].n_unit.unique()
        for u_ind, n_unit in enumerate(n_units):
            indexes = unique_combinations[file, n_unit]
            corpus.loc[indexes, pred_columns] = y_pred[f_ind, u_ind, :]
            # also write final predictions in target columns
            corpus.loc[indexes, target_columns] = y_pred[f_ind, u_ind, :]

    return corpus


def synthesise_anbysyn_testset(corpus, contour_generators, params):
    '''
    synthesise individual and summed contours for whole corpus.
    '''
    log = logging.getLogger('anbysyn-synth')
    pred_columns = corpus.columns[-params.vowel_pts-1:]  # write here the synthesis
    print()
    # First synthesise all the contours
    log.info('Synthesising all the contours in test corpus ...')
    for contour_type in corpus.contourtype.unique():
        print('\rSynthesising for contour type : {}'.format(contour_type), end='')
        contour_generator = contour_generators[contour_type]
        mask_row = corpus.contourtype == contour_type
        X = corpus.loc[mask_row, 'ramp1':'ramp4'].values.astype('float32')
        if not params.use_strength:
            y_pred = contour_generator.predict(X)
        else:
            if params.strength_method == 'manual':
                strengths = corpus.loc[mask_row, 'strength'].values.astype('float32')
                y_pred = contour_generator.predict(X, strengths=strengths)

            elif 'context' in params.strength_method:
                contexts = corpus.loc[mask_row,
                                      params.context_columns].values.astype('float32')
                y_pred, strengths = contour_generator.predict(X,
                                                              contexts=contexts)
                corpus.loc[mask_row, 'strength'] = strengths

        corpus.loc[mask_row, pred_columns] = y_pred

    # now sum them up
    log.info('Summing all the contours in test corpus ...')
    for file in corpus.file.unique():
        print('\rSumming for file {}'.format(file), end='')
        mask_file = corpus.file == file
        n_units = corpus[mask_file].n_unit.max()
        for n_unit in range(n_units+1):
            mask_unit = corpus.n_unit == n_unit
            mask_row = mask_file & mask_unit
            y_pred = corpus.loc[mask_row, pred_columns].values
            y_pred_sum = np.sum(y_pred, axis=0)
            corpus.loc[mask_row, params.target_columns] = y_pred_sum
        print('\r', end='')

    return corpus
