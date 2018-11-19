#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ProsoDeep - a Python prosody analysis system based on the modelling paradigm of
the Superposition of Functional Contours (SFC) prosody model [1] and comprising
the following prosody models:
    - Superposition of Functional Contours (SFC) model, original Python
    implementation was known as PySFC [2]
    - Weighted Superposition of Functional Contours (WSFC) model [3]
    - Variational Prosody Model (VPM) [4]
    - Variational Recurrent Prosody Model (VRPM) [5]

[1] Bailly, Gérard, and Bleicke Holm. "SFC: a trainable prosodic model."
    Speech communication 46, no. 3 (2005): 348-364.

@authors:
    Branislav Gerazov Oct 2017

Copyright 2017 by GIPSA-lab, Grenoble INP, Grenoble, France.

See the file LICENSE for the licence associated with this software.
"""
import pandas as pd
from matplotlib import pyplot as plt
import pickle
from datetime import datetime
import logging
import os
import shutil
import sys
import re
import argparse
from prosodeep import (prosodeep_params, prosodeep_corpus, prosodeep_learn,
                       prosodeep_dsp, prosodeep_eval, prosodeep_plot)

start_time = datetime.now()  # start stopwatch
print()

#%% enable to run with older backend on GPU server
plt.switch_backend('agg')  # to be run using ssh
#plt.switch_backend('Qt5Agg')  # to be run using ssh
#font = {'family' : 'normal',  # increase font for plotting
#        'weight' : 'bold',
#        'size'   : 22}
#matplotlib.rc('font', **font)

#%% parse input arguments
parser = argparse.ArgumentParser()
parser = prosodeep_params.create_parser(parser)
args = parser.parse_args()

#% init system/model parameters
params = prosodeep_params.Params(args)

#% mkdirs
if os.path.isdir(params.save_path):
    if params.remove_folders:  # delete them
        shutil.rmtree(params.save_path, ignore_errors=False)

os.makedirs(params.save_path, exist_ok=True)
if params.plot_f0s:
    os.makedirs(params.save_path+'/f0s', exist_ok=True)
if params.plot_contours:
    os.makedirs(params.save_path+'/all_f0', exist_ok=True)
    if params.plot_duration:
        os.makedirs(params.save_path+'/all_dur', exist_ok=True)
    if params.plot_expansion:
        os.makedirs(params.save_path+'/all_exp', exist_ok=True)
if params.plot_eval:
    os.makedirs(params.save_path+'/eval_f0', exist_ok=True)

#% logger setup
logging.basicConfig(filename=params.save_path+'/prosodeep.log', filemode='w',
                    format='%(asctime)s %(name)-12s: %(levelname)-8s: %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)

# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s: %(message)s')
console.setFormatter(formatter)  # tell the handler to use this format
logging.getLogger('').addHandler(console)  # add the handler to the root logger

# start logging

if args is not None and not args.ignore:
    settings = ''
    for arg in args.__dict__:
        try:
            settings += f'{arg}={getattr(params, arg)}, '
        except:
            pass
    logging.info(settings)

if params.use_cuda:
    logging.info('Using GPU')
else:
    logging.info('Using CPU')

#%% load corpus
pkl_corpus_name = params.pkl_path + params.corpus_name + '.pkl'
if params.load_corpus and os.path.isfile(pkl_corpus_name):
    logging.info('Found corpus ' + params.corpus_name + '.pkl')
    logging.info('Loading corpus ...')
    with open(pkl_corpus_name, 'rb') as f:
        data = pickle.load(f)
        fpro_stats, corpus, f0_data, utterances, phone_set, phone_cnts, _ = data
        f0_ref, isochrony_clock, isochrony_gravity, disol, stats = fpro_stats

#    # add context if it isn't there
    resave = False
    if not all([s in corpus.columns for s in params.context_columns]):
        corpus = prosodeep_corpus.add_context(corpus, params)
        resave = True
#
    if resave:
        with open(pkl_corpus_name, 'wb') as f:
            fpro_stats = f0_ref, isochrony_clock, isochrony_gravity, disol, stats
            # to avoid bad headers
            data = (fpro_stats, corpus, f0_data, utterances, phone_set,
                    phone_cnts, params)
            pickle.dump(data, f, -1)  # last version
#
#%% rebuild corpus
else:
    logging.info('Building corpus ' + params.corpus_name + '.pkl')
    fpro_stats, corpus, f0_data, utterances, phone_set, phone_cnts = \
        prosodeep_corpus.build_corpus(params)

    f0_ref, isochrony_clock, isochrony_gravity, disol, stats = fpro_stats

    if not params.do_all_phrases:
        corpus = prosodeep_corpus.remove_phrase_types(corpus, params)

    corpus = prosodeep_corpus.downcast_corpus(corpus, params.columns)

    # add context one-hot vectors
    if (params.vae and 'context' in params.vae_input) or (
            params.use_strength and params.strength_method == 'context'
            ) or 'rnn_vae' in params.model_type:
        corpus = prosodeep_corpus.add_context(corpus, params)

    # add contour counts
    corpus = prosodeep_corpus.add_contour_generator_count(corpus, params)

    with open(params.pkl_path + params.corpus_name + '.pkl', 'wb') as f:
        fpro_stats = f0_ref, isochrony_clock, isochrony_gravity, disol, stats
        # to avoid bad headers
        data = (fpro_stats, corpus, f0_data, utterances, phone_set, phone_cnts,
                params)
        pickle.dump(data, f, -1)  # last version

    os.system('spd-say "Building the corpus is done."')

#%% load processed corpus
pkl_process_corp_name = params.pkl_path + params.processed_corpus_name + '.pkl'
if params.load_processed_corpus and \
        os.path.isfile(pkl_process_corp_name):
    logging.info('Loading processed corpus '+ pkl_process_corp_name)
    with open(pkl_process_corp_name, 'rb') as f:
        data = pickle.load(f)
        if any(x in params.model_type for x in ['deep', 'baseline']):
            (corpus, f0_data, fpro_stats, utterances, dict_files, dict_models,
             dict_losses, dict_scope_counts, params) = data
        else:
            (corpus, f0_data, fpro_stats, utterances, dict_files,
             dict_contour_generators,
             dict_losses, dict_scope_counts, params) = data
    if params.use_test_set:
            corpus_test = corpus['corpus_test']
            corpus = corpus['corpus']
else:
    logging.info('Processing corpus ...')

#%% if not all phrase types remove those not necessary
    if not params.do_all_phrases:
        corpus = corpus[corpus.phrasetype.isin(params.phrase_types)]
        ## TODO the error will not be for all the utterances

# fix and add columns to the corpus
    corpus = prosodeep_corpus.scale_and_expand_corpus(corpus, params)

# split corpus into train and test
    if params.use_test_set:
        train_ind, test_ind = prosodeep_corpus.split_corpus(
                corpus, params.test_size, stratify=params.test_stratify,
                random_state=42)
        corpus, corpus_test = corpus.iloc[train_ind], corpus.iloc[test_ind]
    else:
        corpus_test = None

# get the scope counts
    logging.info('Counting scopes ...')
    dict_scope_counts ={}
    ### TODO - get rid of loop for phrase_list
    for phrase_type in params.phrase_list:
        dict_scope_counts[phrase_type] = prosodeep_corpus.contour_scope_count(
                corpus, phrase_type=phrase_type, max_scope=40)
    #%% phrase loop - if not bunched!
    # init dictionaries per phrase type
    dict_contour_generators = {}
    dict_models = {}
    dict_losses = {}
    dict_files = {}

    if params.use_pretrained_models:  # preload models
        if 'anbysyn' in params.model_type:
            with open(params.pretrained_models, 'rb') as f:
                data = pickle.load(f)
                _, _, _, _, _, pretrained_models, _, _, _ = data

            if params.database == 'liu':
                pretrained_models = pretrained_models['all']  # all for chen -> liu
            else:
                pretrained_models = pretrained_models['all']  # DC can be the reference

        else:  # deep models
            with open(params.pretrained_models, 'rb') as f:
                data = pickle.load(f)
                _, _, _, _, _, pretrained_models, _, _, _ = data
            pretrained_models = pretrained_models['all']  # all for chen -> liu
            contour_generators = pretrained_models.contour_generators
    else:
        contour_generators = None

    for phrase_type in params.phrase_list:
        logging.info('='*42)
        logging.info('Training for phrase {} from {} ...'.format(
                      phrase_type, params.phrase_types))

        # init contour generators
        logging.info('Initialising contour generators and masks ...')
        contour_generators = {}
        contour_keys = []
        for function_type in params.phrase_types + params.function_types:
            if function_type in dict_scope_counts[phrase_type].keys():
                contour_keys.append(function_type)


        if 'anbysyn' in params.model_type:
            for contour_type in contour_keys:
                if (params.use_pretrained_models
                        and contour_type in pretrained_models.keys()):
                    if params.database == 'liu':
                        # for liu only keep tones
                        if contour_type not in params.tones: # + ['WB']:
                            contour_generators[contour_type] = \
                                    prosodeep_learn.construct_contour_generator(
                                            contour_type, params)
                            continue
                        # copy only the contour generator layers
                        contour_generator = prosodeep_learn.construct_contour_generator(
                                contour_type, params)
                        contour_generator_pre = pretrained_models[contour_type]
                        for l in ['hidden0', 'out_contour']:
                            layer = getattr(
                                    contour_generator_pre.contour_generator, l)
                            # freeze if necessary
                            if params.freeze:
                                for p in layer.parameters():
                                    p.requires_grad = False
                            setattr(contour_generator.contour_generator, l, layer)
                        contour_generators[contour_type] = contour_generator
                    else:
                        contour_generator = pretrained_models[contour_type]
                        contour_generator.reg_strengths = params.reg_strengths
                        contour_generator.reg_strengths_mean = params.reg_strengths_mean
                        contour_generator.reset_optimizer()
                        if params.freeze:
                            for l in ['hidden0', 'out_contour']:
                                layer = getattr(contour_generator.contour_generator, l)
                                for p in layer.parameters():
                                    p.requires_grad = False
                        contour_generators[contour_type] = contour_generator
                else:
                    contour_generators[contour_type] = \
                        prosodeep_learn.construct_contour_generator(
                                contour_type, params)

        # save them in dictonary
        dict_contour_generators[phrase_type] = contour_generators

        # create masks
        (files, mask_all_files, mask_file_dict,
        mask_contours, n_units_dict, mask_unit_dict
        ) = prosodeep_corpus.create_masks(
                corpus, contour_keys, params,
                phrase_type=phrase_type)
        dict_files[phrase_type] = files

        start_train = datetime.now()
        #%% normalise input ramps
        if params.normalisation_type == 'minmax':
            corpus, feats_min, feats_max = prosodeep_dsp.normalise_min_max(
                    corpus, params)
            corpus_test, _, _ = prosodeep_dsp.normalise_min_max(
                    corpus_test, params, feats_min=feats_min, feats_max=feats_max)
        #%% model training
        if params.model_type == 'anbysyn':
            (corpus,
             dict_contour_generators[phrase_type],
             dict_losses[phrase_type],
             losses_DC) = prosodeep_learn.analysis_by_synthesis(
                     corpus, mask_all_files,
                     mask_file_dict, mask_contours,
                     n_units_dict, mask_unit_dict,
                     contour_keys,
                     contour_generators, params)
        else:
            if 'rnn' in params.model_type:
                (corpus, model,
                 batch_losses,
                 epoch_cnt, epoch_losses) = prosodeep_learn.train_rnn_model(
                         corpus,
                         contour_generators=dict_contour_generators[phrase_type],
                         params=params)
            elif any(x in params.model_type
                     for x in ['baseline', 'deep', 'deep_vae']):
                (corpus, model,
                 batch_losses,
                 epoch_cnt, epoch_losses) = prosodeep_learn.train_model(
                         corpus,
                         contour_generators=dict_contour_generators[phrase_type],
                         params=params)

            dict_losses[phrase_type] = (batch_losses, epoch_cnt, epoch_losses)
            dict_models[phrase_type] = model

    #%% save results
    # downcast
    corpus.loc[:, 'f01':] = corpus.loc[:, 'f01':].apply(pd.to_numeric,
                                                        downcast='float')

    if params.save_processed_data:
        with open(params.pkl_path + params.processed_corpus_name+'.pkl', 'wb') as f:
            if params.use_test_set:
                corpus_dict = {'corpus':corpus, 'corpus_test':corpus_test}
            else:
                corpus_dict = corpus
            if any(x in params.model_type for x in ['deep', 'baseline']):
                data = (corpus_dict, f0_data, fpro_stats, utterances, dict_files,
                        dict_models,
                        dict_losses, dict_scope_counts, params)
            else:
                data = (corpus_dict, f0_data, fpro_stats, utterances, dict_files,
                        dict_contour_generators,
                        dict_losses, dict_scope_counts, params)
            pickle.dump(data, f, -1)  # last version

    end_time = datetime.now()
    dif_time = end_time - start_train
    logging.info('='*42)
    logging.info('Finished training in {}'.format(dif_time))
    os.system('spd-say "Training complete."')

#%% make a DataFrame from utterances
# db_utterances = pd.DataFrame(data=list(utterances.values()),
                             # index=utterances.keys(), columns=['utterance'])
#db_utterances["length"] = db_utterances.utterance.apply(lambda x: len(x.split()))

#%% plot contours
colors = prosodeep_plot.init_colors(params)
if params.plot_contours:
    logging.info('='*42)
    logging.info('='*42)
    logging.info('Plotting final iterations ...')
    for phrase_type, files in dict_files.items():
        if params.plot_n_files is not None:
            l = files[:params.plot_n_files]
        elif params.plot_last_n_files is not None:
            l = files[-params.plot_last_n_files:]
        else:
            l = files
        if params.database == 'morlec' and len(params.phrase_types) > 1:
            # plot last from all phrasetypes
            m = []
            for file in l:
                nr = re.match('(.*)_(\d*).*',file).groups()[1]
                for phrase in params.phrase_types:
                    m.append(phrase+'_'+nr+'.TextGrid')
            l = m

        if params.database == 'liu':  # plot 1st iteration from all sentences
            m = []
            for file in l:
                if re.match('F1_1.*1\.TextGrid', file):
                    m.append(file)
            l = m[:6]  # reduce plotting

        for file in l:
            #% plot one file
            # file = 'DC_328.TextGrid'
            if 'baseline' not in params.model_type:  # synthesise individual contours
                if 'rnn' in params.model_type:
                    corpus = prosodeep_learn.synthesise_rnn_contours(
                            corpus, dict_models[phrase_type], file, params)
                elif 'deep' in params.model_type:
                    corpus = prosodeep_learn.synthesise_deep_contours(
                            corpus, dict_models[phrase_type], file, params)
#
            logging.info('Plotting f0 and dur for file {} ...'.format(file))
            prosodeep_plot.plot_contours(params.save_path+'/'+phrase_type+'_f0/',
                                   file, utterances,
                                   corpus, colors, params, plot_contour='f0',
                                  show_plot=params.show_plot)
            if params.plot_duration:
                prosodeep_plot.plot_contours(
                        params.save_path+'/'+phrase_type+'_dur/',
                        file, utterances,
                        corpus, colors, params,
                        plot_contour='dur', show_plot=params.show_plot)

#%% plot expansion
if params.plot_expansion:
    logging.info('Plotting expansions ...')
    if 'deep' in params.model_type:  # no expansion for baseline
        for phrase_type in params.phrase_list:
            prosodeep_plot.plot_expansion(params.save_path+'/'+phrase_type+'_exp/',
                                    dict_models[phrase_type].contour_generators,
                                    colors, dict_scope_counts[phrase_type], params,
                                    show_plot=params.show_plot)
    elif 'anbysyn' in params.model_type:
        for phrase_type, contour_generators in dict_contour_generators.items():
            scope_counts = dict_scope_counts[phrase_type]
            prosodeep_plot.plot_expansion(params.save_path+'/'+phrase_type+'_exp/',
                                    contour_generators,
                                    colors, dict_scope_counts[phrase_type],
                                    params, show_plot=params.show_plot)

#%% plot losses
logging.info('='*42)
logging.info('Plotting losses ...')
if any(x in params.model_type for x in ['deep', 'baseline']):
    for phrase_type, losses in dict_losses.items():
        batch_losses, epoch_cnt, epoch_losses = losses
        prosodeep_plot.plot_batch_losses(params.save_path, batch_losses, losses,
                                         epoch_cnt, epoch_losses,
                                   log_scale=False, show_plot=params.show_plot)
else:
    for phrase_type, losses in dict_losses.items():
        losses = dict_losses[phrase_type]
        prosodeep_plot.plot_losses(params.save_path, phrase_type, losses,
                                   log_scale=True,
                                   show_plot=params.show_plot)

#%% final losses
if 'deep' not in params.model_type:
    logging.info('Plotting final losses ...')
    prosodeep_plot.plot_final_losses(dict_losses, params,
                                     show_plot=params.show_plot)

#%% evaluate performance
logging.info('Evaluating reconstruction performance ...')
if corpus_test is None:
    corpus_test = corpus
else:  # synthesise test corpus data!
    if 'anbysyn' in params.model_type:  # anbysyn synthesise contours
            corpus_test = prosodeep_learn.synthesise_anbysyn_testset(
                    corpus_test, contour_generators, params)
    elif 'rnn' in params.model_type:
        for phrase_type, files in dict_files.items():
            corpus_test = prosodeep_learn.synthesise_rnn_testset(
                    corpus_test,
                    dict_models[phrase_type],
                    params)
    elif any(x in params.model_type for x in ['deep', 'baseline']):
        for phrase_type, files in dict_files.items():
            corpus_test = prosodeep_learn.synthesise_deep_testset(
                    corpus_test,
                    dict_models[phrase_type],
                    params)

eval_pkl_name = params.pkl_path + params.processed_corpus_name + '_eval.pkl'
if params.load_eval_corpus and os.path.isfile(eval_pkl_name):
    with open(eval_pkl_name, 'rb') as f:
        data = pickle.load(f)
        corpus_eval, corpus_eval_sum = data
else:
    corpus_eval = None
    corpus_eval_sum = None
    for eval_ref in params.eval_refs:
        for eval_unit in params.eval_units:
            for eval_segment in params.eval_segments:
                for eval_weight in params.eval_weights:
                    corpus_eval = prosodeep_eval.eval_performance(
                            corpus_test, f0_data, params, eval_unit, eval_ref,
                            eval_weight, eval_segment, corpus_eval=corpus_eval)

                    # drop nans due to good files:
                    corpus_eval.dropna(inplace=True)
                    mask_row = prosodeep_eval.get_mask(
                            corpus_eval, eval_unit, eval_ref,
                            eval_weight, eval_segment)
                    print(corpus_eval.loc[mask_row].describe())
                    sys.stdout.flush()

                    corpus_eval_sum = prosodeep_eval.eval_sum(
                            corpus_eval, eval_unit, eval_ref,
                            eval_weight, eval_segment, params,
                            corpus_eval_sum=corpus_eval_sum)

    if params.save_eval_data:
        with open(eval_pkl_name, 'wb') as f:
            data = corpus_eval, corpus_eval_sum
            pickle.dump(data, f, -1)  # last version

spreadname = '{}/{}_evaluation_stats.xls'.format(
        params.save_path, params.processed_corpus_name)
writer = pd.ExcelWriter(spreadname)
mask_row = corpus_eval_sum.measure == 'wrmse'
corpus_eval_sum.loc[mask_row].to_excel(writer,'wrmse')
mask_row = corpus_eval_sum.measure == 'wcorr'
corpus_eval_sum.loc[mask_row].to_excel(writer,'wcorr')
writer.save()
logging.info('Evaluation data saved in {}.'.format(spreadname))

#%% plot eval data
logging.info('Plotting performance statistics ...')
if params.plot_eval:
    for eval_ref in params.eval_refs:
        for eval_unit in params.eval_units:
            for eval_segment in params.eval_segments:
                for eval_weight in params.eval_weights:
                        combination = '{}_{}_{}_{}'.format(
                                eval_segment, eval_ref,
                                eval_unit, eval_weight)
                        mask_row = prosodeep_eval.get_mask(
                                corpus_eval, eval_unit, eval_ref,
                                eval_weight, eval_segment)
                        prosodeep_plot.plot_histograms(
                                corpus_eval[mask_row].wrmse,
                                corpus_eval[mask_row].wrmse.mean(),
                                corpus_eval[mask_row].wrmse.median(),
                                params.save_path,
                                plot_type='rmse_'+combination,
                                show_plot=params.show_plot)
                        prosodeep_plot.plot_histograms(
                                corpus_eval[mask_row].wcorr,
                                corpus_eval[mask_row].wcorr.mean(),
                                corpus_eval[mask_row].wcorr.median(),
                                params.save_path,
                                plot_type='corr_'+combination,
                                show_plot=params.show_plot)

#%% wrap up
end_time = datetime.now()
dif_time = end_time - start_time
logging.info('='*42)
prompt = 'Finished in {}'.format(dif_time)
logging.info(prompt)
logging.info('='*42)
print()
print()
os.system('spd-say "Finished boss!"')

#%% shut down
#import time
#time.sleep(30*60)
#os.system('/sbin/poweroff')

