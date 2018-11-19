#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pyprosodeep - plotting functions.

@authors:
    Branislav Gerazov Nov 2017

Copyright 2017 by GIPSA-lab, Grenoble INP, Grenoble, France.

See the file LICENSE for the licence associated with this software.
"""
import numpy as np
import pandas as pd
import logging
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import os
import glob
import shutil
import torch
from prosodeep import prosodeep_corpus
#%%
def init_colors(params):
    '''
    Init color palet for plots.

    Parameters
    ==========
    phrase_types : list
        Phrase types to reserve colors for.
    function_types : list
        Function types to reserve colors for.
    '''
    phrase_types = params.phrase_types
    function_types = params.function_types

    plt.style.use('default')
    color_phrase = 'C1'
    colors = {'orig' : 'C7',
             'recon' : 'C0'}
    for phrase_type in phrase_types:
        colors[phrase_type] = color_phrase

    for i, function_type in enumerate(function_types):
        if len(function_type) > 2:
            function_type = function_type[:2]
        if function_type not in colors.keys():
            colors[function_type] = 'C' + str(np.mod(i+2, 10))
    #             'DC' : color_phrase, 'QS' : color_phrase,
    #             'DG' : 'C2',
    #             'DD' : 'C3',
    #             'EM' : 'C4',
    #             'XX' : 'C5',
    #             'DV' : 'C6',
    #             'ID' : 'C8',
    #             'IT' : 'C9'}

    return colors

#%%
def plot_histograms(data_all, data_mean, data_median, save_path, plot_type=None, show_plot=False):
    '''
    Used for obtaining f0_ref and duration isochrony clock.

    Parameters
    ==========
    data_all : DataSeries
        Holds all samples from the distribution.
    data_mean : float
        Mean of all the data.
    data_median : float
        Median of all the data.
    data_kde : float
        Kernel Density Estimation peak fo the data.
    save_path : str
        Save path.
    plot_type : str
        To plot f0, phone or syll stats.
    show_plot : bool
        Whether to close the plot.
    '''
    log = logging.getLogger('plt_hist')
    fig = plt.figure()
    sns.set(color_codes=True, style='ticks')
    ax = sns.distplot(data_all)

    kde_x, kde_y = ax.get_lines()[0].get_data()
    data_kde = kde_x[np.argmax(kde_y)]

    ax.axvline(data_mean, c='C1', lw=2, alpha=.7, label='mean')
    ax.axvline(data_median, c='C2', lw=2, alpha=.7, label='median')
    ax.axvline(data_kde, c='C4', lw=2, alpha=.7, label='kde')
    ax.legend()
    if plot_type == 'f0':
        plt.title('F0 histogram\nmean = {} Hz, median = {} Hz, kde = {} Hz'.format(
                    int(data_mean), int(data_median), int(data_kde)))
        plt.xlabel('Frequency [Hz]')
    elif plot_type == 'phone':
        plt.title('Phone duration histogram\nmean = {} ms, median = {} ms, kde = {} ms'.format(
                    int(data_mean*1e3), int(data_median*1e3), int(data_kde*1e3)))
        plt.xlabel('Duration [s]')
    elif plot_type == 'syll':
        plt.title('Syllable duration histogram\nmean = {} ms, median = {} ms, kde = {} ms'.format(
                    int(data_mean*1e3), int(data_median*1e3), int(data_kde*1e3)))
        plt.xlabel('Duration [s]')
    elif 'rmse' in plot_type:
        if 'semitones' in plot_type:
            plt.title('RMSE histogram\nmean = {:.2f} st, median = {:.2f} st, kde = {:.2f} st'.format(
                    data_mean, data_median, data_kde))
            plt.xlabel('WRMSE [semitones]')
        if 'quartertones' in plot_type:
            plt.title('RMSE histogram\nmean = {:.2f} qt, median = {:.2f} qt, kde = {:.2f} qt'.format(
                    data_mean, data_median, data_kde))
            plt.xlabel('WRMSE [quartertones]')
        if 'log' in plot_type:
            plt.title('RMSE histogram\nmean = {:.2f} logHz, median = {:.2f} logHz, kde = {:.2f} logHz'.format(
                    data_mean, data_median, data_kde))
            plt.xlabel('WRMSE [log Hz]')
        if 'hz' in plot_type:
            plt.title('RMSE histogram\nmean = {:.2f} Hz, median = {:.2f} Hz, kde = {:.2f} Hz'.format(
                    data_mean, data_median, data_kde))
            plt.xlabel('WRMSE [Hz]')
    elif 'corr' in plot_type:
        plt.title('Correlation histogram\nmean = {:.3f}, median = {:.3f}, kde = {:.3f}'.format(
                    data_mean, data_median, data_kde))
        plt.xlabel('Correlation')

    plt.ylabel('Density')
    filename = '{}/histogram_{}_stats.png'.format(save_path, plot_type)
    plt.savefig(filename, dpi='figure')
    log.info('Histograms saved in {}'.format(filename))
    fig.tight_layout()

    if not show_plot:
        plt.close(fig)
#%%
def plot_contours(save_path, file, utterances, corpus, colors, params, plot_contour='f0', show_plot=False):
    '''
    Plot prosodeep contours.

    Parameters
    ----------
    save_path : str
    file : str
    utterances : dict
    corpus : pandas DataFrame
    colors : dict
    scale : float
    plot_contour : str
        Which contour to plot. Avalable options 'f0' and 'dur'
    show_plot : bool
        Whether to keep plot open.

    params
    ======
    iterations : int
        Number of iterations of analysis-by-synthesis loop.
    function_types : list
        List of available function types.
    learn_rate : float
        Learning rate of contour generators.
    max_iter : int
        Maximum number of iterations for the contour generators.
    orig_columns : list
        List of original columns.
    database : str
        Name of database.
    vowel_pts : int
        Number of samples taken from the vocalic nuclei.
    '''
    log = logging.getLogger('plt_contour')
    plt.style.use('default')
    matplotlib.rcParams.update({'font.size': 14})

    i = params.iterations -1
    function_types = params.function_types
#    learn_rate = params.learn_rate
#    max_iter = params.max_iter
    orig_columns = params.orig_columns
    target_columns = params.target_columns
    database = params.database
    vowel_pts = params.vowel_pts
    use_strength = params.use_strength
    if plot_contour == 'f0':
        scale = params.f0_scale
    else:
        scale = params.dur_scale
#%% init
    barename = file.split('.',1)[0]
    mask_file = corpus['file'] == file
    corpus_file = corpus[mask_file]
    corpus_file = corpus_file.reset_index()
    contour_type = corpus.loc[mask_file,'phrasetype'].iloc[0]
    mask_contour = corpus['contourtype'] == contour_type
    mask_row = mask_file & mask_contour
    rus = corpus.loc[mask_row, 'unit'].values
    f0_orig = corpus.loc[mask_row,'f00':'f0{}'.format(vowel_pts-1)].values
    f0_orig = f0_orig.ravel()  # flatten to 1D
    dur_orig = corpus.loc[mask_row,'dur'].values
    n_rus = rus.size

    ## open fig and setup axis
    if database in ['morlec', 'yi']:
#        fig = plt.figure(figsize=(10,8))
        if 'baseline' in params.model_type:
            fig = plt.figure(figsize=(8,4))
        else:
            fig = plt.figure(figsize=(8,6))
    else:
        if 'baseline' in params.model_type:
            fig = plt.figure(figsize=(19,4))
        else:
            fig = plt.figure(figsize=(19,10))
    ax1 = fig.add_subplot(111)
    plt.grid(True)
    ax1.set_xlabel('Rhythmic unit')
    ax2 = ax1.twiny()
    ax2.set_xticks(np.arange(n_rus)+1)
    ax2.set_xticklabels(rus)
#    ax2.set_xlabel('{}, learn_rate {}, iteration {}'.format(barename, learn_rate, i+1))
    # plt.title('{} : {}'.format(barename, utterances[barename].lower()),
    #           y=1.05)

    if plot_contour == 'f0':
        ax1.set_ylabel('Normalised f0')
        x_axis = np.arange(1-int(vowel_pts/2)/vowel_pts, n_rus + 1/2, 1/vowel_pts)
        ax1.plot(x_axis, f0_orig/scale, c=colors['orig'], marker='o', ms=3.5, lw=3, label='f0 target')
        # predicted values without the durcoeff
        pred_columns = [column + '_it{:03}'.format(i) for column in orig_columns[:-1]]

    elif plot_contour == 'dur':
        ax1.set_ylabel('Duration coefficient')
        x_axis = np.arange(1, n_rus + 1/2, 1)
        ax1.plot(x_axis, dur_orig/scale, c=colors['orig'], marker='o', ms=3.5, lw=3, label='dur target')
        pred_columns = 'dur_it{:03}'.format(i)

    else:
        raise ValueError('Contour unsupported!')

    contourtypes = corpus.loc[mask_file, 'contourtype'].values
    strengths = corpus.loc[mask_file, 'strength'].values

    # for deep model final recon is already in target f0
    if any(x in params.model_type for x in ['deep','baseline']) \
            or not any([c in function_types for c in contourtypes.tolist()]):  # only phrase component

        if plot_contour == 'f0':
            f0_pred = corpus.loc[mask_row, target_columns[:-1]].values
            f0_pred = f0_pred.ravel()
            ax1.plot(x_axis, f0_pred/scale, c=colors['recon'], marker='o', ms=3.5,
                     lw=3, alpha=.8, label='f0 recon')
        elif plot_contour == 'dur':
            dur_pred = corpus.loc[mask_row, target_columns[-1]].values
            ax1.plot(x_axis, dur_pred/scale, c=colors['recon'], marker='o', ms=3.5,
                     lw=3, alpha=.8, label='dur recon')
        ylims = ax1.get_ylim()
        ax2.set_xlim(ax1.get_xlim())

    if 'baseline' not in params.model_type:
        if any([c in function_types for c in contourtypes.tolist()]):  # non-phrase contours
            if 'deep' in params.model_type:
                if plot_contour == 'f0':
                    pred_columns = corpus.columns[-params.vowel_pts-1:-1]
                    # this is where the individual ones are
                elif plot_contour == 'dur':
                    pred_columns = corpus.columns[-1]

            n_units = corpus.loc[mask_file, 'n_unit'].values
    #        ramp4 = corpus.loc[mask_file, 'ramp4'].values  # we use this to separate contours
    #        ramp2 = corpus.loc[mask_file, 'ramp2'].values  # we use this to find landmarks
            ## go through ramp4 and decompose the contours
    #        end_inds = np.where(ramp4 == 0)[0]
    #        start_inds = np.r_[0, end_inds[:-1]+1]
    #        corpus_file = corpus[mask_file]
    #        corpus_file = corpus_file.reset_index()
            marks = corpus_file['marks']
            start_inds = marks.index[marks.str.contains('start')].values
            landmark_inds = marks.index[marks.str.contains('mark')].values
            end_inds = marks.index[marks.str.contains('end')].values

            n_contours = end_inds.size
            contour_levels = np.empty(n_contours)
            contour_landmarks = np.empty(n_contours, dtype=int)   # position of landmark in contour
            contour_labels = []
            contour_cnt = 0

            if plot_contour == 'f0':
                f0_preds = corpus.loc[mask_file, pred_columns].values
                contour_array = np.empty((50, vowel_pts*n_rus))  # position of contours in plot, at least 2 levels
                contours = np.empty((n_contours, vowel_pts*n_rus))   # one contour per row

            elif plot_contour == 'dur':
                dur_preds = corpus.loc[mask_file, pred_columns].values
                contour_array = np.empty((50, n_rus))  # position of contours in plot, at least 2 levels
                contours = np.empty((n_contours, n_rus))   # one contour per row

            contour_array.fill(np.nan)
            contours.fill(np.nan)

            # do plotting planning by fitting contours in a plotting matrix
            level = 0  # uncomment this if contours should respect annotation levels
            last_tone = False
            last_level = 0
            for start, landmark_ind, end in zip(start_inds, landmark_inds, end_inds):
    #            print(start,landmark_ind, end)  # debug
                if plot_contour == 'f0':
                    # check empty
                    if contourtypes[start] in params.tones:  # (for tones only?)
                        last_tone = True
                        if level > last_level:
                            last_level = level
                        level = 0  # start over when fitting contours in the layers
                    else:
                        if last_tone:
                            level =  last_level + 1  # move WB to next level
    #                        last_tone = False

                    while not np.all(np.isnan(contour_array[level,
                                               n_units[start]*vowel_pts : (n_units[end]+1)*vowel_pts])):
                        level += 1
                        if level == contour_array.shape[0]:  # array not big enough!
                            log.error('{} not enough levels for contours - skipping!'.format(barename))
                            break

                    # set f0s
                    f0s_contour = f0_preds[start : end+1, :].ravel()
                    contour_array[level, n_units[start]*vowel_pts : (n_units[end]+1)*vowel_pts] = f0s_contour
                    contours[contour_cnt, n_units[start]*vowel_pts : (n_units[end]+1)*vowel_pts] = f0s_contour

                elif plot_contour == 'dur':
                    # check empty
                    if contourtypes[start] in params.tones:  # (for tones only?)
                        last_tone = True
                        if level > last_level:
                            last_level = level
                        level = 0  # start over when fitting contours in the layers
                    else:
                        if last_tone:
                            level =  last_level + 1  # move WB to next level
    #                        last_tone = False
                    while not np.all(np.isnan(contour_array[level,
                                                            n_units[start] : n_units[end]+1])):
                        level += 1
                        if level == contour_array.shape[0]:  # array not big enough!
                            log.error('{} not enough levels for contours - skipping!'.format(barename))
                            break
                    # set durs
                    durs_contour = dur_preds[start : end+1]
                    contour_array[level, n_units[start] : n_units[end]+1] = durs_contour
                    contours[contour_cnt, n_units[start] : n_units[end]+1] = durs_contour

                contour_levels[contour_cnt] = level
                if use_strength and strengths[start] != 1:
                    contour_labels.append('{} {:.2f}'.format(contourtypes[start], strengths[start]))
                else:
                    contour_labels.append(contourtypes[start])

    #            landmark_ind = np.where(ramp2[start:end+1] == 0)[0][0] + start
    #            landmark_ind = np.where(landmark_inds >= start & landmark_inds <= end)[0][0]
                contour_landmarks[contour_cnt] = n_units[landmark_ind]+1

                contour_cnt += 1
            # preplanning done

            # trim contour_array
            contour_array = contour_array[:level+1]
            # sum and plot predictions
            y_pred_sum = np.nansum(contour_array, axis=0)

            # prepare plotting parameters
            if plot_contour == 'f0':
                if 'deep' not in params.model_type:  # done already
                    ax1.plot(x_axis, y_pred_sum/scale, c=colors['recon'], marker='o', lw=3, ms=3.5,
                             alpha=.8, label='f0 recon')
                offset_coef = 300  # for plotting
    #            offset_text = 30  # for the text labels
                offset_text = 100  # yanpin
                upper_margin = 100  # for the plot borders
                lower_margin = 200  # for the plot borders
                y_ticks = np.arange(-300, 300, 100)
                ylims = ax1.get_ylim()
    #            f0_min =  - np.ceil(np.abs(ylims[0]) / offset_coef) * offset_coef - offset_coef/2
    #            f0_min =  - np.ceil(np.abs(ylims[0]) / 100) * 100
                f0_min =  - 300  # yanpin

            elif plot_contour == 'dur':
                if 'deep' not in params.model_type:
                    ax1.plot(x_axis, y_pred_sum/scale, c=colors['recon'], marker='o', lw=3, ms=3.5,
                             alpha=.8, label='dur recon')
                offset_coef = 1  # for plotting
                offset_text = .2  # for plotting
                upper_margin = 1  # for the plot borders
                lower_margin = 1  # for the plot borders
                y_ticks = np.arange(-2, 4, 1)
                ylims = ax1.get_ylim()
                f0_min =  - np.ceil(np.abs(ylims[0]) / offset_coef) * offset_coef

            # do the actual plotting
            for level, label, contour, landmark in zip(contour_levels, contour_labels,
                                                       contours, contour_landmarks):
                if plot_contour == 'f0':
                    landmark_ind = np.where(x_axis >= landmark)[0][0]  # address the next one
                elif plot_contour == 'dur':
                    landmark_ind = np.where(x_axis == landmark)[0][0]  # address the next one
                landmark = x_axis[landmark_ind]
    #            contour_min = np.nanmin(contour)
    #            contour_min = np.ceil(np.abs(contour_min)/ offset_coef) * offset_coef
    #            offset = np.max([offset_coef, contour_min])
                offset = f0_min - (level+1)*offset_coef
                plt.axhline(y=offset, c='C7', ls='--', lw=2)
    #            plt.plot([landmark, landmark],
    #                     [-4e3,
    #                      contour[landmark_ind]/scale+offset],
    #                     c=colors[label], ls='--', lw=1, alpha=.8)
                # TODO for XX it doesn't mark it at the landmark
                if ' ' in label:  # then it has strength
                    color = colors[label.split(' ')[0][:2]]  # [:2] for EMc in yi
                else:
                    color = colors[label[:2]]

                plt.plot([landmark, landmark],
                         [contour[landmark_ind]/scale+offset-offset_coef/8,
                          contour[landmark_ind]/scale+offset +offset_coef/8],
                         c=color, ls='-', lw=2, alpha=.8)
    #            plt.text(landmark, contour[landmark_ind]/scale+offset+offset_text,
    #                     label, color=colors[label], fontweight='bold',
    #                     bbox=dict(facecolor='w', lw=0, alpha=0.5))
                plt.text(landmark-.2, np.max((offset+offset_text,
                                              contour[landmark_ind]/scale+offset+offset_text/2)),
                         label, color=color, fontweight='bold')
    #                     bbox=dict(facecolor='w', lw=0, alpha=0.8))
                plt.plot(x_axis, contour/scale+offset, c=color,
                         marker='o', ms=3.5, lw=3, alpha=.8)
    #            f0_min =  offset - np.ceil(np.abs(np.nanmin(contour)) / offset_coef) * offset_coef

            # delete ticks from y-axis
            ax1.set_yticks(y_ticks)
            ax1.set_yticklabels(y_ticks)

            offset = f0_min - np.max(contour_levels + 1)*offset_coef
    #        final_limit = f0_min - np.max(contour_levels)*offset_coef-lower_margin
            final_limit = offset-lower_margin
            ax1.axis([0,n_rus+1,final_limit, ylims[1]+upper_margin])
            ax2.axis([0,n_rus+1,final_limit, ylims[1]+upper_margin])

        if plot_contour == 'f0':
            ax1.legend(loc='upper right')
        else:
            ax1.legend(loc='upper left')
#%% save
    plt.tight_layout()
    plt.savefig('{}{}_{}_{}.png'.format(save_path, barename, plot_contour,
                                        params.processed_corpus_name), dpi='figure')
    if not show_plot:
        plt.close(fig)

#    # for publication
    ax1.legend(loc='lower left')
    plt.title('')

##    ax1.set_xlabel('')
##    ax1.set_xticklabels([])
#
#    ax2.set_xticklabels([])
#
#    ax1.set_ylabel('')
#    ax1.set_yticklabels([])
#    ax1.legend(loc='upper right')


def plot_losses(save_path, phrase_type, losses,
                log_scale=False, show_plot=False):
    '''
    Plot losses of training contour generators.

    Parameters
    ==========
    save_path : str
        Path to save figure in.
    phrase_type : str
        Type of phrase to plot losses for.
    losses : dict
        Losses for each function type.
    show_plot : bool
        Whether to close plot.
    '''
    plt.style.use('default')
    fig = plt.figure(figsize=(10,8))
    plt.grid(True)
    plt.ylabel('Squared-loss')
    plt.xlabel('Iteration')
    if type(losses) == dict:  # we have contour generators
        for key in losses:
            plt.plot(losses[key], lw=3, alpha=0.75, label=key)
    else:  # for baseline
        plt.plot(losses[2], lw=3, alpha=0.75, label='loss')
    plt.legend()
    if log_scale:
        plt.yscale('log')
#    plt.savefig('losses_{}_learn{}_maxit{}_it{}.png'.format(corpus_name, learn_rate, max_iter, iterations),
    plt.savefig('{}/losses_{}.png'.format(save_path, phrase_type), dpi='figure')
    if not show_plot:
        plt.close(fig)

def plot_batch_losses(save_path, batch_losses, losses, epoch_cnt, epoch_losses,
                      log_scale=False, show_plot=False):
    '''
    Plot losses of training contour generators.

    Parameters
    ==========
    save_path : str
        Path to save figure in.
    phrase_type : str
        Type of phrase to plot losses for.
    losses : dict
        Losses for each function type.
    show_plot : bool
        Whether to close plot.
    '''
    plt.style.use('default')
    fig = plt.figure(figsize=(15,8))
    plt.grid(True)
    plt.ylabel('Mean square error')
    plt.xlabel('Epoch')
    ymin = 0
    ymax = np.max(batch_losses) + 10
#    plt.axvline(epoch_cnt[0], ymin=ymin, ymax=ymax, c='C3', lw=2, alpha=.4,
#                label='epoch bounds')
#    for x in epoch_cnt[1:]:
#        plt.axvline(x, ymin=ymin, ymax=ymax, c='C3', lw=2, alpha=.4)

    x_epoch = np.arange(1, epoch_cnt.size+1)
    x_batch = np.linspace(0, x_epoch[-1], batch_losses.size)
    plt.plot(x_batch, batch_losses, label='mse per batch')
#    offset = batch_losses.size // epoch_cnt.size / 2  # center the mean errors
#    plt.plot(epoch_cnt-offset, epoch_losses, 'o-', c='w', ms=14, lw=8, alpha=.8)  # for contrast
#    plt.plot(epoch_cnt-offset, epoch_losses, 'o-', c='C2', ms=12, lw=6, alpha=.8, label='mean mse per epoch')
    plt.plot(x_epoch, epoch_losses, 'o-', c='w', ms=10, lw=6, alpha=.8)  # for contrast
    plt.plot(x_epoch, epoch_losses, 'o-', c='C2', ms=8, lw=4, alpha=.8, label='mean mse per epoch')
    plt.grid(True)
    plt.legend()
#        plt.xticks(epoch_cnt, np.arange(epoch_cnt.size))
    if log_scale:
        plt.yscale('log')
    plt.axis([0, x_epoch[-1], ymin, ymax])
    plt.savefig('{}/losses_batch.png'.format(save_path), dpi='figure')
    if not show_plot:
        plt.close(fig)

def plot_expansion(save_path, contour_generators, colors, scope_counts, params, show_plot=False):
    '''
    Plot expansion of contour generators.

    Parameters
    ==========
    save_path : str
        Save path for figures.
    contour_generators : dict
        Dictionary of contour generators.
    colors : dict
        Dictionary of colors.
    phrase_type : str - defunct!
        Phrase type for plotting expansion.
    scope_counts : dict
        Dictionary of scope counts for each function type for chosen phrase type.
    show_plot : bool
        Whether to keep the plot open after saving.

    params
    ======
    left_max : int
        Maximum length of left scope.
    right_max : int
        Maximum length of right scope.
    f0_scale : float
        f0 scaling coefficient.
    dur_scale : float
        dur_coeff scaling coefficient.
    end_marks : list
        List of function types with left context only.
    vowel_pts : int
        Number of vowel samples.
    '''
    plt.style.use('default')
    left_max = params.left_max
    right_max = params.right_max
    phrase_max = params.phrase_max
    f0_scale = params.f0_scale
#    dur_scale = params.dur_scale
    phrase_types = params.phrase_types
    tones = params.tones
    end_marks = params.end_marks
    vowel_pts = params.vowel_pts
    model_type = params.model_type
#    contour_types = params.phrase_types + params.function_types
#%% test
#    left_max = 5
#    right_max = 5
#    show_plot=True
#    scope_counts = prosodeepdata.contour_scope_count(corpus, 'DC')
#    colors = prosodeepplot.init_colors(phrase_types)
    # create ramps
#    ramps_all = []
    left_scopes = np.arange(1,left_max+1)
    right_scopes = np.arange(1,right_max+1)
    scope_max = left_max + right_max
#    for left_scope in left_scopes:
#        ramps_row = []
#        for right_scope in right_scopes:
#            #%
#            # create ramps
#            contour_scope = left_scope + right_scope
#            ramp_global = np.arange(contour_scope)[::-1]
##            landmark_unit = left_scope  # position of landmark
#            unit_scope_1 = np.arange(left_scope)  # including landmark unit
#            unit_scope_2 = np.arange(right_scope)
#            ramp_local = np.c_[np.r_[unit_scope_1[::-1], unit_scope_2[::-1]],
#                               np.r_[np.linspace(0, 10, num=left_scope)[::-1],
#                                     # if there is 1 unit this will favor the 0
#                                     np.linspace(0, 10, num=right_scope)],
#                               np.r_[unit_scope_1, unit_scope_2]]
#
#            ramps = np.c_[ramp_local, ramp_global]
#            ramps_row.append(ramps)
#
#        ramps_all.append(ramps_row)
    #% left_ramps - only with left scope
#    ramps_left = []
#    for contour_scope in range(1,phrase_max+1):
#        ramp_global = np.arange(contour_scope)[::-1]
#        ramp_local = np.c_[ramp_global,
#                           np.linspace(10, 0, num=contour_scope),
#                           ramp_global[::-1]]
#        ramps = np.c_[ramp_local, ramp_global]
#        ramps_left.append(ramps)

    ramps_all = prosodeep_corpus.generate_ramps(left_max, right_max)
    ramps_left = prosodeep_corpus.generate_ramps(phrase_max)
#%% plot expansions
    for contour_type, contour_generator in contour_generators.items():
        if contour_type not in scope_counts.keys():  # if it's in the corpus - fix for EMc
            continue
#        if contour_type not in contour_types:
#            continue
#        contour_type = 'C3'
#        contour_generator = contour_generators[contour_type]
# TODO : take care of scope trunctuation!!!
#%%
        if 'deep' in model_type:
            contour_generator.cpu()

        x_offset = 10
        y_offset = 400

        if contour_type in phrase_types + end_marks:

            if contour_type in phrase_types:
                y_offset = 200
                fig = plt.figure(figsize=(10,12))
#                fig = plt.figure(figsize=(4,6))
                ramps_left_trim = ramps_left  # draw all
            else:
#                continue
                fig = plt.figure(figsize=(6,8))
                ramps_left_trim = ramps_left[:left_max+1]
            ax1 = fig.add_subplot(111)
            plt.grid(True)
            ax1.set_xlabel('Rhythmic unit')
            plt.title('Expansion for {}'.format(contour_type))
            ax1.set_ylabel('Normalised f0')
            plt.grid(True)
            x_axis_f0 = np.arange(-int(vowel_pts/2)/vowel_pts, len(ramps_left_trim)+1/2, 1/vowel_pts) + 1
            x_axis_f0 = - x_axis_f0[::-1]
    #        y_ticks_orig = np.arange(-200,300,100)
            y_ticks = np.empty(0)
    #        y_tick_labels = np.empty(0, dtype=int)
            y_tick_labels = []
#            if 'rnn' in model_type:
#                h_rnn = None

            for i, row in enumerate(ramps_left_trim):
                row_offset = i * y_offset
                if row.ndim == 1:  # add 0 axis
                    row = row[np.newaxis, :]
                X = row
                if 'deep' in model_type:
                    x = torch.tensor(X, dtype=torch.float32)
                    if 'rnn' in model_type:
                        x = x.unsqueeze(dim=1)  # for batch size
                        hx, cx = None, None
                        if params.vae:
                            contexts = x.new_zeros(1, params.n_context)  # for the batch size
                            contexts[:, 0] = 1  # this is DC or No emphasis

                        if params.rnn_inputs is not None:
                            x = x[:,:,params.rnn_inputs]
                            # x = x.unsqueeze(2)
                        if params.vae:
#                                cntxt = contexts
                            if contour_generator.rnn_model == 'lstm':
                                y, _ = contour_generator(x, contexts, hx, cx)
                            else:
                                y, _ = contour_generator(x, contexts, hx)

                        else:
                            if contour_generator.rnn_model == 'lstm':
                                y = contour_generator(x, hx, cx)
                            else:
                                y = contour_generator(x, hx)

                        y_pred = y[:,0,:]

                    elif params.vae:
                        if 'context' in params.vae_input:
                            contexts = x.new_zeros(x.shape[0], params.n_context)
                            contexts[:,0] = 1
                            x_input = torch.cat((x, contexts), dim=1)
                            y_pred, _, _, _ = contour_generator(x_input)
                        else:
                            y_pred, _, _, _ = contour_generator(x)
                    elif params.use_strength:
                        if params.strength_method == 'manual':
                            strengths = x.new_ones(1).unsqueeze_(0)
                            y_pred = contour_generator(x, strengths=strengths)

                        elif 'context' in params.strength_method:
                            y_pred, _ = contour_generator(x, contexts=None)
                    else:
                        y_pred = contour_generator(x)

                    y_pred = y_pred.data.numpy()

                else:
                    if not params.use_strength:
                        y_pred = contour_generator.predict(X)
                    else:
                        if params.strength_method == 'manual':
                            y_pred = contour_generator.predict(X, strengths=1)

                        elif 'context' in params.strength_method:
                            y_pred, _ = contour_generator.predict(X, contexts=None)

                f0_pred = y_pred[:,:-1].ravel()/f0_scale

                scope_count = scope_counts[contour_type][i+1,0]

                plt.axhline(y=row_offset, c='C7', ls='--', lw=2)
                if scope_count > 0:
                    plt.plot(x_axis_f0[-f0_pred.size:], f0_pred + row_offset,
                             c=colors[contour_type[:2]], marker='o', ms=3.5, lw=3, alpha=.8)
                else:
                    plt.plot(x_axis_f0[-f0_pred.size:], f0_pred + row_offset,
                             c=colors[contour_type[:2]], marker='o', ms=3.5, lw=3, alpha=.3)
                plt.text(-.2, f0_pred[-1] + row_offset, scope_count,
                         color=colors[contour_type[:2]], fontweight='bold',
                         bbox=dict(facecolor='w', lw=0, alpha=0.5))
                y_ticks = np.r_[y_ticks, row_offset]
    #            y_ticks = np.r_[y_ticks, y_ticks_orig+row_offset]
    #            y_tick_labels = np.r_[y_tick_labels, 'RU{}'.format(i+1)]
                y_tick_labels.append('RU{:02}'.format(i+1))

            # set ticks
            ax1.set_yticks(y_ticks)
            ax1.set_yticklabels(y_tick_labels)
            ax1.set_xticks(x_axis_f0[-int(vowel_pts/2)-1::-vowel_pts])
            ax1.set_xticklabels(np.arange(-1, -scope_max-1, -1))

            # set ticks
            ax1.set_yticks(y_ticks)
            ax1.set_yticklabels(y_tick_labels)

        else:  # all of the other contours
            if contour_type in tones:  # left and right scopes
                ramps_all_trim = ramps_all[:2]
                fig = plt.figure(figsize=(10,8))
                x_offset = 3
                y_offset = 200
            else:
                ramps_all_trim = ramps_all
                fig = plt.figure(figsize=(18,14))

            ax1 = fig.add_subplot(111)
            plt.grid(True)
            ax1.set_xlabel('Rhythmic unit')
            plt.title('Expansion for {}'.format(contour_type))
            ax1.set_ylabel('Normalised f0')
            plt.grid(True)
            y_ticks = np.empty(0)
    #        y_tick_labels = np.empty(0, dtype=int)
            y_tick_labels = []
            x_ticks = np.empty(0)
            x_tick_labels = np.empty(0, dtype=int)
            x_tick_labels = []

            for i, row in enumerate(ramps_all_trim):  # rows are left_scope in range(1,left_max+1):
                row_offset = i * y_offset
                plt.axhline(y=row_offset, c='C7', ls='--', lw=1.5)
                y_ticks = np.r_[y_ticks, row_offset]
    #            y_tick_labels = np.r_[y_tick_labels, 0]
                y_tick_labels.append('L{:02}'.format(i+1))

                if contour_type in tones:
                    row = [None] + row[:1]

                for j, ramps in enumerate(row):  # columns are right_scope in range(1,right_max+1):
                    if ramps is None:
                        ramps = ramps_left[i]

                    offset = j * x_offset
                    if i == 0:  # if first row plot vertical
                        x_ticks = np.r_[x_ticks, offset]
    #                    x_tick_labels = np.r_[x_tick_labels, 0]

                        if contour_type in tones:
                            x_tick_labels.append('R{:02}'.format(j))  # it also has R00!
                        else:
                            x_tick_labels.append('R{:02}'.format(j+1))
                        plt.axvline(x=offset, c='C7', ls='--', lw=1.5)
    #                     plt.plot([landmark, landmark],
    #                     [-2e3, contour[landmark_ind]/scale+offset],
    #                     c=colors[label], ls='--', lw=2, alpha=.8)

                    if contour_type in tones:
                        scope_count = scope_counts[contour_type][i+1,j]
                    else:
                        scope_count = scope_counts[contour_type][i+1,j+1]
                    if ramps.ndim == 1:  # add 0 axis
                        ramps = ramps[np.newaxis, :]
                    X = ramps
                    if 'deep' in model_type:
                        x = torch.tensor(X, dtype=torch.float32)
                        if 'rnn' in model_type:
                            hx, cx = None, None
                            if params.vae:
                                contexts = x.new_zeros(1, params.n_context)
                                contexts[:,0] = 1  # this is DC

                            x = x.unsqueeze(dim=1)  # for batch size
                            if params.rnn_inputs is not None:
                                x = x[:,:,params.rnn_inputs]
                                x = x.unsqueeze(2)
                            if params.vae:
                                if contour_generator.rnn_model == 'lstm':
                                    y, _ = contour_generator(x, contexts, hx, cx)
                                else:
                                    y, _ = contour_generator(x, contexts, hx)

                            else:
                                if contour_generator.rnn_model == 'lstm':
                                    y = contour_generator(x, hx, cx)
                                else:
                                    y = contour_generator(x, hx)

                            y_pred = y[:,0,:]

                        elif params.vae:
                            if 'context' in params.vae_input:
                                contexts = x.new_zeros(x.shape[0], params.n_context)
                                contexts[:,0] = 1  # for DC and no emphasis
                                x_input = torch.cat((x, contexts), dim=1)
                                y_pred, _, _, _ = contour_generator(x_input)
                            else:
                                y_pred, _, _, _ = contour_generator(x)
                        elif params.use_strength:
                            if params.strength_method == 'manual':
                                strengths = x.new_ones(1).unsqueeze_(0)
                                y_pred = contour_generator(x, strengths=strengths)

                            elif 'context' in params.strength_method:
                                y_pred, _ = contour_generator(x, contexts=None)
                        else:
                            y_pred = contour_generator(x)

                        y_pred = y_pred.data.numpy()

                    else:
                        if not params.use_strength:
                            y_pred = contour_generator.predict(X)
                        else:
                            if params.strength_method == 'manual':
                                y_pred = contour_generator.predict(X, strengths=1)

                            elif 'context' in params.strength_method:
                                y_pred, _ = contour_generator.predict(X, contexts=None)

                    f0_pred = y_pred[:,:-1].ravel()/f0_scale

                    l = left_scopes[i]

                    r = right_scopes[j]
                    if contour_type in tones:
                        r = r - 1  # R00
                    #                x_axis_f0 = np.r_[np.arange(-l-2/3, -2/3, 1/3),  # 0 on the end of R1
                    #                                  np.arange(-2/3, r-2/3, 1/3)]
                    # 0 in the start of R1:
                    x_axis_f0 = np.r_[np.arange(-l, 0, 1/vowel_pts),
                                      np.arange(0, r, 1/vowel_pts)] + 1/2/vowel_pts  # to make it in the middle
                    if scope_count > 0:
                        plt.plot(x_axis_f0 + offset, f0_pred + row_offset,
                             c=colors[contour_type[:2]], marker='o', ms=2.5, lw=2, alpha=.8)
                    else:
                        plt.plot(x_axis_f0 + offset, f0_pred + row_offset,
                             c=colors[contour_type[:2]], marker='o', ms=2.5, lw=2, alpha=.3)

                    plt.text(x_axis_f0[-1] + offset + .4,
                             f0_pred[-1] + row_offset, scope_count,
                             color=colors[contour_type[:2]], fontweight='bold',
                             bbox=dict(facecolor='w', lw=0, alpha=0.5))

            # set ticks
            ax1.set_yticks(y_ticks)
            ax1.set_yticklabels(y_tick_labels)
            ax1.set_xticks(x_ticks)
            ax1.set_xticklabels(x_tick_labels)

#%% save

        plt.savefig('{}expansion_{}.png'.format(save_path, contour_type), dpi='figure')
        if not show_plot:
            plt.close(fig)

def plot_final_losses(dict_losses, params, show_plot=False):
    '''
    Plot final losses for all contour generators using a heat map.

    Parameters
    ==========
    dict_losses : dict
        Dictionary of lossess for all phrase types.
    show_plot : bool
        Whether to close plot.

    params
    ------
    save_path : str
        Save path for figure.
    iterations : int
        Number of iterations of analysis-by-synthesis loop.
    max_iter : int
        Maximum number of itterations.
    '''
    plt.style.use('default')
    save_path = params.save_path
    iterations = params.iterations
    max_iter = params.max_iter

    pd_losses = pd.DataFrame(dict_losses)
    pd_losses = pd_losses.applymap(lambda x: x[-1] if x is not np.NaN else 0)
    sns.set()
    fig = plt.figure(figsize=(8,6))
    sns.heatmap(pd_losses, annot=True, cbar=False, cmap="YlGnBu")
    plt.xlabel('Phrase type')
    plt.ylabel('Contour generator')
    plt.title('Losses after {} iterations with {} maxiter per loop'.format(iterations, max_iter))
    plt.savefig('{}/final_losses_heat.png'.format(save_path), dpi='figure')
    if not show_plot:
        plt.close(fig)

def plot_worst(corpus, params, phrase=None, n_files=None):
    '''
    Copy plots for 100 worst files, or all files for a particular phrase type
    in descending order according to RMS.

    Parameters
    ==========
    corpus : pandas DataFrame
        Holds all the data.
    phrase : str
        Phrase type to copy for, if None copy for all.
    n_files : int
        Number of files to copy, if None copy all.

    params
    ======
    orig_columns : list
        List with column names from corpus holding original data.
    iterations : iter
        Number of iterations of analysis-by-synthesis loop.
    save_path : str
        Save path for figure.
    '''
    orig_columns = params.orig_columns
    iterations = params.iterations
    save_path = params.save_path

    # analyse losses per file and get a list of files
    pred_columns = [column + '_it{:03}'.format(iterations-1) for column in orig_columns]
    df_errors = pd.Series()
    for file in corpus['file'].unique():
        mask_file = corpus['file'] == file
        n_units = np.max(corpus.loc[mask_file, 'n_unit'].values)
        error_file = np.array([])
        for n_unit in range(n_units+1):
            mask_unit = corpus['n_unit'] == n_unit
            mask_row = mask_file & mask_unit
            y_pred = corpus.loc[mask_row, pred_columns].values
            y_pred_sum = np.sum(y_pred, axis=0)
            y_orig = corpus.loc[mask_row, orig_columns].values  # every row should be the same
            y_error = y_orig[0,:] - y_pred_sum  # this will automatically tile row to matrix
            error_file = np.r_[error_file, y_error[:-1]]  # without the dur coeff
        df_errors[file] = np.mean(error_file**2)

    df_errors_sorted = df_errors.sort_values(ascending=False)
    df_errors_sorted.rename('RMS')

    # write to excel
    writer = pd.ExcelWriter(save_path+'/errors_lineup.xls')
    df_errors_sorted.to_excel(writer,'Sheet1')
    writer.save()

    # plot
    if phrase is None:
        os.mkdir(save_path+'/errors_lineup/')
        for cnt, file in enumerate(df_errors_sorted.index[:n_files]):
            bare = file.split('.')[0]
            phrase_type = bare[:2]
            src = save_path+'/'+phrase_type+'_f0/'+bare+'_*.png'
            dst = '{}/errors_lineup/{:03}_{:03}rms_{}.png'.format(
                    save_path, cnt, int(df_errors_sorted.iloc[cnt]), bare)
            shutil.copyfile(glob.glob(src)[0], dst)

    else:
        os.mkdir(save_path+'/errors_lineup_'+phrase+'/')
        for cnt, file in enumerate(df_errors_sorted.index):
            bare = file.split('.')[0]
            phrase_type = bare[:2]
            if phrase_type == phrase:
                src = save_path+'/'+phrase_type+'_f0/'+bare+'_*.png'
                dst = '{}/errors_lineup_{}/{:03}_{:03}rms_{}.png'.format(
                        save_path, phrase, cnt, int(df_errors_sorted.iloc[cnt]), bare)
                shutil.copyfile(glob.glob(src)[0], dst)

def plot_eval(tuple_plot, params, show_plot=False):
    '''
    Plot original and synthesised f0 contours used for performance evaluation.
    '''
    # unpack parameter tuple:
    figname, eval_unit, barename, ref, \
        pitch_marks, f0_marks, t_f0_orig, f0_orig, \
            vowel_pitch_ts, f0_marks_synth, t_f0_synth, f0_synth, \
                vowels_ind, phone_starts, phone_ends = tuple_plot

    f0_min = params.f0_min
    f0_max = params.f0_max

    phrase_start, phrase_end = phone_starts[1], phone_ends[-2]



    # plot for Hz and semitone (all the log ones are the same just scaled)

    #%%
    fig = plt.figure(figsize=(18,8))
    plt.style.use('default')

    # plot phrase bounds
    plt.vlines([phrase_start, phrase_end], -1000, 1000, color='r', lw=2, alpha=0.2)
    plt.fill_betweenx([-1000, 1000], x1=0, x2=phrase_start, facecolor='r', alpha=0.2,
               label='phrase bounds')
    plt.fill_betweenx([-1000, 1000], x1=phrase_end, x2=t_f0_orig[-1], facecolor='r', alpha=0.1)

    # plot vowels bounds
    for phone_ind in vowels_ind:
        start = phone_starts[phone_ind]
        end = phone_ends[phone_ind]
        plt.vlines([start, end], -1000, 1000, color='g', lw=2, alpha=0.2)
        plot = plt.fill_betweenx([-1000, 1000], x1=start, x2=end, facecolor='g', alpha=0.2)
    plot.set_label('vowels')

    plt.plot(pitch_marks, f0_marks, '-o', ms=6, c='C7', alpha=.5)
    plt.plot(t_f0_orig, f0_orig, c='C7', lw=4, alpha=.5, label='f0 target')

    plt.plot(vowel_pitch_ts, f0_marks_synth, 'o', ms=10, alpha=.9, c='w')
    plt.plot(t_f0_synth, f0_synth, c='w', lw=5, alpha=.9)
    plt.plot(vowel_pitch_ts, f0_marks_synth, 'o', ms=9, alpha=.9, c='C0')
    plt.plot(t_f0_synth, f0_synth, c='C0', lw=4, alpha=.9, label='f0 recon')

    # relative axis
#            fig_min = np.min(np.r_[f0_marks, f0_marks_synth])
#            fig_min = fig_min - .2*np.abs(fig_min)
#            fig_max = np.max(np.r_[f0_marks, f0_marks_synth])
#            fig_max = fig_max + .2*np.abs(fig_max)

    # fixed axis
    fig_min = f0_min
    fig_max = f0_max
    if eval_unit == 'semitones':
        fig_min =  12 / np.log(2) * np.log(fig_min / ref)
        fig_max =  12 / np.log(2) * np.log(fig_max / ref)
    elif eval_unit == 'quartertones':
        fig_min =  24 / np.log(2) * np.log(fig_min / ref)
        fig_max =  24 / np.log(2) * np.log(fig_max / ref)
    elif eval_unit == 'log':
        fig_min =  np.log(fig_min / ref)
        fig_max =  np.log(fig_max / ref)

    plt.axis([0, t_f0_orig[-1], fig_min, fig_max])

    plt.title(barename)
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [{}]'.format(eval_unit))
    plt.grid(True)
    legend = plt.legend(frameon=True, loc='upper left')
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('white')

#%% save fig
    plt.savefig(figname, dpi='figure')

    if not show_plot:
        plt.close(fig)