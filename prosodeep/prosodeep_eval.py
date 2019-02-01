# -*- coding: utf-8 -*-
"""
ProsoDeep - evaluation of performance functions.

@authors:
    Branislav Gerazov Nov 2017

Copyright 2019 by GIPSA-lab, Grenoble INP, Grenoble, France.

See the file LICENSE for the licence associated with this software.
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import logging
import tgt
from natsort import natsorted
import os
import sys
from prosodeep import prosodeep_dsp, prosodeep_plot

#%% funcs
def eval_performance(corpus, f0_data, params,
                     eval_unit, eval_ref, eval_weight, eval_segment,
                     corpus_eval=None):
    '''
    Evaluates f0 reconstruction performance on the corpus.

    Parameters
    ----------
    corpus : pandas DataFrame
        Holds all data generated with Pyprosodeep.

    f0_data : dict
        Holds all original f0 info (pitch marks and interpolations).

    params : obj
        Holds all evaluation parameters.

    corpus_eval : pandas DataFrame
        Holds all evaluation results, if None it's initialised.
    '''
#%% debug
#    corpus_eval=None
#    eval_unit = 'hz'
#    eval_ref = 'f0_ref'
#    eval_weight = 'none'
#    eval_segment = 'whole'
#    eval_unit = 'semitones'
#    eval_ref = 'f0_ref'
#    eval_weight = 'none'
#    eval_segment = 'vowels'
#%% load variables from params
    log = logging.getLogger('eval_corpus')

#    datafolder = params.datafolder
    textgrid_folder = params.datafolder

#    re_folder = params.re_folder
    re_vowels = params.re_vowels
    f0_ref = params.f0_ref

    target_columns = params.target_columns
    vowel_marks = params.vowel_marks
    vowel_pts = params.vowel_pts
    f0_scale = params.f0_scale

    eval_columns = params.eval_columns

#    eval_unit = params.eval_unit
#    eval_ref = params.eval_ref
#    eval_segment = params.eval_segment
#    eval_weight = params.eval_weight

    eval_voiced_offset = params.eval_voiced_offset

    plot_eval = params.plot_eval
    plot_eval_n_files = params.plot_eval_n_files

    log.info('Evaluating Pyprosodeep performance for {}, {}, {}, {} ...'.format(
            eval_unit, eval_ref, eval_weight, eval_segment))

    # read filenames
#    filenames = natsorted([f for f in os.listdir(datafolder) if re_folder.match(f)])
    filenames = corpus['file'].unique()  # takes into account only processed files

    #%% init corpus performance dataframe
    if corpus_eval is None:
        log.info('initialising eval_corpus...')
        corpus_eval = pd.DataFrame(columns=eval_columns)


#%% go through files and synthesise f0 contours
#    segmental_data = {}
    for file_cnt, filename in enumerate(filenames):
        if plot_eval_n_files is not None:
            if file_cnt >= plot_eval_n_files:
                plot_eval = False
#%% debug
#        file_cnt = 106
#        filename = filenames[file_cnt]
        barename = filename.split('.')[0]
        print('\rEvaluating performance on file {}'.format(filename), end='')

        ## check for NaNs
#        mask_file = corpus['file'] == filename
#        if

        # original f0s
        f0_data_file = f0_data[barename]
        pitch_marks, f0_marks_data, f0_marks_smooth, \
            t_f0_orig, f0_orig_data, _, _ = f0_data_file
            # phrase start and end calculated below

        # read textgrids and extract segment durations
        try:
            textgrid = tgt.read_textgrid(textgrid_folder+filename)
        except:
            log.error(sys.exc_info()[0])
            log.error('Can''t read file {} - skipping!'.format(filename))
            raise

        n_phones = len(textgrid.tiers[0])
        phone_starts = np.zeros(n_phones)
        phone_ends = np.zeros(n_phones)
        phone_durs = np.zeros(n_phones)
        phones = []
        vowels = np.empty(n_phones, dtype=np.bool)

        for i, seg in enumerate(textgrid.tiers[0]):  # tier 0 should always be the phones
            phone = seg.text
            phones.append(phone)
            vowels[i] = True if re_vowels.match(phone) else False

            phone_start = seg.start_time
            phone_end = seg.end_time
            phone_dur = seg.duration()

            phone_starts[i] = phone_start
            phone_ends[i] = phone_end
            phone_durs[i] = phone_dur

        phrase_start = phone_starts[1]
        phrase_end = phone_ends[-2]
#        segmental_data[filename] = (phones, vowels, phone_starts, phone_ends, phone_durs)
        wav_len = phone_ends[-1]

#%% synthesise f0
        # take predicted f0s from corpus:
#        mask_file = mask_file_dict[filename]
#        n_units = n_units_dict[filename]
#        print(filename)
        mask_file = corpus['file'] == filename
        n_units = np.max(corpus.loc[mask_file, 'n_unit'].values)
        f0s_array = np.zeros((n_units+1, vowel_pts))
        for n_unit in range(n_units+1):
#            mask_unit = mask_unit_dict[n_unit]
            mask_unit = corpus['n_unit'] == n_unit
            mask_row = mask_file & mask_unit
            # all rows in corpus should be the same
            f0s_array[n_unit,:] = corpus.loc[mask_row, target_columns].values[0,:vowel_pts] # write new targets
#        f0_marks_synth = f0s_array.ravel()  # this are the f0s themselves

#%% find sampling points
        vowels_ind = np.where(vowels)[0]
        vowel_pitch_ts = np.array(())  # this are the times where we have f0 sampled
        for i, f0s_vec in enumerate(f0s_array):
            # find times corresponding to f0s_vec
            phone_ind = vowels_ind[i]
            phone_start = phone_starts[phone_ind]
            phone_dur = phone_durs[phone_ind]
            pitch_time_samples = phone_start + phone_dur * np.array(vowel_marks)

            # add them to list
            vowel_pitch_ts = np.r_[vowel_pitch_ts, pitch_time_samples]

        vowel_pitch_ts = np.array(vowel_pitch_ts)

#%% reverse scale and semitones
        f0_marks_synth = f0s_array.ravel() / f0_scale

        f0_orig = f0_orig_data
        f0_orig[f0_orig <= 0] = 1  # Hz to compensate for crazy extrapolation at the end
        f0_marks = f0_marks_data
#        f0_orig = np.log(f0_orig / f0_ref) * 240 / np.log(2)
#        f0_marks = np.log(f0_marks / f0_ref) * 240 / np.log(2)
        f0_marks_synth_Hz =  f0_ref * np.exp(f0_marks_synth / 240 * np.log(2))

        if eval_ref == '1hz':  # otherwise is f0_ref
            ref = 1
        elif eval_ref == 'f0_ref':
            ref = f0_ref


        if eval_unit == 'hz':
            f0_marks_synth =  f0_marks_synth_Hz
        elif eval_unit == 'semitones':
            f0_marks_synth =  12 / np.log(2) * np.log(f0_marks_synth_Hz / ref)
            f0_orig = 12 / np.log(2) * np.log(f0_orig / ref)
            f0_marks = 12 / np.log(2) * np.log(f0_marks / ref)
        elif eval_unit == 'quartertones':
            f0_marks_synth =  24 / np.log(2) * np.log(f0_marks_synth_Hz / ref)
            f0_orig =  24 / np.log(2) * np.log(f0_orig / ref)
            f0_marks =  24 / np.log(2) * np.log(f0_marks / ref)
        elif eval_unit == 'log':
            f0_marks_synth =  np.log(f0_marks_synth_Hz / ref)
            f0_orig =  np.log(f0_orig / ref)
            f0_marks =  np.log(f0_marks / ref)

#%% interpolate contour
        _, t_f0_synth, f0_synth = prosodeep_dsp.f0_smooth(barename, vowel_pitch_ts, f0_marks_synth,
                                                    wav_len, params,
                                                    show_plot=False,
                                                    plot_f0s=False,
                                                    smooth=False)

#%% plot
#        show_plot = True
        show_plot = params.show_plot
        save_path = params.save_path
        figname = '{}/eval_f0/eval_{}_{}.png'.format(save_path, eval_unit, barename)

        if eval_unit in ['hz','semitones'] and \
            (show_plot or (plot_eval and not os.path.isfile(figname))):

            tuple_plot = (figname, eval_unit, barename, ref,
                        pitch_marks, f0_marks, t_f0_orig, f0_orig,
                        vowel_pitch_ts, f0_marks_synth, t_f0_synth, f0_synth,
                        vowels_ind, phone_starts, phone_ends)

            prosodeep_plot.plot_eval(tuple_plot, params, show_plot=show_plot)

#%% calculate weight
        assert np.all(t_f0_synth == t_f0_orig)  # should be so

        # TODO find soft weight (pov or energy)

        # segment is a hard weight - binary mask
        if eval_segment == 'whole':  # whole phrase
            weight_mask = (t_f0_orig > phrase_start) & (t_f0_orig < phrase_end)

        elif eval_segment in ['voiced', 'vowels']:
            weight_mask = np.zeros(t_f0_orig.size, dtype=bool)  # all false
            for pitch_mark in pitch_marks:
                start = pitch_mark - eval_voiced_offset
                end = pitch_mark + eval_voiced_offset
                weight_mask = weight_mask | ((t_f0_orig > start) & (t_f0_orig < end))

            if eval_segment == 'vowels':
                vowel_mask = np.zeros(t_f0_orig.size, dtype=bool)  # all false
                for vowel_ind in vowels_ind:
                    start = phone_starts[vowel_ind]
                    end = phone_ends[vowel_ind]
                    vowel_mask = vowel_mask | ((t_f0_orig > start) & (t_f0_orig < end))
                weight_mask = weight_mask & vowel_mask  # keep only voiced part of vowels

        weight = weight_mask * 1

#%% calculate performance metric

        wrmse = prosodeep_dsp.wrmse(f0_synth, f0_orig, w=weight)

#        if eval_ref == '1hz' or eval_unit == 'hz':
#            # to get the Hermes 98 perceptual categories wcorr should be normalised
#            normalise = True
#        else:
#            normalise = False
        wcorr = prosodeep_dsp.wcorr(f0_synth, f0_orig, w=weight, normalise=True)

#%% append to corpus
#            'filename ref unit segment weight rmse corr'
        row = pd.Series([filename, eval_ref, eval_unit, eval_segment,
                         eval_weight, wrmse, wcorr], index=eval_columns)
        corpus_eval = corpus_eval.append([row], ignore_index=True)

    print('\r',end='')  # delete line at the end

    return corpus_eval

def get_mask(corpus_eval, eval_unit, eval_ref, eval_weight, eval_segment):
    '''
    Get row mask for the given combination of parameters.
    '''
    mask_ref = corpus_eval.ref == eval_ref
    mask_unit = corpus_eval.unit == eval_unit
    mask_segment = corpus_eval.segment == eval_segment
    mask_weight = corpus_eval.weight == eval_weight
    mask_row = mask_ref & mask_unit & mask_segment & mask_weight
    return mask_row

def eval_sum(corpus_eval, eval_unit, eval_ref, eval_weight, eval_segment, params, corpus_eval_sum=None):
    '''
    Evaluates Pyprosodeep performance on the corpus.

    Parameters
    ----------
    params : obj
        Holds all evaluation parameters.

    corpus_eval_sum : pandas DataFrame
        Holds all evaluation results, if None it's initialised.
    '''
#%% load variables from params
    log = logging.getLogger('eval_sum')
    eval_sum_columns = params.eval_sum_columns

    log.info('Summing Pyprosodeep performance evaluation for {}, {}, {}, {} ...'.format(
            eval_unit, eval_ref, eval_weight, eval_segment))

    # if None init sum corpus performance dataframe
    if corpus_eval_sum is None:
        log.info('initialising eval_corpus...')
        corpus_eval_sum = pd.DataFrame(columns=eval_sum_columns)

#    'measure segment ref unit weight mean std'
    mask_row = get_mask(corpus_eval, eval_unit, eval_ref, eval_weight, eval_segment)

    row = pd.Series(['wrmse', eval_segment, eval_ref, eval_unit, eval_weight,
                     corpus_eval.loc[mask_row,'wrmse'].mean(),
                     corpus_eval.loc[mask_row,'wrmse'].std()],
                     index=eval_sum_columns)

    corpus_eval_sum = corpus_eval_sum.append([row], ignore_index=True)

    row = pd.Series(['wcorr', eval_segment, eval_ref, eval_unit, eval_weight,
                     corpus_eval.loc[mask_row,'wcorr'].mean(),
                     corpus_eval.loc[mask_row,'wcorr'].std()],
                     index=eval_sum_columns)

    corpus_eval_sum = corpus_eval_sum.append([row], ignore_index=True)

    return corpus_eval_sum
