# -*- coding: utf-8 -*-
"""
ProsoDeep - corpus related utility functions.

@authors:
    Branislav Gerazov Nov 2017

Copyright 2019 by GIPSA-lab, Grenoble INP, Grenoble, France.

See the file LICENSE for the licence associated with this software.
"""
import numpy as np
import pandas as pd
import logging
from natsort import natsorted
import os
import re
import sys
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
# KFold, GroupKFold, StratifiedKFold, StratifiedShuffleSplit
from prosodeep import prosodeep_data

def corpus_to_fpro(corpus, params):
    """
    This writes the test corpus to fpro files that can be used for synthesis
    with TDPSOLA.

    --deprecated--
    """
#    target_columns = params.target_columns
#    filenames = corpus.file.unique()
#    for file_cnt, filename in enumerate(filenames):
#        barename = filename.split('.')[0]
#        fproname = barename + '.fpro'
#        mask_file = corpus.file == filename
#        n_units = np.max(corpus.loc[mask_file, 'n_unit'].values)
#        f0s_list = []
#        dur_list = []
#        for n_unit in range(n_units+1):
#            mask_unit = corpus.n_unit == n_unit
#            mask_row = mask_file & mask_unit
#            # all rows in corpus should be the same
#            f0s = corpus.loc[mask_row, target_columns].values[0,:-1]
#            dur = corpus.loc[mask_row, target_columns].values[0,-1]
#            f0s_list.append(f0s.tolist())
#            dur_list.append(dur)
#        prosodeep_data.write_fpro(fproname, f0s_list, dur_list)


def build_corpus(params):
    '''
    Build corpus from all files in datafolder and save it to corpus_name. All parameters
    passed through object params.

    Parameters
    ----------
    datafolder : str
        Folder where the input data files are located.
    file_type : str
        Whether to use fpros or TextGrids to read the data.

    phrase_types : list
        Attitude functions found in database.

    function_types : list
        Functions other than attitudes found in database.

    end_marks : list
        Functions that end the scope, i.e. they have only left context.

    database : str
        Which database are we using - sets a bunch of predifined parameters.

    vowel_marks : list
        Points where to sample vocalic nuclei.

    use_ipcgs : bool
        Use IPCGs or regular syllables as rhtythmic units. Morlec uses IPCGs,
        Chen uses syllables.

    show_plot : bool
        Whether to keep the plots open of the f0_ref, dur_stat and syll_stat if it is
        necessary to extract them when file_type is TextGrid.

    save_path : str
        Save path for the plotted figures.
    '''
    log = logging.getLogger('build_corpus')

    datafolder = params.datafolder
    phrase_types = params.phrase_types
    function_types = params.function_types
#    tone_context = params.tone_context
    end_marks = params.end_marks
#    database = params.database
    file_type = params.file_type
    columns = params.columns

    re_folder = params.re_folder
    re_vowels = params.re_vowels
    use_ipcgs = params.use_ipcgs
    f0_ref = params.f0_ref
    isochrony_clock = params.isochrony_clock

    # read filenames
    filenames = natsorted([f for f in os.listdir(datafolder) if re_folder.match(f)])
    # build a pandas corpus
    corpus = pd.DataFrame(columns=columns)

#% read files loop
    phone_set = []
    utterances = {}
    f0_datas = {}
    phone_duration_means=None
    for file_cnt, filename in enumerate(filenames):
        log.info('Reading file {}'.format(filename))
        barename = filename.split('.')[0]

        if file_type == 'TextGrid':
            try:
                fpro_lists, f0_data, phone_duration_means, f0_ref, isochrony_clock = \
                        prosodeep_data.read_textgrid(filename, params,
                                                     phone_duration_means=phone_duration_means,
                                                     f0_ref=f0_ref,
                                                     isochrony_clock=isochrony_clock)
            except:
                log.error(sys.exc_info()[0])
                log.error('{} read error!'.format(filename))
                continue

            fpro_stats, f0s, units, durunits, durcoeffs, dursegs, \
                    phones, orthographs, phrase, levels = fpro_lists

        f0_datas[barename] = f0_data

        phone_set = phone_set + phones.tolist()
        # get the utterance text
        utterance = orthographs[orthographs != '*'].tolist()
        utterance = [' '+s if s not in ['.',',','?','!'] else s for s in utterance]
        utterance = ''.join(utterance)[1:]
        utterances[barename]= utterance
        if fpro_stats is not None:  # might have a bad header
            f0_ref, isochrony_clock, r_clock, disol, stats = fpro_stats

#%  get the rhythmic units (rus) and order them
        # we can't use np.unique for this
        rus = []
        ru_f0s = []
        ru_coefs = []
        ru_map = np.zeros(phones[1:-1].size, dtype=int)
        cnt = -1  # count of unit, skip first silence
        unit = ''
        for i, phone in enumerate(phones[1:-1]):
            # run through the phones and detect vowels to accumulate the ru-s
            if len(unit) == 0:  # at start of loop and after deleting unit
                cnt += 1
                unit = units[1:-1][i]
                rus.append(unit)
                unit = unit[len(phone):]  # delete phone from unit
            else:
                assert unit.startswith(phone), \
                    log.error('{} - unit doesn''t start with phone!'.format(filename))
                unit = unit[len(phone):]  # delete phone from unit
            if re_vowels.match(phone):
                ru_f0s.append(f0s[1:-1,:][i,:])
                ru_coefs.append(durcoeffs[1:-1][i])
            ru_map[i] = cnt

        ru_f0s = np.array(ru_f0s)
        # if there are nans change them to 0s, so the learning doesn't crash:
        if np.any(np.isnan(ru_f0s)):
            log.warning('No pitch marks for unit {} in {}!'.format(
                              rus[np.where(np.any(np.isnan(ru_f0s), axis=1))[0][0]], filename))
            ru_f0s[np.isnan(ru_f0s)] = 0

        ru_coefs = np.array(ru_coefs)
        rus = np.array(rus)
        assert ru_coefs.size == rus.size, \
            log.error('{} - error ru_coefs.size != rus.size'.format(filename))

        #% check phrase contour
        if 'FF' not in phrase[1]:
            log.error('{} Start of phrase is not in first segment - skipping!'.format(filename))
            continue
        # this covers phr:FF too
        if phrase[-1] == '*':
            log.error('{} End of phrase is not in finall segment - skipping!'.format(filename))
            continue

#        extract phrase type and power
        phrase_type = phrase[-1].strip(':')
        phrase_type, strength = get_strength(phrase_type)

        # target vector for
        phrase_targets = np.c_[ru_f0s, ru_coefs]
        # generate input ramps for phrase
        ramp_global = np.arange(rus.size)[::-1]
        ramp_local = np.c_[ramp_global,
                           np.linspace(10, 0, num=rus.size),
                           ramp_global[::-1]]
        phrase_ramps = np.c_[ramp_local, ramp_global].T
        mask_rus = np.arange(rus.size)  # the scope of the contour
        marks = get_marks(mask_rus, mask_rus[-1])
        for unit, l in enumerate(np.c_[phrase_ramps.T, phrase_targets]):
            row = pd.Series([filename, phrase_type, phrase_type, marks[unit],
                             rus[unit], mask_rus[unit], strength] + l.tolist(),
                          columns)
            corpus = corpus.append([row], ignore_index=True)

        #% do the rest of the contours
        landmarks = phrase_types + function_types
        if levels.size > 0:
            for level in levels[:,1:]:  # without the initial silence
                mask = []
                iscontour = False
                found_landmark = False
                for cnt, entry in enumerate(level.tolist()):
                    entry = entry.strip(':')

                    if entry not in function_types:
                        entry = entry[:2]  # this eliminates reduced tones

                    if not iscontour:
                        if 'FF' in entry:  # contour start
                            # if it's the first IPCG or it's one with a vowel we add the mask
                            # this is to solve IPCG-Syll overlap issues:
                            # wordlevel: game DG kupe, but IPCG level: gam (e DG k) up e
                            # i.e. where is ek? it is in the left unit but not the right!
                            # this is because of the mapping function we use
                            # between segments and units (IPCGs)
                            if use_ipcgs:
                                if cnt == 0 or re_vowels.match(phones[cnt+1]):  #+1 because we skip silence
                                    mask.append(cnt)
                            else:  # use sylls - easier
                                mask.append(cnt)

                            iscontour = True

                        if entry in landmarks:
                            log.warning('{} landmark {} starts contour - skipping!'.format(filename, entry))

                    elif iscontour:
                        if entry == '*':  # normal *
                            mask.append(cnt)

                        elif (entry in end_marks and \
                                not found_landmark) or \
                             (entry in landmarks and \
                                not found_landmark and \
                                    entry not in end_marks and \
                                    cnt == len(level.tolist())-1): # end of contour,
                                 # Cs can also end contours if at the end:
                                 # :C4 :FF * :C2
                            contour_type = entry

                            if use_ipcgs:
                                if cnt < level.size - 1 and re_vowels.match(phones[cnt+1]):
                                    # the segment with FF is not part of scope if its the end silence
                                    # and if it is not a vowel (from the word that follows)
                                    mask.append(cnt)
                                    landmark_ind = cnt
                                else:
                                    landmark_ind = cnt - 1
                            else:
                                landmark_ind = cnt - 1

                            # EMlong - take all EMs to have right scope to the
                            # end of the phrase
                            if contour_type in ['EM', 'EMc'] and params.tone_levels:
                                contour_type = 'EM'
                                for cnt2 in range(cnt, level.size-1):
                                    mask.append(cnt2)

                            ru_landmark_ind = ru_map[landmark_ind]
                            mask_rus = np.array(np.unique(ru_map[mask]), dtype=int)  # map to the rus
                            marks = get_marks(mask_rus, ru_landmark_ind)
                            contour_type, strength = get_strength(contour_type)

                            # if it's a tone contour take care of context
                            if params.tone_levels:  # if it is chinese
                                if params.tone_scope and \
                                (params.tone_scope != params.ann_tone_scope):
                                    # it is not the context used in the data
                                    if contour_type in params.tones:  # if it is a tone contour
                                        mask_rus, marks = process_tone_scope(mask_rus, ru_landmark_ind,
                                                                               rus, params.tone_scope)

                            corpus = append_contour_to_corpus(corpus, columns, filename, phrase_type,
                                                                  contour_type, marks, rus, mask_rus,
                                                                  strength, landmark_ind,
                                                                  ru_map, ru_f0s, ru_coefs, params)
                            ## for EMlong
                            if contour_type in ['EM', 'EMc'] and params.tone_levels:
                                break

                            if use_ipcgs:
                                if re_vowels.match(phones[cnt+1]):
                                    # the segment with FF is not part of scope if its not a vowel
                                    mask = [cnt]
                                else:
                                    mask = []

                            else:  # when using syllables it's part of the next scope
                                mask = [cnt]

                            found_landmark = False  # you cannot daisy chain XX

                        elif entry in landmarks and \
                                not found_landmark and \
                                    entry not in end_marks:   # if first landmark
                            found_landmark = True
                            contour_type = entry

                            if use_ipcgs:
                                if cnt < level.size - 1 and re_vowels.match(phones[cnt+1]):
                                    # the segment with FF is not part of scope if its the end silence
                                    # and if it is not a vowel (from the word that follows)
                                    landmark_ind = cnt
                                else:
                                    landmark_ind = cnt - 1
                            else:
                                landmark_ind = cnt - 1

                            mask.append(cnt)


                        elif 'FF' in entry:  # end of contour and maybe start of next one
                            if use_ipcgs:
                                if cnt < level.size - 1 and re_vowels.match(phones[cnt+1]):
                                    # the segment with FF is not part of scope if its the end silence
                                    # and if it is not a vowel (from the word that follows)
                                    mask.append(cnt)
#                            else:  # it's not part of the scope

                            if not found_landmark:
                                log.warning('{} no landmark in level {} - skipping!'.format(
                                        filename, level))
                                # this will also find the case where FF is not the start of the next contour
                                # e.g. FF * * * DD * * FF * * FF * * * XX * FF
                            else:
                                ru_landmark_ind = ru_map[landmark_ind]
                                mask_rus = np.array(np.unique(ru_map[mask]), dtype=int)  # map to the rus
                                marks = get_marks(mask_rus, ru_landmark_ind)
                                contour_type, strength = get_strength(contour_type)

                                # if it's a tone contour take care of context
                                if params.tone_levels:  # if it is chinese
                                    if params.tone_scope and \
                                    (params.tone_scope != params.ann_tone_scope):
                                        # it is not the context used in the data
                                        if contour_type in params.tones:  # if it is a tone contour
                                            mask_rus, marks = process_tone_scope(mask_rus, ru_landmark_ind,
                                                                                   rus, params.tone_scope)

                                corpus = append_contour_to_corpus(corpus, columns, filename, phrase_type,
                                                                  contour_type, marks, rus, mask_rus,
                                                                  strength, landmark_ind,
                                                                  ru_map, ru_f0s, ru_coefs, params)

                            # it might be the start of the next one like in DC_368.fpro!
                            # e.g. FF * * * DD * * * FF * * * XX * FF
                            found_landmark = False

                            if use_ipcgs:
                                if re_vowels.match(phones[cnt+1]):
                                    # the segment with FF is not part of scope if its not a vowel
                                    mask = [cnt]
                                else:
                                    mask = []

                            else:  # when using syllables it's part of the next scope
                                mask = [cnt]

                        elif entry in landmarks and found_landmark:  # end of contour
                            # if not first landmark - ie it's the start of another contour
                            # process previous contour
                            if use_ipcgs:
                                if re_vowels.match(phones[cnt+1]):  #+1 because we skip silence
                                    # also if its a vowel its from the next scope
                                    mask.append(cnt)  # well apparently it is, except the final silence
#                            else: # don't append if it's syllables

                            ru_landmark_ind = ru_map[landmark_ind]
                            mask_rus = np.array(np.unique(ru_map[mask]), dtype=int)  # map to the rus
                            marks = get_marks(mask_rus, ru_landmark_ind)
                            contour_type, strength = get_strength(contour_type)

                            # if it's a tone contour take care of context
                            if params.tone_levels:  # if it is chinese
                                if params.tone_scope and \
                                (params.tone_scope != params.ann_tone_scope):
                                    # it is not the context used in the data
                                    if contour_type in params.tones:  # if it is a tone contour
                                        mask_rus, marks = process_tone_scope(mask_rus, ru_landmark_ind,
                                                                               rus, params.tone_scope)

                            corpus = append_contour_to_corpus(corpus, columns, filename, phrase_type,
                                                                  contour_type, marks, rus, mask_rus,
                                                                  strength, landmark_ind,
                                                                  ru_map, ru_f0s, ru_coefs, params)

                            # update mask to new contour
                            # keep the mask starting from previous landmark
                            # keep landmark if its on vowel
                            if use_ipcgs:
                                if re_vowels.match(phones[landmark_ind+1]):
                                    # +1 because we skip silence
                                    mask = [i for i in mask if i >= landmark_ind]
                                else:
                                    mask = [i for i in mask if i > landmark_ind]
                            else:  # for sylls keep after the landmark
                                mask = [i for i in mask if i > landmark_ind]

                            contour_type = entry  # new contour's type
                            if use_ipcgs:
                                landmark_ind = cnt
                            else:
                                landmark_ind = cnt - 1

                            mask.append(cnt)

                        else:
                            log.warning('{} unknown landmark {} in {} - skipping!'.format(
                                    filename, entry, level))
    phone_set = np.array(phone_set)
    phone_set, phone_cnts = np.unique(phone_set, return_counts=True)

#%
    return fpro_stats, corpus, f0_datas, utterances, phone_set, phone_cnts


def create_masks(corpus, contour_keys, params, phrase_type='all'):
    '''
    Create masks to address the data in the corpus.

    Parameters
    ==========
    corpus : pandas data frame
        Holds all data from corpus.
    phrase_type : str
        Type of phrase type to make the mask, all for bunching them together.
    contour_keys : list
        Types of functions used for contour generators.

    params
    ======
    good_files_only : bool
        Whether to use only a subset of the files.
    good_files : list
        The subset of good files.
    database : str
        Name of database.
    '''
    # init masks
    if phrase_type != 'all':
        mask_phrase = corpus['phrasetype'] == phrase_type
        files = natsorted(corpus[mask_phrase]['file'].unique())
    else:
        mask_phrase = corpus.file.notnull()  # TODO: there's probably a better way to do it
        files = natsorted(corpus['file'].unique())

    mask_file_dict = {}
    n_units_dict = {}
    mask_unit_dict = {}
    re_file_nr = re.compile('(.*)_(\d*).*')
    mask_contours = {}
    for contour_type in contour_keys:
        mask_contours[contour_type] = corpus['contourtype'] == contour_type
    # make a mask for all files for training the contours
    mask_all_files = mask_phrase

    for file in files:
        prefix, file_nr =  re_file_nr.match(file).groups()
        mask_file = corpus['file'] == file
        mask_file_dict[file] = mask_file
        n_units = np.max(corpus.loc[mask_file, 'n_unit'].values)
        n_units_dict[file] = n_units
        for n_unit in range(n_units+1):
            mask_unit = corpus['n_unit'] == n_unit
            mask_unit_dict[n_unit] = mask_unit

    return files, mask_all_files, mask_file_dict, \
        mask_contours, n_units_dict, mask_unit_dict


def downcast_corpus(corpus, columns):
    '''
    Take care of dtypes and downcast.

    Parameters
    ==========
    corpus : pandas DataFrame
        Holds all the data.
    columns : list
        The columns of the DataFrame.
    '''
    log = logging.getLogger('down_corpus')
    log.info('Converting columns to numeric ...')  # TODO why some n_units are strings?
    start_colum = corpus.columns.tolist().index('f00')
    corpus[columns[start_colum:]] = corpus[columns[start_colum:]].apply(pd.to_numeric, downcast='float')
    for c in ['n_unit','ramp1','ramp3','ramp4']:
        corpus.loc[:, c] = corpus.loc[:, c].apply(pd.to_numeric, downcast='unsigned')
    corpus['ramp2'] = corpus['ramp2'].apply(pd.to_numeric, downcast='float')

    return corpus

def scale_and_expand_corpus(corpus, params):
    '''
    Take care of scale and expand corpus.

    Parameters
    ==========
    corpus : pandas DataFrame
        Holds all input features and all predictions by the contour generators.

    params
    ======
    columns : list
        Columns of corpus DataFrame.
    orig_columns : list
        Columns holding original f0 and dur_coeff in corpus DataFrame.
    target_columns : list
        Columns holding the f0 and dur_coeff tagets used to train the contour
        generators.
    iterations : int
        Number of iterations to run analysis-by-synthesis loop.
    f0_scale : float
        Scaling factor for the f0s to downscale them near to the dur_coeffs.
    dur_scale : float
        Scaling factor for the dur_coeffs to upscale them near to the f0s.
    '''
    log = logging.getLogger('exp_corpus')

    orig_columns = params.orig_columns
    target_columns = params.target_columns
    iterations = params.iterations
    f0_scale = params.f0_scale
    dur_scale = params.dur_scale
    model_type = params.model_type
    use_only_last_iteration = params.use_only_last_iteration  # for deep models

    log.info('Applying scaling to columns ...')
    corpus.loc[:,'dur'] = corpus.loc[:,'dur'] * dur_scale
    corpus.loc[:, orig_columns[0] : orig_columns[-2]] = corpus.loc[:,
                                                                   orig_columns[0] :
                                                                   orig_columns[-2]] \
                                                                   * f0_scale

    log.info('Expanding initial columns ...')
    new_columns = target_columns.copy()

    if any(x in model_type for x in ['deep','rnn','baseline']) \
            and use_only_last_iteration:
        pred_columns = [column + '_it{:03}'.format(iterations-1) for column in orig_columns]
        new_columns += pred_columns
    else:
        for i in range(iterations):
            pred_columns = [column + '_it{:03}'.format(i) for column in orig_columns]
            new_columns += pred_columns
    corpus = pd.concat([corpus,
                        pd.DataFrame(np.nan, index=corpus.index,
                                     columns=new_columns)],
                       axis=1)
    return corpus


def append_contour_to_corpus(corpus, columns, barename, phrase_type,
                             contour_type, marks, rus, mask_rus, strength, landmark_ind,
                             ru_map, ru_f0s, ru_coefs, params):  #, debug=False):
    """
    Process contour and append to corpus.

    Parameters
    ==========
    corpus : pandas DataFrame
        Holds all the data.
    columns : list
        The columns of the DataFrame.
    barename : str
        Name of file.
    phrase_type : str
        Type of phrase component in file.
    contour_type : str
        Type of contour to add.
    rus : list
        List of Rhythmic units, e.g. IPCGs or syllables, in contour.
    mask : list
        Mask of unit number in the utterance for each unit in the contour.
    landmark_ind : int
        Position of the function type designator.
    ru_map : list
        Shows where each phone in utterance belongs to in rus.
    ru_f0s : ndarray
        f0s for each of the units in rus.
    ru_coefs : ndarray
        dur_coeffs for each unit in rus.
    """
    log = logging.getLogger('append2corpus')
    # find targets for contour
    # TODO cast int to int because some are string??
    contour_targets = np.c_[ru_f0s[mask_rus],
                            ru_coefs[mask_rus]]
    # generate ramps for contour
    contour_scope = contour_targets.shape[0]
    landmark_unit = ru_map[landmark_ind] - mask_rus[0]  # relative position of landmark

    default_scope = None
    default_landmark_unit = None
    if contour_type in params.tones:
        if params.tone_scope == 'right' and contour_scope != 2:
           default_scope = 2
           default_landmark_unit = 0
        elif params.tone_scope == 'left' and contour_scope != 2:
           default_scope = 2
           default_landmark_unit = 1
        elif params.tone_scope == 'both' and contour_scope != 3:
           default_scope = 3
           default_landmark_unit = 1

    contour_ramps = generate_input_ramps(contour_scope, landmark_unit,
                                         default_scope=default_scope,
                                         default_landmark_unit=default_landmark_unit)
    log.debug('-'*42)
    log.debug(contour_type, mask_rus, landmark_ind)
    log.debug(contour_targets)
    log.debug(contour_ramps)
    log.debug('-'*42,'\n')

    # add to corpus
    for l in np.c_[marks, rus[mask_rus], mask_rus,
                   contour_ramps, contour_targets]:
        row = pd.Series([barename, phrase_type, contour_type] + l.tolist()[:3] +
                        [strength] + l.tolist()[3:],
                        columns)
        corpus = corpus.append([row], ignore_index=True)
    return corpus


def generate_input_ramps(contour_scope, landmark_unit, default_scope=None, default_landmark_unit=None):
    '''
    Generate input ramps for SFC based on scope and landmark position. Can also
    generate default scope and landmark ramps and then truncate to current one.
    This is useful for modelling tones with extended scopes that are at the start
    or end of an utterance.
    '''
    if default_scope is not None:
        contour_scope, contour_scope_trunc = default_scope, contour_scope
        landmark_unit, landmark_unit_trunc = default_landmark_unit, landmark_unit

    ramp_global = np.arange(contour_scope)[::-1]

    unit_scope_1 = np.arange(landmark_unit+1)  # including landmark unit
    unit_scope_2 = np.arange(contour_scope - landmark_unit - 1)
    ramp_local = np.c_[np.r_[unit_scope_1[::-1], unit_scope_2[::-1]],
                       np.r_[np.linspace(0, 10, num=unit_scope_1.size)[::-1],
                             # if there is 1 unit this will favor the 0
                             np.linspace(0, 10, num=unit_scope_2.size)],
                       np.r_[unit_scope_1, unit_scope_2]]

    contour_ramps = np.c_[ramp_local, ramp_global]
    if default_scope is not None:  # truncate
        start = landmark_unit - landmark_unit_trunc
        end = start + contour_scope_trunc
        contour_ramps = contour_ramps[start : end, :]
    return contour_ramps


def contour_scope_count(corpus, phrase_type='all', max_scope=20):
    '''
    Count all the scope contexts of a contour.

    Parameters
    ==========
    corpus : pandas DataFrame
        Holds all the data.
    phrase_type : str
        Phrase contour subset within which to count. If None bunch all together.
    max_scope : int
        Maximum scope to take into account
    '''
    log = logging.getLogger('scope_count')
    if phrase_type != 'all':  # split by phrase_type
        corpus_phrase = corpus[corpus['phrasetype'] == phrase_type]
        corpus_phrase = corpus_phrase.reset_index()
    else:
        corpus_phrase = corpus
    contour_types = corpus_phrase['contourtype'].unique()
    scope_counts = {}
    for contour_type in contour_types:
        scope_counts[contour_type] = np.zeros((max_scope,max_scope),dtype=int)

    marks = corpus_phrase['marks']
    start_inds = marks.index[marks.str.contains('start')].values
    landmark_inds = marks.index[marks.str.contains('mark')].values
    end_inds = marks.index[marks.str.contains('end')].values
    for start, land, end in zip(start_inds, landmark_inds, end_inds):
        contour_type = corpus_phrase.loc[start]['contourtype']
        scope_tot = end - start + 1
        scope_left = land - start + 1  # +1 puts the RU with the landmark in left context
        scope_right = end - land
        assert (scope_tot == scope_left + scope_right) , log.error('Scope length doesn''t match!')
        scope_counts[contour_type][scope_left, scope_right] = \
            scope_counts[contour_type][scope_left, scope_right] + 1
    return scope_counts

def add_context(corpus, params):
    '''
    For each contour in the corpus add context, where context is a boolean value
    signifying cooccurence of this contour with the current one.
    Second version (tailored to liu) makes it apply only when there's a landmark
    in the other contour.
    '''
    log = logging.getLogger('add_context')
    columns = params.context_columns
    insert_loc = corpus.columns.get_loc('f00')
    log.info('Adding context columns...')
    for column in columns[::-1]:
        if column not in corpus.columns:
            corpus.insert(insert_loc, column, [0] * len(corpus.index))

    log.info('Adding context for all files in corpus ...')
    filenames = corpus.file.unique().tolist()
    for file in filenames:
        mask_file = corpus.file == file
        marks = corpus.loc[mask_file, 'marks']
        start_inds = marks.index[marks.str.contains('start')].values
        mark_inds = marks.index[marks.str.contains('mark')].values
        end_inds = marks.index[marks.str.contains('end')].values
        for start, mark, end in zip(start_inds, mark_inds, end_inds):
            contour_type = corpus.loc[start].contourtype

            # attitudes have no context by default
            if 'onehot' not in params.context_type \
                    and contour_type in params.all_phrase_types:
                continue

            if 'tone' in params.context_type \
                    and contour_type not in params.tones:  # skip
                continue

            contours_in_context = []  # accumulate contexts

            if 'onehot' in params.context_type:
                contours_in_context.append(contour_type)
                if 'att' in params.context_type:  # add phrase
                    contours_in_context.append(corpus.loc[start].phrasetype)

            elif 'att' in params.context_type:  # add phrase
                contours_in_context.append(corpus.loc[start].phrasetype)

            elif 'all' in params.context_type:  # add all
                # any part of another contour anywhere during the current one
                for i in range(start, end+1):  # include last ru
                    n_unit = corpus[mask_file].loc[i].n_unit
                    mask_n_unit = corpus[mask_file].n_unit == n_unit
                    contours = corpus[mask_file][mask_n_unit].contourtype.tolist()
                    contours.remove(contour_type)  # remove self reference
                    for contour in contours:
                        if contour not in contours_in_context:  # update list
                            contours_in_context.append(contour)
                for contour in contours_in_context:
                    corpus.loc[start : end, contour] = 1

            n_unit = corpus[mask_file].loc[mark].n_unit
            if 'mark' in params.context_type:
                for mark in mark_inds:
                    if corpus[mask_file].loc[mark].n_unit == n_unit:
                        contour = corpus[mask_file].loc[mark].contourtype
                        if contour not in contours_in_context:  # update list
                            contours_in_context.append(contour)

            if 'emph' in params.context_type:
                # mark left and right subscope of emphasis

                # this overloads all and mark for EM:
                if 'EM' in contours_in_context:
                    contours_in_context.remove('EM')
                if 'EMc' in contours_in_context:
                    contours_in_context.remove('EMc')

                for start2, mark2, end2 in zip(start_inds, mark_inds, end_inds):
                    contour = corpus[mask_file].loc[mark2].contourtype
                    if contour in ['EM','EMc']:
                        # if it is EM or EMc take left subscope
                        for i in range(start2, mark2):
                            if corpus[mask_file].loc[i].n_unit == n_unit:
                                if 'EMp' not in contours_in_context:
                                    contours_in_context.append('EMp')

                        if corpus[mask_file].loc[mark2].n_unit == n_unit:
                                if 'EM' not in contours_in_context:
                                # might be there from mark
                                    contours_in_context.append('EM')

                        # add post emphasis to all
                        for i in range(corpus[mask_file].loc[mark2].n_unit + 1,
                                       corpus[mask_file].loc[end_inds[0]].n_unit + 1):
                                       # till the end of phrase
                            if i == n_unit:
                                assert contour not in contours_in_context  # should not be
                                contours_in_context.append('EMc')

            for contour in contours_in_context:  # this works with duplicate contours
                corpus.loc[start : end, contour] = 1
    return corpus

def add_contour_generator_count(corpus, params):
    '''
    For each rhythmic unit (row in corpus) add number of contour generators of
    each type that is co-occuring.
    '''
    # add context columns
    log = logging.getLogger('add_contcount')
    columns = params.contour_count_columns
    ins_loc = corpus.columns.get_loc('f00')
    log.info('Adding contour count columns ...')
    try:  # if they are not there
        for column in columns[::-1]:
            corpus.insert(ins_loc, column, [0] * len(corpus.index))
    except:
        log.info('Contour count columns already present, clearing them ...')
        corpus[columns] = 0

    # loop through files
    log.info('Adding contour count for all files in corpus ...')
    filenames = corpus.file.unique().tolist()
    for file in filenames:
        mask_file = corpus.file == file
        n_unit_max = int(corpus.loc[mask_file, 'n_unit'].max())
        for n_unit in range(0, n_unit_max + 1):
            mask_n_unit = corpus.n_unit == n_unit  # TODO: more efficient to create masks before
            mask_row = mask_file & mask_n_unit
            contours = corpus[mask_row].contourtype.tolist()
            for contour in contours:
                corpus.loc[mask_row, 'cc' + contour] += 1  # including multiple occurances

    return corpus


def remove_phrase_types(corpus, params):
    """
    Removes phrase types not present in the phrase type list.
    """
    mask = None
    for ptype in corpus.phrasetype.unique():
        if ptype not in params.phrase_types:
            if mask is None:
                mask = corpus.phrasetype == ptype
            else:
                mask = mask | corpus.phrasetype == ptype
    corpus = corpus.loc[~mask]
    corpus = corpus.reset_index(drop=True)
    return corpus


def get_marks(mask_rus, ru_landmark_ind):
    '''
    Generates start, landmark and end marks for a given contour.
    '''
    marks = [''] * mask_rus.size
    if len(marks) == 1:
        marks = ['startendmark']
    else:
        if mask_rus[0] == ru_landmark_ind:
            marks[0] = 'startmark'
            marks[-1] = 'end'
        else:
            marks[0] = 'start'
            if mask_rus[-1] == ru_landmark_ind:
                marks[-1] = 'endmark'  # landmark end
            else:
                marks[np.where(mask_rus == ru_landmark_ind)[0][0]] = 'landmark'
                marks[-1] = 'end'  # landmark end

    return marks


def get_strength(contour_type):
    '''
    Extracts strength from contour_type string if any, else returns 1.
    '''
    if '*' in contour_type:
        contour_type, strength = contour_type.split('*')
        strength = float(strength)
    else:
        strength = 1

    return contour_type, strength


def process_tone_scope(mask_rus, ru_landmark_ind, rus, tone_scope):
    '''
    Changes contour scope to include neighboring syllables based on
    required tone_scope.
    '''
    if tone_scope == 'right':
        if ru_landmark_ind != rus.size - 1:  # not the last syllable
            mask_rus = np.array([ru_landmark_ind, ru_landmark_ind + 1])
        else:
            mask_rus = np.array([ru_landmark_ind])

    elif tone_scope == 'left':
        if ru_landmark_ind != 0:  # not first syllable
            mask_rus = np.array([ru_landmark_ind -1, ru_landmark_ind])
        else:
            mask_rus = np.array([ru_landmark_ind])

    elif tone_scope == 'both':
        if ru_landmark_ind != 0:  # not first syllable
            mask_rus = np.array([ru_landmark_ind -1, ru_landmark_ind])
        else:
            mask_rus = np.array([ru_landmark_ind])
        if ru_landmark_ind != rus.size - 1:  # not the last syllable
            mask_rus = np.r_[mask_rus, np.array([ru_landmark_ind + 1])]

    elif tone_scope == 'no':
        mask_rus = np.array([ru_landmark_ind])

    marks = get_marks(mask_rus, ru_landmark_ind)

    return mask_rus, marks


def split_corpus(corpus, test_size, stratify=True, random_state=None):
    """
    Split corpus into train and test. Can be also used to split train into
    train and validation for analysis by synthesis.
    Two ways to do it:
        - split aware of contour content and makes a stratification of the data
        keeping the original percentage of contours in both sets.
        - using files as groups in GroupsShuffleSplit
    """
    files = corpus.file.tolist()
    filenames = corpus.file.unique()
    file_ind = np.arange(len(filenames))

    if stratify:
        # loop through files and find their contour content
        all_contours = corpus.contourtype.unique().tolist()
        #    file_contours = {}
        file_contours = []
        for file in filenames:
            mask = corpus.file == file
            contours = sorted(corpus[mask].contourtype.unique().tolist())
            contour_code = 0
            for contour in contours:
                ind = all_contours.index(contour)
                # encode in an integer:
                contour_code += 2**ind
            file_contours.append(contour_code)
        file_contours = np.array(file_contours)

        sss = StratifiedShuffleSplit(test_size=test_size,
                                     random_state=random_state)
        train_ind, test_ind = sss.split(file_ind, file_contours).__next__()
        train_filenames = filenames[train_ind]
        test_filenames = filenames[test_ind]

        mask_train = corpus.file.isin(train_filenames)
        mask_test = corpus.file.isin(test_filenames)
        #%
        corpus_train = corpus[mask_train]
        corpus_test = corpus[mask_test]
        assert len(corpus) == len(corpus_train) + len(corpus_test)
        # TODO: make our own algorithm

    else:  # use files as groups
        groups = [np.where(filenames == file)[0] for file in files]
        gss = GroupShuffleSplit(test_size=test_size,
                                random_state=random_state)
        train_ind, test_ind = gss.split(groups, groups, groups).__next__()

    return train_ind, test_ind

def reformat_corpus_static(corpus,
                           params,
                           contours_in_graph=None):  # for baseline
    """
    Reformat the data so it can be used in static graph training.

    Outputs
    =======
    X : ndarray
        shape is (n_samples x n_input x n_modules_in_graph)
        where n_input is [strength + n_syll_ramps + n_context_columns]
    masks : ndarray
        gives the active modules in each row of X
        its shape is (n_samples x n_modules)
    y : ndarray
        targets for each row, shape is (n_samples x n_output)
    contours_in_graph : list
        names of the n_modules
    unique_combinations : dictionary
        maps unique (file, n_unit) combinations to the indexes that correspond
        to the active modules in corpus
    """
    log = logging.getLogger('refrmt_stat')
    log.info('Reformating data from corpus for static graph ...')
    context_columns = params.context_columns
    contour_count_columns = params.contour_count_columns
    n_context = params.n_context
    vowel_pts = params.vowel_pts
    n_output = vowel_pts + 1
    orig_columns = params.orig_columns

    feature_columns = ['strength', 'ramp1', 'ramp2', 'ramp3', 'ramp4']

    corpus_X = corpus.copy().sort_values(['file','n_unit'])
    corpus_X = corpus_X.loc[:, ['file', 'contourtype', 'n_unit'] +
                            feature_columns + context_columns]
# init index
    if contours_in_graph is None:
        max_contour_counts = corpus.loc[:, contour_count_columns].max()
        contours_in_graph = []
        for contour, max_contour in max_contour_counts.iteritems():
            if max_contour == 0:
                pass
            if max_contour == 1:
                contours_in_graph = contours_in_graph + [contour[2:]]
            else:
                for i in range(1, max_contour+1):
                    contours_in_graph = contours_in_graph + ['{}{}'.format(
                                                              contour[2:], i)]
    else:  # get max_contour_counts from provided contours_in_graph
        max_contour_counts = {}
        for item in contours_in_graph:
            contour_type = item[:2]
            contour_count = item[2:]
            if contour_count == '':
                contour_count = 1
            else:
                contour_count = int(contour_count)
            if contour_type in max_contour_counts.keys():  # adjust count
                if max_contour_counts['cc'+contour_type] < contour_count:
                    # should always be True
                    max_contour_counts['cc'+contour_type] = contour_count
            else:
                max_contour_counts['cc'+contour_type] = contour_count


#% fill in data
    corpus_sorted = corpus.copy().sort_values(['file','n_unit'])
    unique_combinations = corpus_sorted.groupby(['file','n_unit']).groups
    corpus_X_new = pd.DataFrame([], columns=contours_in_graph, index=unique_combinations.keys())
    for ind, row in corpus_X.iterrows():
        X_new_ind = (row['file'], row['n_unit'])
        contexts = row[context_columns].tolist()  # .astype('float32')
        features = row[feature_columns].tolist()  # .values.astype('float32')

        # determine column where to go
        contour_type = row['contourtype']
        max_contour = max_contour_counts['cc'+contour_type]
        if max_contour == 1:
            column = contour_type
        else:  # more than 1
            for i in range(1, max_contour+1):
                column = '{}{}'.format(contour_type, i)
                if corpus_X_new.loc[[X_new_ind], column].isnull().values[0]:
                    break

        corpus_X_new.at[X_new_ind, column] = features + contexts

    # create input matrix
    n_samples = len(corpus_X_new)
    n_feats =  1 + 4 + n_context
    n_modules = len(contours_in_graph)  # we'll make it 3D
    X = np.empty((n_samples, n_feats, n_modules), 'float32')
    X.fill(np.nan)
    for i, row in enumerate(corpus_X_new.iterrows()):
        for j, col in enumerate(row[1].tolist()):
            if col is not np.nan:
                X[i, :, j] = np.array(col).astype('float32')

#% create output matrix
    y_target = np.zeros((n_samples, n_output))
    for i, ind in enumerate(corpus_X_new.index):
        ind_in_corpus = unique_combinations[ind][0]
        y_target[i, :] = corpus.loc[ind_in_corpus, orig_columns]

#% create mask
    masks = ~corpus_X_new.isnull()
    masks = masks.values.astype('float32')

    return X, masks, y_target, contours_in_graph, unique_combinations


def reformat_corpus_rnn(corpus, params,
                        contours_in_graph=None):  # for baseline
    """
    Reformat the data so it can be used in static graph training.

    Outputs
    =======
    X : ndarray
        shape is (n_files, seq_len, n_input, n_modules)
        where n_input is [strength + n_syll_ramps + n_context_columns]
    cg_masks : ndarray
        gives the active modules in each row of X
        its shape is (n_files, seq_len, n_modules)
    contour_starts : ndarray
        gives the start of each contour
        its shape is (n_files, seq_len, n_modules)
    y : ndarray
        targets for each row, shape is (n_files, seq_len, n_output)
    contours_in_graph : list
        names of the n_modules
    unique_combinations : dictionary
        maps unique (file, n_unit) combinations to the indexes that correspond
        to the active modules in corpus
    """
    log = logging.getLogger('refrmt_rnn')
    log.info('Reformating data from corpus for RNN graph ...')
    context_columns = params.context_columns
    contour_count_columns = params.contour_count_columns
    n_context = params.n_context
    vowel_pts = params.vowel_pts
    n_output = vowel_pts + 1
    orig_columns = params.orig_columns

    feature_columns = ['strength', 'ramp1', 'ramp2', 'ramp3', 'ramp4']
    n_feature = len(feature_columns)

    corpus_X = corpus.copy().sort_values(['file','n_unit'])
    unique_combinations = corpus_X.groupby(['file','n_unit']).groups
# init index
    if contours_in_graph is None:
        max_contour_counts = corpus.loc[:, contour_count_columns].max()
        contours_in_graph = []
        for contour, max_contour in max_contour_counts.iteritems():
            if max_contour == 0:
                pass

            if max_contour == 1:
                contours_in_graph = contours_in_graph + [contour[2:]]
            else:
                for i in range(1, max_contour+1):
                    contours_in_graph = contours_in_graph + ['{}{}'.format(
                                                              contour[2:], i)]
    else:
        max_contour_counts = {}
        for item in contours_in_graph:
            contour_type = item[:2]
            contour_count = item[2:]
            if contour_count == '':
                contour_count = 1
            else:
                contour_count = int(contour_count)
            if contour_type in max_contour_counts.keys():  # adjust count
                if max_contour_counts['cc'+contour_type] < contour_count:
                    # should always be True
                    max_contour_counts['cc'+contour_type] = contour_count
            else:
                max_contour_counts['cc'+contour_type] = contour_count

    n_modules = len(contours_in_graph)

    # fill in data
    filenames = corpus.file.unique()
    n_files = filenames.size
    seq_len = corpus.n_unit.max() + 1
    X = np.empty([n_files, seq_len, n_feature + n_context, n_modules],
                 'float32')
    X.fill(np.nan)
    cg_masks = np.zeros([n_files, seq_len, n_modules])
    contour_starts = np.zeros([n_files, seq_len, n_modules])
    y = np.zeros([n_files, seq_len, n_output])

    for f_ind, file in enumerate(filenames):
        mask_file = corpus.file == file
        # we have to do starts and ends here - not to mix the continuity!
        marks = corpus[mask_file].marks
        start_inds = marks.index[marks.str.contains('start')].values
        end_inds = marks.index[marks.str.contains('end')].values
        for start, end in zip(start_inds, end_inds):
            contour_type = corpus.loc[start].contourtype
            # determine column where to go
            n_start = corpus.loc[start].n_unit
            max_contour = max_contour_counts['cc'+contour_type]
            if max_contour == 1:
                i = 1
                c_ind = contours_in_graph.index(contour_type)
            else:  # more than 1
                for i in range(1, max_contour+1):
                    column = '{}{}'.format(contour_type, i)
                    c_ind = contours_in_graph.index(column)
                    if np.isnan(X[f_ind, n_start, 0, c_ind]):
                        break
            contour_starts[f_ind, n_start, c_ind] = 1
            for ind in range(start, end+1):
                n_unit = corpus.loc[ind].n_unit
                ys = corpus.loc[ind, orig_columns].values.astype('float32')
                y[f_ind, n_unit, :] = ys

                features = corpus.loc[ind, feature_columns].values.astype('float32')
                contexts = corpus.loc[ind, context_columns].values.astype('float32')
                feat_conts = np.r_[features, contexts]

                X[f_ind, n_unit, :, c_ind] = feat_conts

    # create mask
    cg_masks = ~np.isnan(X)
    cg_masks = cg_masks[:,:,0,:]
    cg_masks = cg_masks.astype('float32')

    return X, cg_masks, contour_starts, y, contours_in_graph, unique_combinations


def generate_ramps(left_max, right_max=0, loop=True):
    """
    Generate ramps as dummy input for contour generators. Used in plotting
    expansion and evaluation of latent space.

    loop controls whether to generate all ramps up to specified scope.
    """
    ramps_all = []
    left_scopes = np.arange(1,left_max+1)
    right_scopes = np.arange(1,right_max+1)

    for left_scope in left_scopes:
        if not loop:
            left_scope = left_max
        ramps_row = []
        if right_max == 0:  # just do left scope
            ramp_global = np.arange(left_scope)[::-1]
            ramp_local = np.c_[ramp_global,
                               np.linspace(10, 0, num=left_scope),
                               ramp_global[::-1]]
            ramps_row = np.c_[ramp_local, ramp_global]
            if not loop:
                break
        else:
            for right_scope in right_scopes:
                if not loop:
                    right_scope = right_max
                # create ramps
                contour_scope = left_scope + right_scope
                ramp_global = np.arange(contour_scope)[::-1]
                unit_scope_1 = np.arange(left_scope)  # including landmark unit
                unit_scope_2 = np.arange(right_scope)
                ramp_local = np.c_[np.r_[unit_scope_1[::-1], unit_scope_2[::-1]],
                                   np.r_[np.linspace(0, 10, num=left_scope)[::-1],
                                         # if there is 1 unit this will favor the 0
                                         np.linspace(0, 10, num=right_scope)],
                                   np.r_[unit_scope_1, unit_scope_2]]

                ramps = np.c_[ramp_local, ramp_global]
                if not loop:
                    ramps_row = ramps
                    break
                ramps_row.append(ramps)

        ramps_all.append(ramps_row)

    if loop:
        return ramps_all
    else:
        return ramps_row
