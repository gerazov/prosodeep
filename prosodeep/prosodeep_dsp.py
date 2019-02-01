#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ProsoDeep - functions for signal processing.

@authors:
    Branislav Gerazov Nov 2017

Copyright 2019 by GIPSA-lab, Grenoble INP, Grenoble, France.

See the file LICENSE for the licence associated with this software.
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy import signal as sig
import numpy.linalg as linalg

def f0_smooth(barename, pitch_ts, f0s, wav_len, params, show_plot=False, plot_f0s=False, smooth=True):
    '''
    Smooth the f0.

    Parameters
    ==========
    pitch_ts : ndarray
        Pitch marks timepoints.
    f0s : ndarray
        f0s at those timepoints.
    plot : bool
        Plot smoothing results.
    process : bool
        Whether to apply smoothing or just to interpolate (used for synthesis).
    '''
    save_path = params.save_path
    use_median = params.use_median
    median_order = params.median_order
    use_lowpass = params.use_lowpass
    lowpass_fg = params.lowpass_fg
    lowpass_order = params.lowpass_order

    fs = params.pitch_fs
    t = np.arange(0, wav_len, 1/fs)
    interfunc = interp1d(pitch_ts, f0s, kind='linear', bounds_error=False, fill_value='extrapolate')
    f0s_t = interfunc(t)

    if smooth:
### median
        if use_median:
            f0s_t = sig.medfilt(f0s_t, kernel_size=median_order)
### lp filter
        if use_lowpass:
        #    b_iir, a_iir = sig.iirfilter(order, np.array(fl/(fs/2)), btype='lowpass', ftype='butter')
            b_fir = sig.firwin(lowpass_order, lowpass_fg, window='hamming', pass_zero=True, nyq=fs/2)
        #    f0s_t_lp = sig.lfilter(b_iir, a_iir, f0s_t)
            f0s_t = sig.filtfilt(b_fir, [1], f0s_t)
    #    else:
    #        f0s_t_lp = f0s_t_med

        # now find the points you need:
        interfunc_back = interp1d(t, f0s_t, kind='linear', bounds_error=True)
        f0s_smooth = interfunc_back(pitch_ts)

    else:  # just interpolation
        f0s_smooth = None

    # plot
    if show_plot or plot_f0s:
        fig = plt.figure(figsize=(18,8))
        # plot filter response
#        plt.subplot(2,1,1)
    #    w, h_spec = sig.freqz(b_iir, a_iir)
#        w, h_spec = sig.freqz(b_fir, 1)
#        plt.plot(w/np.pi*fs/2,
#                 20*np.log10(np.abs(h_spec)))
#        plt.xlabel('Frequency [Hz]')
#        plt.ylabel('Amplitude [dB]')
#        plt.grid('on')
#        plt.subplot(2,1,2)
        plt.plot(pitch_ts, f0s, 'o', ms=5, c='C0')
        plt.plot(t, f0s_t, c='C0')
        plt.plot(t, f0s_t, alpha=.7, c='C1')
        plt.plot(pitch_ts, f0s_smooth, 'o', ms=5, c='C1')
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.grid('on')

        # save
        plt.savefig('{}/f0s/{}.png'.format(save_path, barename), dpi='figure')

        if not show_plot:
            plt.close(fig)

    return f0s_smooth, t, f0s_t


def wrmse(f01, f02, w=None):
    """
    This calculates weighted RMSE between f01 and f02, using the weight w.

    Parameters
    ----------
    f01 : ndarray
        First signal.
    f02 : ndarray
        Second signal.
    w : ndarray
        weight vector, if None then apply no weighting.
    """
    assert (f01.size == f02.size)
    if w is not None:
        top = np.sum(w * (f01 - f02)**2)
        down = np.sum(w)
    else:
        top = np.sum((f01 - f02)**2)
        down = f01.size  # the mean in RMSE

    wrmse = np.sqrt(top / down)

    return wrmse

def wcorr(f01, f02, w=None, normalise=False):
    """
    This calculates weighted correlation between f01 and f02, using the weight w.

    Parameters
    ----------
    f01 : ndarray
        First signal.
    f02 : ndarray
        Second signal.
    w : ndarray
        weight vector, if None then apply no weighting.
    normalise : bool
        Normalize to 0 mean both signals.
    """
    assert (f01.size == f02.size)  # doesn't have to be but should
    if np.any(np.isnan(f01)):  # should not happen anymore
        f01[np.isnan(f01)] = 0
    if np.any(np.isnan(f02)):  # should not happen anymore
        f02[np.isnan(f02)] = 0

    if normalise:
        f01 = f01 - np.mean(f01)
        f02 = f02 - np.mean(f02)

    if w is not None:
        top = np.sum(w * f01 * f02)
        down1 = np.sum(w * f01**2)
        down2 = np.sum(w * f02**2)
        wcorr = top / np.sqrt(down1*down2)

    else:
        down1 = np.sum(f01**2)
        down2 = np.sum(f02**2)
        wcorr = np.sum(f01 * f02) / np.sqrt(down1*down2)

    return wcorr

def frame_up(fs, sig, size=512, hop=None, pad=True):
    if hop is None:
        hop = size//2

    if pad:
        # This ensures that frames are aligned in the centre
        sig = np.pad(sig, (size//2, size//2), 'edge')

    n_frames = (sig.size - (size-hop)) // hop
    indf = hop * np.arange(n_frames)
    indf = np.matrix(indf)
    indf = indf.transpose()
    indf = np.tile(indf, size)
    ind = indf + np.arange(size)

    frames = sig[ind]  # sooo smooth : )

    t_frames = np.arange(n_frames) * hop / fs
    return t_frames, frames

def get_energy(fs, sig, win_size, hop_size, win, extract='energy'):

    t_frames, frames = frame_up(fs, sig, win_size, hop_size)
    frames = frames * win
    if extract == 'energy':
        energy = np.sum(frames**2, axis=1)
    elif extract == 'amps':
        energy = np.max(np.abs(frames), axis=1)

    return t_frames, energy

def normalise_min_max(corpus, params, input_feats=False,
                      feats_min=None, feats_max=None):
    if input_feats:
        feats = corpus
    else:
        feats = corpus.loc[:, 'ramp1' : 'ramp4'].values
    if feats_min is None:
        feats_min = feats.min(axis=0)
        feats_max = feats.max(axis=0)
    feats_0to1 = (feats - feats_min) / (feats_max - feats_min)
    feats_norm = feats_0to1 * (params.feat_max - params.feat_min)
    feats_norm = feats_norm + params.feat_min

    if input_feats:
        return feats_norm, feats_min, feats_max
    else:
        corpus.loc[:, 'ramp1' : 'ramp4'] = feats_norm
        return corpus, feats_min, feats_max
