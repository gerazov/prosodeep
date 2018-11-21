title: Code
author: gerazov
date: 2018-11-03
save_as: code.html
url: code

# PySFC

GitHub link: <https://github.com/bgerazov/PySFC>

PySFC comprises the following modules:

 * `sfc.py` - main module that controls the application of the SFC model to a chosen dataset.
 * `sfc_params.py` - parameter setting module that includes:
      * data related parameters - type of input data, phrase level functional contours, local functional contours,
      * SFC hyperparameters - number of points to be sampled from the pitch, number of iterations for analysis-by-synthesis, as well as NNCG parameters,
      * SFC execution parameters - use of preprocessed data, use of trained models, plotting.
 * `sfc_corpus.py` - holds all the functions that are used to consolidate and work with the corpus of data that is directly fed and output from the SFC model. The corpus is a [Pandas](http://pandas.pydata.org/) data frame object, which allows easy data access and analysis.
 * `sfc_data.py` - comprises functions that read the input data files and calculate the `f_0` and duration coefficients,
 * `sfc_learn.py` - holds the SFC training function `analysis_by_synthesis()` and the function for NNCG initialisation,
 * `sfc_dsp.py` - holds DSP functions for smoothing the pitch contour based on SciPy,
 * `sfc_plot.py` - holds the plotting functions based on [matplotlib](http://matplotlib.org/) and [seaborn](http://seaborn.pydata.org/).

Currently, PySFC supports the proprietary SFC `fpro` file format as well as standard Praat `TextGrid` annotations. Pitch is calculated based on Praat `PointProcess` pitch mark files, but integration of state-of-the-art pitch extractors is planned for the future.
PySFC also brings added value, by adding the possibility to adjust the number of samples to be taken from the pitch contour at each rhythmical unit vowel nucleus, and with its extended plotting capabilities for data and performance analysis.
