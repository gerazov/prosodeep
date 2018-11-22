title: Code
author: gerazov
date: 2018-11-03
save_as: code.html
url: code


# Python ecosystem

[Python](https://www.python.org/) was chosen as an implementation language because of the powerful scientific computing environment that is completely based on free software. The code is built upon [NumPy](http://www.numpy.org/) within the [SciPy](https://www.scipy.org/) ecosystem. The neural network models and their training were implemented in [PyTorch](https://pytorch.org/), which is a powerful deep learning platform centered on Python that allows for rapid model prototyping and easy debugging.
Great attention was put on code readability, which is also one of the features of good Python, augmented with detailed functions docstrings, and comments. The code is segmented in [Spyder](https://pythonhosted.org/spyder/) cells for rapid prototyping.
Other packages used in the code include:

- [scikit-learn](https://scikit-learn.org/) -- for machine learning utility functions,
- [pandas](http://pandas.pydata.org/) -- for data structuring and manipulation,
- [matplotlib](http://matplotlib.org/) -- for result plotting, and
- [seaborn](http://seaborn.pydata.org/) -- for plotting histograms.

Finally, the whole code has been licensed as [free software](http://fsf.org/) with a [GNU General Public License v3](http://www.gnu.org/licenses/). The code can be found on GitHub: <https://github.com/gerazov/prosodeep>

# Modules

All of the ProsoDeep prosody models are implemented within one code base package named `prosodeep`. The package comprises the following modules:

 * `prosodeep.py` -- main module that controls the application of the chosen model to a chosen dataset.
 * `prosodeep_params.py` -- parameter setting module that includes (this list is not exhaustive):
      * model selection -- which type of contour generators to use, and what  training approach to use, i.e. joint backpropagation training or analysis-by-synthesis,
      * model hyperparameters -- number of points to be sampled from the pitch, number of training iterations, number of hidden units, layers, type of activation, learning rate, regularisation parameters, type of optimizer, early stopping, validation and test set ratios,
      * data related parameters -- type of input data, phrase level functional contours, local functional contours,
      * general control parameters -- use of preprocessed data, use of trained models, plotting, saving.
 * `prosodeep_corpus.py` -- holds all the functions that are used to consolidate and work with the corpus of data that is directly fed and output from the SFC model. The corpus is a Pandas data frame object, which allows easy data access and analysis.
 * `prosodeep_data.py` -- comprises functions that read the input data files and calculate the $f_0$ and duration coefficients,
 * `prosodeep_learn.py` -- holds the training functions for backpropagation and analysis_by_synthesis,
 * `prosodeep_model.py` -- holds all of the neural network models used by the various prosody models,
 * `prosodeep_dsp.py` -- holds DSP functions for smoothing the pitch contour based on SciPy,
 * `prosodeep_plot.py` -- holds the plotting functions based on matplotlib and seaborn.

Currently, `prosodeep` supports the standard Praat `TextGrid` annotations, and calculates pitch based on Praat `PointProcess` pitch mark files. We plan to integrate state-of-the-art pitch extractors in the near future, e.g. the [Kaldi](http://kaldi-asr.org/) pitch extractor.


# PySFC

Unlike the `prosodeep` package, the PySFC package, named `sfc` only offers the SFC modelling paradigm. Since it does not use deep learning it is entirely based on scikit-learn for the machine learning.
The PySFC implementation is available as Free Software on GitHub: <https://github.com/bgerazov/PySFC>

The `sfc` package comprises the following modules:

 * `sfc.py` -- main module that controls the application of the SFC model to a chosen dataset.
 * `sfc_params.py` -- parameter setting module that includes:
      * data related parameters -- type of input data, phrase level functional contours, local functional contours,
      * SFC hyperparameters -- number of points to be sampled from the pitch, number of iterations for analysis-by-synthesis, as well as NNCG parameters,
      * SFC execution parameters -- use of preprocessed data, use of trained models, plotting.
 * `sfc_corpus.py` -- holds all the functions that are used to consolidate and work with the corpus of data that is directly fed and output from the SFC model. The corpus is a Pandas data frame object, which allows easy data access and analysis.
 * `sfc_data.py` -- comprises functions that read the input data files and calculate the `f_0` and duration coefficients,
 * `sfc_learn.py` -- holds the SFC training function `analysis_by_synthesis()` and the function for NNCG initialisation,
 * `sfc_dsp.py` -- holds DSP functions for smoothing the pitch contour based on SciPy,
 * `sfc_plot.py` -- holds the plotting functions based on matplotlib and seaborn.

The PySFC supports the proprietary SFC `fpro` file format in addition to the standard Praat `TextGrid` annotations. As in `prosodeep` pitch is calculated based on Praat `PointProcess` pitch mark files.

PySFC also brings added value to the SFC by adding the possibility to adjust the number of samples to be taken from the pitch contour at each rhythmical unit vowel nucleus, and with its extended plotting capabilities for data and performance analysis.
