title: PySFC
author: gerazov
date: 2018-11-03
save_as: pysfc.html
url: pysfc

# PySFC

PySFC is a [Python](https://www.python.org/) implementation of the [SFC model]({filename}project.md#sfc) that was created with two goals: *i*) to make the SFC more accessible to the scientific community, and *ii*) to serve as a foundation for future improvements of the prosody model.
The PySFC also implements a minimum set of tools necessary to make the system self-contained and fully functional.

Python was chosen as an implementation language because of the powerful scientific computing environment that is completely based on free software. It is based on [NumPy](http://www.numpy.org/) within the [SciPy](https://www.scipy.org/) ecosystem. The neural networks and their training have been facilitated through the use of the Multi Layer Perceptron (MLP) regressor in the powerful [scikit-learn](http://scikit-learn.org/stable/index.html) module.
Great attention was put on code readability, which is also one of the features of good Python, augmented with detailed functions docstrings, and comments. The code is segmented in [Spyder](https://pythonhosted.org/spyder/) cells for rapid prototyping. Finally, the whole implementation has been licensed as [free software](http://fsf.org/) with a [GNU General Public License v3](http://www.gnu.org/licenses/). The code can be found on GitHub: <https://github.com/bgerazov/PySFC>

# Modules

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

# Example Plots

Here are a few example plots with PySFC just to show case what it can do. The plotted files are included as examples in the `examples/` directory.

![alt text]({static}../images/r1_DC_393.png)
**Figure 1.** Example PySFC intonation decomposition for the French utterance: *Son bagou pourrait faciliter la communauté.* into constituent functional contours: declaration (DC), dependency to the left/right (DG/DD), and cliticisation (XX, DV).

![alt text]({static}../images/r1_chinese_003.png)
**Figure 2.** Example PySFC intonation decomposition for the Chinese utterance: *Tā men céng zài jī cāng nèi géi lǔ kè diǎn gē hè shēng rì,
céng ná zhē shuí guǒ nái fěn qù tàn wàng yóu tā men zhuǎn sòng qù yī yuàn de lǔ kè chǎn fù.* into constituent functional contours: declaration (DC), tones (C0-4), word boundaries (WB), and independence (ID).

![alt text]({static}../images/r1_expansion_DD.png)
**Figure 3.** Example PySFC expansion in left and right context for the dependency to the right (DD) functional contour, numbers next to the plots show the number of occurences of that scope in the data.

![alt text]({static}../images/r1_losses_DC.png)

**Figure 4.** Example PySFC plots of `f_0` reconstruction losses for all NNCGs for attitude DC per iteration for French.
