# ProsoDeep
**Deep understanding and modelling of the hierarchical structure of Prosody**

The [ProsoDeep project](https://gerazov.github.io/prosodeep/)
seeks to gain a deeper understanding of the hierarchical structure of the language of prosody through the utilisation of deep models.
The models are designed to facilitate the advanced exploration of prosodic phenomena in lingustics, as well as advancements in speech technologies that rely both on the synthesis of prosody, e.g. expressive text-to-speech (TTS) systems, and its analysis, e.g. automatic speech recognition (ASR) and speech emotion recognition (SER).

# The models

The different models developed within the ProsoDeep project are based on the [Superposition of Functional Contours (SFC) model](https://gerazov.github.io/prosodeep/project#sfc), which is a top-down approach that aims to decompose prosodic contours into their constituent functionally relevant elementary contours, named also *prosodic prototypes* or *clichés*  \[[sfc](#References)\]. They include the:

  - [PySFC](https://gerazov.github.io/prosodeep/pysfc) model -- a [Python](https://www.python.org/) implementation of the original SFC model \[[pysfc](#References)\],
  - [Weighted SFC (WSFC)](https://gerazov.github.io/prosodeep/wsfc) model -- that incorporates the modelling of prominence of the extracted prosodic prototypes \[[wsfc](#References)\],
  - [Variational Prosody Model (VPM)](https://gerazov.github.io/prosodeep/vpm) -- that models the linguistic conext specific variability of the prosodic prototypes \[[vpm](#References)\], and
  - [Variational Recurrent Prosody Model (VRPM)](https://gerazov.github.io/prosodeep/vrpm) -- that decouples the context specific variability from function scope \[[vrpm](#References)\].

# Code

The [code](https://gerazov.github.io/prosodeep/code.md) implementation for all of the models is written in [Python](https://www.python.org/) and is available as free software under a [GNU General Public License v3](http://www.gnu.org/licenses/).

The code includes a code base package named `prosodeep` and a main execution script `prosodeep.py`. The script controls the execution flow and carries out data loading, model initialisation, training and evaluation.

## Python ecosystem

[Python](https://www.python.org/) was chosen as an implementation language because of the powerful scientific computing environment that is completely based on free software. The code is built upon [NumPy](http://www.numpy.org/) within the [SciPy](https://www.scipy.org/) ecosystem. The neural network models and their training were implemented in [PyTorch](https://pytorch.org/), which is a powerful deep learning platform centered on Python that allows for rapid model prototyping and easy debugging.
Great attention was put on code readability, which is also one of the features of good Python, augmented with detailed functions docstrings, and comments. The code is segmented in [Spyder](https://pythonhosted.org/spyder/) cells for rapid prototyping.
Other packages used in the code include:

- [scikit-learn](https://scikit-learn.org/) -- for machine learning utility functions,
- [pandas](http://pandas.pydata.org/) -- for data structuring and manipulation,
- [matplotlib](http://matplotlib.org/) -- for result plotting, and
- [seaborn](http://seaborn.pydata.org/) -- for plotting histograms.

Finally, the whole code has been licensed as [free software](http://fsf.org/) with a [GNU General Public License v3](http://www.gnu.org/licenses/). The code can be found on GitHub: <https://github.com/gerazov/prosodeep>

## Modules

 The `prosodeep` package comprises the following modules:
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

The PySFC implementation can be found at <https://github.com/gerazov/pysfc>

# Acknowledgement

This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the [Marie Skłodowska-Curie Actions (MSCA)](http://ec.europa.eu/research/mariecurieactions/) grant agreement No 745802: “ProsoDeep: Deep understanding and modelling of the hierarchical structure of Prosody”.

# References

## ProsoDeep

- `[vrpm]` Gerazov Branislav, Gérard Bailly, Omar Mohammed, Yi Xu, and Philip N. Garner, “Embedding Context-Dependent Variations of Prosodic Contours using Variational Encoding for Decomposing the Structure of Speech Prosody,” Workshop on Prosody and Meaning: Information Structure and Beyond, Aix-en-Provence, France, 8 November 2018. \[[pdf](https://hal.archives-ouvertes.fr/hal-01927872/document)\]
- `[vpm]` Gerazov Branislav, Gérard Bailly, Omar Mohammed, Yi Xu, and Philip N. Garner, “A Variational Prosody Model for the decomposition and synthesis of speech prosody,” in ArXiv e-prints, 22 June 2018. <https://arxiv.org/abs/1806.08685>
- `[wsfc]` Gerazov Branislav, Gérard Bailly, and Yi Xu, “A Weighted Superposition of Functional Contours model for modelling contextual prominence of elementary prosodic contours,” in Proceedings of Interspeech, Hyderabad, India, 02 – 07 Sep, 2018. \[[pdf](https://www.isca-speech.org/archive/Interspeech_2018/pdfs/1286.pdf)\]
- `[pysfc-tones]` Gerazov Branislav, Gérard Bailly, and Yi Xu, “The significance of scope in modelling tones in Chinese,” in Tonal Aspects of Languages, Berlin, Germany, 18 – 20 Jun, 2018. \[[pdf](http://public.beuth-hochschule.de/~mixdorff/tal2018/180620_poster_session/TAL_2018_paper_10.pdf)\]
- `[pysfc]` Gerazov Branislav and Gérard Bailly, “PySFC – A System for Prosody Analysis based on the Superposition of Functional Contours Prosody Model,” in Speech Prosody, Poznan, Poland, 13 – 16 June, 2018. \[[pdf](https://hal.archives-ouvertes.fr/hal-01821214/document)\]

## SFC
- `[sfc]` Bailly, Gérard, and Bleicke Holm, “SFC: a trainable prosodic model,” Speech communication 46, no. 3: 348-364, 2005. \[[pdf](https://hal.archives-ouvertes.fr/hal-00416724/document)\]
- `[sfc-av]` Barbulescu Adela, Rémi Ronfard, and Gérard Bailly, “Exercises in Speaking Style: A Generative Audiovisual Prosodic Model for Virtual Actors,” Computer Graphics Forum, 37-6:40-51, 2017.
\[[pdf](https://hal.inria.fr/hal-01643334/document)\]

## GCR
- `[gcr]` Honnet, Pierre-Edouard, Branislav Gerazov, Aleksandar Gjoreski, and Philip N. Garner, “Intonation modelling using a muscle model and perceptually weighted matching pursuit,” Speech Communication, 97:81--93, March 2018. \[[pdf](https://infoscience.epfl.ch/record/233571/files/Honnet_SPECOM_2018.pdf)\]
- `[gcr-wcad]` Gerazov, Branislav, Pierre-Edouard Honnet, Aleksandar Gjoreski, Philip N. Garner, “Weighted correlation based atom decomposition intonation
modelling,” in Proceedings of Interspeech, Dresden, Germany, 6 -- 10 September, 2015.
\[[pdf](https://www.isca-speech.org/archive/interspeech_2015/papers/i15_1601.pdf)\]
