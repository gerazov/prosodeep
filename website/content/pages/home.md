title: ProsoDeep
date: 2018-11-03
url:
save_as: index.html

# ProsoDeep
**Deep understanding and modelling of the hierarchical structure of Prosody**

The [ProsoDeep project]({filename}project.md)
seeks to gain a deeper understanding of the hierarchical structure of the language of prosody through the utilisation of deep models.
The results will facilitate the advancement of speech technologies that rely both on the synthesis of prosody, e.g. text-to-speech (TTS) systems, and its analysis, e.g. speech recognition and speech emotion recognition (SER).

# The models

The different models developed within the ProsoDeep project are based on the [Superposition of Functional Contours (SFC) model]({filename}project.md#sfc), which is a top-down approach based on the decomposition of prosodic contours into functionally relevant elementary contours, named also *prosodic prototypes* or *clichés* \[[sfc]({filename}refs.md)\]. They include the:

  - [PySFC]({filename}pysfc.md) model -- a [Python](https://www.python.org/) implementation of the original SFC model \[[pysfc]({filename}refs.md)\],
  - [Weighted SFC (WSFC)]({filename}wsfc.md) model -- that incorporates the modelling of prominence of the extracted prosodic prototypes \[[wsfc]({filename}refs.md)\],
  - [Variational Prosody Model (VPM)]({filename}vpm.md) -- that models the linguistic conext specific variability of the prosodic prototypes \[[vpm]({filename}refs.md)\], and
  - [Variational Recurrent Prosody Model (VRPM)]({filename}vrpm.md) -- that decouples the context specific variability from function scope \[[vrpm]({filename}refs.md)\].

# Code

The [code]({filename}code.md) implementation for all of the models is available as free software under a [GNU General Public License v3](http://www.gnu.org/licenses/) on GitHub <https://github.com/gerazov/prosodeep>

The PySFC implementation can be found at <https://github.com/gerazov/pysfc>

# Acknowledgement

This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the [Marie Skłodowska-Curie Actions (MSCA)](http://ec.europa.eu/research/mariecurieactions/) grant agreement No 745802: “ProsoDeep: Deep understanding and modelling of the hierarchical structure of Prosody”.
