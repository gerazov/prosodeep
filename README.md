# ProsoDeep
**Deep understanding and modelling of the hierarchical structure of Prosody**

The [ProsoDeep project](https://gerazov.github.io/prosodeep/)
seeks to gain a deeper understanding of the hierarchical structure of the language of prosody through the utilisation of deep models.
The models are designed to facilitate the advanced exploration of prosodic phenomena in lingustics, as well as advancements in speech technologies that rely both on the synthesis of prosody, e.g. expressive text-to-speech (TTS) systems, and its analysis, e.g. automatic speech recognition (ASR) and speech emotion recognition (SER).

# The models

The different models developed within the ProsoDeep project are based on the [Superposition of Functional Contours (SFC) model](https://gerazov.github.io/prosodeep/project#sfc), which is a top-down approach that aims to decompose prosodic contours into their constituent functionally relevant elementary contours, named also *prosodic prototypes* or *clichés* \[[sfc](#SFC)\]. They include the:

  - [PySFC]({filename}pysfc.md) model -- a [Python](https://www.python.org/) implementation of the original SFC model \[[pysfc]({filename}refs.md)\],
  - [Weighted SFC (WSFC)]({filename}wsfc.md) model -- that incorporates the modelling of prominence of the extracted prosodic prototypes \[[wsfc]({filename}refs.md)\],
  - [Variational Prosody Model (VPM)]({filename}vpm.md) -- that models the linguistic conext specific variability of the prosodic prototypes \[[vpm]({filename}refs.md)\], and
  - [Variational Recurrent Prosody Model (VRPM)]({filename}vrpm.md) -- that decouples the context specific variability from function scope \[[vrpm]({filename}refs.md)\].

# Code

The [code]({filename}code.md) implementation for all of the models is available as free software under a [GNU General Public License v3](http://www.gnu.org/licenses/) on GitHub <https://github.com/gerazov/prosodeep>

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
