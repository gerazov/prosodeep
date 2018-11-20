title: WSFC
author: gerazov
date: 2018-11-03
save_as: wsfc.html
url: wsfc

# WSFC

The Weighted SFC (WSFC) model is an extension to the SFC that can capture the prominence of each functional prototype contour in the final prosody \[[wsfc]({filename}refs.md)\]. It does so through incorporating weighted contour generators (WCGs) in which weights are used to scale the prosodic contours based on their linguistic context, shown in Fig. 1 (all figures taken from \[[wsfc]({filename}refs.md)\]). The WSFC has been shown to be able to successfully capture the impact of attitude and emphasis on prototype prominence.

<img class="center" style="width: 35%;" src="/images/wsfc_wcg.png">
<p class="caption">
**Fig. 1** -- Weighted contour generator introduced in the WSFC
that features the SFC contour generator submodule (left) gated by the
weighting module (right).
</p>

# Modelling prominence

The WSFC has been shown to be able to successfully capture the impact of attitude and emphasis on prototype prominence. In Fig. 2 we can see that the syntactic function contours, while full-blown for the declaration (DC) attitude, are significantly diminished when dor the incredulous question (DI). In Fig. 3 we can see the effect of emphasis (word focus) on increasing the prominence of the tone function contours.

<img class="center" style="width: 60%;" src="/images/wsfc_dc.png">
<img class="center" style="width: 60%;" src="/images/wsfc_di.png">
<p class="caption">
**Fig. 2** -- Decomposition of the melody of the French utterance *“Les gamins coupaient des rondins.”* with the WSFC for declaration (top) and incredulous question (bottom) into constituent functional contours: syntactical dependencies to the left and right (DG, DD), and clitics (XX). Activations of XX, DG and DD are strongly reduced when solicited in the DI context.
</p>

<img class="center" style="width: 90%;" src="/images/wsfc_chen.png">
<p class="caption">
**Fig. 3** -- WSFC decomposition of the intonation of the Chinese utterance: _“Yè Liàng hài pà **Zhào Lì** shuì jiào zuò mèng.”_, into component contours: declaration (DC), tone contour 4 (C4), word boundary (WB), and emphasis (EM).
</p>
