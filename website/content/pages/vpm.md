title: VPM
author: gerazov
date: 2018-11-03
save_as: vpm.html
url: vpm

# VPM

To Variational Prosody Model (VPM) is able to capture a part of the prosodic prototype variance \[[vpm]({filename}refs.md)\]. Its variational CGs (VCGs) shown in Fig. 1 (all figures taken from \[[vpm]({filename}refs.md)\]), use the linguistic context input to map out a *prosodic latent* space for each contour.

<img class="center" style="width: 25%;" src="images/vpm_vcg.png">
<p class="caption">
**Fig. 1** -- Variational contour generator introduced in the VPM
that features a variational encoding mapping function context and rhythmic unit position into a prosodic latent space. </p>

Unlike the [SFC]({filename}pysfc.md), which uses analysis-by-synthesis, the VPM integrates all the VCGs within a single network architecture shown in Fig. 2, and trains them jointly using backpropagation. This eliminates the ad hoc distribution of errors and leads to better modelling performance.

<img class="center" style="width: 45%;" src="images/vpm_arch.png">
<p class="caption">
**Fig. 2** -- VPM architecture comprising VCGs for each function and allowing their joint training. </p>

# Prosodic latent space

The mapped two-dimensional latent space can be used to visualise the captured context-specific variation, shown in Figs. 3 and 4. Since the VCGs are still based on synthesising the contours based on rhythmic unit position input, the mapped prosodic latent space is amenable for exploration only for short contours, such as Chinese tones or clitics.

<img class="center" style="width: 55%;" src="images/vpm_lat_xx.png">
<p class="caption">
**Fig. 3** -- The prosodic latent space of the clitic functional contour (XX) demonstrates that a full blown contour is only generated in context of the declaration (DC) and exclamation (EX) attitudes, while it is largely diminished for the rest. </p>
<img class="center" style="width: 56%;" src="images/vpm_lat_t3.png">
<p class="caption">
**Fig. 4** -- Prosodic latent space of the tone 3 functional contour in context of no-, pre-, on- and post-emphasis (None, EMp, EM, and EMc), as extracted by the VPM. </p>
