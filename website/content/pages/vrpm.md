title: VRPM
author: gerazov
date: 2018-11-03
save_as: vrpm.html
url: vrpm

# VRPM

The Variational Recurrent Prosody Model (VRPM) \[[vrpm]({filename}refs.md)\] is an extension of the VPM that is built upon variance embedding and recurrent neural network based contour generators (VRCGs), shown in Fig. 1. We use a variational encoder to embed the context-dependent variance in a latent space that is used to initialise a long short term memory (LSTM) recurrent network. The LSTM then uses rhythmic unit positions to generate the prosodic contour.
Like the [VPM]({filename}vpm.md), the VRPM integrates all the VRCGs within a single network architecture and trains them jointly using backpropagation.

<img class="center" style="width: 70%;" src="/images/vrpm_vrcg.png">
<p class="caption">
**Fig. 1** -- Variational recurrent contour generator introduced in the VRPM
that features a variational encoding mapping function context into a prosodic latent space that is sampled to initialise the LSTM to generate the contour. </p>

# Prosodic latent space

The approach used in the VRPM effectively decouples the *prosodic latent space* from the length of the contour's scope. This facilitates the exploration of the latent space even for longer contours, as shown in Fig. 2 for the left-dependency (DG) contour.

<img class="center" style="width: 90%;" src="/images/vrpm_dg.png">
<p class="caption">
**Fig. 2** -- Prosodic latent space of left-dependency function contour (DG) structured based on attitude context with attitude codes same as in Fig. 2; again DC and EX elicit full-blown contours, with EX inducing larger contour prominence. </p>
