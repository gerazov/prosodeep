title: VRPM
author: gerazov
date: 2018-11-03
save_as: vrpm.html
url: vrpm

# VRPM

The Variational Recurrent Prosody Model (VRPM) is an extension of the VPM that is built upon variance embedding and recurrent neural network based contour generators (VRCGs). We use a variational encoder to embed the context-dependent variance in a latent space that is used to initialise a long short term memory (LSTM) recurrent network. The LSTM then uses rhythmic unit positions to generate the prosodic contour. This approach decouples the *prosodic latent space* from the length of the contour's scope, thus it can now be readily explored even for longer contours.

<!-- Fig.~\ref{fig:dg} shows the embedded variance in the prosodic latent space of the left-dependency contour solicited in 6 different attitudes. We can clearly see that the declaration and especially exclamation attitudes give a full contour realisation, while the other induce its suppression.  -->
