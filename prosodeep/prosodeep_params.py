#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ProsoDeep - class used to set all parameters.

@authors:
    Branislav Gerazov Nov 2017

Copyright 2017 by GIPSA-lab, Grenoble INP, Grenoble, France.

See the file LICENSE for the licence associated with this software.
"""
import re
import torch


def create_parser(parser):
    # ignore defaults here - if editing next cell
    parser.add_argument("-i", "--ignore",
                        action='store_false',
                        default=True, dest='ignore',
                        help="don't ignore parser arguments")
    # general flow
    parser.add_argument("-d", "--data",
                        choices=['morlec', 'chen','liu'],
                        default='chen', dest='database',
                        help='database used')
    parser.add_argument("-m", "--model",
                        choices=['baseline','baseline_rnn',
                                 'baseline_context','baseline_rnn_context',
                                 'anbysyn', 'deep', 'deep_vae',
                                 'deep_rnn', 'deep_rnn_vae'],
                        default='deep_vae', dest='model_type',
                        help='model arhitecture')
    parser.add_argument("--rnn",
                        choices=['rnn', 'gru', 'lstm'],
                        default='lstm', dest='rnn_model',
                        help='recurrent arhitecture')
    parser.add_argument("--act",
                        choices=['tanh', 'relu'],
                        default='tanh', dest='activation',
                        help='activation of hidden units for MLP,'
                        ' for VE and RNN it is set to tanh')

    # training parameters
    parser.add_argument("-l", "--lr", type=float,
                        default=0.001, dest='learn_rate',
                        help='learning rate')
    parser.add_argument("--iter", type=int,
                        default=3000, dest='iterations',
                        help='epochs of model training in deep learning, '
                        'iterations of loop in anbysyn')
    parser.add_argument("--max_iter", type=int,
                        default=20, dest='max_iter',
                        help='maximum iterations of contour generator training '
                        'in anbysyn')
    parser.add_argument("-r", "--reg", type=float,
                        default=1e-5, dest='l2',
                        help='l2 regularisation')
    parser.add_argument("-b", "--batch",
                        default='256', dest='batch_size',
                        choices='all auto 1 8 16 32 64 128 256'.split(),
                        help='batch size')
    # hidden
    parser.add_argument("--hidden", type=int,
                        nargs='+', default=[17], dest='hidden_units',
                        help='list of hidden layer sizes of contour generator')
    parser.add_argument("--hidden_c", type=int,
                        nargs='+', default=[10], dest='n_hidden_context',
                        help='list of hidden layer sizes of strength module')
    parser.add_argument("--hidden_vae", type=int,
                        nargs='+', default=[10], dest='n_hidden_vae',
                        help='list of hidden layer sizes for variational encoder '
                        'in deep_rnn_vae')
    parser.add_argument("--hidden_rnn", type=int,
                        nargs='+', default=None, dest='n_hidden_rnn',
                        help='list of hidden layer sizes for rnn, if None take '
                        'from --hidden')
    # weighting in WSFC
    parser.add_argument("-s", "--strength",
                        action='store_true',
                        default=False, dest='use_strength',
                        help='use strength weighting in contour generators')
    parser.add_argument("--sreg", type=float,
                        default=0, dest='reg_strengths',
                        help='regularisation of strengths')
    parser.add_argument("--sregmean", type=float,
                        default=0, dest='reg_strengths_mean',
                        help='regularisation of strengths mean')
    parser.add_argument('--sregphrase',
                        action='store_true',
                        default=False, dest='sreg_phrases',
                        help='regularise phrases only')
    # pretraining and freezing
    parser.add_argument("--pre",
                        action='store_true',
                        default=False, dest='use_pretrained_models',
                        help='train (chen) or use (liu) pretrained models')
    parser.add_argument("--freeze",
                        action='store_true',
                        default=False, dest='freeze_pretrained_models',
                        help='freeze pretrained models')
    # tonescope
    parser.add_argument("--tone",
                        choices=['right', 'no', 'left', 'both'],
                        default='right', dest='tone_scope',
                        help='scope of tone functions')
    # optimizer
    parser.add_argument("--optim", choices=['adam', 'adadelta', 'adagrad',
                                            'sparseadam',
                                            'adamax','asgd','sgd','lbfgs',
                                            'rmsprop','rprop',
                                            'sgdr'],
                        default='adam', dest='optimization',
                        help='optimization algorithm')
    # vae
    parser.add_argument("--vaeinput",
                        default='_mmd_context', dest='vae_input',
                        help='input to the VCGs: _ separated '
                        'context, mmd, or kld')
    parser.add_argument("--regvae", type=float,
                        default=0.3, dest='reg_vae',
                        help='regularisation coefficient of VCG loss (KLD, MMD)')
    parser.add_argument("--latdim", type=int,
                        default=2, dest='n_latent',
                        help='latent space dimension')
    parser.add_argument("--vae_as_input",
                        action='store_true',
                        default=False, dest='vae_as_input',
                        help='use latent space sample as SOS in deep_rnn_vae')
    parser.add_argument("--vae_as_hidden",
                        action='store_true',
                        default=False, dest='vae_as_input_hidd',
                        help='use latent space as SOS input AND hidden init')

    parser.add_argument("--device", type=int,
                        default=0, dest='device',
                        help='cuda device to select if multiple available')

    return parser

class Params:
    def __init__(self, args=None):

        #############
#%%     General flow
        #############

        self.load_corpus = True  # load pickled corpus
        self.load_processed_corpus = True
        self.load_eval_corpus = False  # load evaluation data
        self.remove_folders = False  # remove folders if they exist

        #####################
#%%     Prosody model params
        #####################

        self.database = 'chen'  # morlec, chen, liu
        if args is not None and not args.ignore:
            self.database = args.database

        if self.database == 'morlec':
            self.do_all_phrases = True  # if not bunched do you do only one phrase type (DC)
        elif self.database == 'chen':
            self.do_all_phrases = False
        elif self.database == 'liu':
            self.do_all_phrases = True

        self.use_pretrained_models = False # implemented for liu to be based on chen
        self.freeze = False  # freeze pretrained contour generators

        self.tone_scope = 'no'  # chinese tone context in modelling: left, no, right, both
        self.tone_context = None  # use context for Chinese tones: list of tones, all, None

        self.do_all_speakers = False  # for liu
        self.speaker = 'F1'  # do only F1

#        self.vowel_marks=[.1,.5,.9]  # original SFC
        self.vowel_marks=[.1,.3,.5,.7,.9]  # samples at percentage of vowel nucleus
        self.vowel_pts = len(self.vowel_marks)

        # normalisation
        self.normalisation_type = 'minmax'
        self.feat_min = 0.01  # from Merlin
        self.feat_max = 0.99

        #% contour generators params
        self.model_type = 'baseline_rnn'  # model architecture:
        # baseline - a baseline dnn that we feed with the input features
        # baseline_context - context used as part of input features
        # baseline_rnn - a baseline rnn that we feed with the input features
        # baseline_rnn_context - context used as part of input features
        # anbysyn - traditional analysis-by-synthesis like in the SFC
        # deep_dyn - (deprecated) dynamic graphs based on unique context combniations
        # deep - one big graph using masking based on context
        # deep_vae - one big graph using masking with VAEs
        # deep_rnn - one big graph with rnns
        # deep_rnn_vae - one big graph with vae initialised rnns
        self.rnn_model = 'lstm'  # rnn, gru, lstm
#        self.use_layer = True  # use layers in rnn instead of cells
        # always use layers - they're more efficient
        self.rnn_inputs = None  # index of ramp or None to use all ramps
#        self.rnn_inputs = 0  # use only first ramp - rus till landmark/end
        self.n_hidden_rnn = [17]  # the length determines the number of layers
        # but all the layers have n_hidden_rnn[0] hidden units

        self.n_hidden_vae = [10]
        self.vae_as_input = True  # use vae latent sample as input SOS to generate hx
        self.vae_as_input_hidd = False  # use vae sample as input SOS and hx init

        self.baseline_context = False

        # from input args: (this one has to be here)
        if args is not None and not args.ignore:
            self.model_type = args.model_type
            self.rnn_model = args.rnn_model

        if any(s in self.model_type for s in ['vae', 'rnn', 'baseline']) \
                and torch.cuda.is_available():
            self.use_cuda = True
            if args is not None and not args.ignore:
                torch.cuda.set_device(args.device)
        else:
            self.use_cuda = False

        if 'vae' in self.model_type:
            self.vae = True
        else:
            self.vae = False
        self.n_latent = 2
        self.vae_input = '_mmd_context'  # can be: context, kld, mmd
        self.reg_vae = .3  # regularisation coeff of vae loss (kld or mmd)

        self.use_only_last_iteration = True  # not to expand with columns for every epoch
                                             # only valid for deep models

        if 'vae' in self.model_type or 'rnn' in self.model_type:
            self.use_strength = False
        self.n_hidden_context = [10]

        self.use_strength = False
        self.reg_strengths = 0  # regularisation of strengths to be close to 1
        self.reg_strengths_mean = 10.  # regularisation of strengths' means to be close to 1
                                       # variance reducition depends on batch size!
        self.sreg_phrases_only = False  # regularise just the phrases or all contours
                                        # this is needed when using pretrained frozen contours

        self.strength_method = 'context'  # options: manual, context,
                                          # deepcontext is deprecated

        if self.model_type == 'anbysyn':
            self.max_iter = 150  # number of iterations at each stepping
                                 # of the contour generators
            self.iterations = 20  # to run analysis by synthesis
        else:
            self.max_iter = None
            self.iterations = 1  # this is epochs for deep model

        self.batch_size = 8  # all, auto or number,
        self.hidden_units = [17]  # for the contour generators
        self.learn_rate = .001  # learning rate - default for adam is 0.001
        if 'baseline' in self.model_type:
            self.l2 = 1e-4  # 1e-4 default
        else:
            self.l2 = 1e-5  # Merlin

        self.adjust_max_iter = False  # adjust number of iterations
                # so that the total learning steps equals the amount
                # as for a 32 batch size - for batch_size analysis

        self.f0_scale = .05  # originally .05 in the SFC
        self.dur_scale = 10  # 10/.05 = 200 (originally 10)

        self.optimization = 'adam'  # can be 'adam', 'adadelta', 'adagrad', 'sparseadam',
                                    # 'adamax','asgd','sgd','lbfgs','rmsprop','rprop',
                                    # 'sgdr'

        # deprecated
#        self.regularise_compensation = False  # punish mutual compensation of CGs
#        self.reg_comp = 1
        self.activation = 'tanh'  # relu or tanh for MLP. For VE and RNN set to tanh

        self.early_stopping = True
        self.early_thresh = 0.0001  # change in RMS to decrease patience 1e-4
        self.patience = 20  # number of epochs to wait if there is no improvement 20

        if self.early_stopping:
            self.use_validation = True  # split train data into train and val
                                        # for early stopping
                                        # validation only works for early stopping
        else:
            self.use_validation = False

        self.validation_size = 0.1  # percentage of data to keep for validation
        # GroupShuffleSplit also allows here to specify the number of files that
        # are left for the test set

        self.use_test_set = True
        self.test_size = 0.1  # percentage of data to keep for test set, if int
                              # then number of files
        self.test_stratify = False  # TODO: fix stratification

        # build suffix for folders and pickles
#        self.suffix = 'keeptargets'
#        self.suffix = 'maskKL'
        self.suffix = ''
        if self.early_stopping:
            self.suffix += '_early'
            if self.use_validation:
                self.suffix += f'val{self.validation_size}'
        if self.use_test_set:
            self.suffix += f'test{self.test_size}'
            if self.test_stratify:
                self.suffix += 'strat'

        self.suffix += ''


        ###################
#%%     arg params overload
        ###################
        if args is not None and not args.ignore:
            self.use_pretrained_models = args.use_pretrained_models
            self.freeze = args.freeze_pretrained_models
            self.tone_scope = args.tone_scope
            self.use_strength = args.use_strength
            self.n_hidden_context = args.n_hidden_context
            self.reg_strengths = args.reg_strengths
            self.reg_strengths_mean = args.reg_strengths_mean
            self.sreg_phrases_only = args.sreg_phrases
            self.max_iter = args.max_iter
            self.iterations = args.iterations
            if args.batch_size not in ['all','auto']:
                args.batch_size = int(args.batch_size)
            self.batch_size = args.batch_size
            self.hidden_units = args.hidden_units
            self.learn_rate = args.learn_rate
            self.l2 = args.l2
            self.activation = args.activation

            self.optimization = args.optimization
            self.vae_input = args.vae_input
            self.reg_vae =  args.reg_vae
            self.n_latent =  args.n_latent
            self.n_hidden_vae = args.n_hidden_vae
            self.vae_as_input = args.vae_as_input
            self.vae_as_input_hidd = args.vae_as_input_hidd
            if args.n_hidden_rnn is None:
                self.n_hidden_rnn = [self.hidden_units[0]]
            else:
                self.n_hidden_rnn = args.n_hidden_rnn

        #################
#%%     database params
        #################

        self.file_type = 'TextGrid' # from what to build the corpus
        self.phrase_list = ['all']
##########
### morlec data
##########

        if self.database == 'morlec':
            # define context vector input
            self.context_type = 'att'  # attitudes
#            'all'  # any contour apearing within the scope of the current one
#            'att'  # attitudes only
#            'mark'  # landmark context - all that fall on the same landmark
#            'emph'  # emphasis context - pre EMp, on EM, and post EMc emphasis
#            'tone'  # tones only - make context vectors only for tones
#                       (can be combined) with previous two

            self.datapath = '../../data/_modalites/'

            self.all_phrase_types = 'DC QS EX SC EV DI'.split()
            if self.do_all_phrases:
                # these are the phrase types we have
                self.phrase_types = self.all_phrase_types
                #    DC - declaration
                #    QS - question
                #    EX - exclamation
                #    SC - suspicious irony
                #    EV - obviousness
                #    DI - incredulous question
                self.corpus_name = self.database+'_v8_{}pts_{}cont'.format(
                        self.vowel_pts, self.context_type)
            else:
                self.phrase_types = ['DC']  # just do it for DC
                self.corpus_name = self.database+'_v8_{}pts_{}phrase_{}cont'.format(
                        self.vowel_pts, self.phrase_types, self.context_type)

            ## function types in data
            self.function_types = 'DD DG XX DV EM ID IT'.split()
            #    DD - clause on the right depends on the left
            #    DG - clause on the left depends on the right
            #    XX - clitic on the left (les enfants) - downstepping for function words
            #    DV - like XX - downstepping for auxiliaries
            #    ID - independancy (separated by a , )
            #    IT - interdependancy

            self.end_marks = 'XX DV EM'.split()  # contours with only left context
            self.tones = []

            name = self.corpus_name
            name += '_' + self.model_type
            if 'rnn' in self.model_type:
                name += '_rh{}'.format(self.n_hidden_rnn)
                if self.rnn_model != 'rnn':
                    name += '_' + self.rnn_model
                    if self.vae:
                        name += '_vh{}'.format(self.n_hidden_vae)
                        if self.vae_as_input:
                            name += '_vai'
                            if self.vae_as_input_hidd:
                                name += 'ah'
                        name += '_lat{}'.format(self.n_latent)
                        name += '_regvae{}'.format(self.reg_vae)
                    name += '_vh{}'.format(self.n_hidden_vae)
                    name += '_lat{}'.format(self.n_latent)
                    name += '_regvae{}'.format(self.reg_vae)
            elif self.vae:
                name += self.vae_input
                name += '_lat{}'.format(self.n_latent)
                name += '_regvae{}'.format(self.reg_vae)

            name += '_batch{}'.format(self.batch_size)
            if not self.vae and self.use_strength:
                name += '_{}strength'.format(self.strength_method)
                name += '_reg{}_regm{}'.format(
                        self.reg_strengths,
                        self.reg_strengths_mean)
                if self.use_pretrained_models and self.sreg_phrases_only:
                    name += '_sregphrase'

                name += '_hc{}'.format(self.n_hidden_context)

            if self.model_type == 'anbysyn':
                name += '_lr{}_maxit{}_it{}_h{}_l2{}'.format(
                    self.learn_rate, self.max_iter, self.iterations,
                    self.hidden_units, self.l2)
            else:
                name += '_lr{}_it{}_h{}_l2{}'.format(
                    self.learn_rate, self.iterations,
                    self.hidden_units, self.l2)

            if not self.do_all_phrases:
                name += '_phrase{}'.format(self.phrase_types)

            if self.use_pretrained_models:
                name += '_pre'
                if self.freeze:
                    name += 'freeze'

            if self.optimization != 'adam':
                name += self.optimization

            name += self.suffix
            self.processed_corpus_name = name

            if self.use_pretrained_models:
                self.pretrained_models = "pkls/_interspeech_strength_final/" +  "morlec_v7_5pts_anbysyn_batch256_contextstrength_reg0_regm10.0_hc[10]_lr0.001_maxit20_it20_h[17]_l20_phrase['DC'].pkl"

            if self.file_type=='TextGrid':
                self.datafolder = self.datapath + '_grid/'
                self.re_folder = re.compile(r'^.*\d\.TextGrid$')

            self.f0_folder = self.datapath + '_pca/'

##########
### chen
##########

        elif self.database == 'chen':
            if self.use_pretrained_models:  # train pretrained models for liu
                self.context_type = 'emph'
                self.use_pretrained_models = False
            else:
                self.context_type = 'all'
#            'all'  # any contour apearing within the scope of the current one
#            'mark'  # landmark context - all that fall on the same landmark
#            'emph'  # emphasis context - pre EMp, on EM, and post EMc emphasis
#            'tone'  # tones only - make context vectors only for tones
#                       (can be combined) with previous two
            self.datapath = '../../data/chinese/'
            self.ann_tone_scope = 'right'  # in the annotations
            if self.tone_scope is None:
                self.tone_scope = self.ann_tone_scope

            self.corpus_name = self.database +'_v7_{}pts_{}scope'.format(
                    self.vowel_pts, self.tone_scope)
            self.corpus_name += '_{}cont'.format(self.context_type)
            # tone_context eploration
            if self.tone_context is not None:
                self.corpus_name += '_{}tonecont'.format(''.join(self.tone_context))

            self.all_phrase_types = 'DC QS'.split()

            if self.do_all_phrases:
                self.phrase_types = self.all_phrase_types
            else:
                self.phrase_types = ['DC']  # just do it for DC

            ## build the name
            name = self.corpus_name
            name += '_' + self.model_type

            if 'rnn' in self.model_type:
                name += '_rh{}'.format(self.n_hidden_rnn)
                if self.rnn_model != 'rnn':
                    name += '_' + self.rnn_model
                if self.vae:
                    name += '_vh{}'.format(self.n_hidden_vae)
                    if self.vae_as_input:
                        name += '_vai'
                        if self.vae_as_input_hidd:
                                name += 'ah'
                    name += '_lat{}'.format(self.n_latent)
                    name += '_regvae{}'.format(self.reg_vae)

            elif self.vae:
                name += self.vae_input
                name += '_lat{}'.format(self.n_latent)
                name += '_regvae{}'.format(self.reg_vae)

            name += '_batch{}'.format(self.batch_size)
            if not self.vae and self.use_strength:
                name += '_{}strength'.format(self.strength_method)

                name += '_reg{}_regm{}'.format(
                        self.reg_strengths,
                        self.reg_strengths_mean)
                if self.use_pretrained_models and self.sreg_phrases_only:
                    name += '_sregphrase'
                name += '_hc{}'.format(self.n_hidden_context)

            if self.model_type == 'anbysyn':
                name += '_lr{}_maxit{}_it{}_h{}_l2{}'.format(
                    self.learn_rate, self.max_iter, self.iterations,
                    self.hidden_units, self.l2)
            else:
                name += '_lr{}_it{}_h{}_l2{}'.format(
                    self.learn_rate, self.iterations,
                    self.hidden_units, self.l2)

            if not self.do_all_phrases:
                name += '_phrase{}'.format(self.phrase_types)

            if self.use_pretrained_models:
                name += '_pre'
                if self.freeze:
                    name += 'freeze'

            if self.optimization != 'adam':
                name += self.optimization

            name += self.suffix
            self.processed_corpus_name = name

            ## the function types
            self.tones = 'C1 C2 C3 C4'.split()  # used for context
            new_tones = []
            if self.tone_context  is not None:
                if self.tone_context == 'all':
                    self.tone_context = self.tones
                for tone_current in self.tone_context:
                    for tone_previous in self.tones:
                        new_tones += [tone_current + tone_previous]
            self.tones += new_tones
            self.function_types = self.tones + 'WB ID IT'.split()
            #    WB - word boundary

            self.end_marks = ['WB']  # contours with only left context

            if self.file_type=='TextGrid':
                if self.tone_context is None:
                    self.datafolder = self.datapath + '_grid/'
                else:
                    self.datafolder = self.datapath + '_grid/_context/'
                self.re_folder = re.compile(r'^chen_\d*\.TextGrid$')  # eliminate the '_pred' fpros

            self.f0_folder = self.datapath + '_pca/'

##########
### liu data
##########

        elif self.database == 'liu':
            self.context_type = 'tonemarkemph'
#            'mark'  # landmark context - all that fall on the same landmark
#            'emph'  # emphasis context - pre EMp, on EM, and post EMc emphasis
#            'tone'  # tones only - make context vectors only for tones
#                       (can be combined) with previous two

            self.datapath = '../../data/chinese/'
            self.ann_tone_scope = 'right'  # in the annotations
            if self.tone_scope is None:
                self.tone_scope = self.ann_tone_scope

            self.corpus_name = self.database +'_{}_v7_{}pts_{}scope'.format(
                    self.speaker, self.vowel_pts, self.tone_scope)
            self.corpus_name += '_{}cont'.format(self.context_type)
            # tone_context eploration
            if self.tone_context is not None:
                self.corpus_name += '_{}tonecont'.format(''.join(self.tone_context))

            self.all_phrase_types = 'DC QS QI'.split()
            if self.do_all_phrases:
                self.phrase_types = self.all_phrase_types
            else:
                self.phrase_types = ['DC']  # just do it for DC

            ## build the name
            name = self.corpus_name
            name += '_' + self.model_type
            if 'rnn' in self.model_type:
                name += '_rh{}'.format(self.n_hidden_rnn)
                if self.rnn_model != 'rnn':
                    name += '_' + self.rnn_model
                if self.vae:
                    name += '_vh{}'.format(self.n_hidden_vae)
                    if self.vae_as_input:
                        name += '_vai'
                        if self.vae_as_input_hidd:
                                name += 'ah'
                    name += '_lat{}'.format(self.n_latent)
                    name += '_regvae{}'.format(self.reg_vae)

            elif self.vae:
                name += self.vae_input
                name += '_lat{}'.format(self.n_latent)
                name += '_regvae{}'.format(self.reg_vae)

            name += '_batch{}'.format(self.batch_size)
            if not self.vae and self.use_strength:
                name += '_{}strength'.format(self.strength_method)

                name += '_reg{}_regm{}'.format(
                        self.reg_strengths,
                        self.reg_strengths_mean)
                if self.use_pretrained_models and self.sreg_phrases_only:
                    name += '_sregphrase'
                name += '_hc{}'.format(self.n_hidden_context)

            if self.model_type == 'anbysyn':
                name += '_lr{}_maxit{}_it{}_h{}_l2{}'.format(
                    self.learn_rate, self.max_iter, self.iterations,
                    self.hidden_units, self.l2)
            else:
                name += '_lr{}_it{}_h{}_l2{}'.format(
                    self.learn_rate, self.iterations,
                    self.hidden_units, self.l2)

            if not self.do_all_phrases:
                name += '_phrase{}'.format(self.phrase_types)

            if self.use_pretrained_models:
                name += '_pre'
                if self.freeze:
                    name += 'freeze'

            if self.optimization != 'adam':
                name += self.optimization

            name += self.suffix
            self.processed_corpus_name = name

            if self.use_pretrained_models:
                # adapt name from liu to chen
                self.pretrained_models = 'pkls/'
                self.pretrained_models += self.processed_corpus_name.replace('liu_F1','chen')
                self.pretrained_models = self.pretrained_models.replace('_pre',"_phrase['DC']")
                if 'freeze' in self.pretrained_models:
                    self.pretrained_models = self.pretrained_models.replace('freeze',"")
                self.pretrained_models = self.pretrained_models.replace('_sregphrase',"")
                self.pretrained_models = self.pretrained_models.replace('tonemarkemph','emph')
                self.pretrained_models += '.pkl'

            ## function types
            self.function_types = 'C1 C2 C3 C4 WB EM EMc'.split()
            self.tones = 'C1 C2 C3 C4'.split()  # used for context
            self.end_marks = 'WB EM EMc'.split()  # contours with only left context

            if self.file_type == 'TextGrid':
                self.datafolder = self.datapath + '_grid/'
                if not self.do_all_speakers:
                    self.re_folder = re.compile(r'^'+self.speaker+r'_\d*\.TextGrid$')
                else:
                    self.re_folder = re.compile(r'^[FM]\d_\d*\.TextGrid$')

            self.f0_folder = self.datapath + '_pca/'

        ##################
#%%     context
        ##################
        # attitudes have no context by default
        # context types are:
        #    'all'  # any contour apearing within the scope of the current one
        #    'att'  # attitudes only
        #    'emph'  # full emphasis context - pre EMp, on EM, and post EMc emphasis
        #    'mark'  # landmark context - all that fall on the same landmark
        #    'tone'  # tones only - make context vectors only for tones
        #    'dummy' # for pretraining
        # all and att are mutually exclusive
        # emph can be combined with either all or att or mark, or used alone
        # mark can be combined with att
        # mark and all are mutually exclusive
        # mark and emph are not mutually exclusive - mark then applies for the rest i.e. WB

        if 'all' in self.context_type:
            self.context_columns = self.phrase_types + self.function_types
            if 'emph' in self.context_type:
                self.context_columns += 'EMp EMc'.split()

        elif 'att' in self.context_type:
            self.context_columns = self.phrase_types

        elif 'emph' in self.context_type:
            self.context_columns = 'EMp EM EMc'.split()

#        elif self.context_type = 'dummy':
#            self.context_columns = 'EMp EM EMc'.split()

        self.n_context = len(self.context_columns)

        self.contour_count_columns = ['cc' + c for c
                                      in self.phrase_types + self.function_types]
        ##################
#%%     read data params
        ##################
        # read textgrids
        self.disol = 0  # we won't use this
        self.isochrony_gravity = 0.2  # take this as a given
        self.f0_method = 'pitch_marks'
        self.f0_ref_method = 'all'
        # Method to use to accumulate stats if f0_stats is None. Can be:
        #  all - all files
        #  DC - DC files (works for Morlec)
        #  vowels - just vowel segments
        #  vowel_marks - at the vowel marks in the vowel segments

        # stats
        self.f0_stats = self.database + '_f0_stats_all'
        self.dur_stats = self.database + '_phone_dur_stats'
        self.syll_stats = self.database + '_syll_dur_stats'

### morlec data
        if self.database == 'morlec':
            if self.f0_method == 'pitch_marks':
                self.re_f0 = re.compile(r'^.*.PointProcess$')
            elif self.f0_method == 'pca':
                self.re_f0 = re.compile(r'^.*.pca$')

            self.use_ipcgs = True
            self.re_vowels = re.compile('[aeouiyx]')

            self.isochrony_clock = .190
            self.f0_ref = 126
            self.f0_min = 80
            self.f0_max = 300

            # levels in the textgrid - phones should always be 0
            self.syll_level = 1
            self.orthographs_level = 3
            self.phrase_level = 4
            self.tone_levels = None

### chen data
        elif self.database == 'chen':
            if self.f0_method == 'pitch_marks':
                self.re_f0 = re.compile(r'^chen_\d*\.PointProcess$')
            elif self.f0_method == 'pca':
                self.re_f0 = re.compile(r'^chen_\d*\.pca$')

            self.use_ipcgs = False  # use syllables
            self.re_vowels = re.compile('.*[aeouiv].*')
            # v in yu

            self.isochrony_clock = .214

            self.f0_ref = 270
            self.f0_min = 100
            self.f0_max = 450

            # levels in the textgrid - phones should always be 0
            self.syll_level = 1
            self.tone_levels = [2, 3]
            self.orthographs_level = 4
            self.phrase_level = 7

### liu data
        elif self.database == 'liu':
            if not self.do_all_speakers:
                if self.f0_method == 'pitch_marks':
                    self.re_f0 = re.compile(r'^'+self.speaker+r'_\d*\.PointProcess$')
                elif self.f0_method == 'pca':
                    self.re_f0 = re.compile(r'^'+self.speaker+r'_\d*\.pca$')
            else:
                if self.f0_method == 'pitch_marks':
                    self.re_f0 = re.compile(r'^[FM]\d_\d*\.PointProcess$')
                elif self.f0_method == 'pca':
                    self.re_f0 = re.compile(r'^[FM]\d_\d*\.pca$')

            self.f0_stats = self.database + self.speaker + '_f0_stats_all'
            self.dur_stats = self.database + self.speaker + '_phone_dur_stats'
            self.syll_stats = self.database + self.speaker + '_syll_dur_stats'

            self.use_ipcgs = False  # use syllables
            self.re_vowels = re.compile('.*[aeouiv].*')
            # v in yu

            self.isochrony_clock = .193

            self.f0_ref = 277
            self.f0_min = 100
            self.f0_max = 450

            # levels in the textgrid - phones should always be 0
            self.syll_level = 1
            self.tone_levels = [2, 3]
            self.orthographs_level = 4
            self.phrase_level = 7


        ############
#%%     f0 smoothing
        ############
        self.pitch_fs = 200  # every 5 ms
        self.use_median = True
        self.median_order = 5
        self.use_lowpass = False
        self.lowpass_fg = 30
        self.lowpass_order = 8


        ###############
#%%     corpus params
        ###############

        self.columns = \
            'file phrasetype contourtype marks unit n_unit strength ramp1 ramp2 ramp3 ramp4'.split()
        self.columns_in = len(self.columns)
        self.columns += ['f0{}'.format(x) for x in range(len(self.vowel_marks))] + ['dur']
        self.orig_columns = self.columns[self.columns_in:]
        self.target_columns = ['target_'+column for column in self.orig_columns]

        ##########
#%%     evaluation
        ##########

        # for the dataFrame
        self.eval_columns = 'filename ref unit segment weight wrmse wcorr'.split()
        self.eval_sum_columns = 'measure segment ref unit weight mean std'.split()

        self.eval_units = ['semitones']  # possible choices: Hz, semitones, quartertones, log
        self.eval_refs = ['f0_ref']  # possible choices: f0_ref, 1hz
        self.eval_weights = ['none']  # choices: none pov energy both
        self.eval_segments = ['vowels']  # choices: whole vowels voiced
        self.eval_voiced_offset = .005  # duration within which speech is voiced arround each pitch mark

        #######################
#%%     plotting and saving
        #######################
        self.show_plot = False  # used in all plotting - whether to close the plots immediately.
        self.plot_f0s = False  # in sfc_dsp - plots all f0 contours

        self.plot_contours = True # this controls whole block
        if self.database == 'morlec':
            self.plot_n_files = None  # plor first X files, if None plot all
            self.plot_last_n_files = 1  # plor last X files, if None plot all
        elif self.database == 'chen':
            self.plot_n_files = 1  # plor first X files, if None plot all
            self.plot_last_n_files = None # plor last X files, if None plot all
        elif self.database == 'liu':
            self.plot_n_files = None  # plor first X files, if None plot all
            self.plot_last_n_files = None # plor last X files, if None plot all

        self.plot_duration = False

        self.plot_expansion = False  # expansion plots
        self.left_max = 5
        self.right_max = 5
        self.phrase_max = 10

        # evaluation plots
        self.plot_eval = False
        self.plot_eval_n_files = 20  # plor first X files, if None plot all

        # figure save path
        self.save_path = 'figures/{}'.format(self.processed_corpus_name)

        # pkl save path
        self.pkl_path = 'pkls/'

        # save processed corpus data
        self.save_processed_data = False
        self.save_eval_data = False

