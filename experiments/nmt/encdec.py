import numpy
import logging
import pprint
import operator
import itertools

import theano
import theano.tensor as TT
from theano.ifelse import ifelse
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from groundhog.layers import\
        Layer,\
        MultiLayer,\
        SoftmaxLayer,\
        HierarchicalSoftmaxLayer,\
        LSTMLayer, \
        RecurrentLayer,\
        RecursiveConvolutionalLayer,\
        UnaryOp,\
        Shift,\
        LastState,\
        DropOp,\
        Concatenate
from groundhog.models import LM_Model
from groundhog.datasets import PytablesBitextIterator
from groundhog.utils import sample_zeros, sample_weights_orth, init_bias, sample_weights_classic
import groundhog.utils as utils

logger = logging.getLogger(__name__)

def create_padded_batch(state, x, y, return_dict=False):
    """A callback given to the iterator to transform data in suitable format

    :type x: list
    :param x: list of numpy.array's, each array is a batch of phrases
        in some of source languages

    :type y: list
    :param y: same as x but for target languages

    :param new_format: a wrapper to be applied on top of returned value

    :returns: a tuple (X, Xmask, Y, Ymask) where
        - X is a matrix, each column contains a source sequence
        - Xmask is 0-1 matrix, each column marks the sequence positions in X
        - Y and Ymask are matrices of the same format for target sequences
        OR new_format applied to the tuple

    Notes:
    * actually works only with x[0] and y[0]
    * len(x[0]) thus is just the minibatch size
    * len(x[0][idx]) is the size of sequence idx
    """

    mx = state['seqlen']
    my = state['seqlen']
    if state['trim_batches']:
        # Similar length for all source sequences
        mx = numpy.minimum(state['seqlen'], max([len(xx) for xx in x[0]]))+1
        # Similar length for all target sequences
        my = numpy.minimum(state['seqlen'], max([len(xx) for xx in y[0]]))+1

    # Batch size
    n = x[0].shape[0]

    X = numpy.zeros((mx, n), dtype='int64')
    Y = numpy.zeros((my, n), dtype='int64')
    Xmask = numpy.zeros((mx, n), dtype='float32')
    Ymask = numpy.zeros((my, n), dtype='float32')

    # Fill X and Xmask
    for idx in xrange(len(x[0])):
        # Insert sequence idx in a column of matrix X
        if mx < len(x[0][idx]):
            X[:mx, idx] = x[0][idx][:mx]
        else:
            X[:len(x[0][idx]), idx] = x[0][idx][:mx]

        # Mark the end of phrase
        if len(x[0][idx]) < mx:
            X[len(x[0][idx]):, idx] = state['null_sym_source']

        # Initialize Xmask column with ones in all positions that
        # were just set in X
        Xmask[:len(x[0][idx]), idx] = 1.
        if len(x[0][idx]) < mx:
            Xmask[len(x[0][idx]), idx] = 1.

    # Fill Y and Ymask in the same way as X and Xmask in the previous loop
    for idx in xrange(len(y[0])):
        Y[:len(y[0][idx]), idx] = y[0][idx][:my]
        if len(y[0][idx]) < my:
            Y[len(y[0][idx]):, idx] = state['null_sym_target']
        Ymask[:len(y[0][idx]), idx] = 1.
        if len(y[0][idx]) < my:
            Ymask[len(y[0][idx]), idx] = 1.

    null_inputs = numpy.zeros(X.shape[1])

    # We say that an input pair is valid if both:
    # - either source sequence or target sequence is non-empty
    # - source sequence and target sequence have null_sym ending
    # Why did not we filter them earlier?
    for idx in xrange(X.shape[1]):
        if numpy.sum(Xmask[:,idx]) == 0 and numpy.sum(Ymask[:,idx]) == 0:
            null_inputs[idx] = 1
        if Xmask[-1,idx] and X[-1,idx] != state['null_sym_source']:
            null_inputs[idx] = 1
        if Ymask[-1,idx] and Y[-1,idx] != state['null_sym_target']:
            null_inputs[idx] = 1

    valid_inputs = 1. - null_inputs

    # Leave only valid inputs
    X = X[:,valid_inputs.nonzero()[0]]
    Y = Y[:,valid_inputs.nonzero()[0]]
    Xmask = Xmask[:,valid_inputs.nonzero()[0]]
    Ymask = Ymask[:,valid_inputs.nonzero()[0]]
    if len(valid_inputs.nonzero()[0]) <= 0:
        return None

    # Unknown words
    X[X >= state['n_sym_source']] = state['unk_sym_source']
    Y[Y >= state['n_sym_target']] = state['unk_sym_target']

    if return_dict:
        return {'x' : X, 'x_mask' : Xmask, 'y': Y, 'y_mask' : Ymask}
    else:
        return X, Xmask, Y, Ymask

def get_batch_iterator(state):

    class Iterator(PytablesBitextIterator):

        def __init__(self, *args, **kwargs):
            PytablesBitextIterator.__init__(self, *args, **kwargs)
            self.batch_iter = None
            self.peeked_batch = None

        def get_homogenous_batch_iter(self):
            while True:
                k_batches = state['sort_k_batches']
                batch_size = state['bs']
                data = [PytablesBitextIterator.next(self) for k in range(k_batches)]
                x = numpy.asarray(list(itertools.chain(*map(operator.itemgetter(0), data))))
                y = numpy.asarray(list(itertools.chain(*map(operator.itemgetter(1), data))))
                lens = numpy.asarray([map(len, x), map(len, y)])
                order = numpy.argsort(lens.max(axis=0)) if state['sort_k_batches'] > 1 \
                        else numpy.arange(len(x))
                for k in range(k_batches):
                    indices = order[k * batch_size:(k + 1) * batch_size]
                    batch = create_padded_batch(state, [x[indices]], [y[indices]],
                            return_dict=True)
                    if batch:
                        yield batch

        def next(self, peek=False):
            if not self.batch_iter:
                self.batch_iter = self.get_homogenous_batch_iter()

            if self.peeked_batch:
                # Only allow to peek one batch
                assert not peek
                logger.debug("Use peeked batch")
                batch = self.peeked_batch
                self.peeked_batch = None
                return batch

            if not self.batch_iter:
                raise StopIteration
            batch = next(self.batch_iter)
            if peek:
                self.peeked_batch = batch
            return batch

    train_data = Iterator(
        batch_size=int(state['bs']),
        target_file=state['target'][0],
        source_file=state['source'][0],
        can_fit=False,
        queue_size=1000,
        shuffle=state['shuffle'],
        use_infinite_loop=state['use_infinite_loop'],
        max_len=state['seqlen'])
    return train_data

class RecurrentLayerWithSearch(Layer):
    """A copy of RecurrentLayer from groundhog"""

    def __init__(self, rng,
                 n_hids,
                 c_dim=None,
                 # cleaned by Zhaopeng Tu, 2016-01-28
                 # gather all the options
                 state=None,
                 # ==================================
                 scale=.01,
                 activation=TT.tanh,
                 bias_fn='init_bias',
                 bias_scale=0.,
                 init_fn='sample_weights',
                 gating=False,
                 reseting=False,
                 dropout=1.,
                 gater_activation=TT.nnet.sigmoid,
                 reseter_activation=TT.nnet.sigmoid,
                 weight_noise=False,
                 name=None):
        logger.debug("RecurrentLayerWithSearch is used")

        self.grad_scale = 1
        assert gating == True
        assert reseting == True
        assert dropout == 1.
        assert weight_noise == False
        updater_activation = gater_activation

        if type(init_fn) is str or type(init_fn) is unicode:
            init_fn = eval(init_fn)
        if type(bias_fn) is str or type(bias_fn) is unicode:
            bias_fn = eval(bias_fn)
        if type(activation) is str or type(activation) is unicode:
            activation = eval(activation)
        if type(updater_activation) is str or type(updater_activation) is unicode:
            updater_activation = eval(updater_activation)
        if type(reseter_activation) is str or type(reseter_activation) is unicode:
            reseter_activation = eval(reseter_activation)
        
        self.scale = scale
        self.activation = activation
        self.n_hids = n_hids
        self.bias_scale = bias_scale
        self.bias_fn = bias_fn
        self.init_fn = init_fn
        self.updater_activation = updater_activation
        self.reseter_activation = reseter_activation
        self.c_dim = c_dim

        # added by Zhaopeng Tu, 2016-01-28
        self.state = state

        assert rng is not None, "random number generator should not be empty!"

        super(RecurrentLayerWithSearch, self).__init__(self.n_hids,
                self.n_hids, rng, name)

        self.params = []
        self._init_params()

    def _init_params(self):
        self.W_hh = theano.shared(
                self.init_fn(self.n_hids,
                self.n_hids,
                -1,
                self.scale,
                rng=self.rng),
                name="W_%s"%self.name)
        self.params = [self.W_hh]
        self.G_hh = theano.shared(
                self.init_fn(self.n_hids,
                self.n_hids,
                -1,
                self.scale,
                rng=self.rng),
                name="G_%s"%self.name)
        self.params.append(self.G_hh)
        self.R_hh = theano.shared(
                self.init_fn(self.n_hids,
                self.n_hids,
                -1,
                self.scale,
                rng=self.rng),
                name="R_%s"%self.name)
        self.params.append(self.R_hh)
        self.A_cp = theano.shared(
                sample_weights_classic(self.c_dim,
                    self.n_hids,
                    -1,
                    10 ** (-3),
                    rng=self.rng),
                name="A_%s"%self.name)
        self.params.append(self.A_cp)
        self.B_hp = theano.shared(
                sample_weights_classic(self.n_hids,
                    self.n_hids,
                    -1,
                    10 ** (-3),
                    rng=self.rng),
                name="B_%s"%self.name)
        self.params.append(self.B_hp)
        self.D_pe = theano.shared(
                numpy.zeros((self.n_hids, 1), dtype="float32"),
                name="D_%s"%self.name)
        self.params.append(self.D_pe)

        # added by Zhaopeng Tu, 2015-10-29
        if self.state.get('maintain_coverage', False):
            self.C_covp = theano.shared(
                    sample_weights_classic(self.state['coverage_dim'],
                        self.n_hids,
                        -1,
                        10 ** (-3),
                        rng=self.rng),
                    name="C_%s"%self.name)
            self.params.append(self.C_covp)

            # added by Zhaopeng Tu, 2015-11-10
            # for coverage model II,  z_t = f (z_{t-1}, a_{t-1}, s_{t-1}, h)
            # where a denotes alignment scores, s denotes state before, and h denotes annotation of input sentence (in this code, is c)
            # a:  (source_length, target_num)     target_num == source_num
            # s:  (target_num, dim)
            # h:  (source_num, c_dim)          c:  (source_length, source_num, c_dim)
            # z:  (source_length, source_num, cov_dim)
            # the following parameters are for the above function
            if self.state.get('use_recurrent_coverage', False):
                self.W_cc = theano.shared(
                        self.init_fn(self.state['coverage_dim'],
                        self.state['coverage_dim'],
                        -1,
                        self.scale,
                        rng=self.rng),
                        name="Cov_W_%s"%self.name)
                self.params.append(self.W_cc)
                
                # modified by Zhaopeng Tu, 2016-01-19
                # for alignment probabilities
                if self.state.get('use_probability_for_recurrent_coverage', False):
                    self.Cov_inputer_p = theano.shared(
                            numpy.zeros((1, self.state['coverage_dim']), dtype="float32"),
                            name="Cov_inputer_p_%s"%self.name)
                    self.params.append(self.Cov_inputer_p)
                
                # for input annotations
                if self.state.get('use_input_annotations_for_recurrent_coverage', False):
                    self.Cov_inputer_c = theano.shared(
                            sample_weights_classic(self.c_dim,
                                self.state['coverage_dim'],
                                -1,
                                10 ** (-3),
                                rng=self.rng),
                            name="Cov_inputer_c_%s"%self.name)
                    self.params.append(self.Cov_inputer_c)

                # for decoding states
                if self.state.get('use_decoding_state_for_recurrent_coverage', False):
                    self.Cov_inputer_h = theano.shared(
                            sample_weights_classic(self.n_hids,
                                self.state['coverage_dim'],
                                -1,
                                10 ** (-3),
                                rng=self.rng),
                            name="Cov_inputer_h_%s"%self.name)
                    self.params.append(self.Cov_inputer_h)

                if self.state.get('use_recurrent_gating_coverage', False):
                    self.G_cc = theano.shared(
                            self.init_fn(self.state['coverage_dim'],
                                self.state['coverage_dim'],
                                -1,
                                self.scale,
                                rng=self.rng),
                            name="Cov_G_%s"%self.name)
                    self.params.append(self.G_cc)
                    self.R_cc = theano.shared(
                            self.init_fn(self.state['coverage_dim'],
                                self.state['coverage_dim'],
                                -1,
                                self.scale,
                                rng=self.rng),
                            name="Cov_R_%s"%self.name)
                    self.params.append(self.R_cc)

                    # for alignment probabilities, the necessary input for coverage
                    # modified by Zhaopeng Tu, 2016-01-20
                    if self.state.get('use_probability_for_recurrent_coverage', False):
                        self.Cov_updater_p = theano.shared(
                                numpy.zeros((1, self.state['coverage_dim']), dtype="float32"),
                                name="Cov_updater_p_%s"%self.name)
                        self.params.append(self.Cov_updater_p)
                        self.Cov_reseter_p = theano.shared(
                                numpy.zeros((1, self.state['coverage_dim']), dtype="float32"),
                                name="Cov_reseter_p_%s"%self.name)
                        self.params.append(self.Cov_reseter_p)
     
                    # for input annotations
                    if self.state.get('use_input_annotations_for_recurrent_coverage', False):
                        self.Cov_updater_c = theano.shared(
                                sample_weights_classic(self.c_dim,
                                    self.state['coverage_dim'],
                                    -1,
                                    10 ** (-3),
                                    rng=self.rng),
                                name="Cov_updater_c_%s"%self.name)
                        self.params.append(self.Cov_updater_c)
                        self.Cov_reseter_c = theano.shared(
                                sample_weights_classic(self.c_dim,
                                    self.state['coverage_dim'],
                                    -1,
                                    10 ** (-3),
                                    rng=self.rng),
                                name="Cov_reseter_c_%s"%self.name)
                        self.params.append(self.Cov_reseter_c)

                    # for decoding states
                    if self.state.get('use_decoding_state_for_recurrent_coverage', False):
                        self.Cov_updater_h = theano.shared(
                                sample_weights_classic(self.n_hids,
                                    self.state['coverage_dim'],
                                    -1,
                                    10 ** (-3),
                                    rng=self.rng),
                                name="Cov_updater_h_%s"%self.name)
                        self.params.append(self.Cov_updater_h)
                        self.Cov_reseter_h = theano.shared(
                                sample_weights_classic(self.n_hids,
                                    self.state['coverage_dim'],
                                    -1,
                                    10 ** (-3),
                                    rng=self.rng),
                                name="Cov_reseter_h_%s"%self.name)
                        self.params.append(self.Cov_reseter_h)


        self.params_grad_scale = [self.grad_scale for x in self.params]

       
    def set_decoding_layers(self, c_inputer, c_reseter, c_updater):
        self.c_inputer = c_inputer
        self.c_reseter = c_reseter
        self.c_updater = c_updater
        for layer in [c_inputer, c_reseter, c_updater]:
            self.params += layer.params
            self.params_grad_scale += layer.params_grad_scale
        
    def set_coverage_decoding_layers(self, cov_inputer, cov_reseter, cov_updater):
        self.cov_inputer = cov_inputer
        self.cov_reseter = cov_reseter
        self.cov_updater = cov_updater
        for layer in [cov_inputer, cov_reseter, cov_updater]:
            self.params += layer.params
            self.params_grad_scale += layer.params_grad_scale

    def set_fertility_layers(self, fertility_inputer):
        self.fertility_inputer = fertility_inputer
        self.params += fertility_inputer.params
        self.params_grad_scale += fertility_inputer.params_grad_scale
    
    def coverage_updater(self, 
                           coverage_before, 
                           probs,
                           c,
                           cndim,
                           state_before=None,
                           # added by Zhaopeng Tu, 2015-11-11
                           given_cov_state_below=None,
                           given_cov_gater_below=None,
                           given_cov_reseter_below=None,
                           # added by Zhaopeng Tu, 2015-12-16
                           fertility=None):
        # added by Zhaopeng Tu, 2015-11-10
        if self.state.get('use_recurrent_coverage', False):
            # for coverage
            W_cc = self.W_cc
            if self.state.get('use_probability_for_recurrent_coverage', False):
                Cov_inputer_p = self.Cov_inputer_p
            if self.state.get('use_input_annotations_for_recurrent_coverage', False):
                Cov_inputer_c = self.Cov_inputer_c
            if self.state.get('use_decoding_state_for_recurrent_coverage', False):
                Cov_inputer_h = self.Cov_inputer_h

            if self.state.get('use_recurrent_gating_coverage', False):
                G_cc = self.G_cc
                R_cc = self.R_cc
                if self.state.get('use_probability_for_recurrent_coverage', False):
                    Cov_updater_p = self.Cov_updater_p
                    Cov_reseter_p = self.Cov_reseter_p
                if self.state.get('use_input_annotations_for_recurrent_coverage', False):
                    Cov_updater_c = self.Cov_updater_c
                    Cov_reseter_c = self.Cov_reseter_c
                if self.state.get('use_decoding_state_for_recurrent_coverage', False):
                    Cov_updater_h = self.Cov_updater_h
                    Cov_reseter_h = self.Cov_reseter_h

        # updating coverage model
        if self.state.get('use_recurrent_coverage', False):
            # GRU for coverage updating
            # for coverage model II,  z_t = f (z_{t-1}, a_{t-1}, s_{t-1}, h)
            # where a denotes alignment scores, s denotes state before, and h denotes annotation of input sentence (in this code, is c)
            
            source_len = c.shape[0]
            source_num = c.shape[1]
            target_num = state_before.shape[0]
            dim = self.n_hids

            cov_state_below = TT.zeros((source_len, target_num, self.state['coverage_dim']), dtype='float32')
            if self.state.get('use_recurrent_gating_coverage', False):
                cov_gater_below = TT.zeros((source_len, target_num, self.state['coverage_dim']), dtype='float32')
                cov_reseter_below = TT.zeros((source_len, target_num, self.state['coverage_dim']), dtype='float32')

            # alignment probabilities
            if self.state.get('use_probability_for_recurrent_coverage', False):
                cov_state_below += TT.dot(probs.dimshuffle(0,1,'x'), Cov_inputer_p)
                if self.state.get('use_recurrent_gating_coverage', False):
                    cov_gater_below += TT.dot(probs.dimshuffle(0,1,'x'), Cov_updater_p)
                    cov_reseter_below += TT.dot(probs.dimshuffle(0,1,'x'), Cov_reseter_p)

            if self.state.get('use_input_annotations_for_recurrent_coverage', False):
                if not given_cov_state_below:
                    # this is the bias for coverege
                    cov_state_below += utils.dot(c, Cov_inputer_c).reshape((source_len, source_num, self.state['coverage_dim']))
                    if self.state.get('use_recurrent_gating_coverage', False):
                        cov_gater_below += utils.dot(c, Cov_updater_c).reshape((source_len, source_num, self.state['coverage_dim']))
                        cov_reseter_below += utils.dot(c, Cov_reseter_c).reshape((source_len, source_num, self.state['coverage_dim']))
                else:
                    cov_state_below += given_cov_state_below
                    if self.state.get('use_recurrent_gating_coverage', False):
                        cov_gater_below += given_cov_gater_below
                        cov_reseter_below += given_cov_reseter_below

            if self.state.get('use_decoding_state_for_recurrent_coverage', False):
                # here we can decide whether to use 's_{t-1}' and 'h' or not
                # add past translation information
                cov_state_below += ReplicateLayer(source_len)(utils.dot(state_before, Cov_inputer_h)).out
                if self.state.get('use_recurrent_gating_coverage', False):
                    cov_gater_below += ReplicateLayer(source_len)(utils.dot(state_before, Cov_updater_h)).out
                    cov_reseter_below += ReplicateLayer(source_len)(utils.dot(state_before, Cov_reseter_h)).out

            if self.state.get('use_recurrent_gating_coverage', False):
                # Reset gate:
                # optionally reset the hidden state.
                cov_reseter = self.reseter_activation(TT.dot(coverage_before, R_cc) +
                        cov_reseter_below)
                reseted_cov_state_before = cov_reseter * coverage_before

                # Feed the input to obtain potential new state.
                # here state_below = W*E*y_{i-1} + C*c_i
                preactiv_coverage = TT.dot(reseted_cov_state_before, W_cc) + cov_state_below
                coverage = self.activation(preactiv_coverage)

                # Update gate:
                # optionally reject the potential new state and use the new one.
                cov_updater = self.updater_activation(TT.dot(coverage_before, G_cc) +
                        cov_gater_below)
                coverage = cov_updater * coverage + (1-cov_updater) * coverage_before
            else:
                coverage = self.activation(TT.dot(coverage_before, W_cc)+cov_state_below)
        else:
            # accumulated coverage
            if self.state.get('use_fertility_model', False):
                # if fertility.shape[1] == 1:
                if cndim == 2:
                    # for beam search, in which target num is the beam size (default 10)
                    fertility_probs = probs/TT.addbroadcast(fertility,1)
                else:
                    fertility_probs = probs/fertility
                coverage = TT.unbroadcast(fertility_probs.dimshuffle(0,1,'x'), 2)
            else:
                coverage = TT.unbroadcast(probs.dimshuffle(0,1,'x'), 2)

            if self.state.get('coverage_accumulated_operation', False) == 'additive':
                # use additive coverage
                coverage = coverage_before + coverage
            elif self.state.get('coverage_accumulated_operation', False) == 'subtractive':
                # use subtractive coverage
                coverage = coverage_before - coverage
            else:
                raise Exception("Not a valid accumulated operation: %s" % self.state.get('coverage_accumulated_operation', False))

        return coverage


    def step_fprop(self,
                   state_below,
                   state_before,
                   # added by Zhaopeng Tu, 2016-01-21
                   previous_word=None,
                   # added by Zhaopeng Tu, 2015-10-29
                   coverage_before=None,
                   # added by Zhaopeng Tu, 2015-11-11
                   given_cov_state_below=None,
                   given_cov_gater_below=None,
                   given_cov_reseter_below=None,
                   # added by Zhaopeng Tu, 2015-12-16
                   fertility=None,
                   #================
                   gater_below=None,
                   reseter_below=None,
                   mask=None,
                   c=None,
                   c_mask=None,
                   p_from_c=None,
                   use_noise=True,
                   no_noise_bias=False,
                   step_num=None,
                   return_alignment=False):
        """
        Constructs the computational graph of this layer.

        :type state_below: theano variable
        :param state_below: the input to the layer

        :type mask: None or theano variable
        :param mask: mask describing the length of each sequence in a
            minibatch

        :type state_before: theano variable
        :param state_before: the previous value of the hidden state of the
            layer

        :type updater_below: theano variable
        :param updater_below: the input to the update gate

        :type reseter_below: theano variable
        :param reseter_below: the input to the reset gate

        :type use_noise: bool
        :param use_noise: flag saying if weight noise should be used in
            computing the output of this layer

        :type no_noise_bias: bool
        :param no_noise_bias: flag saying if weight noise should be added to
            the bias as well
        """

        updater_below = gater_below

        W_hh = self.W_hh
        G_hh = self.G_hh
        R_hh = self.R_hh
        A_cp = self.A_cp
        B_hp = self.B_hp
        D_pe = self.D_pe

        # added by Zhaopeng Tu, 2015-10-29
        if self.state.get('maintain_coverage', False):
            # for coverage
            C_covp = self.C_covp

        # The code works only with 3D tensors
        cndim = c.ndim
        if cndim == 2:
            c = c[:, None, :]
        
        # Warning: either source_num or target_num should be equal,
        #          or one of them sould be 1 (they have to broadcast)
        #          for the following code to make any sense.
        source_len = c.shape[0]
        source_num = c.shape[1]
        target_num = state_before.shape[0]
        dim = self.n_hids

        # Form projection to the tanh layer from the previous hidden state
        # Shape: (source_len, target_num, dim)
        p_from_h = ReplicateLayer(source_len)(utils.dot(state_before, B_hp)).out

        # Form projection to the tanh layer from the source annotation.
        if not p_from_c:
            p_from_c =  utils.dot(c, A_cp).reshape((source_len, source_num, dim))

        # Sum projections - broadcasting happens at the dimension 1.
        p = p_from_h + p_from_c

        # added by Zhaopeng Tu, 2015-10-29
        if self.state.get('maintain_coverage', False):
            p_from_coverage = utils.dot(coverage_before, C_covp).reshape((source_len, target_num, dim))
            p += p_from_coverage

        # Apply non-linearity and project to energy.
        energy = TT.exp(utils.dot(TT.tanh(p), D_pe)).reshape((source_len, target_num))
        if c_mask:
            # This is used for batches only, that is target_num == source_num
            energy *= c_mask

        # Calculate energy sums.
        normalizer = energy.sum(axis=0)

        # Get probabilities.
        probs = energy / normalizer

        ctx = (c * probs.dimshuffle(0, 1, 'x')).sum(axis=0)

        # moved by Zhaopeng Tu, 2015-12-07
        # here we need to update coverage, as calculating ctx
        # the updated coverage would be used for decoding, if needed
        if self.state.get('maintain_coverage', False):
            coverage = self.coverage_updater(coverage_before, probs, c, cndim,
                                             state_before=state_before, 
                                             given_cov_state_below=given_cov_state_below, given_cov_gater_below=given_cov_gater_below, given_cov_reseter_below=given_cov_reseter_below,
                                             fertility=fertility)
       
        state_below += self.c_inputer(ctx).out
        reseter_below += self.c_reseter(ctx).out
        updater_below += self.c_updater(ctx).out

        # Reset gate:
        # optionally reset the hidden state.
        reseter = self.reseter_activation(TT.dot(state_before, R_hh) +
                reseter_below)
        reseted_state_before = reseter * state_before

        # Feed the input to obtain potential new state.
        # here state_below = W*E*y_{i-1} + C*c_i
        preactiv = TT.dot(reseted_state_before, W_hh) + state_below
        h = self.activation(preactiv)

        # Update gate:
        # optionally reject the potential new state and use the new one.
        updater = self.updater_activation(TT.dot(state_before, G_hh) +
                updater_below)
        h = updater * h + (1-updater) * state_before

        if mask is not None:
            if h.ndim ==2 and mask.ndim==1:
                mask = mask.dimshuffle(0,'x')
            h = mask * h + (1-mask) * state_before
        
        # h is the prior result from scan, which is provided to fn in scan
        results = [h, ctx]
        if return_alignment:
            results += [probs]

        if self.state.get('maintain_coverage', False):
            results += [coverage]
        
        return results


    def fprop(self,
              state_below,
              mask=None,
              init_state=None,
              # added by Zhaopeng Tu, 2015-12-16
              fertility=None,
              # added by Zhaopeng Tu, 2016-01-21
              target_words=None,
              #================
              gater_below=None,
              reseter_below=None,
              c=None,
              c_mask=None,
              nsteps=None,
              batch_size=None,
              use_noise=True,
              truncate_gradient=-1,
              no_noise_bias=False,
              return_alignment=False):

        updater_below = gater_below

        if theano.config.floatX=='float32':
            floatX = numpy.float32
        else:
            floatX = numpy.float64
        if nsteps is None:
            # state_below:  X
            nsteps = state_below.shape[0]
            if batch_size and batch_size != 1:
                nsteps = nsteps / batch_size
        if batch_size is None and state_below.ndim == 3:
            batch_size = state_below.shape[1]
        if state_below.ndim == 2 and \
           (not isinstance(batch_size,int) or batch_size > 1):
            state_below = state_below.reshape((nsteps, batch_size, self.n_in))
            if target_words:
                target_words = target_words.reshape((nsteps, batch_size, self.state['rank_n_approx']))
            if updater_below:
                updater_below = updater_below.reshape((nsteps, batch_size, self.n_in))
            else:
                # added by Zhaopeng Tu, 2016-02-26
                updater_below = TT.zeros((nsteps, batch_size, self.n_in), dtype='float32')

            if reseter_below:
                reseter_below = reseter_below.reshape((nsteps, batch_size, self.n_in))
            else:
                # added by Zhaopeng Tu, 2016-02-26
                reseter_below = TT.zeros((nsteps, batch_size, self.n_in), dtype='float32')

        if not init_state:
            if not isinstance(batch_size, int) or batch_size != 1:
                init_state = TT.alloc(floatX(0), batch_size, self.n_hids)
            else:
                init_state = TT.alloc(floatX(0), self.n_hids)

        p_from_c =  utils.dot(c, self.A_cp).reshape(
                (c.shape[0], c.shape[1], self.n_hids))

        if self.state.get('maintain_coverage', False) and self.state.get('use_recurrent_coverage', False) and self.state.get('use_input_annotations_for_recurrent_coverage', False):
            # this is the bias for coverege
            cov_inputer_from_c = utils.dot(c, self.Cov_inputer_c).reshape((c.shape[0], c.shape[1], self.state['coverage_dim']))
            if self.state.get('use_recurrent_gating_coverage', False):
                cov_gater_from_c = utils.dot(c, self.Cov_updater_c).reshape((c.shape[0], c.shape[1], self.state['coverage_dim']))
                cov_reseter_from_c = utils.dot(c, self.Cov_reseter_c).reshape((c.shape[0], c.shape[1], self.state['coverage_dim']))
            else:
                cov_gater_from_c = TT.zeros(cov_inputer_from_c.shape)
                cov_reseter_from_c = TT.zeros(cov_inputer_from_c.shape)
        else:
            cov_inputer_from_c = TT.zeros((c.shape[0], c.shape[1], 1))
            cov_gater_from_c = TT.zeros((c.shape[0], c.shape[1], 1))
            cov_reseter_from_c = TT.zeros((c.shape[0], c.shape[1], 1))

        if not fertility:
            fertility = TT.zeros((c.shape[0], c.shape[1], 1))
            total_fertility_num = (fertility*c_mask).sum(axis=0)

        # mask always exists in this function, see build_decoder() for more details
        # The general order of function parameters to fn is:
        # sequences (if any), prior result(s) (if needed), non-sequences (if any)
        if not self.state.get('maintain_coverage', False): 
            sequences = [state_below, target_words, mask, updater_below, reseter_below]
            non_sequences = [c, c_mask, p_from_c] 
            #              seqs        | out |  non_seqs
            fn = lambda x, y, m, g, r,   h,    c1, cm, pc, l, pl : self.step_fprop(x, h, mask=m,
                    previous_word=y,
                    gater_below=g, reseter_below=r,
                    c=c1, p_from_c=pc, c_mask=cm,
                    use_noise=use_noise, no_noise_bias=no_noise_bias,
                    return_alignment=return_alignment)
        else:
            # combined by Zhaopeng Tu, 2016-01-29
            sequences = [state_below, target_words, mask, updater_below, reseter_below]
            non_sequences = [c, c_mask, p_from_c, cov_inputer_from_c, cov_gater_from_c, cov_reseter_from_c, fertility] 
            #              seqs        | out          |  non_seqs
            fn = lambda x, y, m, g, r,   h, coverage,   c1, cm, pc, cx, cg, cr, f : self.step_fprop(x, h, mask=m,
                    previous_word=y,
                    coverage_before=coverage,
                    # added by Zhaopeng Tu, 2015-12-17
                    fertility=f,
                    gater_below=g, reseter_below=r,
                    given_cov_state_below=cx, given_cov_gater_below=cg, given_cov_reseter_below=cr,
                    c=c1, p_from_c=pc, c_mask=cm,
                    use_noise=use_noise, no_noise_bias=no_noise_bias,
                    return_alignment=return_alignment)

        # outputs_info is the list of Theano variables or dictionaries describing the initial state of the outputs computed recurrently.
        # If outputs_info is an empty list or None, scan assumes that no tap is used for any of the outputs.
        # In this case, only init_state and init_coverage would be used recurrently.
        outputs_info = [init_state, None]
        if return_alignment:
            outputs_info.append(None)
        if self.state.get('maintain_coverage', False):
            # init_coverage = TT.zeros((c.shape[0], c.shape[1], self.state['coverage_dim']), dtype='float32')
            if self.state.get('use_linguistic_coverage', False) and self.state.get('coverage_accumulated_operation', False) == 'subtractive':
                init_coverage = TT.unbroadcast(TT.ones((c.shape[0], c.shape[1], self.state['coverage_dim']), dtype='float32'), 2)
            else:
                init_coverage = TT.unbroadcast(TT.zeros((c.shape[0], c.shape[1], self.state['coverage_dim']), dtype='float32'), 2)
            outputs_info.append(init_coverage)

        rval, updates = theano.scan(fn,
                        sequences=sequences,
                        non_sequences=non_sequences,
                        outputs_info=outputs_info,
                        name='layer_%s'%self.name,
                        truncate_gradient=truncate_gradient,
                        n_steps=nsteps)
        self.out = rval
        self.rval = rval
        self.updates = updates

        return self.out


class ReplicateLayer(Layer):

    def __init__(self, n_times):
        self.n_times = n_times
        super(ReplicateLayer, self).__init__(0, 0, None)

    def fprop(self, x):
        # This is black magic based on broadcasting,
        # that's why variable names don't make any sense.
        a = TT.shape_padleft(x)
        padding = [1] * x.ndim
        b = TT.alloc(numpy.float32(1), self.n_times, *padding)
        self.out = a * b
        return self.out

class PadLayer(Layer):

    def __init__(self, required):
        self.required = required
        Layer.__init__(self, 0, 0, None)

    def fprop(self, x):
        if_longer = x[:self.required]
        padding = ReplicateLayer(TT.max([1, self.required - x.shape[0]]))(x[-1]).out
        if_shorter = TT.concatenate([x, padding])
        diff = x.shape[0] - self.required
        self.out = ifelse(diff < 0, if_shorter, if_longer)
        return self.out

class ZeroLayer(Layer):

    def fprop(self, x):
        self.out = TT.zeros(x.shape)
        return self.out


def none_if_zero(x):
    if x == 0:
        return None
    return x

class Maxout(object):

    def __init__(self, maxout_part):
        self.maxout_part = maxout_part

    def __call__(self, x):
        shape = x.shape
        if x.ndim == 1:
            shape1 = TT.cast(shape[0] / self.maxout_part, 'int64')
            shape2 = TT.cast(self.maxout_part, 'int64')
            x = x.reshape([shape1, shape2])
            x = x.max(1)
        else:
            shape1 = TT.cast(shape[1] / self.maxout_part, 'int64')
            shape2 = TT.cast(self.maxout_part, 'int64')
            x = x.reshape([shape[0], shape1, shape2])
            x = x.max(2)
        return x

def prefix_lookup(state, p, s):
    if '%s_%s'%(p,s) in state:
        return state['%s_%s'%(p, s)]
    return state[s]

class EncoderDecoderBase(object):

    def _create_embedding_layers(self):
        logger.debug("_create_embedding_layers")
        self.approx_embedder = MultiLayer(
            self.rng,
            n_in=self.state['n_sym_source']
                if self.prefix.find("enc") >= 0
                else self.state['n_sym_target'],
            n_hids=[self.state['rank_n_approx']],
            activation=[self.state['rank_n_activ']],
            name='{}_approx_embdr'.format(self.prefix),
            **self.default_kwargs)

        # We have 3 embeddings for each word in each level,
        # the one used as input,
        # the one used to control resetting gate,
        # the one used to control update gate.
        self.input_embedders = [lambda x : 0] * self.num_levels
        self.reset_embedders = [lambda x : 0] * self.num_levels
        self.update_embedders = [lambda x : 0] * self.num_levels
        embedder_kwargs = dict(self.default_kwargs)
        embedder_kwargs.update(dict(
            n_in=self.state['rank_n_approx'],
            n_hids=[self.state['dim'] * self.state['dim_mult']],
            activation=['lambda x:x']))
        for level in range(self.num_levels):
            self.input_embedders[level] = MultiLayer(
                self.rng,
                name='{}_input_embdr_{}'.format(self.prefix, level),
                **embedder_kwargs)
            if prefix_lookup(self.state, self.prefix, 'rec_gating'):
                self.update_embedders[level] = MultiLayer(
                    self.rng,
                    learn_bias=False,
                    name='{}_update_embdr_{}'.format(self.prefix, level),
                    **embedder_kwargs)
            if prefix_lookup(self.state, self.prefix, 'rec_reseting'):
                self.reset_embedders[level] =  MultiLayer(
                    self.rng,
                    learn_bias=False,
                    name='{}_reset_embdr_{}'.format(self.prefix, level),
                    **embedder_kwargs)

    def _create_inter_level_layers(self):
        logger.debug("_create_inter_level_layers")
        inter_level_kwargs = dict(self.default_kwargs)
        inter_level_kwargs.update(
                n_in=self.state['dim'],
                n_hids=self.state['dim'] * self.state['dim_mult'],
                activation=['lambda x:x'])

        self.inputers = [0] * self.num_levels
        self.reseters = [0] * self.num_levels
        self.updaters = [0] * self.num_levels
        for level in range(1, self.num_levels):
            self.inputers[level] = MultiLayer(self.rng,
                    name="{}_inputer_{}".format(self.prefix, level),
                    **inter_level_kwargs)
            if prefix_lookup(self.state, self.prefix, 'rec_reseting'):
                self.reseters[level] = MultiLayer(self.rng,
                    name="{}_reseter_{}".format(self.prefix, level),
                    **inter_level_kwargs)
            if prefix_lookup(self.state, self.prefix, 'rec_gating'):
                self.updaters[level] = MultiLayer(self.rng,
                    name="{}_updater_{}".format(self.prefix, level),
                    **inter_level_kwargs)

    def _create_transition_layers(self):
        logger.debug("_create_transition_layers")
        self.transitions = []
        rec_layer = eval(prefix_lookup(self.state, self.prefix, 'rec_layer'))
        add_args = dict()
        if rec_layer == RecurrentLayerWithSearch:
            add_args = dict(c_dim=self.state['c_dim'])
            # cleaned by Zhaopeng Tu, 2016-01-28
            add_args['state'] = self.state
        for level in range(self.num_levels):
            self.transitions.append(rec_layer(
                    self.rng,
                    n_hids=self.state['dim'],
                    activation=prefix_lookup(self.state, self.prefix, 'activ'),
                    bias_scale=self.state['bias'],
                    init_fn=(self.state['rec_weight_init_fn']
                        if not self.skip_init
                        else "sample_zeros"),
                    scale=prefix_lookup(self.state, self.prefix, 'rec_weight_scale'),
                    weight_noise=self.state['weight_noise_rec'],
                    dropout=self.state['dropout_rec'],
                    gating=prefix_lookup(self.state, self.prefix, 'rec_gating'),
                    gater_activation=prefix_lookup(self.state, self.prefix, 'rec_gater'),
                    reseting=prefix_lookup(self.state, self.prefix, 'rec_reseting'),
                    reseter_activation=prefix_lookup(self.state, self.prefix, 'rec_reseter'),
                    name='{}_transition_{}'.format(self.prefix, level),
                    **add_args))

class Encoder(EncoderDecoderBase):

    def __init__(self, state, rng, prefix='enc', skip_init=False):
        self.state = state
        self.rng = rng
        self.prefix = prefix
        self.skip_init = skip_init

        self.num_levels = self.state['encoder_stack']

        # support multiple gating/memory units
        if 'dim_mult' not in self.state:
            self.state['dim_mult'] = 1.
        if 'hid_mult' not in self.state:
            self.state['hid_mult'] = 1.

    def create_layers(self):
        """ Create all elements of Encoder's computation graph"""

        self.default_kwargs = dict(
            init_fn=self.state['weight_init_fn'] if not self.skip_init else "sample_zeros",
            weight_noise=self.state['weight_noise'],
            scale=self.state['weight_scale'])

        self._create_embedding_layers()
        self._create_transition_layers()
        self._create_inter_level_layers()
        self._create_representation_layers()

    def _create_representation_layers(self):
        logger.debug("_create_representation_layers")
        # If we have a stack of RNN, then their last hidden states
        # are combined with a maxout layer.
        self.repr_contributors = [None] * self.num_levels
        for level in range(self.num_levels):
            self.repr_contributors[level] = MultiLayer(
                self.rng,
                n_in=self.state['dim'],
                n_hids=[self.state['dim'] * self.state['maxout_part']],
                activation=['lambda x: x'],
                name="{}_repr_contrib_{}".format(self.prefix, level),
                **self.default_kwargs)
        self.repr_calculator = UnaryOp(
                activation=eval(self.state['unary_activ']),
                name="{}_repr_calc".format(self.prefix))

    def build_encoder(self, x,
            x_mask=None,
            use_noise=False,
            approx_embeddings=None,
            return_hidden_layers=False):
        """Create the computational graph of the RNN Encoder

        :param x:
            input variable, either vector of word indices or
            matrix of word indices, where each column is a sentence

        :param x_mask:
            when x is a matrix and input sequences are
            of variable length, this 1/0 matrix is used to specify
            the matrix positions where the input actually is

        :param use_noise:
            turns on addition of noise to weights
            (UNTESTED)

        :param approx_embeddings:
            forces encoder to use given embeddings instead of its own

        :param return_hidden_layers:
            if True, encoder returns all the activations of the hidden layer
            (WORKS ONLY IN NON-HIERARCHICAL CASE)
        """

        # Low rank embeddings of all the input words.
        # Shape in case of matrix input:
        #   (max_seq_len * batch_size, rank_n_approx),
        #   where max_seq_len is the maximum length of batch sequences.
        # Here and later n_words = max_seq_len * batch_size.
        # Shape in case of vector input:
        #   (seq_len, rank_n_approx)
        if not approx_embeddings:
            approx_embeddings = self.approx_embedder(x)

        # Low rank embeddings are projected to contribute
        # to input, reset and update signals.
        # All the shapes: (n_words, dim)
        input_signals = []
        reset_signals = []
        update_signals = []
        for level in range(self.num_levels):
            input_signals.append(self.input_embedders[level](approx_embeddings))
            update_signals.append(self.update_embedders[level](approx_embeddings))
            reset_signals.append(self.reset_embedders[level](approx_embeddings))

        # Hidden layers.
        # Shape in case of matrix input: (max_seq_len, batch_size, dim)
        # Shape in case of vector input: (seq_len, dim)
        hidden_layers = []
        for level in range(self.num_levels):
            # Each hidden layer (except the bottom one) receives
            # input, reset and update signals from below.
            # All the shapes: (n_words, dim)
            if level > 0:
                input_signals[level] += self.inputers[level](hidden_layers[-1])
                update_signals[level] += self.updaters[level](hidden_layers[-1])
                reset_signals[level] += self.reseters[level](hidden_layers[-1])
            hidden_layers.append(self.transitions[level](
                    input_signals[level],
                    nsteps=x.shape[0],
                    batch_size=x.shape[1] if x.ndim == 2 else 1,
                    mask=x_mask,
                    gater_below=none_if_zero(update_signals[level]),
                    reseter_below=none_if_zero(reset_signals[level]),
                    use_noise=use_noise))
        if return_hidden_layers:
            assert self.state['encoder_stack'] == 1
            return hidden_layers[0]

        # If we no stack of RNN but only a usual one,
        # then the last hidden state is used as a representation.
        # Return value shape in case of matrix input:
        #   (batch_size, dim)
        # Return value shape in case of vector input:
        #   (dim,)
        if self.num_levels == 1 or self.state['take_top']:
            c = LastState()(hidden_layers[-1])
            if c.out.ndim == 2:
                c.out = c.out[:,:self.state['dim']]
            else:
                c.out = c.out[:self.state['dim']]
            return c

        # If we have a stack of RNN, then their last hidden states
        # are combined with a maxout layer.
        # Return value however has the same shape.
        contributions = []
        for level in range(self.num_levels):
            contributions.append(self.repr_contributors[level](
                LastState()(hidden_layers[level])))
        # I do not know a good starting value for sum
        c = self.repr_calculator(sum(contributions[1:], contributions[0]))
        return c

class Decoder(EncoderDecoderBase):

    EVALUATION = 0
    SAMPLING = 1
    BEAM_SEARCH = 2

    def __init__(self, state, rng, prefix='dec',
            skip_init=False, compute_alignment=True):
        self.state = state
        self.rng = rng
        self.prefix = prefix
        self.skip_init = skip_init
        # self.compute_alignment = compute_alignment
        # modified by Zhaopeng Tu, 2016-02-29
        self.compute_alignment = True

        # Actually there is a problem here -
        # we don't make difference between number of input layers
        # and outputs layers.
        self.num_levels = self.state['decoder_stack']

        if 'dim_mult' not in self.state:
            self.state['dim_mult'] = 1.

    def create_layers(self):
        """ Create all elements of Decoder's computation graph"""

        self.default_kwargs = dict(
            init_fn=self.state['weight_init_fn'] if not self.skip_init else "sample_zeros",
            weight_noise=self.state['weight_noise'],
            scale=self.state['weight_scale'])

        self._create_embedding_layers()
        self._create_transition_layers()
        self._create_inter_level_layers()
        self._create_initialization_layers()
        self._create_decoding_layers()
        self._create_readout_layers()
        
        # added by Zhaopeng Tu, 2015-12-16
        # fertility layer
        if self.state['maintain_coverage'] and self.state['use_fertility_model'] and self.state['use_linguistic_coverage']:
            self._create_fertility_layer()
            if self.state['search']:
                assert self.num_levels == 1
                self.transitions[0].set_fertility_layers(self.fertility_inputer)

        if self.state['search']:
            assert self.num_levels == 1
            self.transitions[0].set_decoding_layers(
                    self.decode_inputers[0],
                    self.decode_reseters[0],
                    self.decode_updaters[0])
     
    def _create_initialization_layers(self):
        logger.debug("_create_initialization_layers")
        self.initializers = [ZeroLayer()] * self.num_levels
        if self.state['bias_code']:
            for level in range(self.num_levels):
                self.initializers[level] = MultiLayer(
                    self.rng,
                    n_in=self.state['dim'],
                    n_hids=[self.state['dim'] * self.state['hid_mult']],
                    activation=[prefix_lookup(self.state, 'dec', 'activ')],
                    bias_scale=[self.state['bias']],
                    name='{}_initializer_{}'.format(self.prefix, level),
                    **self.default_kwargs)

    def _create_decoding_layers(self):
        logger.debug("_create_decoding_layers")
        self.decode_inputers = [lambda x : 0] * self.num_levels
        self.decode_reseters = [lambda x : 0] * self.num_levels
        self.decode_updaters = [lambda x : 0] * self.num_levels
        self.back_decode_inputers = [lambda x : 0] * self.num_levels
        self.back_decode_reseters = [lambda x : 0] * self.num_levels
        self.back_decode_updaters = [lambda x : 0] * self.num_levels

        decoding_kwargs = dict(self.default_kwargs)
        decoding_kwargs.update(dict(
                n_in=self.state['c_dim'],
                n_hids=self.state['dim'] * self.state['dim_mult'],
                activation=['lambda x:x'],
                learn_bias=False))

        if self.state['decoding_inputs']:
            for level in range(self.num_levels):
                # Input contributions
                self.decode_inputers[level] = MultiLayer(
                    self.rng,
                    name='{}_dec_inputer_{}'.format(self.prefix, level),
                    **decoding_kwargs)
                # Update gate contributions
                if prefix_lookup(self.state, 'dec', 'rec_gating'):
                    self.decode_updaters[level] = MultiLayer(
                        self.rng,
                        name='{}_dec_updater_{}'.format(self.prefix, level),
                        **decoding_kwargs)
                # Reset gate contributions
                if prefix_lookup(self.state, 'dec', 'rec_reseting'):
                    self.decode_reseters[level] = MultiLayer(
                        self.rng,
                        name='{}_dec_reseter_{}'.format(self.prefix, level),
                        **decoding_kwargs)

    # added by Zhaopeng Tu, 2015-12-16
    def _create_fertility_layer(self):
        logger.debug("_create_fertility_layers")
        decoding_kwargs = dict(self.default_kwargs)
        decoding_kwargs.update(dict(
                n_in=self.state['c_dim'],
                n_hids=1,
                activation=['TT.nnet.sigmoid'],
                learn_bias=False))

        # Input contributions
        self.fertility_inputer = MultiLayer(
            self.rng,
            name='{}_fertility_inputer'.format(self.prefix),
            **decoding_kwargs)


    def _create_readout_layers(self):
        softmax_layer = self.state['softmax_layer'] if 'softmax_layer' in self.state \
                        else 'SoftmaxLayer'

        logger.debug("_create_readout_layers")

        readout_kwargs = dict(self.default_kwargs)
        readout_kwargs.update(dict(
                n_hids=self.state['dim'],
                activation='lambda x: x',
            ))
 
        self.repr_readout = MultiLayer(
                self.rng,
                n_in=self.state['c_dim'],
                learn_bias=False,
                name='{}_repr_readout'.format(self.prefix),
                **readout_kwargs)
        
        # Attention - this is the only readout layer
        # with trainable bias. Should be careful with that.
        self.hidden_readouts = [None] * self.num_levels
        for level in range(self.num_levels):
            self.hidden_readouts[level] = MultiLayer(
                self.rng,
                n_in=self.state['dim'],
                name='{}_hid_readout_{}'.format(self.prefix, level),
                **readout_kwargs)

        self.prev_word_readout = 0
        if self.state['bigram']:
            self.prev_word_readout = MultiLayer(
                self.rng,
                n_in=self.state['rank_n_approx'],
                n_hids=self.state['dim'],
                activation=['lambda x:x'],
                learn_bias=False,
                name='{}_prev_readout_{}'.format(self.prefix, level),
                **self.default_kwargs)

        if self.state['deep_out']:
            act_layer = UnaryOp(activation=eval(self.state['unary_activ']))
            drop_layer = DropOp(rng=self.rng, dropout=self.state['dropout'])
            self.output_nonlinearities = [act_layer, drop_layer]
            self.output_layer = eval(softmax_layer)(
                    self.rng,
                    self.state['dim'] / self.state['maxout_part'],
                    self.state['n_sym_target'],
                    sparsity=-1,
                    rank_n_approx=self.state['rank_n_approx'],
                    name='{}_deep_softmax'.format(self.prefix),
                    use_nce=self.state['use_nce'] if 'use_nce' in self.state else False,
                    **self.default_kwargs)
        else:
            self.output_nonlinearities = []
            self.output_layer = eval(softmax_layer)(
                    self.rng,
                    self.state['dim'],
                    self.state['n_sym_target'],
                    sparsity=-1,
                    rank_n_approx=self.state['rank_n_approx'],
                    name='dec_softmax',
                    sum_over_time=True,
                    use_nce=self.state['use_nce'] if 'use_nce' in self.state else False,
                    **self.default_kwargs)

    def build_decoder(self, c, y,
            c_mask=None,
            y_mask=None,
            step_num=None,
            mode=EVALUATION,
            given_init_states=None,
            # added by Zhaopeng Tu, 2015-10-30
            coverage_before=None,
            # added by Zhaopeng Tu, 2015-12-17
            fertility=None,
            T=1):
        """Create the computational graph of the RNN Decoder.

        :param c:
            representations produced by an encoder.
            (n_samples, dim) matrix if mode == sampling or
            (max_seq_len, batch_size, dim) matrix if mode == evaluation

        :param c_mask:
            if mode == evaluation a 0/1 matrix identifying valid positions in c

        :param y:
            if mode == evaluation
                target sequences, matrix of word indices of shape (max_seq_len, batch_size),
                where each column is a sequence
            if mode != evaluation
                a vector of previous words of shape (n_samples,)

        :param y_mask:
            if mode == evaluation a 0/1 matrix determining lengths
                of the target sequences, must be None otherwise

        :param mode:
            chooses on of three modes: evaluation, sampling and beam_search

        :param given_init_states:
            for sampling and beam_search. A list of hidden states
                matrices for each layer, each matrix is (n_samples, dim)

        :param T:
            sampling temperature
        """

        # Check parameter consistency
        if mode == Decoder.EVALUATION:
            assert not given_init_states
        else:
            assert not y_mask
            assert given_init_states
            # added by Zhaopeng Tu, 2015-10-30
            if self.state['maintain_coverage']:
                assert coverage_before
                if self.state['use_linguistic_coverage']:
                    assert self.state['coverage_dim'] == 1
                assert (self.state['use_linguistic_coverage'] or self.state['use_recurrent_coverage'])
                assert not (self.state['use_linguistic_coverage'] and self.state['use_recurrent_coverage'])
            if mode == Decoder.BEAM_SEARCH:
                assert T == 1

        # For log-likelihood evaluation the representation
        # be replicated for conveniency. In case backward RNN is used
        # it is not done.
        # Shape if mode == evaluation
        #   (max_seq_len, batch_size, dim)
        # Shape if mode != evaluation
        #   (n_samples, dim)
        if not self.state['search']:
            if mode == Decoder.EVALUATION:
                c = PadLayer(y.shape[0])(c)
            else:
                assert step_num
                c_pos = TT.minimum(step_num, c.shape[0] - 1)

        # Low rank embeddings of all the input words.
        # Shape if mode == evaluation
        #   (n_words, rank_n_approx),
        # Shape if mode != evaluation
        #   (n_samples, rank_n_approx)
        approx_embeddings = self.approx_embedder(y)

        # Low rank embeddings are projected to contribute
        # to input, reset and update signals.
        # All the shapes if mode == evaluation:
        #   (n_words, dim)
        # where: n_words = max_seq_len * batch_size
        # All the shape if mode != evaluation:
        #   (n_samples, dim)
        input_signals = []
        reset_signals = []
        update_signals = []
        for level in range(self.num_levels):
            # Contributions directly from input words.
            input_signals.append(self.input_embedders[level](approx_embeddings))
            update_signals.append(self.update_embedders[level](approx_embeddings))
            reset_signals.append(self.reset_embedders[level](approx_embeddings))

            # Contributions from the encoded source sentence.
            if not self.state['search']:
                current_c = c if mode == Decoder.EVALUATION else c[c_pos]
                input_signals[level] += self.decode_inputers[level](current_c)
                update_signals[level] += self.decode_updaters[level](current_c)
                reset_signals[level] += self.decode_reseters[level](current_c)

        # Hidden layers' initial states.
        # Shapes if mode == evaluation:
        #   (batch_size, dim)
        # Shape if mode != evaluation:
        #   (n_samples, dim)
        init_states = given_init_states
        if not init_states:
            init_states = []
            for level in range(self.num_levels):
                init_c = c[0, :, -self.state['dim']:]
                init_states.append(self.initializers[level](init_c))
        
        # added by Zhaopeng Tu, 2015-12-16
        # calculate the fertility for each source word
        if self.state['maintain_coverage'] and self.state['use_fertility_model'] and self.state['use_linguistic_coverage']:
            if not fertility:
                fertility = self.state['max_fertility'] * self.fertility_inputer(c).out
                # the out using utils.dot()
                # elif 'float' in inp.dtype and inp.ndim == 3:
                #     shape0 = inp.shape[0]
                #     shape1 = inp.shape[1]
                #     shape2 = inp.shape[2]
                #     return TT.dot(inp.reshape((shape0*shape1, shape2)), matrix)
                # since matrix is in the shape (shape2, 1)
                # then fertility should be in the shape (c.shape0, c.shape1, 1) for EVALUATION mode
                # in other modes, fertility is in the shape (c.shape0, 1), which is consistent with probs (1 == target_num)
                if mode == Decoder.EVALUATION:
                    # here we make fertility has the same shape with alignment probs
                    # comments added by Zhaopeng Tu, 2015-12-28
                    # note that here fertility is not masked, so that even the padded positions (0) can be divided by fertility
                    fertility = fertility.reshape((c.shape[0], c.shape[1]))

        # Hidden layers' states.
        # Shapes if mode == evaluation:
        #  (seq_len, batch_size, dim)
        # Shapes if mode != evaluation:
        #  (n_samples, dim)
        hidden_layers = []
        contexts = []
        previous_contexts = []
        # Default value for alignment must be smth computable
        alignment = TT.zeros((1,))
        for level in range(self.num_levels):
            if level > 0:
                input_signals[level] += self.inputers[level](hidden_layers[level - 1])
                update_signals[level] += self.updaters[level](hidden_layers[level - 1])
                reset_signals[level] += self.reseters[level](hidden_layers[level - 1])
            add_kwargs = (dict(state_before=init_states[level])
                        if mode != Decoder.EVALUATION
                        else dict(init_state=init_states[level],
                            batch_size=y.shape[1] if y.ndim == 2 else 1,
                            nsteps=y.shape[0]))
            if self.state['search']:
                add_kwargs['c'] = c
                add_kwargs['c_mask'] = c_mask
                add_kwargs['return_alignment'] = self.compute_alignment
                # added by Zhaopeng Tu, 2016-01-21
                # for context gate
                if mode != Decoder.EVALUATION:
                    add_kwargs['previous_word'] = approx_embeddings
                else:
                    add_kwargs['target_words'] = approx_embeddings
                # added by Zhaopeng Tu, 2015-12-16
                # calculate the fertility for each source word
                if self.state['maintain_coverage'] and self.state['use_linguistic_coverage']:
                    if self.state['use_fertility_model']:
                        add_kwargs['fertility'] = fertility

                if mode != Decoder.EVALUATION:
                    add_kwargs['step_num'] = step_num
                    if self.state['maintain_coverage']:
                        # added by Zhaopeng Tu, 2015-10-30
                        add_kwargs['coverage_before'] = coverage_before
            result = self.transitions[level](
                    input_signals[level],
                    mask=y_mask,
                    gater_below=none_if_zero(update_signals[level]),
                    reseter_below=none_if_zero(reset_signals[level]),
                    one_step=mode != Decoder.EVALUATION,
                    use_noise=mode == Decoder.EVALUATION,
                    **add_kwargs)
            # added by Zhaopeng Tu, 2015-10-30
            # for coverage
            if self.state['maintain_coverage']:
                coverage = result[-1]
                result = result[:-1]
        
            if self.state['search']:
                if self.compute_alignment:
                    #This implicitly wraps each element of result.out with a Layer to keep track of the parameters.
                    #It is equivalent to h=result[0], ctx=result[1] etc. 
                    h, ctx, alignment = result
                    if mode == Decoder.EVALUATION:
                        alignment = alignment.out
                else:
                    #This implicitly wraps each element of result.out with a Layer to keep track of the parameters.
                    #It is equivalent to h=result[0], ctx=result[1]
                    h, ctx = result
            else:
                h = result
                if mode == Decoder.EVALUATION:
                    ctx = c
                else:
                    ctx = ReplicateLayer(given_init_states[0].shape[0])(c[c_pos]).out
            hidden_layers.append(h)
            contexts.append(ctx)

        # In hidden_layers we do no have the initial state, but we need it.
        # Instead of it we have the last one, which we do not need.
        # So what we do is discard the last one and prepend the initial one.
        if mode == Decoder.EVALUATION:
            for level in range(self.num_levels):
                hidden_layers[level].out = TT.concatenate([
                    TT.shape_padleft(init_states[level].out),
                        hidden_layers[level].out])[:-1]

        # The output representation to be fed in softmax.
        # Shape if mode == evaluation
        #   (n_words, dim_r)
        # Shape if mode != evaluation
        #   (n_samples, dim_r)
        # ... where dim_r depends on 'deep_out' option.
        readout = self.repr_readout(contexts[0])

        for level in range(self.num_levels):
            if mode != Decoder.EVALUATION:
                read_from = init_states[level]
            else:
                read_from = hidden_layers[level]
            read_from_var = read_from if type(read_from) == theano.tensor.TensorVariable else read_from.out
            if read_from_var.ndim == 3:
                read_from_var = read_from_var[:,:,:self.state['dim']]
            else:
                read_from_var = read_from_var[:,:self.state['dim']]
            if type(read_from) != theano.tensor.TensorVariable:
                read_from.out = read_from_var
            else:
                read_from = read_from_var
            readout += self.hidden_readouts[level](read_from)

        if self.state['bigram']:
            if mode != Decoder.EVALUATION:
                check_first_word = (y > 0
                    if self.state['check_first_word']
                    else TT.ones((y.shape[0]), dtype="float32"))
                # padright is necessary as we want to multiply each row with a certain scalar
                readout += TT.shape_padright(check_first_word) * self.prev_word_readout(approx_embeddings).out
            else:
                if y.ndim == 1:
                    readout += Shift()(self.prev_word_readout(approx_embeddings).reshape(
                        (y.shape[0], 1, self.state['dim'])))
                else:
                    # This place needs explanation. When prev_word_readout is applied to
                    # approx_embeddings the resulting shape is
                    # (n_batches * sequence_length, repr_dimensionality). We first
                    # transform it into 3D tensor to shift forward in time. Then
                    # reshape it back.
                    readout += Shift()(self.prev_word_readout(approx_embeddings).reshape(
                        (y.shape[0], y.shape[1], self.state['dim']))).reshape(
                                readout.out.shape)
        for fun in self.output_nonlinearities:
            readout = fun(readout)

        if mode == Decoder.SAMPLING:
            sample = self.output_layer.get_sample(
                    state_below=readout,
                    temp=T)
            # Current SoftmaxLayer.get_cost is stupid,
            # that's why we have to reshape a lot.
            self.output_layer.get_cost(
                    state_below=readout.out,
                    temp=T,
                    target=sample)
            log_prob = self.output_layer.cost_per_sample
            if self.state['maintain_coverage']:
                return [sample] + [log_prob] + hidden_layers + [coverage]
            else:
                return [sample] + [log_prob] + hidden_layers
        elif mode == Decoder.BEAM_SEARCH:
            if self.compute_alignment:
                return self.output_layer(
                        state_below=readout.out,
                        temp=T).out, alignment
            else:
                return self.output_layer(
                        state_below=readout.out,
                        temp=T).out
        elif mode == Decoder.EVALUATION:
            return (self.output_layer.train(
                    state_below=readout,
                    target=y,
                    mask=y_mask,
                    reg=None),
                    alignment)
        else:
            raise Exception("Unknown mode for build_decoder")


    def sampling_step(self, *args):
        """Implements one step of sampling

        Args are necessary since the number (and the order) of arguments can vary"""

        args = iter(args)

        # Arguments that correspond to scan's "sequences" parameteter:
        step_num = next(args)
        assert step_num.ndim == 0

        # Arguments that correspond to scan's "outputs" parameteter:
        prev_word = next(args)
        assert prev_word.ndim == 1
        # skip the previous word log probability
        assert next(args).ndim == 1

        prev_hidden_states = [next(args) for k in range(self.num_levels)]
        assert prev_hidden_states[0].ndim == 2
        if self.state['maintain_coverage']:
            # added by Zhaopeng Tu, 2015-10-30
            # previous coverage
            prev_coverage = next(args)

        # Arguments that correspond to scan's "non_sequences":
        c = next(args)
        assert c.ndim == 2
        T = next(args)
        assert T.ndim == 0
        if self.state['maintain_coverage'] and self.state['use_linguistic_coverage'] and self.state['use_fertility_model']:
            fertility = next(args)

        decoder_args = dict(given_init_states=prev_hidden_states, T=T, c=c)
        if self.state['maintain_coverage']:
            # added by Zhaopeng Tu, 2015-10-30
            decoder_args['coverage_before']=prev_coverage
            if self.state['use_fertility_model'] and self.state['use_linguistic_coverage']:
                decoder_args['fertility'] = fertility

        sample, log_prob = self.build_decoder(y=prev_word, step_num=step_num, mode=Decoder.SAMPLING, **decoder_args)[:2]
        hidden_states = self.build_decoder(y=sample, step_num=step_num, mode=Decoder.SAMPLING, **decoder_args)[2:]
        return [sample, log_prob] + hidden_states

    def build_initializers(self, c):
        return [init(c).out for init in self.initializers]
    
    # added by Zhaopeng Tu, 2015-12-17
    # for fertility model
    def build_fertility_computer(self, c):
        fertility = self.state['max_fertility'] * self.fertility_inputer(c).out
        return fertility

    def build_sampler(self, n_samples, n_steps, T, c):
        states = [TT.zeros(shape=(n_samples,), dtype='int64'),
                TT.zeros(shape=(n_samples,), dtype='float32')]

        init_c = c[0, -self.state['dim']:]
        states += [ReplicateLayer(n_samples)(init(init_c).out).out for init in self.initializers]
        # added by Zhaopeng Tu, 2015-10-30
        # init_coverage
        if self.state['maintain_coverage']:
            # in sampling, init_c is two-dimension (source_length*c_dim), same for init_coverage
            # modified by Zhaopeng Tu, 2015-12-18, big mistake here!!!
            # coverage should be always 3D, the first two dimensions are consistent with alignment probs
            # while the last one is the coverage dim
            if self.state['use_linguistic_coverage'] and self.state['coverage_accumulated_operation'] == 'subtractive':
                init_coverage = TT.unbroadcast(TT.ones((c.shape[0], n_samples, self.state['coverage_dim']), dtype='float32'), 2)
            else:
                init_coverage = TT.unbroadcast(TT.zeros((c.shape[0], n_samples, self.state['coverage_dim']), dtype='float32'), 2)
            states.append(init_coverage)

        if not self.state['search']:
            c = PadLayer(n_steps)(c).out

        # Pad with final states
        non_sequences = [c, T]
        if self.state['maintain_coverage'] and self.state['use_linguistic_coverage'] and self.state['use_fertility_model']:
            fertility = self.state['max_fertility'] * self.fertility_inputer(c).out
            non_sequences.append(fertility)

        outputs, updates = theano.scan(self.sampling_step,
                outputs_info=states,
                non_sequences=non_sequences,
                sequences=[TT.arange(n_steps, dtype="int64")],
                n_steps=n_steps,
                name="{}_sampler_scan".format(self.prefix))
        if self.state['maintain_coverage']:
            if self.state['use_fertility_model'] and self.state['use_linguistic_coverage']:
                return (outputs[0], outputs[1], outputs[-1], fertility), updates
            else:
                return (outputs[0], outputs[1], outputs[-1]), updates
        else:
            return (outputs[0], outputs[1]), updates

    def build_next_probs_predictor(self, c, step_num, y, init_states, coverage_before=None, fertility=None):
        return self.build_decoder(c, y, mode=Decoder.BEAM_SEARCH,
                given_init_states=init_states, step_num=step_num, coverage_before=coverage_before, fertility=fertility)

    def build_next_states_computer(self, c, step_num, y, init_states, coverage_before=None, fertility=None):
        return self.build_decoder(c, y, mode=Decoder.SAMPLING,
                given_init_states=init_states, step_num=step_num, coverage_before=coverage_before, fertility=fertility)[2:]

class RNNEncoderDecoder(object):
    """This class encapsulates the translation model.

    The expected usage pattern is:
    >>> encdec = RNNEncoderDecoder(...)
    >>> encdec.build(...)
    >>> useful_smth = encdec.create_useful_smth(...)

    Functions from the create_smth family (except create_lm_model)
    when called complile and return functions that do useful stuff.
    """

    def __init__(self, state, rng,
            skip_init=False,
            compute_alignment=True):
        """Constructor.

        :param state:
            A state in the usual groundhog sense.
        :param rng:
            Random number generator. Something like numpy.random.RandomState(seed).
        :param skip_init:
            If True, all the layers are initialized with zeros. Saves time spent on
            parameter initialization if they are loaded later anyway.
        :param compute_alignment:
            If True, the alignment is returned by the decoder.
        """

        self.state = state
        self.rng = rng
        self.skip_init = skip_init
        self.compute_alignment = compute_alignment

    def build(self):
        logger.debug("Create input variables")
        self.x = TT.lmatrix('x')
        self.x_mask = TT.matrix('x_mask')
        self.y = TT.lmatrix('y')
        self.y_mask = TT.matrix('y_mask')
        self.inputs = [self.x, self.y, self.x_mask, self.y_mask]

        # Annotation for the log-likelihood computation
        training_c_components = []

        logger.debug("Create encoder")
        self.encoder = Encoder(self.state, self.rng,
                prefix="enc",
                skip_init=self.skip_init)
        self.encoder.create_layers()

        logger.debug("Build encoding computation graph")
        forward_training_c = self.encoder.build_encoder(
                self.x, self.x_mask,
                use_noise=True,
                return_hidden_layers=True)

        logger.debug("Create backward encoder")
        self.backward_encoder = Encoder(self.state, self.rng,
                prefix="back_enc",
                skip_init=self.skip_init)
        self.backward_encoder.create_layers()

        logger.debug("Build backward encoding computation graph")
        backward_training_c = self.backward_encoder.build_encoder(
                self.x[::-1],
                self.x_mask[::-1],
                use_noise=True,
                approx_embeddings=self.encoder.approx_embedder(self.x[::-1]),
                return_hidden_layers=True)
        # Reverse time for backward representations.
        backward_training_c.out = backward_training_c.out[::-1]

        if self.state['forward']:
            training_c_components.append(forward_training_c)
        if self.state['last_forward']:
            training_c_components.append(
                    ReplicateLayer(self.x.shape[0])(forward_training_c[-1]))
        if self.state['backward']:
            training_c_components.append(backward_training_c)
        if self.state['last_backward']:
            training_c_components.append(ReplicateLayer(self.x.shape[0])
                    (backward_training_c[0]))
        self.state['c_dim'] = len(training_c_components) * self.state['dim']
        
        logger.debug("Create decoder")
        self.decoder = Decoder(self.state, self.rng,
                skip_init=self.skip_init, compute_alignment=self.compute_alignment)
        self.decoder.create_layers()
        logger.debug("Build log-likelihood computation graph")
        self.predictions, self.alignment = self.decoder.build_decoder(
                c=Concatenate(axis=2)(*training_c_components), c_mask=self.x_mask,
                y=self.y, y_mask=self.y_mask)

        # Annotation for sampling
        sampling_c_components = []

        logger.debug("Build sampling computation graph")
        self.sampling_x = TT.lvector("sampling_x")
        self.n_samples = TT.lscalar("n_samples")
        self.n_steps = TT.lscalar("n_steps")
        self.T = TT.scalar("T")
        self.forward_sampling_c = self.encoder.build_encoder(
                self.sampling_x,
                return_hidden_layers=True)
        self.backward_sampling_c = self.backward_encoder.build_encoder(
                self.sampling_x[::-1],
                approx_embeddings=self.encoder.approx_embedder(self.sampling_x[::-1]),
                return_hidden_layers=True).out[::-1]
        if self.state['forward']:
            sampling_c_components.append(self.forward_sampling_c)
        if self.state['last_forward']:
            sampling_c_components.append(ReplicateLayer(self.sampling_x.shape[0])
                    (self.forward_sampling_c[-1]))
        if self.state['backward']:
            sampling_c_components.append(self.backward_sampling_c)
        if self.state['last_backward']:
            sampling_c_components.append(ReplicateLayer(self.sampling_x.shape[0])
                    (self.backward_sampling_c[0]))

        self.sampling_c = Concatenate(axis=1)(*sampling_c_components).out
        sample_results, self.sampling_updates =\
            self.decoder.build_sampler(self.n_samples, self.n_steps, self.T,
                    c=self.sampling_c)
        self.sample = sample_results[0]
        self.sample_log_prob = sample_results[1]
        if self.state['maintain_coverage']:
            self.sample_coverage = sample_results[2]
            if self.state['use_fertility_model'] and self.state['use_linguistic_coverage']:
                self.sample_fertility = sample_results[3]
 
        logger.debug("Create auxiliary variables")
        self.c = TT.matrix("c")
        self.step_num = TT.lscalar("step_num")
        self.current_states = [TT.matrix("cur_{}".format(i))
                for i in range(self.decoder.num_levels)]
        self.gen_y = TT.lvector("gen_y")
        # added by Zhaopeng Tu, 2015-11-02
        self.coverage_before = TT.tensor3("coverage_before")
        # added by Zhaopeng Tu, 2015-12-17
        self.fertility = TT.matrix("fertility")


    def create_lm_model(self):
        if hasattr(self, 'lm_model'):
            return self.lm_model
        self.lm_model = LM_Model(
            cost_layer=self.predictions,
            sample_fn=self.create_sampler(),
            weight_noise_amount=self.state['weight_noise_amount'],
            indx_word=self.state['indx_word_target'],
            indx_word_src=self.state['indx_word'],
            rng=self.rng)
        self.lm_model.load_dict(self.state)
        logger.debug("Model params:\n{}".format(
            pprint.pformat(sorted([p.name for p in self.lm_model.params]))))
        return self.lm_model

    def create_representation_computer(self):
        if not hasattr(self, "repr_fn"):
            self.repr_fn = theano.function(
                    inputs=[self.sampling_x],
                    outputs=[self.sampling_c],
                    name="repr_fn")
        return self.repr_fn

    # added by Zhaopeng Tu, 2015-12-17
    # for fertility model
    def create_fertility_computer(self):
        if not hasattr(self, "fert_fn"):
            self.fert_fn = theano.function(
                    inputs=[self.sampling_c],
                    outputs=self.decoder.build_fertility_computer(self.sampling_c),
                    name="fert_fn")
        return self.fert_fn


    def create_initializers(self):
        if not hasattr(self, "init_fn"):
            init_c = self.sampling_c[0, -self.state['dim']:]
            self.init_fn = theano.function(
                    inputs=[self.sampling_c],
                    outputs=self.decoder.build_initializers(init_c),
                    name="init_fn")
        return self.init_fn

    def create_sampler(self, many_samples=False):
        if hasattr(self, 'sample_fn'):
            return self.sample_fn
        logger.debug("Compile sampler")
        outputs = [self.sample, self.sample_log_prob]
        # added by Zhaopeng Tu, 2015-12-09
        if self.state['maintain_coverage']:
            outputs.append(self.sample_coverage)
            if self.state['use_fertility_model'] and self.state['use_linguistic_coverage']:
                outputs.append(self.sample_fertility)

        self.sample_fn = theano.function(
                inputs=[self.n_samples, self.n_steps, self.T, self.sampling_x],
                outputs=outputs,
                updates=self.sampling_updates,
                name="sample_fn")
        if not many_samples:
            def sampler(*args):
                # squeeze: Remove broadcastable dimensions from the shape of an array.
                # thus coverage downcasts from 3D to 2D, since the coverage_dim is 1
                return map(lambda x : x.squeeze(), self.sample_fn(1, *args))
            return sampler
        return self.sample_fn

    def create_scorer(self, batch=False):
        if not hasattr(self, 'score_fn'):
            logger.debug("Compile scorer")
            self.score_fn = theano.function(
                    inputs=self.inputs,
                    outputs=[-self.predictions.cost_per_sample],
                    name="score_fn")
        if batch:
            return self.score_fn
        def scorer(x, y):
            x_mask = numpy.ones(x.shape[0], dtype="float32")
            y_mask = numpy.ones(y.shape[0], dtype="float32")
            return self.score_fn(x[:, None], y[:, None],
                    x_mask[:, None], y_mask[:, None])
        return scorer

    def create_next_probs_computer(self):
        if not hasattr(self, 'next_probs_fn'):
            self.next_probs_fn = theano.function(
                    inputs=[self.c, self.step_num, self.gen_y] + self.current_states + [self.coverage_before, self.fertility],
                    outputs=self.decoder.build_next_probs_predictor(
                        self.c, self.step_num, self.gen_y, self.current_states, self.coverage_before, self.fertility),
                    name="next_probs_fn")
        return self.next_probs_fn

    def create_next_states_computer(self):
        if not hasattr(self, 'next_states_fn'):
            self.next_states_fn = theano.function(
                    inputs=[self.c, self.step_num, self.gen_y] + self.current_states + [self.coverage_before, self.fertility],
                    outputs=self.decoder.build_next_states_computer(
                        self.c, self.step_num, self.gen_y, self.current_states, self.coverage_before, self.fertility),
                    name="next_states_fn")
        return self.next_states_fn


    def create_probs_computer(self, return_alignment=False):
        if not hasattr(self, 'probs_fn'):
            logger.debug("Compile probs computer")
            self.probs_fn = theano.function(
                    inputs=self.inputs,
                    outputs=[self.predictions.word_probs, self.alignment],
                    name="probs_fn")
        def probs_computer(x, y):
            x_mask = numpy.ones(x.shape[0], dtype="float32")
            y_mask = numpy.ones(y.shape[0], dtype="float32")
            probs, alignment = self.probs_fn(x[:, None], y[:, None],
                    x_mask[:, None], y_mask[:, None])
            if return_alignment:
                return probs, alignment
            else:
                return probs
        return probs_computer

def parse_input(state, word2idx, line, raise_unk=False, idx2word=None, unk_sym=-1, null_sym=-1):
    if unk_sym < 0:
        unk_sym = state['unk_sym_source']
    if null_sym < 0:
        null_sym = state['null_sym_source']
    seqin = line.split()
    seqlen = len(seqin)
    seq = numpy.zeros(seqlen+1, dtype='int64')
    for idx,sx in enumerate(seqin):
        seq[idx] = word2idx.get(sx, unk_sym)
        if seq[idx] >= state['n_sym_source']:
            seq[idx] = unk_sym
        if seq[idx] == unk_sym and raise_unk:
            raise Exception("Unknown word {}".format(sx))

    seq[-1] = null_sym
    if idx2word:
        idx2word[null_sym] = '<eos>'
        idx2word[unk_sym] = state['oov']
        parsed_in = [idx2word[sx] for sx in seq]
        return seq, " ".join(parsed_in)

    return seq, seqin


def parse_target(state, word2idx, line, raise_unk=False, idx2word=None, unk_sym=-1, null_sym=-1):
    if unk_sym < 0:
        unk_sym = state['unk_sym_target']
    if null_sym < 0:
        null_sym = state['null_sym_target']
    seqin = line.split()
    seqlen = len(seqin)
    seq = numpy.zeros(seqlen+1, dtype='int64')
    for idx,sx in enumerate(seqin):
        seq[idx] = word2idx.get(sx, unk_sym)
        if seq[idx] >= state['n_sym_target']:
            seq[idx] = unk_sym
        if seq[idx] == unk_sym and raise_unk:
            raise Exception("Unknown word {}".format(sx))

    seq[-1] = null_sym
    if idx2word:
        idx2word[null_sym] = '<eos>'
        idx2word[unk_sym] = state['oov']
        parsed_in = [idx2word[sx] for sx in seq]
        return seq, " ".join(parsed_in)

    return seq, seqin
