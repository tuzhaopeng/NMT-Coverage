#!/usr/bin/env python

import argparse
import cPickle
import traceback
import logging
import time
import sys

import numpy

import experiments.nmt
from experiments.nmt import\
    RNNEncoderDecoder,\
    prototype_state,\
    prototype_search_with_coverage_state,\
    parse_input, parse_target

from experiments.nmt.numpy_compat import argpartition

numpy.set_printoptions(threshold=numpy.nan)

logger = logging.getLogger(__name__)

class Timer(object):

    def __init__(self):
        self.total = 0

    def start(self):
        self.start_time = time.time()

    def finish(self):
        self.total += time.time() - self.start_time

class BeamSearch(object):

    def __init__(self, enc_dec):
        self.enc_dec = enc_dec
        state = self.enc_dec.state
        self.eos_id = state['null_sym_target']
        self.unk_id = state['unk_sym_target']

    def compile(self):
        self.comp_repr = self.enc_dec.create_representation_computer()
        # added by Zhaopeng Tu, 2015-12-17, for fertility
        if self.enc_dec.state['maintain_coverage'] and self.enc_dec.state['use_linguistic_coverage'] and self.enc_dec.state['use_fertility_model']:
            self.comp_fert = self.enc_dec.create_fertility_computer()
        self.comp_init_states = self.enc_dec.create_initializers()
        self.comp_next_probs = self.enc_dec.create_next_probs_computer()
        self.comp_next_states = self.enc_dec.create_next_states_computer()

    def search(self, seq, out, ignore_unk=False, minlen=1):
        c = self.comp_repr(seq)[0]
        states = map(lambda x : x[None, :], self.comp_init_states(c))
        dim = states[0].shape[1]
        # added by Zhaopeng Tu, 2015-11-02
        if self.enc_dec.state['maintain_coverage']:
            coverage_dim = self.enc_dec.state['coverage_dim']
            if self.enc_dec.state['use_linguistic_coverage'] and self.enc_dec.state['coverage_accumulated_operation'] == 'subtractive':
                coverages = numpy.ones((c.shape[0], 1, coverage_dim), dtype='float32')
            else:
                coverages = numpy.zeros((c.shape[0], 1, coverage_dim), dtype='float32')
        else:
            coverages = None
        
        if self.enc_dec.state['maintain_coverage'] and self.enc_dec.state['use_linguistic_coverage'] and self.enc_dec.state['use_fertility_model']:
            fertility = self.comp_fert(c)
        else:
            fertility = None

        num_levels = len(states)

        aligns = []
        costs = [0.0]

        for k in range(len(out)):
            # Compute probabilities of the next words for
            # all the elements of the beam.
            last_words = (numpy.array([out[k-1]])
                    if k > 0
                    else numpy.zeros(1, dtype="int64"))

            results = self.comp_next_probs(c, k, last_words, *states, coverage_before=coverages, fertility=fertility)
            log_probs = numpy.log(results[0])
            alignment = results[1]
            # alignment shape: (source_len, target_num) where target_num = 1
            aligns.append(alignment[:,0])

            # Adjust log probs according to search restrictions
            if ignore_unk:
                log_probs[:,self.unk_id] = -numpy.inf
            # TODO: report me in the paper!!!
            if k < minlen:
                log_probs[:,self.eos_id] = -numpy.inf

            # costs = numpy.array(costs)[:, None] - log_probs

            inputs = numpy.array([out[k]])
            states = self.comp_next_states(c, k, inputs, *states, coverage_before=coverages, fertility=fertility)
            if self.enc_dec.state['maintain_coverage']:
                coverages = states[-1]
                states = states[:-1]

        if self.enc_dec.state['maintain_coverage']:
            coverage = coverages[:,0,0]
        # aligns shape:  (target_len, source_len)
        # we reverse it to the shape (source_len, target_len) to show the matrix
        aligns = numpy.array(aligns).transpose()

        if self.enc_dec.state['maintain_coverage']:
            if self.enc_dec.state['use_linguistic_coverage'] and self.enc_dec.state['use_fertility_model']:
                return aligns, costs, coverage, fertility
            else:
                return aligns, costs, coverage
        else:
            return aligns, costs

def indices_to_words(i2w, seq):
    sen = []
    for k in xrange(len(seq)):
        if i2w[seq[k]] == '<eol>':
            break
        sen.append(i2w[seq[k]])
    return sen

def force_decoding(lm_model, seq, out,
        sampler=None, beam_search=None,
        ignore_unk=False, normalize=False,
        alpha=1, verbose=False):
    if lm_model.maintain_coverage:
        if lm_model.use_linguistic_coverage and lm_model.use_fertility_model:
            aligns, costs, coverage, fertility = beam_search.search(seq, out,
                    ignore_unk=ignore_unk)
        else:
            aligns, costs, coverage = beam_search.search(seq, out,
                    ignore_unk=ignore_unk)
    else:
        aligns, costs = beam_search.search(seq, out,
                ignore_unk=ignore_unk)
    if normalize:
        costs = [co / len(out) for co in costs]

    if lm_model.maintain_coverage:
        if lm_model.use_linguistic_coverage and lm_model.use_fertility_model:
            return aligns, costs, coverage, fertility
        else:
            return aligns, costs, coverage
    else:
        return aligns, costs


def parse_args():
    parser = argparse.ArgumentParser(
            "Force decoding a sentence pair to output the alignments")
    parser.add_argument("--state",
            required=True, help="State to use")
    parser.add_argument("--ignore-unk",
            default=False, action="store_true",
            help="Ignore unknown words")
    parser.add_argument("--source",
            help="File of source sentences")
    parser.add_argument("--target",
            help="File of target sentences")
    parser.add_argument("--aligns",
            help="File to save alignments in")
    parser.add_argument("--normalize",
            action="store_true", default=False,
            help="Normalize log-prob with the word count")
    parser.add_argument("--verbose",
            action="store_true", default=False,
            help="Be verbose")
    parser.add_argument("model_path",
            help="Path to the model")
    parser.add_argument("changes",
            nargs="?", default="",
            help="Changes to state")
    return parser.parse_args()

def main():
    args = parse_args()

    state = prototype_search_with_coverage_state()
    with open(args.state) as src:
        state.update(cPickle.load(src))
    state.update(eval("dict({})".format(args.changes)))

    logging.basicConfig(level=getattr(logging, state['level']), format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    rng = numpy.random.RandomState(state['seed'])
    enc_dec = RNNEncoderDecoder(state, rng, skip_init=True, compute_alignment=True)
    enc_dec.build()
    lm_model = enc_dec.create_lm_model()
    lm_model.load(args.model_path)
    indx_word = cPickle.load(open(state['word_indx'],'rb'))
    t_indx_word = cPickle.load(open(state['word_indx_trgt'], 'rb'))

    sampler = None
    beam_search = BeamSearch(enc_dec)
    beam_search.compile()

    idict_src = cPickle.load(open(state['indx_word'],'r'))
    t_idict_src = cPickle.load(open(state['indx_word_target'],'r'))


    fsrc = open(args.source, 'r')
    ftrg = open(args.target, 'r')

    start_time = time.time()

    total_cost = 0.0
    # for i, line in enumerate(fsrc):
    i = 0
    while 1:
        try:
            seqin = fsrc.next().strip()
            seqout = ftrg.next().strip()
        except StopIteration:
            break

        seq, parsed_in = parse_input(state, indx_word, seqin, idx2word=idict_src)
        out, parsed_out = parse_target(state, t_indx_word, seqout, idx2word=t_idict_src)

        if lm_model.maintain_coverage:
            if lm_model.use_linguistic_coverage and lm_model.use_fertility_model:
                aligns, costs, coverage, fertility = force_decoding(lm_model, seq, out, sampler=sampler,
                        beam_search=beam_search, ignore_unk=args.ignore_unk, normalize=args.normalize)
            else:
                aligns, costs, coverage = force_decoding(lm_model, seq, out, sampler=sampler,
                        beam_search=beam_search, ignore_unk=args.ignore_unk, normalize=args.normalize)
        else:
            aligns, costs = force_decoding(lm_model, seq, out, sampler=sampler,
                    beam_search=beam_search, ignore_unk=args.ignore_unk, normalize=args.normalize)
        
        print "Parsed Input:", parsed_in
        print "Parsed Target:", parsed_out
        print 'Aligns:'
        print aligns.tolist()


        if lm_model.maintain_coverage:
            # since we filtered <eos> from trans[best], thus the index adds 1
            print "Coverage:", 
            words = parsed_in.split()
            for k in xrange(len(words)):
                print '%s/%.2f'%(words[k], coverage[k]),
            print ''

            if lm_model.use_linguistic_coverage and lm_model.use_fertility_model:
                print 'Fertility:  ',
                for k in xrange(len(words)):
                    print '%s/%.2f'%(words[k], fertility[k]),
                print ''
        print 

        total_cost += costs[0]
        if (i + 1)  % 100 == 0:
            logger.debug("Current speed is {} per sentence".
                    format((time.time() - start_time) / (i + 1)))
    print "Total cost of the translations: {}".format(total_cost)

    fsrc.close()
    ftrg.close()

if __name__ == "__main__":
    main()
