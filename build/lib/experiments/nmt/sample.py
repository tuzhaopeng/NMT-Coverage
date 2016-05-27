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
    parse_input

from experiments.nmt.numpy_compat import argpartition

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
        self.comp_init_states = self.enc_dec.create_initializers()
        self.comp_next_probs = self.enc_dec.create_next_probs_computer()
        self.comp_next_states = self.enc_dec.create_next_states_computer()

    def search(self, seq, n_samples, ignore_unk=False, minlen=1):
        c = self.comp_repr(seq)[0]
        states = map(lambda x : x[None, :], self.comp_init_states(c))
        dim = states[0].shape[1]
        # added by Zhaopeng Tu, 2015-11-02
        if self.enc_dec.state['maintain_coverage']:
            # in sampling, init_c is two-dimension (source_length*c_dim), same for init_coverage
            if self.enc_dec.state['use_accumulated_coverage'] and self.enc_dec.state['coverage_accumulated_operation'] == 'subtractive':
                coverage_before = numpy.ones((c.shape[0], self.enc_dec.state['coverage_dim']), dtype='float32')
            else:
                coverage_before = numpy.zeros((c.shape[0], self.enc_dec.state['coverage_dim']), dtype='float32')
            coverages = [coverage_before[:,0]]
        else:
            coverage_before = None

        num_levels = len(states)

        fin_trans = []
        fin_costs = []

        trans = [[]]
        costs = [0.0]

        for k in range(3 * len(seq)):
            if n_samples == 0:
                break

            # Compute probabilities of the next words for
            # all the elements of the beam.
            beam_size = len(trans)
            last_words = (numpy.array(map(lambda t : t[-1], trans))
                    if k > 0
                    else numpy.zeros(beam_size, dtype="int64"))
            log_probs = numpy.log(self.comp_next_probs(c, k, last_words, *states, coverage_before=coverage_before)[0])

            # Adjust log probs according to search restrictions
            if ignore_unk:
                log_probs[:,self.unk_id] = -numpy.inf
            # TODO: report me in the paper!!!
            if k < minlen:
                log_probs[:,self.eos_id] = -numpy.inf

            # Find the best options by calling argpartition of flatten array
            next_costs = numpy.array(costs)[:, None] - log_probs
            flat_next_costs = next_costs.flatten()
            best_costs_indices = argpartition(
                    flat_next_costs.flatten(),
                    n_samples)[:n_samples]

            # Decypher flatten indices
            voc_size = log_probs.shape[1]
            trans_indices = best_costs_indices / voc_size
            word_indices = best_costs_indices % voc_size
            costs = flat_next_costs[best_costs_indices]

            # Form a beam for the next iteration
            new_trans = [[]] * n_samples
            new_costs = numpy.zeros(n_samples)
            new_states = [numpy.zeros((n_samples, dim), dtype="float32") for level
                    in range(num_levels)]
            inputs = numpy.zeros(n_samples, dtype="int64")
            for i, (orig_idx, next_word, next_cost) in enumerate(
                    zip(trans_indices, word_indices, costs)):
                new_trans[i] = trans[orig_idx] + [next_word]
                new_costs[i] = next_cost
                for level in range(num_levels):
                    new_states[level][i] = states[level][orig_idx]
                inputs[i] = next_word
            new_states = self.comp_next_states(c, k, inputs, *new_states, coverage_before=coverage_before)
            if self.enc_dec.state['maintain_coverage']:
                coverage_before = new_states[-1]
                new_states = new_states[:-1]
                coverages.append(coverage_before[:,0])

            # Filter the sequences that end with end-of-sequence character
            trans = []
            costs = []
            indices = []
            for i in range(n_samples):
                if new_trans[i][-1] != self.enc_dec.state['null_sym_target']:
                    trans.append(new_trans[i])
                    costs.append(new_costs[i])
                    indices.append(i)
                else:
                    n_samples -= 1
                    fin_trans.append(new_trans[i])
                    fin_costs.append(new_costs[i])
            states = map(lambda x : x[indices], new_states)

        # Dirty tricks to obtain any translation
        if not len(fin_trans):
            if ignore_unk:
                logger.warning("Did not manage without UNK")
                return self.search(seq, n_samples, False, minlen)
            elif n_samples < 100:
                logger.warning("Still no translations: try beam size {}".format(n_samples * 2))
                return self.search(seq, n_samples * 2, False, minlen)
            else:
                fin_trans = trans
                fin_costs = costs
                logger.error("Translation failed")

        fin_trans = numpy.array(fin_trans)[numpy.argsort(fin_costs)]
        fin_costs = numpy.array(sorted(fin_costs))
        if self.enc_dec.state['maintain_coverage']:
            return fin_trans, fin_costs, coverages
        else:
            return fin_trans, fin_costs

def indices_to_words(i2w, seq):
    sen = []
    for k in xrange(len(seq)):
        if i2w[seq[k]] == '<eol>':
            break
        sen.append(i2w[seq[k]])
    return sen

def sample(lm_model, seq, n_samples,
        sampler=None, beam_search=None,
        ignore_unk=False, normalize=False,
        alpha=1, verbose=False):
    if beam_search:
        sentences = []
        if lm_model.maintain_coverage:
            trans, costs, coverages = beam_search.search(seq, n_samples,
                    ignore_unk=ignore_unk, minlen=len(seq) / 2)
        else:
            trans, costs = beam_search.search(seq, n_samples,
                    ignore_unk=ignore_unk, minlen=len(seq) / 2)
        if normalize:
            counts = [len(s) for s in trans]
            costs = [co / cn for co, cn in zip(costs, counts)]
        for i in range(len(trans)):
            sen = indices_to_words(lm_model.word_indxs, trans[i])
            sentences.append(" ".join(sen))
        for i in range(len(costs)):
            if verbose:
                print "{}: {}".format(costs[i], sentences[i])
        if lm_model.maintain_coverage:
            return sentences, costs, coverages, trans
        else:
            return sentences, costs, trans
    elif sampler:
        sentences = []
        all_probs = []
        costs = []

        values, cond_probs = sampler(n_samples, 3 * (len(seq) - 1), alpha, seq)
        for sidx in xrange(n_samples):
            sen = []
            for k in xrange(values.shape[0]):
                if lm_model.word_indxs[values[k, sidx]] == '<eol>':
                    break
                sen.append(lm_model.word_indxs[values[k, sidx]])
            sentences.append(" ".join(sen))
            probs = cond_probs[:, sidx]
            probs = numpy.array(cond_probs[:len(sen) + 1, sidx])
            all_probs.append(numpy.exp(-probs))
            costs.append(-numpy.sum(probs))
        if normalize:
            counts = [len(s.strip().split(" ")) for s in sentences]
            costs = [co / cn for co, cn in zip(costs, counts)]
        sprobs = numpy.argsort(costs)
        if verbose:
            for pidx in sprobs:
                print "{}: {} {} {}".format(pidx, -costs[pidx], all_probs[pidx], sentences[pidx])
            print
        return sentences, costs, None
    else:
        raise Exception("I don't know what to do")


def parse_args():
    parser = argparse.ArgumentParser(
            "Sample (of find with beam-serch) translations from a translation model")
    parser.add_argument("--state",
            required=True, help="State to use")
    parser.add_argument("--beam-search",
            action="store_true", help="Beam size, turns on beam-search")
    parser.add_argument("--beam-size",
            type=int, help="Beam size")
    parser.add_argument("--ignore-unk",
            default=False, action="store_true",
            help="Ignore unknown words")
    parser.add_argument("--source",
            help="File of source sentences")
    parser.add_argument("--trans",
            help="File to save translations in")
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

    state = prototype_state()
    with open(args.state) as src:
        state.update(cPickle.load(src))
    state.update(eval("dict({})".format(args.changes)))

    logging.basicConfig(level=getattr(logging, state['level']), format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    rng = numpy.random.RandomState(state['seed'])
    enc_dec = RNNEncoderDecoder(state, rng, skip_init=True)
    enc_dec.build()
    lm_model = enc_dec.create_lm_model()
    lm_model.load(args.model_path)
    indx_word = cPickle.load(open(state['word_indx'],'rb'))

    sampler = None
    beam_search = None
    if args.beam_search:
        beam_search = BeamSearch(enc_dec)
        beam_search.compile()
    else:
        sampler = enc_dec.create_sampler(many_samples=True)

    idict_src = cPickle.load(open(state['indx_word'],'r'))

    if args.source and args.trans:
        # Actually only beam search is currently supported here
        assert beam_search
        assert args.beam_size

        fsrc = open(args.source, 'r')
        ftrans = open(args.trans, 'w')

        start_time = time.time()

        n_samples = args.beam_size
        total_cost = 0.0
        logging.debug("Beam size: {}".format(n_samples))
        for i, line in enumerate(fsrc):
            seqin = line.strip()
            seq, parsed_in = parse_input(state, indx_word, seqin, idx2word=idict_src)
            if lm_model.maintain_coverage:
                trans, costs, coverages, _ = sample(lm_model, seq, n_samples, sampler=sampler,
                        beam_search=beam_search, ignore_unk=args.ignore_unk, normalize=args.normalize)
            else:
                trans, costs, _ = sample(lm_model, seq, n_samples, sampler=sampler,
                        beam_search=beam_search, ignore_unk=args.ignore_unk, normalize=args.normalize)
            
            if args.verbose:
                print "Parsed Input:", parsed_in

            if len(trans) == 0:
                trans = ['Failed']
                costs = [0.0]

            best = numpy.argmin(costs)
            print >>ftrans, trans[best]
            if args.verbose:
                print "Translation:", trans[best]

                if lm_model.maintain_coverage:
                    # since we filtered <eos> from trans[best], thus the index adds 1
<<<<<<< HEAD
                    coverage = coverages[len(trans[best].split())+1]
=======
                    if trans[0] != 'Failed':
                        coverage = coverages[len(trans[best].split())+1]
                    else:
                        coverage = coverages[0]
>>>>>>> 2bd596987ce6b2c037771e7a8ef6e2f411a788a3
                    print "Coverage:", 
                    words = parsed_in.split()
                    for k in xrange(len(words)):
                        print '%s/%.2f'%(words[k], coverage[k]),
                    print ''
<<<<<<< HEAD
=======
                    print 
>>>>>>> 2bd596987ce6b2c037771e7a8ef6e2f411a788a3

            total_cost += costs[best]
            if (i + 1)  % 100 == 0:
                ftrans.flush()
                logger.debug("Current speed is {} per sentence".
                        format((time.time() - start_time) / (i + 1)))
        print "Total cost of the translations: {}".format(total_cost)

        fsrc.close()
        ftrans.close()
    else:
        while True:
            try:
                seqin = raw_input('Input Sequence: ')
                n_samples = int(raw_input('How many samples? '))
                alpha = None
                if not args.beam_search:
                    alpha = float(raw_input('Inverse Temperature? '))
                seq,parsed_in = parse_input(state, indx_word, seqin, idx2word=idict_src)
                print "Parsed Input:", parsed_in
            except Exception:
                print "Exception while parsing your input:"
                traceback.print_exc()
                continue

            sample(lm_model, seq, n_samples, sampler=sampler,
                    beam_search=beam_search,
                    ignore_unk=args.ignore_unk, normalize=args.normalize,
                    alpha=alpha, verbose=True)

if __name__ == "__main__":
    main()
