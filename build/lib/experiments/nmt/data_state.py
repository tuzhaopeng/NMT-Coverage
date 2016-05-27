dict(
source=["/home/zptu/research/nmt.coverage/data/fbis/binarized_text.zh.h5"],
target=["/home/zptu/research/nmt.coverage/data/fbis/binarized_text.en.h5"],
indx_word="/home/zptu/research/nmt.coverage/data/fbis/ivocab.zh.pkl",
indx_word_target="/home/zptu/research/nmt.coverage/data/fbis/ivocab.en.pkl",
word_indx="/home/zptu/research/nmt.coverage/data/fbis/vocab.zh.pkl",
word_indx_trgt="/home/zptu/research/nmt.coverage/data/fbis/vocab.en.pkl",
null_sym_source=30000,
null_sym_target=30000,
n_sym_source=30001,
n_sym_target=30001,
loopIters=200000,
seqlen=50,
bs=80,
dim=1000,
saveFreq=30,
last_forward = False,
forward = True,
backward = True,
last_backward = False,
##########
# for coverage
maintain_coverage=True,
# for accumulated coverage, the dim can only be 1
coverage_dim=1,

use_coverage_cost = False

# Hard Alignment: at each step, set the align of highest probability to be 1.0, and other aligns to be 0.0; default is soft coverage that uses the real probabilities of aligns
use_hard_alignment=False,

use_coverage_for_alignment=True,
# not recommended, for alignment only yields better performance
use_coverage_for_decoding=False,


#-----------------------
use_accumulated_coverage=False,
# all the below options are for coverage model I -- simple coverage
# Upper bound of the value is 1.0 (for additive) or 0.0 (for subtractive), to eliminate the effect of input sentence length
use_accumulated_coverage_bound=False,
# we define 4 types of accumulated operation, for each position
# additive: sum up the alignment probabilities in the past (coverage starts with 0.0) (default)
# subtractive: minus the alignment probabilities in the past (coverage starts with 1.0)
# max-pooling: use the most representative value (the highest probability till now)(coverage starts with 0.0)
# mean-pooling: use the mean of the alignment probabilities in the past (coverage starts with 0.0)
coverage_accumulated_operation = "additive",


#-----------------------
# settings for recurrent_coverage
use_recurrent_coverage=False,
use_input_annotations_for_recurrent_coverage=False,
use_decoding_state_for_recurrent_coverage=False,
use_recurrent_gating_coverage=True,
##########
)
