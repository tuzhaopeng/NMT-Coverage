dict(
source=["/home/zptu/research/coverage/data/gq/binarized_text.zh.shuf.h5"],
target=["/home/zptu/research/coverage/data/gq/binarized_text.en.shuf.h5"],
indx_word="/home/zptu/research/coverage/data/gq/ivocab.zh.pkl",
indx_word_target="/home/zptu/research/coverage/data/gq/ivocab.en.pkl",
word_indx="/home/zptu/research/coverage/data/gq/vocab.zh.pkl",
word_indx_trgt="/home/zptu/research/coverage/data/gq/vocab.en.pkl",
null_sym_source=16000,
null_sym_target=16000,
n_sym_source=16001,
n_sym_target=16001,
loopIters=1000000,
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
# for linguistic coverage, the dim can only be 1
coverage_dim=10,

#-----------------------
use_linguistic_coverage=False,
# added by Zhaopeng Tu, 2015-12-16
use_fertility_model=True,
max_fertility=2,
coverage_accumulated_operation = "additive",
##########
use_recurrent_coverage=True,
use_recurrent_gating_coverage=True,
use_probability_for_recurrent_coverage=True,
use_input_annotations_for_recurrent_coverage=True,
use_decoding_state_for_recurrent_coverage=True,
)
