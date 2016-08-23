NMT-Coverage
===========================

We are still in the process of releasing our neural machine translation (NMT) code, which alleviates the problem of fluent but inadequate translations that NMT suffers.

**Please refer to an <a href="https://github.com/tuzhaopeng/NMT">improved NMT</a>, which incorporates context gates to obtain a further improvement of 1.6 BLEU over NMT-Coverage**.


In this version, we introduce a coverage mechanism (NMT-Coverage) to indicate whether a source word is translated or not, which proves to alleviate over-translation and under-translation. If you use the code, please cite <a href="http://arxiv.org/abs/1601.04811">our paper</a>:

<pre><code>@InProceedings{Tu:2016:ACL,
  author    = {Tu, Zhaopeng and Lu, Zhengdong and Liu, Yang and Liu, Xiaohua and Li, Hang},
  title     = {Modeling Coverage for Neural Machine Translation},
  booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics},
  year      = {2016},
}
</code></pre>

For any comments or questions, please  email <a href="mailto:tuzhaopeng@gmail.com">the first author</a>.


Installation
------------

NMT-Coverage is developed by <a href="http://www.zptu.net">Zhaopeng Tu</a>, which is on top of lisa-groudhog (https://github.com/lisa-groundhog/GroundHog). It requires Theano0.8 or above version (for the module "scan" used in the trainer).

To install NMT-Coverage in a multi-user setting

``python setup.py develop --user``

For general installation, simply use

``python setup.py develop``

NOTE: This will install the development version of Theano, if Theano is not currently installed.


How to Run?
--------------------------

See experiments/nmt/README.md

