NMT-Coverage
===========================

We are still in the process of releasing our neural machine translation (NMT) code, which alleviates the problem of fluent but inadequate translations that NMT suffers.
In this version, we introduce a coverage mechanism (NMT-Coverage) to indicate whether a source word is translated or not, which proves to alleviate over-translation and under-translation.

If you use our code, please cite our paper:

<pre><code>@InProceedings{Tu:2016:ACL,
  author    = {Tu, Zhaopeng and Lu, Zhengdong and Liu, Yang and Liu, Xiaohua and Li, Hang},
  title     = {Modeling Coverage for Neural Machine Translation},
  booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics},
  year      = {2016},
}
</code></pre>

We are developing another mechanism to further alleviate the inadequate translations, which is complementary to the coverage model. The paper and code will be released soon.

For any comments or questions, please  email <a href="mailto:tuzhaopeng@gmail.com">the first author</a>.


Installation
------------

NMT-Coverage is developed by <a href="http://www.zptu.net">Zhaopeng Tu</a>, which is on top of lisa-groudhog (https://github.com/lisa-groundhog/GroundHog). It requires Theano0.9 or above version.

To install NMT-Coverage in a multi-user setting

``python setup.py develop --user``

For general installation, simply use

``python setup.py develop``

NOTE: This will install the development version of Theano, if Theano is not currently installed.


How to Run?
--------------------------

See experiments/nmt/README.md

