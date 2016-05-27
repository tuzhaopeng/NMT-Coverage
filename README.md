NMT-Coverage
===========================

We are still in the process of releasing our neural machine translation (NMT) code, which alleviates the problem of fluent but inadequate translations that NMT suffers.
In this version, we introduce a coverage mechanism (NMT-Coverage) to indicate whether a source word is translated or not, which proves to alleviate over-translation and under-translation.

If you use our code, please cite it:
> <a href="http://www.zptu.net">Zhaopeng Tu</a>, Zhengdong Lu, Yang Liu, Xiaohua Liu, Hang Li. Modeling Coverage for Neural Machine Translation. <i>ACL 2016</i>.

We are developing another mechanism to further alleviate the inadequate translations, which is complementary to the coverage model. The paper and code will be released soon.

For any comments or questions, please  email <a href="mailto:tuzhaopeng@gmail.com">the first author</a>.


Installation
------------

NMT-Coverage is on top of lisa-groudhog (https://github.com/lisa-groundhog/GroundHog). It requires Theano0.9 or above version.

To install NMT-Coverage in a multi-user setting

``python setup.py develop --user``

For general installation, simply use

``python setup.py develop``

NOTE: This will install the development version of Theano, if Theano is not currently installed.


How to Run?
--------------------------

See experiments/nmt/README.md

