NMT-Coverage by Zhaopeng Tu
===========================

This is an implementation of the coverage model for attention-based NMT. More technique details please refer to the following paper:

Zhaopeng Tu, Zhengdong Lu, Yang Liu, Xiaohua Liu, Hang Li. Modeling Coverage for Neural Machine Translation. ACL 2016.

NMT-Coverage is on top of lisa-groudhog (https://github.com/lisa-groundhog/GroundHog). It requires Theano0.9 or above version

Installation
------------
To install NMT-Coverage in a multi-user setting

``python setup.py develop --user``

For general installation, simply use

``python setup.py develop``

NOTE: This will install the development version of Theano, if Theano is not currently installed.


How to Run?
--------------------------

See experiments/nmt/README.md

