#!/usr/bin/env python
# encoding: utf-8

import os

K = [5, 10, 20]
iters = 120
methods = ['baseline', 'sparse', 'alias']

for met in iter(methods):
    for k in iter(K):
        logname = 'log_' + met + '_' + str(k) + 'k.txt'
        cmd = 'python efficient-lda.py %s %d %d %s' % (met, k, iters, logname)
        os.system(cmd)