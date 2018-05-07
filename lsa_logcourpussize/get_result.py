from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import cPickle as pickle
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

def rSquared(coeffs, x, y):
  p = np.poly1d(coeffs)
  yhat = p(x)                         # or [p(z) for z in x]
  ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
  ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
  sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
  return ssreg / sstot



num_retain = 20
sizes = []
opt = collections.defaultdict(dict)
for path, _, _ in os.walk('.'):
  if 'test_result' not in path or 'logcount'in path:
    continue
  dir_name = path
  lens = int(dir_name.split('_')[-1])
  # corpus size smaller than 1.5m usually is not sufficient
  if lens < 1500000:
    continue
  run_idx = 0
  while True:
    try:
      with open('{}/test_result_{}.pkl'.format(dir_name, run_idx), 'r') as f:
        sim_test = pickle.load(f)
        for k, result in sim_test.iteritems():
          if lens not in opt[k]:
            opt[k][lens] = []
          sorted_result = sorted(enumerate(result), key=lambda x:x[1], reverse=True)
          opt_dims = [x[0] for x in sorted_result[:num_retain]]
          opt[k][lens].append(np.mean(opt_dims))
      run_idx += 1
    except:
      break
means = collections.defaultdict(list)
stds = collections.defaultdict(list)
for k, v in opt.iteritems():
  sizes = []
  for key in sorted(v.keys()):
    print('{}: {}'.format(key, v[key]))
    means[k].append(np.mean(v[key]))
    stds[k].append(np.std(v[key]))
    sizes.append(key)

sizes = np.log(np.array(sizes))
m, b = np.polyfit(sizes, means['wordsim353.csv'], 1)
print('slope={}, intercept={}'.format(m, b))
rsquared = rSquared([m, b], sizes, means['wordsim353.csv'])
print('r squared={}'.format(rsquared))
fig = plt.figure()
plt.plot(sizes, means['wordsim353.csv'], '.')
plt.plot(sizes, m*sizes+b, '-')
plt.xlabel('log of vocabulary size')
plt.ylabel('optimal dimensionality')
fig.tight_layout()
plt.savefig('ws353_vs_corpussize.pdf')
plt.clf()
plt.close()
m, b = np.polyfit(sizes, means['mturk771.csv'], 1)
print('slope={}, intercept={}'.format(m, b))
rsquared = rSquared([m, b], sizes, means['mturk771.csv'])
print('r squared={}'.format(rsquared))
fig = plt.figure()
plt.plot(sizes, means['mturk771.csv'], '.')
plt.plot(sizes, m*sizes+b, '-')
plt.xlabel('log of vocabulary size')
plt.ylabel('optimal dimensionality')
fig.tight_layout()
plt.savefig('mturk771_vs_corpussize.pdf')


