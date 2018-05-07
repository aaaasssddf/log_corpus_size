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
from sklearn import linear_model

def smooth(x,window_len=11,window='hanning'):
  if x.ndim != 1:
    raise ValueError, "smooth only accepts 1 dimension arrays."
  if x.size < window_len:
    raise ValueError, "Input vector needs to be bigger than window size."
  if window_len<3:
    return x
  if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
    raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
  s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
  if window == 'flat': #moving average
    w=np.ones(window_len,'d')
  else:  
    w=eval('np.'+window+'(window_len)')
  y=np.convolve(w/w.sum(),s,mode='same')
  return y[window_len:-window_len+1]

def rSquared(coeffs, x, y):
  p = np.poly1d(coeffs)
  yhat = p(x)                         # or [p(z) for z in x]
  ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
  ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
  sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
  return ssreg / sstot

def expWt(seq, base=0.1):
  wt = 1.0
  total = 0.0
  total_wt = 0.0
  seq = sorted(seq)
  for val in reversed(seq):
    total += val * wt
    total_wt += wt
    wt *= base
  return total / total_wt


def getResultSingleRun(data_dir):
  print("loading {}".format(data_dir))
  sizes = []
  subdirs = []
  for path in os.listdir(data_dir):
    if os.path.isdir(os.path.join(data_dir, path)):
      subdirs.append(os.path.join(data_dir, path))

  """ 
  each subdir is a corpus size.
  each corpus size contains skip-gram trained with
  different dimensionalities.
  each dimensionality contains the performance of the
  model on a set of tasks across all epochs.
  we extract performance for three tasks: Google analogy,
  wordsim353 and mturk771.
  """

  corpus_sizes = []
  tasks = {'analogy', 'ws353', 'mturk771'}
  analogy = collections.defaultdict(dict)
  ws353 = collections.defaultdict(dict)
  mturk771 = collections.defaultdict(dict)

  for subdir in reversed(subdirs):
    corpus_size = int(subdir.split('/')[-1])
    if corpus_size < 1500000:
      continue
    corpus_sizes.append(corpus_size)
    current_dir = '{}/analogy'.format(subdir)
    files = os.listdir(current_dir)
    for filename in files:
      if 'analogy_' in filename:
        dim = int(filename.split('_')[-1].split('.')[0])
        with open(os.path.join(current_dir, filename), 'r') as f:
          analogy_test = pickle.load(f)
        accuracy = expWt(analogy_test)
        analogy[corpus_size][dim] = accuracy

    files = os.listdir('{}/relatedness'.format(subdir))
    current_dir = '{}/relatedness'.format(subdir)
    for filename in files:
      if 'wordsim' in filename:
        dim = int(filename.split('_')[-1].split('.')[0])
        with open(os.path.join(current_dir, filename), 'r') as f:
          ws353_test = pickle.load(f)
        accuracy = expWt(ws353_test)
        ws353[corpus_size][dim] = accuracy

      if 'mturk' in filename:
        dim = int(filename.split('_')[-1].split('.')[0])
        with open(os.path.join(current_dir, filename), 'r') as f:
          mt771_test = pickle.load(f)
        accuracy = expWt(mt771_test)
        mturk771[corpus_size][dim] = accuracy

  analogy_mean = []
  analogy_std = []
  mturk771_mean = []
  mturk771_std = []
  ws353_mean = []
  ws353_std = []

  corpus_sizes = sorted(corpus_sizes)

  num_retain = 5

  for corpus_size in corpus_sizes:
    """ (k, v) = (dim, accuracy)"""
    analogy_result = [(k, v) for k, v in analogy[corpus_size].iteritems()]
    analogy_sorted_result = sorted(analogy_result, key=lambda x: x[1], reverse=True)
    analogy_opt_dims = [x[0] for x in analogy_sorted_result[:num_retain]]
    analogy_mean.append(np.mean(analogy_opt_dims))
    analogy_std.append(np.std(analogy_opt_dims))

    ws353_result = [(k, v) for k, v in ws353[corpus_size].iteritems()]
    ws353_sorted_result = sorted(ws353_result, key=lambda x: x[1], reverse=True)
    ws353_opt_dims = [x[0] for x in ws353_sorted_result[:num_retain]]
    ws353_mean.append(np.mean(ws353_opt_dims))
    ws353_std.append(np.std(ws353_opt_dims))
   
    mturk771_result = [(k, v) for k, v in mturk771[corpus_size].iteritems()]
    mturk771_sorted_result = sorted(mturk771_result, key=lambda x: x[1], reverse=True)
    mturk771_opt_dims = [x[0] for x in mturk771_sorted_result[:num_retain]]
    mturk771_mean.append(np.mean(mturk771_opt_dims))
    mturk771_std.append(np.std(mturk771_opt_dims))

  return corpus_sizes, ws353_mean, mturk771_mean, analogy_mean
 
if __name__ == "__main__":
  num_runs = 5
  c_sizes, w_mean, m_mean, a_mean = [], [], [], []
  for i in range(num_runs):
    data_dir = "data{}".format(i)
    c, w, m, a = getResultSingleRun(data_dir)
    c_sizes.append(c)
    w_mean.append(w)
    m_mean.append(m)
    a_mean.append(a)
  
  corpus_sizes = [np.mean(x) for x in zip(*c_sizes)]
  ws353_mean =  [np.mean(x) for x in zip(*w_mean)]
  mturk771_mean =  [np.mean(x) for x in zip(*m_mean)]
  analogy_mean =  [np.mean(x) for x in zip(*a_mean)]

  print(corpus_sizes)
  print(ws353_mean)
  print(mturk771_mean)
  print(analogy_mean)
  ransac = linear_model.RANSACRegressor()
  sizes = np.log(np.array(corpus_sizes))
  m, b = np.polyfit(sizes, ws353_mean, 1)
  print('slope={}, intercept={}'.format(m, b))
  rsquared = rSquared([m, b], sizes, ws353_mean)
  print('r squared={}'.format(rsquared))
  fig = plt.figure()
  plt.plot(sizes, ws353_mean, '.')
  plt.plot(sizes, m*sizes+b, '-')
  #plt.errorbar(sizes, means['wordsim353.csv'], stds['wordsim353.csv'], linestyle='None', marker='o')
  plt.xlabel('log of vocabulary size')
  plt.ylabel('optimal dimensionality')
  fig.tight_layout()
  plt.savefig('ws353_vs_corpussize.pdf')
  plt.clf()
  plt.close()


  sizes = np.log(np.array(corpus_sizes))
  m, b = np.polyfit(sizes, mturk771_mean, 1)

  print('slope={}, intercept={}'.format(m, b))
  rsquared = rSquared([m, b], sizes, mturk771_mean)
  print('r squared={}'.format(rsquared))
  fig = plt.figure()
  plt.plot(sizes, mturk771_mean, '.')
  plt.plot(sizes, m*sizes+b, '-')
  #plt.errorbar(sizes, means['wordsim353.csv'], stds['wordsim353.csv'], linestyle='None', marker='o')
  plt.xlabel('log of vocabulary size')
  plt.ylabel('optimal dimensionality')
  fig.tight_layout()
  plt.savefig('mturk771_vs_corpussize.pdf')
  plt.clf()
  plt.close()


  sizes = np.log(np.array(corpus_sizes))
  m, b = np.polyfit(sizes, analogy_mean, 1)

  print('slope={}, intercept={}'.format(m, b))
  rsquared = rSquared([m, b], sizes, analogy_mean)
  print('r squared={}'.format(rsquared))
  fig = plt.figure()
  plt.plot(sizes, analogy_mean, '.')
  plt.plot(sizes, m*sizes+b, '-')
  #plt.errorbar(sizes, means['wordsim353.csv'], stds['wordsim353.csv'], linestyle='None', marker='o')
  plt.xlabel('log of vocabulary size')
  plt.ylabel('optimal dimensionality')
  fig.tight_layout()
  plt.savefig('analogy_vs_corpussize.pdf')
  plt.clf()
  plt.close()


