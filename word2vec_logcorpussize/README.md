File | What's in it?
--- | ---
`run.sh` | shell script for running the experiments
`data0.zip` | raw experiment results, when decompressed can be used to generate the coefficients and plots
`get_result.py` | process the raw experiment results and get the coefficients
`word2vec.py` | A version of word2vec implemented using TensorFlow ops and minibatching.
`word2vec_test.py` | Integration test for word2vec.
`word2vec_optimized.py` | A version of word2vec implemented using C ops that does no minibatching.
`word2vec_optimized_test.py` | Integration test for word2vec_optimized.
`word2vec_kernels.cc` | Kernels for the custom input and training ops.
`word2vec_ops.cc` | The declarations of the custom ops.


Copyright 2015 The TensorFlow Authors. All Rights Reserved.  

Licensed under the Apache License, Version 2.0 (the "License");  
you may not use this file except in compliance with the License.  
You may obtain a copy of the License at  

http://www.apache.org/licenses/LICENSE-2.0  

Unless required by applicable law or agreed to in writing, software  
distributed under the License is distributed on an "AS IS" BASIS,  
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
See the License for the specific language governing permissions and  
limitations under the License.  

This directory contains models for unsupervised training of word embeddings
using the model described in:

(Mikolov, et. al.) [Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/abs/1301.3781),
ICLR 2013.

Detailed instructions on how to get started and use them are available in the
tutorials. Brief instructions are below.

* [Word2Vec Tutorial](http://tensorflow.org/tutorials/word2vec)

You will need to compile the ops as follows:

```shell
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
g++ -std=c++11 -shared word2vec_ops.cc word2vec_kernels.cc -o word2vec_ops.so -fPIC -I $TF_INC -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -L$TF_LIB -ltensorflow_framework
```

On Mac, add `-undefined dynamic_lookup` to the g++ command.

(For an explanation of what this is doing, see the tutorial on [Adding a New Op to TensorFlow](https://www.tensorflow.org/how_tos/adding_an_op/#building_the_op_library). The flag `-D_GLIBCXX_USE_CXX11_ABI=0` is included to support newer versions of gcc. However, if you compiled TensorFlow from source using gcc 5 or later, you may need to exclude the flag.)

