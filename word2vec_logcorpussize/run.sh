#!/bin/bash
plist=(2262741 4525483 9050966 17000000 1600000 3200000 6400000 12800000)
logdir="data0"
data="text8.zip"
epochs=(15 15 15 15 15 15 15 15)
threads=5
for ((j=0;j<${#plist[@]};++j)); do
  size=${plist[j]}
  epoch=${epochs[j]}
  for (( i=2; i < 29; ++i)); do
    dim=$((10*i))
    echo "$dim"
    python word2vec_optimized.py --min_count 5 --vocab_size_in 10000 --embedding_size $dim --train_data=$data --eval_data=word2vec/trunk/questions-words.txt   --save_path=./data/ --num_neg_samples 1 --concurrent_steps $threads --corpus_size $size --epochs_to_train $epoch --logdir $logdir&
    dim=$((10*i+2))
    echo "$dim"
    sleep $[ ( $RANDOM % 15 )  + 1 ]s
    python word2vec_optimized.py --min_count 5 --vocab_size_in 10000 --embedding_size $dim --train_data=$data --eval_data=word2vec/trunk/questions-words.txt   --save_path=./data/ --num_neg_samples 1 --concurrent_steps $threads --corpus_size $size --epochs_to_train $epoch  --logdir $logdir&
    dim=$((10*i+4))
    echo "$dim"
    sleep $[ ( $RANDOM % 15 )  + 1 ]s
    python word2vec_optimized.py --min_count 5 --vocab_size_in 10000 --embedding_size $dim --train_data=$data --eval_data=word2vec/trunk/questions-words.txt   --save_path=./data/ --num_neg_samples 1 --concurrent_steps $threads --corpus_size $size --epochs_to_train $epoch  --logdir $logdir&
    dim=$((10*i+6))
    echo "$dim"
    sleep $[ ( $RANDOM % 15 )  + 1 ]s
    python word2vec_optimized.py --min_count 5 --vocab_size_in 10000 --embedding_size $dim --train_data=$data --eval_data=word2vec/trunk/questions-words.txt   --save_path=./data/ --num_neg_samples 1 --concurrent_steps $threads --corpus_size $size --epochs_to_train $epoch  --logdir $logdir&
    dim=$((10*i+8))
    echo "$dim"
    sleep $[ ( $RANDOM % 15 )  + 1 ]s
    python word2vec_optimized.py --min_count 5 --vocab_size_in 10000 --embedding_size $dim --train_data=$data --eval_data=word2vec/trunk/questions-words.txt   --save_path=./data/ --num_neg_samples 1 --concurrent_steps $threads --corpus_size $size --epochs_to_train $epoch  --logdir $logdir
  done
  rm /tmp/tmp*
done
