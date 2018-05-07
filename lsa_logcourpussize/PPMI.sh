#!/bin/bash
plist=(1131370 2262741 4525483 9050966 17000000 1600000 3200000 6400000 12800000)
for dim in ${plist[@]}; do
  echo "$dim"
  python PPMI.py --size $dim &
done
