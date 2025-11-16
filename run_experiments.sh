#!/bin/bash
set -e
for p in 0.0 0.05 0.10 0.50
do
  echo "Running with poison fraction = $p"
  python train.py --poison_fraction $p
done
