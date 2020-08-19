#!/bin/bash

# arg1 = num GPUs on the system
# arg2 = input file containing path names of database shards
# arg3 = Base dir to find model dirs
dh=/homes/brettin/covid19/ML-training-inferencing/descriptor_headers.csv
th=/homes/brettin/covid19/ML-training-inferencing/training_headers.csv
dd=/lambda_stor/data/hsyoo/descriptors/descriptors-ob3
dd=/lambda_stor/data/hsyoo/descriptors/ORD

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export PYTHON_PATH=$DIR

max_device=$(( $1-1 ))

# Some logging stuff
echo "num files in $2 is $(wc -l $2)"

for m in $(find $3 -name "*.autosave.model.h5") ; do
  echo "running  model $m"
  model_base=$(basename $(dirname $m))
  d=0
  # There can be no more than 8 filenames in $2
  for n in $(cat $2) ;  do
    in_base=$(basename $n)
    outdir="./$model_base/$in_base"
    mkdir -p $outdir
    echo "running model $m on input $n on device $d"
    export CUDA_VISIBLE_DEVICES=$d
    python $DIR/reg_go_infer_dg.py  --in $n --model $m --out $outdir &
    d=$(( $d+1 )) 
  done
  echo "waiting for model $m to finish"
  wait
done
