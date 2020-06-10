#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

prefix="/gpfs/alpine/scratch/brettin/med110"
local_prefix="/mnt/bb/$USER"



file_index=$1
file_prefix=$2
model_dir=$3

for i in $(seq $file_index $(($file_index+6-1)) ) ; do

  device=$(($i % 6))
  export CUDA_VISIBLE_DEVICES=$device

  # should test if JSM_GPU_ASSIGNMENTS is empty
  if [ $JSM_GPU_ASSIGNMENTS -eq $device ] ; then
    echo "device $device matches JSM_GPU_ASSIGNMENTS $JSM_GPU_ASSIGNMENTS"
    echo "file prefix: $file_prefix"
    echo "file index: $file_index"
    echo "file offset: $i"

    if [ $i -lt 10 ] ; then
      names_file="$file_prefix""0""$i"
    else
      names_file="$file_prefix""$i"
    fi

    echo "names file: $names_file"

    mkdir -p "$prefix"/DIR."$2"
    mkdir -p "$local_prefix"/DIR."$2"

    echo "looking for models"
    for m in $(find $3 -name "*.autosave.model.h5") ; do
      d=$(basename $(dirname $m ))
      echo "making directory $d"
      mkdir -p $local_prefix/DIR.$2/$d
      echo "running on host: $HOSTNAME with device: $device"
      echo "starting gpu $i: $(date)"
      echo "calling: python $DIR/reg_go_infer_dg.py --in $names_file --out $local_prefix/DIR.$2/$d --model $m"
      python $DIR/reg_go_infer_dg.py --in $names_file --out $local_prefix/DIR.$2/$d --model $m
      echo "finishing gpu $i: $(date)"
    done

    wait

    echo "running ls $local_prefix/DIR.$2/*"
    ls $local_prefix/DIR.$2/*

    echo "running cp -r $local_prefix/DIR.$2/* $prefix/DIR.$2/"
    cp -r $local_prefix/DIR.$2/* $prefix/DIR.$2/

  fi
  echo "done: $(date)"
done


function copy_to_local {
  # function for copying feather files to local
  # and rewriting input file
  infile=$1
  mkdir -p $local_prefix/datasets
  for i in $(cat infile) ; do
    cp -n $i $local_prefix/datasets/
    b=($basename $i)
    # write FILE "$local_prefix/datasets/$b"
    echo "$local_prefix/datasets/$b" >> LOCAL."$infile"
  done
}
