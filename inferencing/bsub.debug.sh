#!/bin/bash
#BSUB -W 02:00
#BSUB -nnodes 1
#BSUB -P med110
#BSUB -J CANCERInfer.Debug
#BSUB -alloc_flags NVME

model_dir="/gpfs/alpine/med110/proj-shared/hsyoo/CANCER/ML-models"

unset OMP_NUM_THREADS

module load ibm-wml-ce
export PATH="/gpfs/alpine/med110/proj-shared/hsyoo/CANCER/ML-training-inferencing/inferencing:$PATH"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# --nnodes has to equal the number of times jsrun is called.

for i in $(seq 0 6 7) ; do
  jsrun -n 6 -a 1 -c 7 -g 1 reg_go_infer_dg.sh $i DEBUG/DEBUG.input $model_dir > DEBUG.input."$i".log 2>&1 &
  sleep 2
done

