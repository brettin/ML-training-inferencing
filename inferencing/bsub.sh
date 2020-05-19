#!/bin/bash
#BSUB -W 2:00
#BSUB -nnodes 16
#BSUB -P med110
#BSUB -alloc_flags NVME

model_dir="/gpfs/alpine/scratch/brettin/med110/ML-models/V5_docking_data_april_24_bestHPO"

#module load gcc/4.8.5
#module load spectrum-mpi/10.3.0.1-20190611
#module load cuda/10.1.168
#export PATH="/ccs/proj/med106/gounley1/summit/miniconda37/bin:$PATH"

module load ibm-wml-ce
export PATH="/autofs/nccs-svm1_proj/med110/brettin/ML-training-inferencing/inferencing:$PATH"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# --nnodes has to equal the number of times jsrun is called.

for i in $(seq 1 6 96) ; do
  jsrun -n 6 -a 1 -c 7 -g 1 reg_go_infer_dg.sh $i ENA.input $model_dir > ENA.input."$i".log 2>&1 &
  sleep 4
done

# for i in $(seq 1 16) ; do
#   jsrun -n 6 -a 1 -c 7 -g 1 ./reg_go_infer_dg.sh.sh $i ZIN.input > ZIN.input."$i".log 2>&1 &
# done


# 0:06:38
# 244450000
# 5 models
# 2 nodes
# 307097.5 inferences per second per node

# if we consider 16 nodes and 96 input files
# and that ENA has 121169 files in ENA.input
# we then see 1263 files per ENA.input.XX file.


# 16 summit nodes
# 5 models
#    Started at Sun May 17 16:57:57 2020
# Terminated at Sun May 17 17:26:33 2020
# 0:28:36
# DIR.ml.ADRP_ADPR_A.dsc.csv.reg.csv	1,211,584,346
# DIR.ml.MPRO-X0104.dsc.csv.reg.csv	1,211,584,346
# DIR.ml.MPRO-X0161.dsc.csv.reg.csv	1,211,584,346
# DIR.ml.MPRO-X0305.dsc.csv.reg.csv	1,211,584,346
# DIR.ml.NSP15_3_6W01.dsc.csv.reg.csv	1,211,584,346

