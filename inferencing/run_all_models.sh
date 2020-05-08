#!/bin/bash
export PATH=$PATH:/homes/brettin/covid19/ML-Code
[ $# -eq 0 ] && {
	echo ""
	echo "Usage: "$(basename $0)" <num_gpus> <input_file_prefix> <models_base_dir>"
	echo ""
	echo "arg1 = num_gpus_on_node"
	echo ""
	echo "arg2 = basename of input file of pkl filenames"
	echo ""
	echo "arg3 = base dir for find models"
	echo ""
	echo "The number of input files should equal number of GPUs."
	echo "The input files basename is LIB.input where actual"
	echo "input file name is LIB.input00 thru LIB.input07 i.e."
	echo ""
	exit 1
}
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

max_device=$(( $1-1 ))
dh=/homes/brettin/covid19/ML-training-inferencing/descriptor_headers.csv
th=/homes/brettin/covid19/ML-training-inferencing/training_headers.csv

mkdir -p DIR.$2
cd DIR.$2

for m in $(find $3 -name "*.autosave.model.h5") ; do
	d=$(basename $(dirname $m ))
	echo "making directory $d"
	mkdir -p $d
	cd $d
	echo $(date)
	for n in $(seq 0 $max_device) ; do 
		export CUDA_VISIBLE_DEVICES=$(( $n % 8 ))
		echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
		echo "running: $DIR/reg_go_infer_batch.py --in ../../$20$n --model $m --dh $dh --th $th > 0$n.log 2>&1 &"
		python $DIR/reg_go_infer_batch.py --in ../../"$2"0$n --model $m --dh $dh --th $th > 0$n.log 2>&1 &
	done
	cd ..
	wait
	echo $(date)
done
cd ..

