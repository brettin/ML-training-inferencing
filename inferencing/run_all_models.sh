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

max_device=$(( $1-1 ))
dh=/homes/brettin/covid19/ML-Code/descriptor_headers.csv
th=/homes/brettin/covid19/ML-Code/training_headers.csv

mkdir -p DIR.$2
cd DIR.$2

for m in $(find $3 -name "*.autosave.model.h5") ; do
	d=$(basename $(dirname $m ))
	mkdir -p $d
	cd $d
	for n in $(seq 0 $max_device) ; do 
		export CUDA_VISIBLE_DEVICES=$(( $n % 8 ))
		echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
		echo "running: reg_go_infer.sh ../../"$2"0$n $m $dh $th > 0$n.log 2>&1 &"
		# reg_go_infer.sh ../../"$2"0$n $m $dh $th > 0$n.log 2>&1 &
	done
	cd ..
	wait
done
cd ..

