#!/bin/bash
[ $# -eq 0 ] && {
        echo ""
        echo "Usage: "$(basename $0)" <file_of_filenames> <model> <dh> <th>"
        echo ""
        echo "This program takes as input a file of csv file paths on which to "
        echo "run reg_go_infer.py against."
        echo ""
        exit 1
}



# takes as input a file of files and a model
export PATH=$PATH:/lambda_stor/homes/brettin/covid19/ML-Code
arg1=$1
model=$2
dh=$3
th=$4

for n in $(cat $arg1) ; do
	b=$(basename $n)
	date
	python /lambda_stor/homes/brettin/covid19/ML-Code/reg_go_infer.py --in $n --out Infer_$b --model $model --dh $dh --th $th
	date
done
