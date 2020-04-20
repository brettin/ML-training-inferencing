#!/bin/bash
[ $# -eq 0 ] && {
        echo ""
        echo "Usage: "$(basename $0)" <file_of_filenames> <model>"
        echo ""
        echo "This program takes as input a file of csv file paths on which to "
        echo "run infer15.py against."
        echo ""
        exit 1
}



# assumes infer15M.py is in your path
# takes as input a file of files and a model
export PATH=$PATH:/lambda_stor/homes/brettin/covid19/ML-Code
arg1=$1
arg2=$2

for n in $(cat $arg1) ; do
	b=$(basename $n)
	date
	python /lambda_stor/homes/brettin/covid19/ML-Code/infer15M.py --in $n --out Infer_$b --model $arg2
	date
done
