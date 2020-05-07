#!/bin/bash
[ $# -eq 0 ] && {
        echo ""
        echo "Usage: "$(basename $0)" <file_of_filenames> <model> <dh> <th>"
        echo ""
        echo "arg1 = file of pkl filenames"
        echo ""
	echo "arg2 = model file"
	echo ""
	echo "arg3 = descritor header file"
	echo ""
	echo "arg4 = training header file"
	echo ""
        exit 1
}
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"



b=$(basename $1)
date
python $DIR/reg_go_infer.py --in $n --out Infer_$b --model $2 --dh $3 --th $4
date
