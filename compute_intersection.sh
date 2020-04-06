#!/bin/bash

source ~/env.sh

# To create input file
# ls *.top1.csv | cut -f3 -d'.' | sort | uniq > intersect.in


# a file of file paths
arg1=$1

for n in $(cat $arg1) ; do

	a="1-Enamine_Infer_DIR.ml."$n".csv.bin.csv.top1.csv"
	b="1-Enamine_Infer_DIR.ml."$n".csv.filtered.bin.csv.top1.csv"
	o=$a.$b.intersection.csv
	python compute_intersection.py --1 $a --2 $b --out $o > $o.log 2>&1
done



# 0-Enamine_Infer_DIR.ml.3CLPro_pocket1_dock.csv.bin.csv.top1.csv
# 0-Enamine_Infer_DIR.ml.3CLPro_pocket1_dock.csv.filtered.bin.csv.top1.csv
# 0-Enamine_Infer_DIR.ml.ADRP_pocket1_dock.csv.bin.csv.top1.csv
# 0-Enamine_Infer_DIR.ml.ADRP_pocket1_dock.csv.filtered.bin.csv.top1.csv
# 0-Enamine_Infer_DIR.ml.CoV_pocket1_dock.csv.bin.csv.top1.csv
# 0-Enamine_Infer_DIR.ml.CoV_pocket1_dock.csv.filtered.bin.csv.top1.csv
# 0-Enamine_Infer_DIR.ml.PLPro_pocket3_dock.csv.bin.csv.top1.csv
# 0-Enamine_Infer_DIR.ml.PLPro_pocket3_dock.csv.filtered.bin.csv.top1.csv

# 1-Enamine_Infer_DIR.ml.3CLPro_pocket1_dock.csv.bin.csv.top1.csv
# 1-Enamine_Infer_DIR.ml.3CLPro_pocket1_dock.csv.filtered.bin.csv.top1.csv
# 1-Enamine_Infer_DIR.ml.ADRP_pocket1_dock.csv.bin.csv.top1.csv
# 1-Enamine_Infer_DIR.ml.ADRP_pocket1_dock.csv.filtered.bin.csv.top1.csv
# 1-Enamine_Infer_DIR.ml.CoV_pocket1_dock.csv.bin.csv.top1.csv
# 1-Enamine_Infer_DIR.ml.CoV_pocket1_dock.csv.filtered.bin.csv.top1.csv
# 1-Enamine_Infer_DIR.ml.PLPro_pocket3_dock.csv.bin.csv.top1.csv
# 1-Enamine_Infer_DIR.ml.PLPro_pocket3_dock.csv.filtered.bin.csv.top1.csv
