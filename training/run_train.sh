#!/bin/bash
mkdir -p DIR.$1
/bin/cp -n $1 DIR.$1
/bin/cp reg_go2.py DIR.$1
cd DIR.$1
CUDA_VISIBLE_DEVICES=$2 python reg_go2.py --ep 600 --in $1 >& output.log 
cd ..
#rm -r DIR.$1
