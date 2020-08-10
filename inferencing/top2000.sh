#!/bin/bash

for n in $(cat done)
  do 
    pushd $n
    for m in $(ls $n | grep csv)
      do 
        find $m -type f -name "*.csv" -exec cat {} + | sort -k2,2 -t, -gr -S 90% | head -n 2000 > $m.top2000 ; 
      done
    popd 
  done

