#!/bin/bash

#cat $1 | awk -F, '{print ($1*(-1.0)) "\t" $4 "\t" $5}' | sort -g | head -n 10000 > $2

cat $1 | awk -F, '{print ($1*(-1.0)) "\t" $4 "\t" $5}' | sort -g | head -n 100 | ssed -f sed.file > $2
