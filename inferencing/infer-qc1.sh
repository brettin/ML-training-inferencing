# current code needs to change .pkl to .csv on inferencing result files.
for n in $(cat done) ; do pushd $n ; for m in $(ls | grep -v top2000) ; do printf "$n\t$m\t" ; find $m -name "*.pred.csv" -type f -exec cat {} + | wc -l ; done ; popd ; done


