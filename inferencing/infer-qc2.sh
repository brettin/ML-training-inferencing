
# Get the dirctory where this script lives
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

for n in $(cat done) ; do pushd $n ; for m in $(ls *.top.csv) ; do printf "$n\t" ; python $DIR/infer-qc2.py $m ; done ; popd ; done
