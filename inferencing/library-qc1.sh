for n in $(ls -d ~/project_work_110/brettin/datasets/*/) ; do
  python library-qc1.py $n
done
