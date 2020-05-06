import pickle
import csv
import sys

pickle_list = pickle.load(open(sys.argv[1], 'rb'))
csv_results = []

i=0
for s, d in pickle_list.items():

    # temp fix for pkl files w/o 'names'
    if len(d[0])==0:
        i=i+1
        id='tmp_id{}'.format(i)
    else:
        id=''.join(d[0])

    data = [s, id] + list(d[1])
    data = ['' if str(i) == 'nan' else i for i in data]
    csv_results.append(data)

with open(sys.argv[2], 'w', newline='') as o_file:
    writer = csv.writer(o_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    writer.writerows(csv_results)
