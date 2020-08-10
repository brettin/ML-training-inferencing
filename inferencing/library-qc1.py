import pandas as pd
import sys
arg1=sys.argv[1]

import os
from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(arg1) if isfile(join(arg1, f))]


for file_name in onlyfiles:
	df=pd.read_feather(arg1 + '/' + file_name, use_threads=False)
	s=df.shape
	print("{}\t{}\t{}".format(arg1, s[0], s[1]))
