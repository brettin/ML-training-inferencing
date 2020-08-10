import pandas as pd
import sys
arg1=sys.argv[1]
df=pd.read_csv (arg1, header=None)
d=df.describe()
print ('name\tcount\tmean\tstd\tmin\t50%\tmax')
print ('{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'.format(arg1,
    d.at['count',2],
    d.at['mean',2],
    d.at['std',2],
    d.at['min',2],
    d.at['50%',2],
    d.at['max',2]))

