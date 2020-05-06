import pandas as pd
import sys
arg1=sys.argv[1]
df=pd.read_csv (arg1, header=None)
d=df.describe()
# print ('name\tcount\tmean\tstd\tmin\t50%\tmax')
print ('{}\t{}\t{:.0f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'.format(arg1,
    d.at['count',1],
    d.at['mean',1],
    d.at['std',1],
    d.at['min',1],
    d.at['50%',1],
    d.at['max',1]))

