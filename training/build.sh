ls *.dsc.csv | parallel " cat {} | cut -f2,3,4,5 -d, --complement  > {}.reg.csv"
