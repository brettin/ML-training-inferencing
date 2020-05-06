

#DSC

#FFP

cut -d, -f1,2,3,4,5 --complement FFP > FFP.rest

paste DSC FFP.rest | ssed -e 's/\t/,/g' > COMB.csv
