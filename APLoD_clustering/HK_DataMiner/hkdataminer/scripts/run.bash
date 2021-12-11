#!/bin/bash
dimension=3
output=faiss_dbscan_output_01.log
rm $output
for n_size in 10000 30000 50000 80000 100000 300000 500000 800000 1000000 3000000 5000000 8000000 10000000 
do
   echo "Running Faiss DBSCAN by nsize = $n_size, dimension = $dimension";
   python faiss_dbscan_general_test.py -n $n_size -d $dimension >> $output  
   echo "--------------------------------------------------------------------";
done
