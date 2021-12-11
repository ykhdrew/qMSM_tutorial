#written on Apr 2, 2019 by Wei Wang

cat -n landmark_used_in_1st_stage.list | grep -E '_0_0_0.mrc|_2.5_0.mrc|_5_0.mrc|_10_0.mrc|_15_0.mrc|_20_0.mrc' | awk '{print $1}' > cluster_1.temp
cat -n landmark_used_in_1st_stage.list | grep -E '_30_0.mrc|_40_0.mrc|_50_0.mrc|_60_0.mrc|_70_0.mrc|_80_0.mrc|_90_0.mrc' | awk '{print $1}' > cluster_0.temp

cat cluster_0.temp | awk '{print 0, $1}' > temp_0.temp
cat cluster_1.temp | awk '{print 1, $1}' > temp_1.temp

cat temp_0.temp temp_1.temp | sort -n -k2 | awk '{print $1}' > landmarks_used_in_1st_stage.labels_

rm -rf temp_*.temp;rm -rf cluster_*.temp;

cat landmark_used_in_1st_stage.list | grep -E '_rotate_0_0_0.mrc|_2.5_0.mrc|_5_0.mrc|_10_0.mrc|_15_0.mrc|_20_0.mrc' > landmark_used_in_2nd_stage.list

j=0
temp=0
while [ $j -lt 6 ]  #2n
do
    cat -n landmark_used_in_2nd_stage.list | grep cluster_${temp}_ | grep -E '_rotate_0_0_0.mrc|_2.5_0.mrc|_5_0.mrc' | awk '{print $1}' > cluster_${j}.temp
    j=$((${j}+1))
    cat -n  landmark_used_in_2nd_stage.list | grep cluster_${temp}_ | grep -E '_10_0.mrc|_15_0.mrc|_20_0.mrc' | awk '{print $1}' > cluster_${j}.temp
    j=$((${j}+1))
    temp=$((${temp}+1))
done

for j in `seq 0 5` #2n-1, 
do
    cat cluster_${j}.temp | awk -v nn=$j '{print nn, $1}' > temp_${j}.temp
done

cat temp_*.temp | sort -n -k2 | awk '{print $1}' > landmarks_used_in_2nd_stage.labels_

rm -rf cluster_*.temp;rm -rf temp_*.temp
    


