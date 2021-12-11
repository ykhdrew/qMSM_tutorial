#we need to convert mrc to vol first, otherwise xmipp_phantom_project cannot be used
#for j in strc_pc*;do xmipp_image_convert -i $j -o `basename $j .mrc`.vol;done

#step 1

for num in `seq 1 3`
do
    #mkdir histographs_new_PC${num}
    while read f1 f2 f3 f4
    do  
	xmipp_image_convert -i ./Select_angle/pc${num}_whole.mrc -o ./Select_angle/pc${num}_whole.vol 
        xmipp_phantom_project -i ./Select_angle/pc${num}_whole.vol -o ./Select_angle/pc_${num}_original_angle_index_${f1}_projection.mrc --angles $f2 $f3 $f4;
#    xmipp_image_histogram -i temp.mrc > histographs_PC${num}/histogram_angle_index${f1}.dat
#    xmipp_image_statistics -i temp.mrc > histographs_PC${num}/statistics_angle_index${f1}.dat
    done < ./Select_angle/index_angles_generating_cores.txt
done

#step 2
#summarize the evidence of each angle in distinguish PCs
#for j in histogram_*;do echo $j;cat $j | awk 'BEGIN{a=0} {for(v=1;v<=NR;v++) if($1>=0.1||$1<=-0.1) a=a+$2;} END{print a}';done | awk '{if(NR%2==1) printf "%s ",$1;else print $1}' > sum_hist.dat    

#step 3
#draw the figure
#go to the histogram folder
#paste ../index_angles_generating_cores.txt sum_hist.dat | sort -n -k6 -r | cat -n | awk '{print $1,$3,$4,$5,$7}' > rank_angle_z_angle_y_angle_z_statistics.dat

