##project 3D 6P1K.vol (real dataset) to 2D images
## num is three, which is core of cpu to accelerate the algorithm
##conf_angle_for_samplings_dataset.txt is the document: the 1st column is conformation type, 2nd-4th column is the euler viewing anlges (zyz)
## viewing angle is randomly choose viewing angles in sphere
num=1;filename=./data/test_data_real/real_${num}.mrcs;while read f1 f2 f3 f4;do xmipp_phantom_project -i ../data/test_data_real/6P1K.vol -o temp_${num}.mrc --angles ${f2} ${f3} ${f4};xmipp_image_convert -i temp_${num}.mrc -o ${filename} --append;done < ./gen_2D_image/conf_angle_for_samplings_dataset1.txt &
num=2;filename=./data/test_data_real/real_${num}.mrcs;while read f1 f2 f3 f4;do xmipp_phantom_project -i ../data/test_data_real/6P1K.vol -o temp_${num}.mrc --angles ${f2} ${f3} ${f4};xmipp_image_convert -i temp_${num}.mrc -o ${filename} --append;done < ./gen_2D_image/conf_angle_for_samplings_dataset2.txt &
num=3;filename=./data/test_data_real/real_${num}.mrcs;while read f1 f2 f3 f4;do xmipp_phantom_project -i ../data/test_data_real/6P1K.vol -o temp_${num}.mrc --angles ${f2} ${f3} ${f4};xmipp_image_convert -i temp_${num}.mrc -o ${filename} --append;done < ./gen_2D_image/conf_angle_for_samplings_dataset3.txt &
