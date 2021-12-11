####project simlation 3D to 2D images, here we use three conformations as an example
###conf_angle_for_samplings_dataset.txt is the document: the 1st column is conformation type, 2nd-4th column is the euler viewing anlges (zyz)
###It is noticed that the proportion of three conformations we set is 0.4, 0.3 and 0.3 as initilaization; viewing angles are randomly choose in the sphere.
###num is the core of cpu to parallelly operate algorithm
num=1;filename=./data/test_data_sim/sim_${num}.mrcs;while read f1 f2 f3 f4 ;do xmipp_phantom_project -i ./data/test_data_sim/${f1}.vol -o temp_${num}.mrc --angles ${f2} ${f3} ${f4};xmipp_image_convert -i temp_${num}.mrc -o ../data/${filename} --append;done < ./gen_2D_image/conf_angle_for_samplings_dataset1.txt.txt &
num=2;filename=./data/test_data_sim/sim_${num}.mrcs;while read f1 f2 f3 f4 ;do xmipp_phantom_project -i ./data/test_data_sim/${f1}.vol -o temp_${num}.mrc --angles ${f2} ${f3} ${f4};xmipp_image_convert -i temp_${num}.mrc -o ../data/${filename} --append;done < ./gen_2D_image/conf_angle_for_samplings_dataset2.txt.txt &
num=3;filename=./data/test_data_sim/sim_${num}.mrcs;while read f1 f2 f3 f4 ;do xmipp_phantom_project -i ./data/test_data_sim/${f1}.vol -o temp_${num}.mrc --angles ${f2} ${f3} ${f4};xmipp_image_convert -i temp_${num}.mrc -o ../data/${filename} --append;done < ./gen_2D_image/conf_angle_for_samplings_dataset3.txt.txt &
