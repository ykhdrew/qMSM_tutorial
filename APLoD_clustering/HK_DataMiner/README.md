# HKDataminer Tutorial - Data Mining Tool for Biomolecular Dynamics
**Version 0.9.1, (c) Song LIU**
## Overview
HK Dataminer is a python library for constructing statistical models for bimolecular dynamics data. The package includes complementary data mining algorithms such as clustering and Markov state models (MSMs).

# Installation guide
## I. Package requirement
* [Python v3.7](https://www.python.org)
* [NumPy v1.1.8](https://numpy.org)
* [SciPy v1.17.0](https://www.scipy.org)
* [MDTraj 1.9.3](mdtraj.org)
* [Scikit-Learn v0.21.3](https://scikit-learn.org)
* [Matplotlib](https://matplotlib.org)
* [Xmipp](http://xmipp.i2pc.es/)
## II.Installing
We highly recommend that you download the Python 3.7 version of Anaconda, which is a completely free enterprise-ready Python distribution for large-scale data processing, predictive analytics, and scientific computing.

# A step-by-step guide of running HKDataminer
## 1.Introduction and Alanine Dipeptide

## 2.Move to tutorial directory Assuming you’ve in the MD Data Miner folder, move to the Tutorial directory.
`cd Tutorial`

## 3.Cluster your data The following command clusters your data using the RMSD metric by k-centers clustering method.
`python ../scripts/test_kcenters.py  -n 500`

After Clustering, the assignments of each conformation are stored as assignments.txt, the cluster centers are sotred as generators.txt.

## 4.Lump microstates into macrostates
Once we have determined the number of macrostates in the system, we will use Perron Cluster CLuster Analysis (PCCA) algorithm to lump microstates into macrostates. We use the doLumping.py script to lump the microstates into macrostates. We could define the number of macrostates using -n option.The command below will build 4 macrostates.

`python ../scripts/doLumping.py -c assignments_kcenters_n_500.txt -m 6`

Examining the macrostate decomposition It is known that the relevant degrees of freedom for alanine dipeptide are the phi and psi backbone angles. Let’s compute the dihedrals and plot the conformations of each macrostate.

## 5.Template matching to obtain popuplations in multiplr conformtions of cryo-EM dataset
Template matching project: aimed to use the templates to match the multiple structures (conformations) so that obtain the equilibrium distributions 
in different conformations.
There are three steps: 1. select the best viewing angle which can distinguish the multiple conformations well. 2. project the 3D volumes into 2D 
images for both templates and experimental volumes. 3. two-stage matching to obtain the populations of multiple conformations.

requirements: python, linux system and xmipp

Dataset: Each dataset is consisted of template structures and experimental structures. We use two expmples to test our algorithm, one is 
simulation dataset, the other is real dataset. We provide simualtion dataset: open.vol, close.vol, intermediate.vol(same as test data and templates) 
and real dataset:6P1K.vol(test data), H_EV2_red.vol, H_EV2_grey.vol, H_EV2_blue.vol(templates)

Preprocess: you can use command: 
`xmipp_xmipp_volume_from_pdb -i open.pdb -o open.vol`     transfer pdb file to vol file

`xmipp_image_convert -i open.vol -o open.mrc`            transfer vol file to mrc file

run our algorithm: `python TSTM.py --datatype='sim' --vol_size=128`, the output is '2nd_stage_brute_force_classification_result.dat' 

and `python ./two_stage_matching/analyze_population.py` to obatin population

# Deployment
HKUST method is developed by [Prof. Xuhui Huang's group](http://compbio.ust.hk)

# Authors
* **Prof. Xuhui Huang** - *Project leader* - [xuhuihuang](http://compbio.ust.hk/public_html/pmwiki-2.2.8/pmwiki.php?n=People.XuhuiHuang)
* **Mr. Song Liu** - *Developer* -[liusong299](https://github.com/liusong299/)
* **Mr. Hanlin Gu** - *Developer* -[ghl1995](https://github.com/ghl1995/)


See also the list of [contributors](https://github.com/liusong299/gromacs-2019-CWBSol/graphs/contributors) who participated in this project.

# License
This project is licensed under the GPL License - see the [LICENSE](LICENSE) file for details

# Acknowledgments

# References:

