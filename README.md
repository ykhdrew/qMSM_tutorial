# A Step-by-step guide for building quasi Markov State Model

This is a tutorial for constructing quasi-Markov State model(qMSM), a new kinetic modelling framework that surpasses the sampling bottleneck in classical MSM

The following content is accompanying the book chapter:
"A Step-by-step Guide on How to Construct quasi-Markov State Models to Study Functional Conformational Changes of Biological Macromolecules" in
"A Practical Guide to Recent Advances in Multiscale Modelling and Simulation of Biomolecules", AIP Publishing, to be Published


### Content

There are 7 notebooks that depict various stages of qMSM construction

1. Featurization [Link](notebook/Featurization.ipynb)
2. Feature selection with Spectral oASIS [Link](notebook/SpectraloASIS-Parallel.ipynb)
3. Dimensionality reduction with tlCA [Link](notebook/TICA.ipynb)
4. APLoD clustering [Link](notebook/APLoD.ipynb)
5. MSM hyperparameter selection with GMRQ [Link](notebook/Gmrq.ipynb)
6. Microstate MSM and Lumping [Link](notebook/micorstate_MSM&PCCA.ipynb)
7. quasi-Markov State Model [Link](notebook/qMSM.ipynb)
8. MFPT calculation and macrostate sampling [Link](notebook/Analysis.ipynb)

For the MD trajectories, you may download them at our dispository at Open Science Framework. (Please refer to the book chapter for the URL)

### Installation
We will use MSMbuilder 3.8.0 and PyEMMA in our tutorial. For installing MSMbuilder, you can use the following script:

bash install_miniconda_MSMbuilder.sh

This will install miniconda(if not installed yet), MSMbuilder as well as libraries used in this tutorial  

For PyEMMA installation, you may refer to http://pyemma.org.

### Authors
* **Prof. Xuhui Huang** - *Project leader* - [xhuang](xhuang@chem.wisc.edu)
* **Andrew Kai-hei Yik** - *Author* 
* **Ilona Christy Unarta** - *Author* 
* **Yunrui Qiu** - *Author* 
* **Siqin Cao** - *Author* 

### License

This tutorial is licensed with <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.)
The library for Quasi-MSM construction is licensed with  <a rel="license" href="http://www.apache.org/licenses/LICENSE-2.0">Apache License, Version 2.0</a>.)

##External Library
APLoD clustering library is part of HK_DataMiner(Apache 2.0): https://github.com/liusong299/HK_DataMiner