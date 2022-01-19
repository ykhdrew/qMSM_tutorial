# A Step-by-step guide for building quasi Markov State Model to Study Functional Conformational Changes of Biological Macromolecules
Version 0.1, (c) Huang Group, Department of Chemistry, University of Wisconsin-Madison

This is a tutorial for constructing quasi-Markov State model(qMSM), a new kinetic modelling framework that encodes non-Markovian dynamics in time-dependent memory kernels.

The following content is accompanying the book chapter:
"A Step-by-step Guide on How to Construct quasi-Markov State Models to Study Functional Conformational Changes of Biological Macromolecules" in
"A Practical Guide to Recent Advances in Multiscale Modelling and Simulation of Biomolecules". AIP Publishing. to be Published.


### Content

There are 8 Jupyter notebooks that depict various stages of qMSM construction

1. Featurization [Link to notebook](notebook/Featurization.ipynb)
2. Feature selection with Spectral oASIS [Link to notebook](notebook/SpectraloASIS-Parallel.ipynb)
3. Dimensionality reduction with tlCA [Link to notebook](notebook/TICA.ipynb)
4. APLoD clustering [Link to notebook](notebook/APLoD.ipynb)
5. MSM hyperparameter selection with GMRQ [Link to notebook](notebook/Gmrq.ipynb)
6. Microstate MSM and Lumping [Link to notebook](notebook/micorstate_MSM&PCCA.ipynb)
7. quasi-Markov State Model [Link to notebook](notebook/qMSM.ipynb)
8. MFPT calculation and macrostate sampling [Link to notebook](notebook/Analysis.ipynb)

For the MD trajectories, you may download them at our dispository on Open Science Framework: https://osf.io/wu2s6/?view_only=c7c5fef31563409babb403669a864572  

### Installation
We will use MSMbuilder 3.8.0 and PyEMMA in our tutorial. For installing MSMbuilder with Anaconda, you can use the following script:

	conda env create -n msmbuilder -f environment.yml

For PyEMMA installation, you may refer to http://pyemma.org.

### Authors
* **Prof. Xuhui Huang** - *Project leader* - [xhuang](xhuang@chem.wisc.edu)
* **Andrew Kai-hei Yik** - *Author* 
* **Ilona Christy Unarta** - *Author* 
* **Yunrui Qiu** - *Author* 
* **Siqin Cao** - *Author* 

### License

This tutorial is licensed with <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
The library for Quasi-MSM construction is licensed with  <a rel="license" href="http://www.apache.org/licenses/LICENSE-2.0">Apache License, Version 2.0</a>.

### External Library
APLoD clustering library is part of HK_DataMiner(Apache 2.0): https://github.com/liusong299/HK_DataMiner
