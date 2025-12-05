# mlgf
Machine learning many-body Green's function and self-energy

Authors: Christian Venturella, Jiachen Li, Christopher Hillenbrand, Tianyu Zhu


Features
--------

- Many-body Green's function calculations with fcmdft: $GW$ theory, coupled cluster theory, and FCI 
- Predicting the many-body Green's function (MBGF) by targeting the self-energy from DFT features
- Local and equivariant represenations of electronic structure data with libdmet 
- Chemical properties dervied from the MBGF
  - photoemission spectrum (i.e. density of states)
  - $GW$ quasiparticle energies
  - quasiparticle renormalization
  - 1-particle density matrix and downstream observables:
    - dipoles
    - quadrupoles
    - IAO partial charges
    - FNO-CCSD energy
  - optical spectrum with $GW$-BSE.

Installation
------------

* Requirements
    - PySCF and all dependencies 
    - fcdmft (by Tianyu Zhu, https://github.com/ZhuGroup-Yale/fcdmft)
    - libdmet (by Zhi-Hao Cui, https://github.com/gkclab/libdmet_preview)
    - scikit-learn for self-energy KRR (https://scikit-learn.org)
    - PyTorch Geometric and all dependencies for MBGF-Net (https://pytorch-geometric.readthedocs.io, https://pytorch.org)
    - pandas for data analysis (https://pandas.pydata.org)

* Optional
    - plotly for data visualization
    - CP2K for AIMD
    - rdkit
    - jupyter and jupyterlab
    
* You need to set environment variable `PYTHONPATH` to export mlgf to Python. 
  E.g. if mlgf is installed in `/opt`, your `PYTHONPATH` should be

        export PYTHONPATH=/opt/mlgf:$PYTHONPATH

References
----------

Cite the following papers for the MLGF workflow, KRR implementation, and PyG data processing and GNN architecture:

* C. Venturella, C. Hillenbrand, J. Li, and T. Zhu, Machine Learning Many-Body Green’s Functions for Molecular Excitation Spectra, J. Chem. Theory Comput. 2024, 20, 1, 143–154

* C. Venturella, J. Li, C. Hillenbrand, X. L. Peralta, J. Liu, T. Zhu, Unified Deep Learning Framework for Many-Body Quantum Chemistry via Green’s Functions; 2024. arXiv:2407.20384

Please cite the following papers in publications utilizing the fcdmft package for MBGF calculation:

* T. Zhu and G. K.-L. Chan, J. Chem. Theory Comput. 17, 727-741 (2021)

* T. Zhu and G. K.-L. Chan, Phys. Rev. X 11, 021006 (2021)

