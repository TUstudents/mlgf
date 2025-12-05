In order of pipeline the scripts are:

1. md_samples.py - generates geometries of molecules from cp2k AIMD trjaectories of small molecules starting from their SMILES
2. generate.py - calculates GW-level and HF-level features (generates datasets for ML)
3. generate_splitxyz.py - functions for calling generate.py with MPI on several xyz files in a folder
4. data_integrity.py - checks integrity of underlying MBGF data
5. ml_features.py/ml_features_pbc.py - functions for creating equivariant DFT features for ML, called my Moldatum.load_chk()
6. get_ml_info.py - post processing functions for extracting properties from MBGF and self-energy
7. model_errors - functions for some ML error metrics
8. qp_energy - computes quasiparticle energies from sigma(iw)
9. get_bse.py - computes BSE excited states

Summary of the checkpoint file central to this workflow (e.g. mol.chk):

- Has keys 'scf' and 'mol' (identical to pyscf chkfile) 
- additional key 'mlf' (for "ML Features) where all MBGF data is stored in addition to DFT features
- important keys in 'mlf' are 'sigmaI' (self-energy on iomega in the MO basis, the ML target), omega_fit (the points sigma is evaluated on)
