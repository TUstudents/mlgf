import os
# os.environ["OMP_NUM_THREADS"] = str(int(os.environ["OMP_NUM_THREADS"]) - 1)

import numpy
import numpy as np
import joblib # import h5py
from sys import argv
import re
import argparse
import time

from pyscf import gto, dft, lib, scf, cc

from fcdmft.gw.mol.gw_gf import GWGF
from fcdmft.gw.mol.gw_ac import GWAC, AC_pade_thiele_full
# from fcdmft.solver import ccgf_mor
from fcdmft.gw.mol.gw_ac import _get_ac_idx, _get_scaled_legendre_roots, thiele, pade_thiele
from fcdmft.gw.mol.gw_gf import get_g0

from mlgf.workflow.generate import do_gw_calculation, do_fcigf_calculation, do_hf_calculation, do_rks_calculation# ,do_ccgf_calculation
# from mlgf.workflow.generate import do_ccsd_calculation, do_ccgfmpi_calculation, do_ccgfmpi_calculation_real

# import threadpoolctl
# threadpoolctl.threadpool_limits(limits=1)

def split_xyz(xyzfile):
    """split xyz file into atom coordinates

    Args:
        xyzfile (str): xyzfile

    Returns:
        str: formatted atom coords string for pyscf mol constructor
    """    
    with open(xyzfile, 'r') as f:
        s = f.read()
    s = s.split('\n')[2:]
    return  ';'.join(s).replace('*^', 'e')

def get_mols_rank(srcxyz, basis, ecp = None):
    """get a list of mols for each MPI rank to generate data on

    Args:
        srcxyz (str): directory with xyzfiles to create mols for

    Returns:
        list[mol]: list of pyscf mol objects
    """   
    xyz_files = [f for f in os.listdir(srcxyz) if '.xyz' in f]
    indices = np.arange(len(xyz_files))
    indices_subset = indices[indices % size == rank]
    xyz_files_subset = [xyz_files[i] for i in indices_subset]

    mols = [None]*len(xyz_files_subset) # indexes of mols on this MPI task
    for i in range(len(xyz_files_subset)):
        try:
            xyz_file = xyz_files_subset[i]
            smiles = xyz_file.replace('.xyz','')
            charge = smiles.count('+') - smiles.count('-')            
            coords = split_xyz(f'{srcxyz}/{xyz_file}')                
            mols[i] = gto.M(atom = coords, basis = basis, verbose=0, parse_arg=False, charge = charge)
            mols[i].my_name = xyz_file.replace('.xyz', '') # use the variable name to as the h5py/joblib file name, i,e. this name is "mol1"
            if not ecp is None:
                mols[i].ecp = ecp
                
        except Exception as e:
            print(f'Exception in mol creation for {xyz_file}: ', str(e))   
    
    mols = [m for m in mols if not m is None]
    return mols

def do_generate_qm9_rks(srcxyz, outdir, basis, xc, ecp = None):
    """Generate DFT calculations from xyz files

    Args:
        srcxyz (str): directory with xyzfiles to create mols for
        outdir (str): directory to create chk file calculation outputs in
        basis (str): basis set for pyscf
        xc (str): DFT functional for pyscf
    """    

    mols = get_mols_rank(srcxyz, basis, ecp = ecp)
            
    for imol, mol in enumerate(mols):
        fname = mol.my_name
        mf_chkfile = os.path.join(outdir, f'{fname}.chk')
    
        # save data in pyscf chkfile under mlf key
        mol_dict = {}
        mf, mol_dict = do_rks_calculation(mol, mf_chkfile, mol_dict = mol_dict,  xc = xc)
        mol_dict['xc'] = xc
        
        if 'mol' not in mol_dict or not isinstance(mol_dict['mol'], str):
            mol_dict['mol'] = mol.dumps()
        
        lib.chkfile.save(mf_chkfile, 'mlf', mol_dict)
        # joblib.dump(mol_dict, f'{outdir}/{fname}.joblib')

def do_generate_qm9_gw(srcxyz, outdir, basis, xc, ecp = None, outcore = False, gw_band_gap_only = False, use_existing_scf = False, nw2 = None):
    """Generate DFT calculations from xyz files

    Args:
        srcxyz (str): directory with xyzfiles to create mols for
        outdir (str): directory to create chk file calculation outputs in
        basis (str): basis set for pyscf
        xc (str): DFT functional for pyscf
        outcore (bool, optional): split up GW calculation of rho response into a loop. Defaults to False.
        gw_band_gap_only (bool, optional): only compute band-gap with GWAC. Defaults to False.
        use_existing_scf (bool, optional): tries to skip existing DFT calculation output if possible. Defaults to False.
        nw2 (_type_, optional): nomega GL points on which to evaluate sigmaI; integration still is carried out on nw = 100. Defaults to None, in which case GWGF computes on the same 100 GL grid used for integration.
    """    

    mols = get_mols_rank(srcxyz, basis, ecp = ecp)
            
    for imol, mol in enumerate(mols):
        fname = mol.my_name
        mf_chkfile = os.path.join(outdir, f'{fname}.chk')
    
        # save data in pyscf chkfile under mlf key
        mol_dict = {}
        mol_dict = do_gw_calculation(mol, mf_chkfile, mol_dict = mol_dict,  xc = xc, outcore = outcore, band_gap_only = gw_band_gap_only, use_existing_scf = use_existing_scf, nw2 = nw2)
        mol_dict['xc'] = xc
        
        if 'mol' not in mol_dict or not isinstance(mol_dict['mol'], str):
            mol_dict['mol'] = mol.dumps()
        
        lib.chkfile.save(mf_chkfile, 'mlf', mol_dict)

def has_ccsd(mlf_chkfile, look_key = 'dm_cc'):
    if not os.path.exists(mlf_chkfile):
        return False
    mlf = lib.chkfile.load(mlf_chkfile, 'mlf')
    if mlf is None:
        return False
    if not look_key in mlf.keys():
        return False
    return True

def do_generate_qm9_ccsd(srcxyz, outdir, basis, ecp = None, redo_ccsd = False):
    """_summary_

    Args:
        srcxyz (str): directory with xyzfiles to create mols for
        outdir (str): directory to create chk file calculation outputs in, note saves the memory-heavy CC amplitudes needed for CCGF
        basis (str): pyscf basis set
        redo_ccsd (bool, optional): skip already done ccsd files if chkfile is found. Defaults to False.

    Returns:
        list[mol]: list of pyscf mols
    """    
    

    mols = get_mols_rank(srcxyz, basis, ecp = ecp)
            
    for imol, mol in enumerate(mols):
        fname = mol.my_name
        mf_chkfile = os.path.join(outdir, f'{fname}.chk')

        mol_dict = {}
        mol_dict = do_ccsd_calculation(mol, mf_chkfile, mol_dict = mol_dict)
        mol_dict['xc'] = 'hf'
        
        lib.chkfile.save(mf_chkfile, 'mlf', mol_dict)
        # joblib_chkfile = mf_chkfile.replace('.chk', '.joblib')
        # joblib.dump(mol_dict, f'{outdir}/{fname}.joblib')
    
    return mols

def do_generate_qm9_ccgf(outdir, basis, redo_ccgf = False, purge_amplitudes = True, nw = 30, gl_grid = False, real_axis_ccgf = False):
    """_summary_

    Args:
        srcxyz (str): directory with xyzfiles to create mols for
        outdir (str): directory to create chk file calculation outputs in, note saves the memory-heavy CC amplitudes needed for CCGF
        basis (str): pyscf basis set
        redo_ccgf (bool, optional): redo ccgf calculation if sigmaI found from previous calc. Defaults to False.
        purge_amplitudes (bool, optional): purge the cluster amplitudes saved from do_generate_qm9_ccsd(). Defaults to True.
        nw (int, optional): number of freq points on which to evaluate sigma. Defaults to 30.
        gl_grid (bool, optional): evaluate sigmaI on a GL grid of iomega. Defaults to False.
        real_axis_ccgf (bool, optional): do real axis ccgf calculation with nw points. Defaults to False.
    """    
    mf_chkfiles = [f'{outdir}/{f}' for f in os.listdir(outdir) if '.chk' in f]  
    mol_dicts =  [lib.chkfile.load(mf_chkfile, 'mlf') for mf_chkfile in mf_chkfiles]     
    if not redo_ccgf:
        mf_chkfiles = [f for f in mf_chkfiles if not has_ccsd(f, look_key = 'sigmaI')]

    if rank == 0:
        verbose = 5
    else:
        verbose = 0

    comm.Barrier()

    for i in range(len(mf_chkfiles)):
        t0 = time.time()

        # load existing data
        mol_dict, mf_chkfile = mol_dicts[i], mf_chkfiles[i]
        
        if rank == 0:
            print(mf_chkfile)

        comm.Barrier() 

        # careful of MPI routine with mpiccgf_mor.py
        if real_axis_ccgf:
            mol_dict = do_ccgfmpi_calculation_real(mf_chkfile, mol_dict, verbose=verbose, purge_amplitudes = purge_amplitudes, nw = nw)
        else:
            mol_dict = do_ccgfmpi_calculation(mf_chkfile, mol_dict, verbose = verbose, purge_amplitudes = purge_amplitudes, nw = nw, gl_grid = gl_grid)
        comm.Barrier() 

        t_tot = time.time() - t0
        if rank == 0:
            print(f'Time to complete CCGF: {t_tot:0.6f}s')
            lib.chkfile.save(mf_chkfile, 'mlf', mol_dict)
            # joblib.dump(mol_dict, joblib_chkfile)
        comm.Barrier()     

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='generate_splitxyz.py')
    parser.add_argument('--src', default='./qm9_mols/', help='path to individual xyz files')
    parser.add_argument('--xc', default='hf', help='exchange correlation functional for dft.RKS()')
    parser.add_argument('-o', '--output-dir', default=os.path.join(os.getcwd(), 'ccpvdz'), help='output directory')
    parser.add_argument('-b', '--basis', default='ccpvdz', help='basis set, e.g. ccpvdz')
    parser.add_argument('--ecp', default = None, help='molecule ECP for some basis sets (e.g. def2-tzvpp)')
    parser.add_argument('--redo_ccsd', action='store_true', help='redo ccsd calculations even if the existing chkfile has dm_cc')
    parser.add_argument('--redo_ccgf', action='store_true', help='if supplied, will NOT skip ccgf calculations when sigmaI is in the keys')
    parser.add_argument('--ccgf_nw', default = 18, help='number of GL points for CC calculation (need higher number of strong correlation)')
    parser.add_argument('--use_existing_scf', action='store_true', help='use existing SCF calc for GW')

    parser.add_argument('--keep_amplitudes', action='store_true', help='if supplied, will not purge the cc amplitudes from the .chk file (memory heavy)')
    parser.add_argument('--outcore', action='store_true', help='if supplied, will run gw with outcore true to trade speed for memory saving, only applies to gw jobs')
    parser.add_argument('--gw_band_gap_only', action='store_true', help='if supplied, will run gw with orbs = [nocc-1, nocc]')
    parser.add_argument('--exp', action='store_true', help='if supplied, will run experimental version of gw calc')
    default_gf = 'gw'
    parser.add_argument('--gf', type=str, default=default_gf, help='One of either ccgf or gw, the level of theory to calculate sigmaI')
    parser.add_argument('--gl_grid', action='store_true', help='if supplied, will make a GL grid instead of pade grid')
    parser.add_argument('--gw_nw2', type=int, required=False, help='The omega grid on which to evaluate sigma in gw calculation, can be fewer than the integration points (omega prime)')
    parser.add_argument('--real_axis_ccgf', action='store_true', help='compute real axis ccgf between -1 and +1')

    from pyscf import __config__
    MAX_MEMORY = getattr(__config__, 'MAX_MEMORY')
    print('MAX_MEMORY in pyscf is: ', MAX_MEMORY)

    
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    comm = MPI.COMM_WORLD


    args = parser.parse_args()
    gf_calc = args.gf

    outdir = os.path.abspath(args.output_dir)
    src = args.src
    xc = args.xc

    if rank == 0:
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        elif not os.path.isdir(outdir):
            raise FileExistsError(f'{outdir} exists but is not a directory')
            comm.Abort()
    comm.Barrier()
    
    basis = args.basis
    purge_amplitudes = not args.keep_amplitudes
    if not os.path.isdir(outdir): os.makedirs(outdir)

    if gf_calc == 'gw':
        do_generate_qm9_gw(src, outdir, basis, xc, ecp = args.ecp, outcore = args.outcore, gw_band_gap_only = args.gw_band_gap_only, use_existing_scf = args.use_existing_scf, nw2 = args.gw_nw2)
    if gf_calc == 'ccsd':
        do_generate_qm9_ccsd(src, outdir, basis, ecp = args.ecp, redo_ccsd = args.redo_ccsd)
    if gf_calc == 'ccgf':
        do_generate_qm9_ccgf(outdir, basis, redo_ccgf = args.redo_ccgf, purge_amplitudes = purge_amplitudes, nw = args.ccgf_nw, gl_grid = args.gl_grid, real_axis_ccgf = args.real_axis_ccgf)    
    if gf_calc == 'rks':
        do_generate_qm9_rks(src, outdir, basis, xc, ecp = arg.ecp)

    comm.Barrier() # unnecessary but can't hurt
    MPI.Finalize()
