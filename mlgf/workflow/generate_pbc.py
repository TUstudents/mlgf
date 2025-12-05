
import numpy
import numpy as np
import joblib # import h5py
import os
from sys import argv
import os
import re
import argparse

from pyscf.pbc import gto, scf, dft, df, tools
from pyscf.pbc.lib import chkfile
from ase.io import read
from ase import Atoms
from pyscf.pbc.tools import pyscf_ase

from fcdmft.gw.mol.gw_gf import GWGF
from fcdmft.gw.mol.gw_ac import AC_pade_thiele_full
from fcdmft.solver import ccgf_mor
from fcdmft.gw.mol.gw_ac import _get_ac_idx, _get_scaled_legendre_roots, thiele, pade_thiele
from fcdmft.gw.mol.gw_gf import get_g0
from fcdmft.gw.pbc.gw_gf import GWGF as GWGF_pbc

from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
comm = MPI.COMM_WORLD


def do_pbc_calculation(xyzfile, box_length):
    # PBC calculation
    mol_dict = {}
    waterbox = read(xyzfile)
    
    # NOTE: modify cell lattice parameter
    atoms = Atoms(symbols=waterbox.symbols,
                positions=waterbox.positions,
                cell=[[box_length, 0.000, 0.000],
                        [0.000, box_length, 0.000],
                        [0.000, 0.000, box_length]],
                pbc=True)
    atoms.wrap()

    cell = gto.Cell()
    cell.atom = pyscf_ase.ase_atoms_to_pyscf(atoms)
    cell.a = atoms.cell
    cell.max_memory = 90000
    cell.pseudo = 'gth-hf-rev'
    cell.basis='gth-dzvp'
    cell.precision = 1e-10
    cell.unit = 'angstrom'
    cell.verbose = 5
    cell.build()

    # NOTE: scf is imported from pyscf.pbc
    chkfname = xyzfile.replace('.xyz', '.chk')
    kmf = scf.RHF(cell).density_fit()
    kmf.exxdiv = None
    kmf.conv_tol = 1e-12
    kmf.chkfile = chkfname
    kmf.kernel()

    # NOTE: gw_gf is imported from fcdmft.gw.pbc
    from fcdmft.gw.pbc.gw_gf import GWGF as GWGF_pbc
    gw = GWGF_pbc(kmf)
    gw.ac = 'pade'
    gw.fullsigma = True
    gw.rdm = True
    gw.eta = 1e-2
    omega = np.linspace(-1,1,201)
    gf, gf0, sigma = gw.kernel(omega=omega, nw=100)

    nocc = kmf.mol.nelectron // 2 # occupation number/number of electrons
    mo_occ = kmf.mo_occ # occupation number of each orbital
    mo_energy = kmf.mo_energy # orbital energy
    mo_coeff = kmf.mo_coeff # orbital coefficient
    ovlp = kmf.get_ovlp() # overlap matrix
    hcore = kmf.get_hcore() # hcore matrix
    vj = kmf.get_j() # Coulomb matrix
    vk = kmf.get_k() # exchange matrix
    fock = kmf.get_fock() # Fock matrix
    dm_hf = kmf.make_rdm1() # HF density matrix

    mol_dict['e_tot'] = kmf.e_tot
    mol_dict['nocc'] = numpy.asarray(nocc)
    mol_dict['mo_occ'] = numpy.asarray(mo_occ)
    mol_dict['mo_energy'] = numpy.asarray(mo_energy)
    mol_dict['mo_coeff'] = numpy.asarray(mo_coeff)
    mol_dict['ovlp'] = numpy.asarray(ovlp)
    mol_dict['hcore'] = numpy.asarray(hcore)
    mol_dict['vj'] = numpy.asarray(vj)
    mol_dict['vk'] = numpy.asarray(vk)
    mol_dict['fock'] = numpy.asarray(fock)
    mol_dict['dm_hf'] = numpy.asarray(dm_hf)


    dm_gw = gw.make_rdm1() # GW density matrix
    freqs = gw.freqs # frequency points for the self-energy, real part
    wts = gw.wts # weights of each frequency point
    ef = gw.ef

    # TZ revision NOTE: modify freqs and sigmaI indices to be consistent with original GW code
    freqs = numpy.concatenate(([0.], freqs))
    iw_cutoff = 5.0
    nw_sigma = sum(freqs < iw_cutoff)
    sigmaI = gw.sigmaI[:,:,:nw_sigma]
    freqs = freqs[:nw_sigma]

    coeff, omega_fit = AC_pade_thiele_full(sigmaI, freqs*1j)

    mol_dict['ef'] = numpy.asarray(ef)
    mol_dict['freqs'] = numpy.asarray(freqs)
    mol_dict['wts'] = numpy.asarray(wts)
    mol_dict['sigmaI'] = numpy.asarray(sigmaI)
    mol_dict['dm_gw'] = numpy.asarray(dm_gw)
    
    mol_dict['coeff'] = numpy.asarray(coeff)
    mol_dict['omega_fit'] = numpy.asarray(omega_fit)+ef 
    return mol_dict


def do_generate_pbc(srcxyz, outdir, box_length):
    
    xyz_files = [f for f in os.listdir(srcxyz) if '.xyz' in f]
    mols = [None]*len(xyz_files)
    mols_task = [] # indexes of mols on this MPI task
    for imol, mol in enumerate(mols):
        if imol % size == rank:
            mols_task.append(imol)
            
    for imol in mols_task:
        xyz_file = xyz_files[imol]
        fname = xyz_file.replace('.xyz', '')
        mf_chkfile = os.path.join(outdir, f'{fname}.chk')
        xyz_file = f'{srcxyz}/{xyz_file}'
        print(xyz_file)
        mol_dict = do_pbc_calculation(xyz_file, box_length)    
        mol_dict['xyzfile'] = xyz_file
        mol_dict['box_length'] = box_length
        joblib.dump(mol_dict, f'{outdir}/{fname}.joblib')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='generate.py')
    parser.add_argument('--src', default='./8/', help='path to individual xyz files')
    parser.add_argument('-o', '--output-dir', default=os.path.join(os.getcwd(), 'ccpvdz'), help='output directory')
    parser.add_argument('--box', required = True, help='ABC in aimd.inp')


    from pyscf import __config__
    MAX_MEMORY = getattr(__config__, 'MAX_MEMORY')
    print('MAX_MEMORY in pyscf is: ', MAX_MEMORY)


    args = parser.parse_args()
    outdir = os.path.abspath(args.output_dir)
    src = args.src
    box_length = args.box

    if rank == 0:
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        elif not os.path.isdir(outdir):
            raise FileExistsError(f'{outdir} exists but is not a directory')
            comm.Abort()

    comm.Barrier()
    
    if not os.path.isdir(outdir): os.makedirs(outdir)
    do_generate_pbc(src, outdir, box_length)
    
    comm.Barrier() # unnecessary but can't hurt
    MPI.Finalize()