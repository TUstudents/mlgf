import os
import sys
import multiprocessing as mp
import argparse

# Symmetry-adapted IAO+PAO (SAIAO) basis
from mlgf.lib.helpers import gGW_mo_saiao, get_chk_saiao, get_chk_saao, get_sigma_fit, find_subset_indices_numpy, get_core_orbital_indices, get_orb_type, get_orb_principal, get_orbtypes
# from mlgf.data import Moldatum

from pyscf import lib, dft
import numpy as np
import numpy
import h5py
import joblib
import scipy
import warnings

from fcdmft.gw.mol.gw_gf import get_g0

def get_mo_features(mol, custom_chkfile):
    """get ML features in MO basis

    Args:
        mol : pyscf mol object
        custom_chkfile (string): mlgf chkfile object from generate.py

    Returns:
        dict: modified mlf dictionary with MO basis features
    """    

    basis_name = 'mo'

    mlf = lib.chkfile.load(custom_chkfile, 'mlf')

    mo_energy = mlf['mo_energy']
    mo_coeff = mlf['mo_coeff']
    S_ao = mlf['ovlp']
    nocc = mlf['nocc']
    mo_occ = mlf['mo_occ']
    nmo = len(mo_energy) #nao =  mol.nao_nr() #len(mo_energy)#
    dm = mlf['dm_hf']
    nelectron = nocc*2
            
    # feature 1: fock matrix
    fock_mo = np.diag(mo_energy)

    # feature 2 : density matrix
    dm_mo = np.diag(mo_occ)

    # feature 3 : hcore matrix
    hcore = mlf['hcore']
    hcore_mo = np.linalg.multi_dot((mo_coeff.T, hcore, mo_coeff))

    # features 4 & 5 : J and K matrices
    vj, vk = mlf['vj'], mlf['vk']
    vj_mo = np.linalg.multi_dot((mo_coeff.T, vj, mo_coeff))
    vk_mo = np.linalg.multi_dot((mo_coeff.T, vk, mo_coeff))

    # feature 6 : mean-field GF (imag freq)
    # GF in MO basis on (ef + iw_n)
    ef = (mo_energy[nocc-1] + mo_energy[nocc]) / 2
    selected_freqs = mlf['omega_fit'] #ALREADY IMAG
    full_sigma = mlf['sigmaI'] # full sigma (>> len(omegaI))
    full_freqs = mlf['freqs']
    omega = ef + selected_freqs # CV note: omega set here

    g0_mo = get_g0(omega, mo_energy, eta=0)
    sigma_fit = get_sigma_fit(full_sigma, full_freqs, selected_freqs)

    # mlf = {}
    mlf[f'dm_{basis_name}'] = dm_mo
    mlf[f'fock_{basis_name}'] = fock_mo
    mlf[f'hcore_{basis_name}'] = hcore_mo
    mlf[f'vj_{basis_name}'] = vj_mo
    mlf[f'vk_{basis_name}'] = vk_mo
    mlf[f'sigma_{basis_name}'] = sigma_fit

    if 'vxc' in mlf.keys():
        vxc_mo = np.linalg.multi_dot((mo_coeff.T, mlf['vxc'], mo_coeff))
        mlf[f'vxc_{basis_name}'] = vxc_mo
    
    return mlf

def get_saao_features(mol, custom_chkfile, C_ao_saao):
    """get ML features in SAAO basis

    Args:
        mol : pyscf mol object
        custom_chkfile (string): mlgf chkfile object from generate.py
        C_ao_saao (np.float64, norb x norb): rotation matrix from AO to SAAO

    Returns:
        dict: modified mlf dictionary with SAAO basis features
    """    


    basis_name = 'saao'

    mlf = lib.chkfile.load(custom_chkfile, 'mlf')
    mlf['C_ao_saao'] = C_ao_saao

    mo_energy = mlf['mo_energy']
    mo_coeff = mlf['mo_coeff']
    S_ao = mlf['ovlp']
    nocc = mlf['nocc']
    nmo = len(mo_energy) #nao =  mol.nao_nr() #len(mo_energy)#
    dm = mlf['dm_hf']
    nelectron = nocc*2
    S_saao = np.linalg.multi_dot((C_ao_saao.T, S_ao, C_ao_saao))
            
    # feature 1: fock matrix
    fock = mlf['fock']
    fock_saao = np.linalg.multi_dot((C_ao_saao.T, fock, C_ao_saao))
    mo_energy_new, mo_coeff_new = scipy.linalg.eigh(fock_saao, S_saao)
    print('max mo energy diff: ', np.max(np.abs(mo_energy_new - mo_energy)))
    # print(mo_energy_new - mo_energy)
    assert ((np.max(np.abs(mo_energy_new - mo_energy)))< 1e-5)
    # transform Fock from SAAO back to AO
    S_saao_inv = np.linalg.inv(S_saao)
    SCS = np.linalg.multi_dot((S_saao_inv, C_ao_saao.T, S_ao))
    fock_ao_check = np.linalg.multi_dot((SCS.T, fock_saao, SCS))
    assert ((np.max(np.abs(fock_ao_check-fock)))<1e-6)

    # feature 2 : density matrix
    dm_saao = np.linalg.multi_dot((SCS, dm, SCS.T))
    assert(abs(np.trace(np.dot(dm_saao, S_saao))-mol.nelectron)<1e-8)

    # transform dm from SAAO back to AO
    dm_ao_new = np.linalg.multi_dot((C_ao_saao, dm_saao, C_ao_saao.T))
    assert ((np.max(np.abs(dm_ao_new-dm)))<1e-6)

    # feature 3 : hcore matrix
    hcore = mlf['hcore']
    hcore_saao = np.linalg.multi_dot((C_ao_saao.T, hcore, C_ao_saao))

    # features 4 & 5 : J and K matrices
    vj, vk = mlf['vj'], mlf['vk']
    vj_saao = np.linalg.multi_dot((C_ao_saao.T, vj, C_ao_saao))
    vk_saao = np.linalg.multi_dot((C_ao_saao.T, vk, C_ao_saao))

    # feature 6 : mean-field GF (imag freq)
    # GF in MO basis on (ef + iw_n)
    ef = (mo_energy[nocc-1] + mo_energy[nocc]) / 2

    
    selected_freqs = mlf['omega_fit'] #ALREADY IMAG
    full_sigma = mlf['sigmaI'] # full sigma (>> len(omegaI))
    full_freqs = mlf['freqs']
    omega = ef + selected_freqs # CV note: omega set here
    g0_mo = get_g0(omega, mo_energy, eta=0)

    # MO to SAAO rotations for hamiltonian-like matrices
    C_saao_mo = np.linalg.multi_dot((S_saao_inv, C_ao_saao.T, S_ao, mo_coeff))
    C_mo_saao = np.linalg.multi_dot((mo_coeff.T, S_ao, C_ao_saao, S_saao_inv))
    
    # GF in SAAO basis
    g0_saao = np.zeros_like(g0_mo)
    
    sigma_fit = get_sigma_fit(full_sigma, full_freqs, selected_freqs)
    C_mo_saao = np.linalg.multi_dot((mo_coeff.T, S_ao, C_ao_saao))
    sigma_saao = gGW_mo_saiao(sigma_fit, C_mo_saao)
    

    
    # mlf = {}
    mlf[f'dm_{basis_name}'] = dm_saao
    mlf[f'fock_{basis_name}'] = fock_saao
    mlf[f'hcore_{basis_name}'] = hcore_saao
    mlf[f'vj_{basis_name}'] = vj_saao
    mlf[f'vk_{basis_name}'] = vk_saao
    mlf[f'gHF_{basis_name}'] = g0_saao
    mlf[f'sigma_{basis_name}'] = sigma_saao
    mlf[f'C_{basis_name}_mo'] = C_saao_mo
    # mlf[f'C_mo_{basis_name}'] = C_mo_saao

    if 'vxc' in mlf.keys():
        vxc_saao = np.linalg.multi_dot((C_ao_saao.T, mlf['vxc'], C_ao_saao))
        mlf[f'vxc_{basis_name}'] = vxc_saao
    
    return mlf

def get_saiao_features(mol, mlf, C_ao_saiao, categorical = True):
    """get ML features in SAIAO basis

    Args:
        mol : pyscf mol object
        custom_chkfile (string): mlgf chkfile object from generate.py
        C_ao_saao (np.float64, norb x norb): rotation matrix from AO to SAIAO
        categorical (boolean): whether to generate integer valued features for quantum numbers and orbital type

    Returns:
        dict: modified mlf dictionary with SAIAO basis features
    """    


    basis_name = 'saiao'
    mlf['C_ao_saiao'] = C_ao_saiao

    mo_energy = mlf['mo_energy']
    mo_coeff = mlf['mo_coeff']
    S_ao = mlf['ovlp']
    nocc = mlf['nocc']
    nmo = len(mo_energy) #nao =  mol.nao_nr() #len(mo_energy)#
    dm = mlf['dm_hf']
    nelectron = nocc*2
            
    # feature 1: density matrix
    dm_saiao = np.linalg.multi_dot((C_ao_saiao.T, S_ao, dm, S_ao, C_ao_saiao))
    abs_diff_particle_number = abs(np.trace(dm_saiao)-nelectron)
    if abs_diff_particle_number > 1e-8:
        warnings.warn(f'dm_saiao particle number diff {abs_diff_particle_number:0.6e}')

    # feature 2 : Fock matrix
    fock = mlf['fock']
    fock_saiao = np.linalg.multi_dot((C_ao_saiao.T, fock, C_ao_saiao))

    # feature 3 : hcore matrix
    hcore = mlf['hcore']
    hcore_saiao = np.linalg.multi_dot((C_ao_saiao.T, hcore, C_ao_saiao))

    # feature 4 & 5 : J and K matrices
    vj, vk = mlf['vj'], mlf['vk']
    vj_saiao = np.linalg.multi_dot((C_ao_saiao.T, vj, C_ao_saiao))
    vk_saiao = np.linalg.multi_dot((C_ao_saiao.T, vk, C_ao_saiao))

    # feature 6 : mean-field GF (imag freq)
    # GF in MO basis on (ef + iw_n)
    ef = (mo_energy[nocc-1] + mo_energy[nocc]) / 2

    # GF in SAIAO basis
    C_saiao_mo = np.linalg.multi_dot((C_ao_saiao.T, S_ao, mo_coeff))
    C_mo_saiao = C_saiao_mo.T
    # With ndim>2 arrays, numpy matmul uses the last two axes

    
    try:
        selected_freqs = mlf['omega_fit'] #ALREADY IMAG
        full_sigma = mlf['sigmaI'] # full sigma (>> len(omegaI))
        full_freqs = mlf['freqs']
        omega = ef + selected_freqs # CV note: omega set here

        sigma_fit = get_sigma_fit(full_sigma, full_freqs, selected_freqs)
        sigma_saiao = gGW_mo_saiao(sigma_fit, C_mo_saiao)
        mlf[f'sigma_{basis_name}'] = sigma_saiao

    except KeyError as e:
        # print('Not generating saiao features for sigma and G: ', str(e))
        pass
    
    # mlf = {}
    mlf[f'dm_{basis_name}'] = dm_saiao
    mlf[f'fock_{basis_name}'] = fock_saiao
    mlf[f'hcore_{basis_name}'] = hcore_saiao
    mlf[f'vj_{basis_name}'] = vj_saiao
    mlf[f'vk_{basis_name}'] = vk_saiao
    mlf[f'C_{basis_name}_mo'] = C_saiao_mo

    if 'vxc' in mlf.keys():
        vxc_saiao = np.linalg.multi_dot((C_ao_saiao.T, mlf['vxc'], C_ao_saiao))
        mlf[f'vxc_{basis_name}'] = vxc_saiao

    if categorical:
        mlf['cat_orbtype_principal'], mlf['cat_orbtype_angular'] = get_orbtypes(mol)
        mlf['cat_orbtype_saiao'] = get_orb_type(mol, dm_saiao)
    return mlf

# running the main method will generate all the dictionaries of ML features in SAIAO basis as joblib in the same folder as the raw conformer data (saved as .chk and .h5)
if __name__ == '__main__':

    print('command line ml_features.py deprecated')
    raise