from mlgf.utils.xyz_traj import read_xyzfile

from pyscf import gto, scf, fci, lib, cc
from pyscf.lib import logger
import os
import joblib
import argparse
import numpy as np
import logging
import time

# from fcdmft.solver import ccgf_mor, fcigf, mpiccgf_mor

from fcdmft.gw.mol.gw_ac import GWAC, _get_ac_idx, _get_scaled_legendre_roots, \
    thiele, AC_pade_thiele_full
from fcdmft.gw.mol.gw_gf import get_g0, GWGF
# from fcdmft.gw.mol.mpigw_ac import GWAC_MPI

from mlgf.lib.helpers import get_pade18


from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
comm = MPI.COMM_WORLD


def my_AC_pade_thiele_full(sigma, omega, step_ratio=2.0/3.0):
    """
    Analytic continuation to real axis using a Pade approximation

    Args:
        sigma : 3D array (norbs, norbs, nomega)
        omega : 1D array (nomega)

    Returns:
        coeff: 2D array (ncoeff, norbs, norbs)
        omega : 1D array (ncoeff)
    """
    norbs, norbs, nw = sigma.shape
    npade = nw // 2 # take even number of points.
    coeff = np.zeros((npade*2, norbs, norbs),dtype=np.complex128)
    for p in range(norbs):
        for q in range(norbs):
            coeff[:,p,q] = thiele(sigma[p,q,:npade*2], omega[:npade*2])
    return coeff, omega[:npade*2]


def do_rks_calculation(mol, chkfile, mol_dict, xc='hf'):
    """
    do DFT calculation
    
    Args:
        mol (mol): pyscf mol
        chkfile (str): pyscf/mlgf chkfile
        mol_dict (dict): dictionary with electronic structure data
        xc (str, optional): dft functional. Defaults to 'hf'.

    Returns:
        dict: dictionary with electronic structure data
    """    
    # Hartree-Fock calculation
    mf = scf.RKS(mol)
    mf.xc = xc
    mf.chkfile = chkfile
    mf.kernel()
    dm = mf.make_rdm1()
    # DFT/HF calculation outputs
    mol_dict['e_tot'] = mf.e_tot
    # occupation number/number of electrons
    mol_dict['nocc'] = mol.nelectron // 2
    # occupation number of each orbital
    mol_dict['mo_occ'] = np.asarray(mf.mo_occ)
    mol_dict['mo_energy'] = np.asarray(mf.mo_energy)    # orbital energy
    mol_dict['mo_coeff'] = np.asarray(mf.mo_coeff)      # orbital coefficient
    mol_dict['ovlp'] = np.asarray(mf.get_ovlp())        # overlap matrix
    mol_dict['hcore'] = np.asarray(mf.get_hcore())      # hcore matrix
    
    vj, vk = mf.get_jk()
    mol_dict['vj'] = np.asarray(vj)                     # Coulomb matrix
    mol_dict['vk'] = np.asarray(vk)                     # exchange matrix
    mol_dict['vxc'] = np.asarray(mf.get_veff() - vj)    # exchange-correlation matrix
    mol_dict['fock'] = np.asarray(mf.get_fock())        # Fock matrix
    mol_dict['dm_hf'] = np.asarray(dm)                  # mean field density matrix
    mol_dict['xc'] = xc

    # the definition of the hamiltonian is a bit tricky here, need to multiply by -0.5 to get the correct definition of vk
    if xc == 'hf':
        mol_dict['vk_hf'] = -0.5*np.asarray(vk)
    else:
        mf_rhf = scf.RHF(mol)
        vk_hf = mf_rhf.get_veff(mol, dm) - mf_rhf.get_j(mol, dm)
        mol_dict['vk_hf'] = np.asarray(vk_hf)
    return mf, mol_dict

def do_hf_calculation(mol, chkfile, mol_dict):
    """do HF calculation

    Args:
        mol (mol): pyscf mol
        chkfile (str): pyscf/mlgf chkfile
        mol_dict (dict): dictionary with electronic structure data

    Returns:
        dict: dictionary with electronic structure data
    """  
    # Hartree-Fock calculation
    mf = scf.RHF(mol)
    mf.chkfile = chkfile
    mf.kernel()
    dm = mf.make_rdm1()
    # DFT/HF calculation outputs
    mol_dict['e_tot'] = mf.e_tot
    # occupation number/number of electrons
    nocc =  mol.nelectron // 2
    mol_dict['nocc'] = nocc
    # occupation number of each orbital
    mol_dict['mo_occ'] = np.asarray(mf.mo_occ)
    mol_dict['mo_energy'] = np.asarray(mf.mo_energy)    # orbital energy
    mol_dict['mo_coeff'] = np.asarray(mf.mo_coeff)      # orbital coefficient
    mol_dict['ovlp'] = np.asarray(mf.get_ovlp())        # overlap matrix
    mol_dict['hcore'] = np.asarray(mf.get_hcore())      # hcore matrix
    
    vj, vk = mf.get_jk()
    mol_dict['vj'] = np.asarray(vj)                     # Coulomb matrix
    mol_dict['vk'] = np.asarray(vk)                     # exchange matrix
    # mol_dict['vxc'] = np.asarray(mf.get_veff() - vj)    # exchange-correlation matrix
    mol_dict['fock'] = np.asarray(mf.get_fock())        # Fock matrix
    mol_dict['dm_hf'] = np.asarray(dm)                  # mean field density matrix
    mol_dict['xc'] = 'hf'

    # for self-energy ACs
    mol_dict['vk_hf'] = mf.get_veff(mol, dm) - mf.get_j(mol, dm)

    
    mol_dict['ef'] = (mf.mo_energy[nocc-1] + mf.mo_energy[nocc]) / 2.0
    return mf, mol_dict

# xc is ignored
def do_fcigf_calculation(mol, chkfile,
                         mol_dict=None, gmres_tol=1e-6, eta=0, xc = 'hf'):
    """do FCI calculation

    Args:
        mol (mol): pyscf mol
        chkfile (str): pyscf/mlgf chkfile
        mol_dict (dict): dictionary with electronic structure data
        gmres_tol (float): GMRES tol for linear solver to get FCIGF, defaults to 1e-6.
        eta (float): band-broadening
        xc (str, optional): dft functional. Defaults to 'hf'.

    Returns:
        dict: dictionary with electronic structure data
    """  
    if mol_dict is None:
        mol_dict = {}
    mf, mol_dict = do_hf_calculation(mol, chkfile, mol_dict=mol_dict)

    # FCI

    cisolver = fci.FCI(mf)
    fci_energy = cisolver.kernel()[0]

    print('FCI energy = ', fci_energy)
    mol_dict['fci_energy'] = fci_energy

    dm_fci = cisolver.make_rdm1(cisolver.ci, mol.nao, mol.nelectron)
    nocc = mol.nelectron // 2
    ef = (mf.mo_energy[nocc-1] + mf.mo_energy[nocc]) / 2.0
    freqs, wts = get_pade18()
    omega = ef + 1j*freqs

    myfcigf = fcigf.FCIGF(cisolver, mf, tol=gmres_tol)

    orbs = range(len(mf.mo_energy))
    g_ip = myfcigf.ipfci_mo(orbs, orbs, omega.conj(), eta).conj()
    g_ea = myfcigf.eafci_mo(orbs, orbs, omega, eta)
    gf = g_ip + g_ea

    gf0 = get_g0(omega, mf.mo_energy, eta=0)
    sigmaI = (np.linalg.inv(gf0.T) - np.linalg.inv(gf.T)).T.copy()

    # AC from sigma(iw) to sigma(w)
    coeff, omega_fit = my_AC_pade_thiele_full(sigmaI, omega)

    for name, obj in zip(['ef', 'dm_fci', 'freqs', 'wts', 'sigmaI', 'coeff', 'omega_fit'],
                         [ef, dm_fci, freqs, wts, sigmaI, coeff, omega_fit]):
        mol_dict[name] = np.asarray(obj)

    return mol_dict

# xc is ignored
def do_ccgf_calculation(mol, chkfile, mol_dict={}, verbose=4, xc = 'hf', nw = 18):
    """do ccgf calc, not MPI parallel

    Args:
        mol (mol): pyscf mol
        chkfile (str): pyscf/mlgf chkfile
        mol_dict (dict, optional): dictionary with electronic structure data. Defaults to {}.
        verbose (int, optional): pyscf verbose. Defaults to 4.
        xc (str, optional): dft functional. Defaults to 'hf'.
        nw (int, optional): nomega for CCGF. Defaults to 18.

    Returns:
        dict: dictionary with electronic structure data. Defaults to {}.
    """    

    if rank == 0:
        verbose = 4
    else:
        verbose = 0

    if mol_dict is None:
        mol_dict = {}

    mf, mol_dict = do_hf_calculation(mol, chkfile, mol_dict=mol_dict)

    log = logger.Logger(mol.stdout, verbose=verbose)
    # CCSD
    mycc = cc.RCCSD(mf)
    mycc.verbose = verbose
    log.info('Starting cc kernel...')
    mycc.kernel()
    log.info('Starting cc lambda solver...')
    mycc.solve_lambda()
    log.info('Getting cc density matrix...')
    dm_cc = mycc.make_rdm1()

    # get ef and freqs (to be consistent with GW)
    nocc = mol.nelectron // 2
    ef = (mf.mo_energy[nocc-1] + mf.mo_energy[nocc]) / 2.0
    freqs, wts =  _get_scaled_legendre_roots(nw=nw) #get_pade18(nocc, ef)

    # run CCGF
    orbs = range(len(mf.mo_energy))
    log.info('Getting CCGF object from fcdmft...')
    myccgf = ccgf_mor.CCGF(mycc, tol=1e-3)
    myccgf.verbose = verbose
    omega = ef + 1j*freqs
    # if two_sided_omega:
    #     omega = np.concatenate((omega.conj()[::-1], omega))
    #     wts = np.concatenate((wts[::-1], wts))
    log.info('Getting IP...')
    g_ip = myccgf.ipccsd_mo(orbs, orbs, omega.conj(), broadening=0).conj()
    log.info('Getting EA...')
    g_ea = myccgf.eaccsd_mo(orbs, orbs, omega, broadening=0)
    gf = g_ip + g_ea

    # get sigma(iw)
    gf0 = get_g0(omega, mf.mo_energy, eta=0)    
    sigmaI = (np.linalg.inv(gf0.T) - np.linalg.inv(gf.T)).T.copy()

    # AC from sigma(iw) to sigma(w)
    coeff, omega_fit = my_AC_pade_thiele_full(sigmaI, omega)

    # get CCGF DOS
    # freqs_dos = np.linspace(-1,1,201)
    # eta = 0.01
    # dos = get_dos_true_mo_ccgf(freqs_dos, eta, coeff, omega_fit, mf.mo_energy)
    # for iw in range(len(dos)):
    #     print (freqs_dos[iw], dos[iw])

    # An example for saving CCGF results
    for name, obj in zip(['ef', 'dm_cc', 'freqs', 'wts', 'sigmaI', 'coeff', 'omega_fit'],
                     [ef, dm_cc, freqs, wts, sigmaI, coeff, omega_fit]):
        mol_dict[name] = np.asarray(obj)

    return mol_dict

def save_ccsd_chk(mycc, chkfile, log, purge_amplitudes = False):
    """save pyscf cc object with amplitudes needed for subsequent CCGF calculation

    Args:
        mycc (pyscf.cc)
        chkfile (_type_): pyscf/mlgf chkfile
        log (pyscf.lib.logger): pyscf logger object
        purge_amplitudes (bool, optional): purge the amplitudes before saving. Defaults to False.
    """    
    if purge_amplitudes:
        lib.chkfile.dump(chkfile, 'ccsd', {})
        return

    ccsd_dict = vars(mycc).copy()
    pop_keys = ['mol', '_scf', 'stdout','_nmo', '_nocc', 'callback', 'frozen']
    amplitude_keys = ['t1', 't2', 'l1', 'l2']
    if purge_amplitudes:
        pop_keys = pop_keys + amplitude_keys
    for key in pop_keys: 
        try:
            ccsd_dict.pop(key)
            if key in amplitude_keys:
                log.info(f'cluster amplitude {key} removed for memory saving.')
        except KeyError as e:
            pass
            # log.info(f'ERROR: {str(e)}, {key} not removed')
        
        if '_keys' in ccsd_dict.keys():
            try:
                ccsd_dict['_keys'].remove(key)
            except ValueError as e:
                pass

    # save to the same chk file as the scf under the new "ccsd" key
    lib.chkfile.dump(chkfile, 'ccsd', ccsd_dict)

# intended to be used like 
def do_ccsd_calculation(mol, chkfile, mol_dict={}, verbose=4, xc = 'hf'):
    """do CCSD calculation and save amplitudes to file

    Args:
        mol (pyscf.mol)
        chkfile (_type_): mlgf/pyscf chkfile
        mol_dict (dict, optional): electronic structure data dict. Defaults to {}.
        verbose (int, optional): pyscf verbose. Defaults to 4.
        xc (str, optional): dft functional. Defaults to 'hf'.

    Returns:
        dict: electronic structure data dict
    """    

    if rank == 0:
        verbose = 4
    else:
        verbose = 0

    if mol_dict is None:
        mol_dict = {}

    mf, mol_dict = do_hf_calculation(mol, chkfile, mol_dict=mol_dict)
    mycc = cc.RCCSD(mf)

    log = logger.Logger(mol.stdout, verbose=verbose)
    # CCSD
    mycc = cc.RCCSD(mf)
    mycc.verbose = verbose
    log.info('Starting cc kernel...')
    mycc.kernel()
    log.info('Starting cc lambda solver...')
    mycc.solve_lambda()
    log.info('Getting cc density matrix...')
    dm_cc = mycc.make_rdm1()

    mol_dict['dm_cc'] = dm_cc
    mol_dict['emp2'] = mycc.emp2
    # mol_dict['e_hf'] = mycc.emp2
    mol_dict['e_corr'] = mycc.e_corr
    save_ccsd_chk(mycc, chkfile, log)

    return mol_dict

# xc is ignored
def do_ccgfmpi_calculation_real(chkfile, mol_dict, verbose=4, xc = 'hf', purge_amplitudes = True, nw = 201, eta = 0.01, tol = 1e-4):
        
    mol = lib.chkfile.load_mol(chkfile)
    mf = scf.RHF(mol)
    mf.xc = xc

    scf_data = lib.chkfile.load(chkfile, 'scf')
    mf.__dict__.update(scf_data)

    mycc = cc.RCCSD(mf)
    cc_data = lib.chkfile.load(chkfile, 'ccsd')
    mycc.__dict__.update(cc_data)

    log = logger.Logger(mol.stdout, verbose=verbose)

    # get ef and freqs (to be consistent with GW)
    nocc = mol.nelectron // 2
    ef = (mf.mo_energy[nocc-1] + mf.mo_energy[nocc]) / 2.0
    omega = np.linspace(-1, 1, int(nw)) + 1j*eta

    # run CCGF
    orbs = range(len(mf.mo_energy))
    log.info('Getting CCGF object from fcdmft...')
    myccgf = mpiccgf_mor.CCGF(mycc, tol=tol, verbose = verbose)
    myccgf.verbose = verbose
    
    # if two_sided_omega:
    #     omega = np.concatenate((omega.conj()[::-1], omega))
    #     wts = np.concatenate((wts[::-1], wts))
    log.info('Getting IP...')
    g_ip = myccgf.ipccsd_mo(orbs, orbs, omega.conj(), broadening=0).conj()
    log.info('Getting EA...')
    g_ea = myccgf.eaccsd_mo(orbs, orbs, omega, broadening=0)
    gf = g_ip + g_ea

    # get sigma(iw)
    gf0 = get_g0(omega, mf.mo_energy, eta=0)  
    if rank == 0:
        t0 = time.time()  
    sigmaI = (np.linalg.inv(gf0.T) - np.linalg.inv(gf.T)).T.copy()
    dos = -np.trace(gf.imag, axis1=0, axis2=1) / np.pi
    print(dos)

    if rank == 0:
        t_inv = time.time() - t0
        log.info(f'time for full dyson to get sigmaI: {t_inv:0.5f}s')


    # saving CCGF results
    for name, obj in zip(['ef', 'dos', 'sigmaI', 'omega'],
                     [ef, dos, sigmaI, omega]):

        mol_dict[name] = np.asarray(obj)
    
    comm.Barrier()
    if rank == 0:
        save_ccsd_chk(mycc, chkfile, log, purge_amplitudes = purge_amplitudes)
    comm.Barrier()
   
    return mol_dict

# xc is ignored
def do_ccgfmpi_calculation(chkfile, mol_dict, verbose=4, xc = 'hf', purge_amplitudes = True, nw = 30, gl_grid = False, tol = 1e-4):
    """do ccgf with MPI parallelization over the orbitals

    Args:
        chkfile (chkfile): pyscf/mlf chkfile
        mol_dict (dict): mlf dictionary
        verbose (int, optional): pyscf verbose. Defaults to 4.
        xc (str, optional): DFT functional for CC starting point. Defaults to 'hf'.
        purge_amplitudes (bool, optional): purge the cluster amplitudes saved from do_generate_qm9_ccsd(). Defaults to True.
        nw (int, optional): number of freq points on which to evaluate sigma. Defaults to 30.
        gl_grid (bool, optional): evaluate sigmaI on a GL grid of iomega. Defaults to False.
        tol (bool, optional): GMRES tol for linear solver to get CCGF, defaults to 1e-4.

    Returns:
        dict: mlf dictionary with CCGF data
    """    
    mol = lib.chkfile.load_mol(chkfile)
    mf = scf.RHF(mol)
    mf.xc = xc

    scf_data = lib.chkfile.load(chkfile, 'scf')
    mf.__dict__.update(scf_data)

    mycc = cc.RCCSD(mf)
    cc_data = lib.chkfile.load(chkfile, 'ccsd')
    mycc.__dict__.update(cc_data)

    log = logger.Logger(mol.stdout, verbose=verbose)

    # get ef and freqs (to be consistent with GW)
    nocc = mol.nelectron // 2
    ef = (mf.mo_energy[nocc-1] + mf.mo_energy[nocc]) / 2.0

    if not gl_grid:
        freqs, wts = get_pade18()
        omega = ef + 1j*freqs
    else:
        freqs, wts = _get_scaled_legendre_roots(nw=int(nw))
        # freqs = np.concatenate([[7.15785522e-05], freqs])
        omega = ef + 1j*freqs
    # freqs, wts =  _get_scaled_legendre_roots(nw=int(nw)) #get_pade18(nocc, ef)
    # omega = ef + 1j*freqs

    # run CCGF
    orbs = range(len(mf.mo_energy))
    log.info('Getting CCGF object from fcdmft...')
    myccgf = mpiccgf_mor.CCGF(mycc, tol=tol, verbose = verbose)
    myccgf.verbose = verbose
    
    # if two_sided_omega:
    #     omega = np.concatenate((omega.conj()[::-1], omega))
    #     wts = np.concatenate((wts[::-1], wts))
    log.info('Getting IP...')
    g_ip = myccgf.ipccsd_mo(orbs, orbs, omega.conj(), broadening=0).conj()
    log.info('Getting EA...')
    g_ea = myccgf.eaccsd_mo(orbs, orbs, omega, broadening=0)
    gf = g_ip + g_ea

    # get sigma(iw)
    gf0 = get_g0(omega, mf.mo_energy, eta=0)  
    if rank == 0:
        t0 = time.time()  
    sigmaI = (np.linalg.inv(gf0.T) - np.linalg.inv(gf.T)).T.copy()

    if rank == 0:
        t_inv = time.time() - t0
        log.info(f'time for full dyson to get sigmaI: {t_inv:0.5f}s')

    # AC from sigma(iw) to sigma(w)
    if not gl_grid:
        _, omega_fit = my_AC_pade_thiele_full(sigmaI, omega)
    else:
        omega_fit = omega


    # saving CCGF results
    for name, obj in zip(['ef', 'freqs', 'wts', 'sigmaI', 'omega_fit'],
                     [ef, freqs, wts, sigmaI, omega_fit]):

        mol_dict[name] = np.asarray(obj)
    
    comm.Barrier()
    if rank == 0:
        save_ccsd_chk(mycc, chkfile, log, purge_amplitudes = purge_amplitudes)
    comm.Barrier()
    mol_dict['gmres_tol'] = tol
    return mol_dict

def do_gw_calculation(mol, chkfile, mol_dict = None, xc = 'hf', outcore = False, band_gap_only = False, use_existing_scf = False, nw2 = None):
    """Do a G0W0 calculation

    Args:
        mol (pyscf.mol):
        chkfile (chkfile): pyscf/mlf chkfile
        mol_dict (dict): mlf dictionary
        xc (str): DFT functional for pyscf
        outcore (bool, optional): split up GW calculation of rho response into a loop. Defaults to False.
        gw_band_gap_only (bool, optional): only compute band-gap with GWAC. Defaults to False.
        use_existing_scf (bool, optional): tries to skip existing DFT calculation output if possible. Defaults to False.
        nw2 (int, optional): nomega GL points on which to evaluate sigmaI; integration still is carried out on nw = 100. Defaults to None, in which case GWGF computes on the same 100 GL grid used for integration.

    Returns:
        dict: mlf dictionary
    """    
    
    if mol_dict is None:
        mol_dict = {}
    # Hartree-Fock calculation
    time_init = time.time()
    
    if use_existing_scf and os.path.isfile(chkfile):
        
        
        scf_data = lib.chkfile.load(chkfile, 'scf')
        
        if scf_data is None:
            mf, mol_dict = do_rks_calculation(mol, chkfile, mol_dict=mol_dict, xc = xc)
        else:
            print(f'Using existing SCF calculation from {chkfile}')
            mf = scf.RKS(mol)
            mf.xc = xc
            mf.__dict__.update(scf_data)
            dm = mf.make_rdm1()
            # DFT/HF calculation outputs
            mol_dict['e_tot'] = mf.e_tot
            # occupation number/number of electrons
            nocc =  mol.nelectron // 2
            mol_dict['nocc'] = nocc
            # occupation number of each orbital
            mol_dict['mo_occ'] = np.asarray(mf.mo_occ)
            mol_dict['mo_energy'] = np.asarray(mf.mo_energy)    # orbital energy
            mol_dict['mo_coeff'] = np.asarray(mf.mo_coeff)      # orbital coefficient
            mol_dict['ovlp'] = np.asarray(mf.get_ovlp())        # overlap matrix
            mol_dict['hcore'] = np.asarray(mf.get_hcore())      # hcore matrix
            
            vj, vk = mf.get_jk()
            mol_dict['vj'] = np.asarray(vj)                     # Coulomb matrix
            mol_dict['vk'] = np.asarray(vk)                     # exchange matrix
            # mol_dict['vxc'] = np.asarray(mf.get_veff() - vj)    # exchange-correlation matrix
            mol_dict['fock'] = np.asarray(mf.get_fock())        # Fock matrix
            mol_dict['dm_hf'] = np.asarray(dm)                  # mean field density matrix
            mol_dict['xc'] = 'hf'

            # for self-energy ACs
            mol_dict['vk_hf'] = mf.get_veff(mol, dm) - mf.get_j(mol, dm)
            mol_dict['ef'] = (mf.mo_energy[nocc-1] + mf.mo_energy[nocc]) / 2.0
            print('Done loading previous SCF')

    else:
        mf, mol_dict = do_rks_calculation(mol, chkfile, mol_dict=mol_dict, xc = xc)
    
    time_rks = time.time() - time_init

    time_init = time.time()
    
    if band_gap_only:
        nocc = mol_dict['nocc']
        gw = GWAC(mf)
        gw.outcore = outcore
        gw.ac = 'pade'
        gw.kernel(orbs=range(nocc-1,nocc+1), nw=100)
        mol_dict['qpe'] = gw.mo_energy
        mol_dict['ef'] = (mf.mo_energy[nocc-1] + mf.mo_energy[nocc]) / 2.0
        return mol_dict

    else:
        # GW calculation
        
        gw = GWGF(mf)
        gw.ac = 'pade'
        gw.verbose = 0
        gw.fullsigma = True
        gw.rdm = True
        gw.eta = 1e-2
        gw.outcore = outcore
        omega_gf = np.linspace(-0.5, 0.5, 1) # we don't need this
        # TZ revision NOTE : change nw to 100 for faster run
        gw.kernel(omega=omega_gf, nw=100, nw2 = nw2)
    time_gw = time.time() - time_init


    ef = gw.ef # Fermi level energy/chemical potential
    dm_gw = gw.make_rdm1() # GW density matrix
    freqs = gw.freqs # frequency points for the self-energy, real part
    wts = gw.wts # weights of each frequency point
    

    # TZ revision NOTE: modify freqs and sigmaI indices to be consistent with original GW code
    if nw2 is None:
        freqs = np.concatenate(([0.], freqs))
        iw_cutoff = 5.0
        nw_sigma = sum(freqs < iw_cutoff)
        sigmaI = gw.sigmaI[:,:,:nw_sigma]
        freqs = freqs[:nw_sigma]
        # this step is cheap, doing it here is fine, coeff=Pade fitting coefficient, omega_fit=fitting frequency
        coeff, omega_fit = AC_pade_thiele_full(sigmaI, freqs*1j+ef)
    else:
        omega_ac = np.concatenate([[0], freqs])
        nw_sigma = sum(omega_ac < 5.0)
        omega_ac = omega_ac[:nw_sigma]
        idx = _get_ac_idx(len(omega_ac), idx_start=1)
        omega_ac = omega_ac[idx]
        sigmaI = gw.sigmaI[:,:,1:]
        coeff = 0
        mol_dict['omega_ac'] = omega_ac*1j+ef
        omega_fit = freqs*1j+ef


    
    
    # GW part
    for name, obj in zip(['ef', 'dm_gw', 'freqs', 'wts', 'sigmaI', 'coeff', 'omega_fit'],
                     [ef, dm_gw, freqs, wts, sigmaI, coeff, omega_fit]):
        mol_dict[name] = np.asarray(obj)

    # print(chkfile, np.abs(np.trace(dm_gw) - np.sum(mol_dict['mo_occ'])))
    mol_dict['time_gw'] = time_gw
    mol_dict['time_rks'] = time_rks
    
    return mol_dict

def _helper(coords, basis, outdir, name, green_function_calculate, verbose):
    mol = gto.M(atom=coords, basis=basis, verbose=verbose, parse_arg=False, xc = 'hf')
    mol.my_name = name
    chkfile = os.path.join(outdir, f'{name}.chk')

    if os.path.isfile(joblib_file):
        mol_dict = lib.chkfile.load(chkfile, 'mlf')
    else:
        mol_dict = {}
    
    calculators = {'gw': do_gw_calculation, 'ccgf': do_ccgf_calculation, 'fcigf': do_fcigf_calculation}
    
    calculators[green_function_calculate](mol, mf_chkfile, mol_dict=mol_dict, xc = xc)

    # if 'mol' not in mol_dict or not isinstance(mol_dict['mol'], str):
    #     mol_dict['mol'] = mol.dumps()

    lib.chkfile.save(chkfile, 'mlf', mol_dict)



if __name__ == '__main__':
    
    print('command line generate.py deprecated')
    raise 

