import os
from mlgf.data import Dataset, Moldatum

from mlgf.lib.helpers import get_sigma_ml, get_vmo, get_sigma_fit, gGW_mo_saiao, find_subset_indices_numpy, sigma_lo_mo
# from mlgf.lib.plots import qp_xy_plot, qp_energy_level_plot
from mlgf.lib.gfhelper import my_AC_pade_thiele_diag
from mlgf.lib.pes import eval_with_pole

from pyscf import scf, lib, dft
import numpy as np
import numpy
import h5py
import argparse
import matplotlib.pyplot as plt
from PIL import Image


import joblib
from fcdmft.gw.mol.gw_ac import pade_thiele, thiele
from fcdmft.gw.mol.gw_gf import get_g0

from scipy.optimize import newton


def get_quasiparticle_energies(omega_fit, ac_coeff, vk, v_mf, mf_mo_energy):
    # self-consistently solve QP equation
    norbs = len(mf_mo_energy)
    qp_energy = np.zeros(norbs)
    for p in range(norbs):
        def quasiparticle(omega):
            sigmaR = pade_thiele(omega, omega_fit, ac_coeff[:, p]).real
            return omega - mf_mo_energy[p] - (sigmaR.real + vk[p, p] - v_mf[p, p])
        try:

            e = newton(quasiparticle, mf_mo_energy[p], tol=1e-6, maxiter=100)
            qp_energy[p] = e
        except RuntimeError:
            qp_energy[p] = np.nan

    return qp_energy

def get_quasiparticle_energies_pes(omega_fit, poles, weights, vk, v_mf, mf_mo_energy):
    # self-consistently solve QP equation
    norbs = len(mf_mo_energy)
    qp_energy = np.zeros(norbs)
    for p in range(norbs):
        def quasiparticle(omega):
            sigmaR = eval_with_pole(poles[p], omega, weights[p])
            return omega - mf_mo_energy[p] - (sigmaR.real + vk[p, p] - v_mf[p, p])
        try:

            e = newton(quasiparticle, mf_mo_energy[p], tol=1e-6, maxiter=100)
            qp_energy[p] = e
        except RuntimeError:
            qp_energy[p] = np.nan

    return qp_energy

def get_fcdmft_qpe(scf_chkfile, xc = 'hf'):
    from pyscf import gto, dft, scf
    from fcdmft.gw.mol.gw_gf import GWGF

    mol = lib.chkfile.load_mol(scf_chkfile)

    mf = dft.RKS(mol)
    mf.xc = xc
    data = lib.chkfile.load(scf_chkfile, 'scf')
    mf.__dict__.update(data)
    gw = GWGF(mf)
    gw.linearized = False
    gw.ac = 'pade'
    gw.fullsigma = True
    gw.rdm = True
    gw.eta = 1e-2
    # TZ revision NOTE : change nw to 100 for faster run
    omega_gf = numpy.linspace(-0.5, 0.5, 1) # we don't need this
    gw.kernel(omega=omega_gf, nw=100)
    return gw.mo_energy

if __name__ == '__main__':
    # configuration variables at the top
    default_src = '/vast/palmer/scratch/zhu/scv22/MD_conformer_gen/pyradine'
    default_basis = 'saiao'
    parser = argparse.ArgumentParser(prog='qp_energy.py')
    parser.add_argument('--src', default=default_src, help='head directory where all molecules subdirectories have checkpoints from generate.py')
    parser.add_argument('--mfile', default=os.path.join(os.getcwd(), 'model.joblib'), help='model filename (default is current directory)')
    parser.add_argument('-c', '--conformer-num', help='conformer number', default=0)
    parser.add_argument('--eta', type=float, default=0.01, help='eta')
    parser.add_argument('--basis_name', type=str, default=default_basis, help='One of either saiao or saao, a symmetrized local basis to express electronic features in')

    args = parser.parse_args()
    model_file = args.mfile
    src = args.src
    if src == default_src:
        print(f'Using default data directory: {default_src}')
    if not os.path.isdir(src):
        raise ValueError(f'{src} is not a valid directory')

    basis_name = args.basis_name
    if basis_name not in ['saiao', 'saao']:
        print('symmetrized basis not available, using saiao.')
        basis_name = default_basis

    conformer_num = args.conformer_num 
    pade_theile_diag = True
    eta = args.eta

    scf_chkfile = f'{src}/mol{conformer_num}.chk'
    mlf = Moldatum.load_chk(scf_chkfile)
    omega_fit = mlf['omega_fit']
    sigma_true = mlf['sigma_saiao']

    sigma_ml = get_sigma_ml(scf_chkfile, model_file, basis_name = basis_name)

    C_lo_mo = mlf[f'C_{basis_name}_mo']
    sigma_true = sigma_lo_mo(sigma_true, C_lo_mo)   
    sigma_ml = sigma_lo_mo(sigma_ml, C_lo_mo) 

    sigma_true_diag = np.diagonal(sigma_true).T
    sigma_ml_diag = np.diagonal(sigma_ml).T

    mf_mo_energy = mlf['mo_energy']
    xc = mlf.get('xc', 'hf').decode('utf-8')
    vk, v_mf = get_vmo(scf_chkfile, xc = xc)

    ac_coeff, omega_fit = my_AC_pade_thiele_diag(sigma_true_diag, omega_fit)
    qpe_true = get_quasiparticle_energies(omega_fit, ac_coeff, vk, v_mf, mf_mo_energy)

    ac_coeff, omega_fit = my_AC_pade_thiele_diag(sigma_ml_diag, omega_fit)
    qpe_ml = get_quasiparticle_energies(omega_fit, ac_coeff, vk, v_mf, mf_mo_energy)

    plt = qp_xy_plot(qpe_true, qpe_ml, mf_mo_energy)
    plt.savefig('QP_pred.png')
    plt = qp_energy_level_plot(qpe_true, qpe_ml)
    plt.savefig('QP_pred_levels.png')

    qpe_gwac_fcdmft = get_fcdmft_qpe(scf_chkfile)
    # qpe_gwac_fcdmft.sort()
    # qpe_true.sort()
    print('MAE this script minus fcdmft qpe: ', np.mean(np.abs(qpe_true-qpe_gwac_fcdmft)))

    homo, lumo = mlf['nocc']-1, mlf['nocc']
    bg_qpe_here = qpe_true[lumo] -  qpe_true[homo]
    bg_qpe_here_ml = qpe_ml[lumo] -  qpe_ml[homo]
    bg_qpe_fcdmft = qpe_gwac_fcdmft[lumo] -  qpe_gwac_fcdmft[homo]
    print('band gap diff here - fcdmft: ', bg_qpe_here-bg_qpe_fcdmft)
    print('band gap diff ml - here: ', bg_qpe_here_ml-bg_qpe_here)


    
