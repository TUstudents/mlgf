import numpy as np
from mlgf.lib.gfhelper import pade_thiele_ndarray, thiele_ndarray
from fcdmft.gw.mol.gw_gf import get_g0
from mlgf.lib.gfhelper import my_AC_pade_thiele_full, my_AC_pade_thiele_diag, get_poles_weights_pes
from mlgf.lib.pes import eval_with_pole

try:
    import baryrat
    has_baryrat = True
except ImportError:
    has_baryrat = False

def get_dos_hf(mo_energy, freqs, eta):
    GHFR = get_g0(freqs, mo_energy, eta)
    return -np.trace(GHFR.imag, axis1=0, axis2=1) / np.pi

def calc_dos_pes_experimental(sigma, freqs, eta, omega_fit, mo_energy):
    """Generate DOS using true sigma in the MO basis with optional full matrix inversion

    Parameters
    ----------
    freqs : array_like, shape (nw,), float
        real frequencies at which to evaluate the DOS
    eta : float
        broadening factor
    ac_coeff : array_like, shape (ncoeff, nmo, nmo), complex
        pade-thiele coefficients
    omega_fit : array_like, shape (ncoeff,), complex
        pade-thiele interpolation points
    mo_energy : array_like, shape (nmo,), float
        MO energies
    vk_minus_vxc : array_like, shape (nmo, nmo), float, optional
        Difference between HF exchange and DFT exchange-correlation potential
    diag : bool, optional
        If True, use diagonal approximation

    Returns
    -------
    array_like, shape (nw,), float
        density of states at the given frequencies
    """

    GHFI = get_g0(omega_fit, mo_energy, 0.)

    # dyson equation to get G on the imaginary axis
    GI = np.linalg.inv(np.linalg.inv(GHFI.T) - sigma.T).T

    # AC fitting to get GR
    poles, weights = get_poles_weights_pes(GI, omega_fit, _pes_mmax = 10, _pes_maxiter = 200, _pes_disp = False)
    # print(poles, weights)
    return calc_dos_pes_GI(freqs, poles, weights)

def calc_dos_pes_GI(freqs, pols, weights):
    norbs = len(pols)
    GR = np.empty((norbs, len(freqs)))
    for ip in range(norbs):
        GR[ip,:] = eval_with_pole(pols[ip], freqs, weights[ip])
    return -np.sum(GR.imag, axis=0) / np.pi

def calc_dos_pes(freqs, eta, pols, weights, omega_fit, mo_energy, vk_minus_vxc):
    norbs = len(mo_energy)

    sigmaR = np.empty((norbs, len(freqs)))
    for ip in range(norbs):
        sigmaR[ip,:] = eval_with_pole(pols[ip], freqs, weights[ip])

    GHFR = np.reciprocal(np.add.outer(-mo_energy, freqs+1j*eta))
    sigmaR += np.expand_dims(np.diagonal(vk_minus_vxc), -1)
    GR = 1.0 / (1.0 / GHFR - sigmaR)
    return -np.sum(GR.imag, axis=0) / np.pi


def calc_dos(freqs, eta, ac_coeff, omega_fit, mo_energy, vk_minus_vxc=None, diag=False):
    """Generate DOS using true sigma in the MO basis with optional full matrix inversion

    Parameters
    ----------
    freqs : array_like, shape (nw,), float
        real frequencies at which to evaluate the DOS
    eta : float
        broadening factor
    ac_coeff : array_like, shape (ncoeff, nmo, nmo), complex
        pade-thiele coefficients
    omega_fit : array_like, shape (ncoeff,), complex
        pade-thiele interpolation points
    mo_energy : array_like, shape (nmo,), float
        MO energies
    vk_minus_vxc : array_like, shape (nmo, nmo), float, optional
        Difference between HF exchange and DFT exchange-correlation potential
    diag : bool, optional
        If True, use diagonal approximation

    Returns
    -------
    array_like, shape (nw,), float
        density of states at the given frequencies
    """

    if not diag:
        GHFR = get_g0(freqs, mo_energy, eta)
        sigmaR = pade_thiele_ndarray(freqs + 1j * eta, omega_fit, ac_coeff)

        if vk_minus_vxc is not None:
            np.add(sigmaR, np.expand_dims(vk_minus_vxc, -1), out=sigmaR)
            # sigmaR += vk_minus_vxc
            
        # dyson equation to get G on the real axis
        GR = np.linalg.inv(np.linalg.inv(GHFR.T) - sigmaR.T).T

        # get DOS
        return -np.trace(GR.imag, axis1=0, axis2=1) / np.pi
    
    else: # diag
        # shape (nmo, nw)
        GHFR = np.reciprocal(np.add.outer(-mo_energy, freqs+1j*eta))

        # get sigma on the real axis
        sigmaR = pade_thiele_ndarray(freqs+1j*eta, omega_fit, ac_coeff)

        if vk_minus_vxc is not None:
            sigmaR += np.expand_dims(np.diagonal(vk_minus_vxc), -1)
        
        # dyson equation
        GR = 1.0 / (1.0 / GHFR - sigmaR)

        # get DOS
        return -np.sum(GR.imag, axis=0) / np.pi

def get_dos_orthbasis_fullinv(freqs, eta, ac_coeff, omega_fit, mo_energy):
    return calc_dos(freqs, eta, ac_coeff, omega_fit, mo_energy, diag=False)

def get_dos_orthbasis_diag(freqs, eta, ac_coeff, omega_fit, mo_energy, vk, vxc):
    return calc_dos(freqs, eta, ac_coeff, omega_fit, mo_energy, vk_minus_vxc=vk-vxc, diag=True)

class NDAAAFit:
    def __init__(self, sigmaI, omega_fit, mo_energy, vk_minus_vxc=None, diag=False, eta=1e-3, fitGF=False, fitDOS=False):
        self.sigmaI = sigmaI
        self.omega_fit = omega_fit
        self.diag = diag
        shape = self.sigmaI.shape
        self.sliceshape = shape[:-1]
        self.mo_energy = mo_energy
        self.vk_minus_vxc = vk_minus_vxc
        self.eta = eta
        self.fitGF=fitGF
        self.fitDOS = fitDOS
        if not has_baryrat:
            raise ImportError("You need baryrat to use NDAAAFit. Install with pip.")
    def kernel(self, **kwargs):
        self.funs = np.empty(self.sliceshape, dtype=object)
        if not self.fitGF:
            tofit = self.sigmaI
        else:
            sig = self.sigmaI.copy()
            if self.vk_minus_vxc is not None:
                sig = sig + np.expand_dims(self.vk_minus_vxc, -1)
            tofit = np.linalg.inv(np.linalg.inv(get_g0(self.omega_fit, self.mo_energy, 0).T) - sig.T).T
            #tofit =  1.0 / (1.0 / get_g0(self.omega_fit, self.mo_energy, 0) - sig)
            #np.nan_to_num(tofit, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            
        if self.fitDOS:
            assert(self.fitGF)
            
            tofit = np.trace(tofit, axis1=0, axis2=1) / np.pi
            self.fun = baryrat.aaa(self.omega_fit, tofit.flatten(), **kwargs)
            print("order ", self.fun.order)
            return
        for idx in np.ndindex(self.sliceshape):
            #self.funs[idx] = baryrat.interpolate_with_degree(self.omega_fit, tofit[idx].flatten(), (8,9))
            self.funs[idx] = baryrat.aaa(self.omega_fit, tofit[idx].flatten(), **kwargs)
        print("maxorder ", max([f.order for f in self.funs.flatten()]))
    def evaluate(self, omega):
        if self.fitDOS:
            return self.fun(omega)
        ret = np.empty(self.sliceshape + (len(omega),), dtype=complex)
        for idx in np.ndindex(self.sliceshape):
            ret[idx] = self.funs[idx](omega)
        return ret

    def getdos(self, freqs):
        if self.fitDOS:
            return -self.evaluate(freqs+1j*self.eta).imag
        if self.fitGF:
            dos= -np.trace(self.evaluate(freqs + 1j*self.eta).imag, axis1=0, axis2=1) / np.pi
            return dos        
        if not self.diag:
            GHFR = get_g0(freqs, self.mo_energy, self.eta)
            sigmaR = self.evaluate(freqs + 1j * self.eta)

            if self.vk_minus_vxc is not None:
                np.add(sigmaR, np.expand_dims(self.vk_minus_vxc, -1), out=sigmaR)
                # sigmaR += vk_minus_vxc
                
            # dyson equation to get G on the real axis
            GR = np.linalg.inv(np.linalg.inv(GHFR.T) - sigmaR.T).T

            # get DOS
            return -np.trace(GR.imag, axis1=0, axis2=1) / np.pi
        
        else:
            GHFR = np.reciprocal(np.add.outer(-self.mo_energy, freqs+1j*self.eta))
            sigmaR = self.evaluate(freqs+1j*self.eta)
            if self.vk_minus_vxc is not None:
                sigmaR += np.expand_dims(np.diagonal(self.vk_minus_vxc), -1)
            GR = 1.0 / (1.0 / GHFR - sigmaR)
            return -np.sum(GR.imag, axis=0) / np.pi

    def getdos2(self, freqs, eta, mo_energy, vk_minus_vxc=None):
        if not self.diag:
            GHFR = get_g0(freqs, mo_energy, eta)
            sigmaR = self.evaluate(freqs + 1j * eta)

            if vk_minus_vxc is not None:
                np.add(sigmaR, np.expand_dims(vk_minus_vxc, -1), out=sigmaR)
                # sigmaR += vk_minus_vxc
                
            # dyson equation to get G on the real axis
            GR = np.linalg.inv(np.linalg.inv(GHFR.T) - sigmaR.T).T

            # get DOS
            return -np.trace(GR.imag, axis1=0, axis2=1) / np.pi
        
        else:
            GHFR = np.reciprocal(np.add.outer(-mo_energy, freqs+1j*eta))
            sigmaR = self.evaluate(freqs+1j*eta)
            if vk_minus_vxc is not None:
                sigmaR += np.expand_dims(np.diagonal(vk_minus_vxc), -1)
            GR = 1.0 / (1.0 / GHFR - sigmaR)
            return -np.sum(GR.imag, axis=0) / np.pi