import numpy as np
from mlgf.lib.pes import hybridization_fitting

def my_AC_pade_thiele_full(sigma, omega, step_ratio=2.0/3.0):
    """
    Analytic continuation to real axis using a Pade approximation

    Args:
        sigma : 3D array (norbs, norbs, nomega)
        omega : 1D array (nomega)

    Returns:
        coeff: 3D array (ncoeff, norbs, norbs)
        omega : 1D array (ncoeff)
    """
    norbs, norbs, nw = sigma.shape
    npade = nw // 2 # take even number of points.
    coeff = thiele_ndarray(sigma[:,:,:npade*2], omega[:npade*2])
    #coeff = np.zeros((npade*2, norbs, norbs),dtype=np.complex128)
    #for p in range(norbs):
    #    for q in range(norbs):
    #        coeff[:,p,q] = thiele(sigma[p,q,:npade*2], omega[:npade*2])
    return coeff, omega[:npade*2]


def my_AC_pade_thiele_diag(sigma, omega, step_ratio=2.0/3.0):
    """
    Analytic continuation to real axis using a Pade approximation
    from Thiele's reciprocal difference method
    Reference: J. Low Temp. Phys. 29, 179 (1977)
    Args:
        sigma : 2D array (norbs, nomega)
        omega : 1D array (nomega)
    Returns:
        coeff : 2D array (ncoeff, norbs)
        omega : 1D array (ncoeff)
    """
    norbs, nw = sigma.shape
    npade = nw // 2 # take even number of points.
    coeff = thiele_ndarray(sigma[:,:npade*2], omega[:npade*2])
    # coeff = np.zeros((npade*2, norbs),dtype=np.complex128)
    # for p in range(norbs):
    #     coeff[:, p] = thiele(sigma[p,:npade*2], omega[:npade*2])
    return coeff, omega[:npade*2]

def get_poles_weights_pes(sigma_fit, omega_fit, _pes_mmax = 5, _pes_maxiter = 200, _pes_disp = False):
    pols, weights = [], []
    norbs = sigma_fit.shape[0]
    for i in range(norbs):
        pol, weight = hybridization_fitting(
            sigma_fit[i, i, :], omega_fit, mmax=_pes_mmax, maxiter=_pes_maxiter, disp=_pes_disp)
        pols.append(pol)
        weights.append(weight)
    return pols, weights

def pade_thiele_ndarray(freqs, zn, coeff):
    """NDarray-friendly analytic continuation using Pade-Thiele method.

    Parameters
    ----------
    freqs : array_like, shape (nfreqs,), complex
        Points in the complex plane at which to evaluate the analytic continuation.
    zn : array_like, shape (ncoeff,), complex
        interpolation points
    coeff : array_like, shape (ncoeff, M1, M2, ...,), complex
        Pade-Thiele coefficients

    Returns
    -------
    array_like, shape (M1, M2, ..., nfreqs,), complex
        Pade-Thiele analytic continuation evaluated at `freqs`
    """
    
    ncoeff = len(coeff)
    
    if freqs.ndim != 1 or zn.ndim != 1:
        raise ValueError('freqs and zn must be 1D arrays')
    if ncoeff != len(zn):
        raise ValueError('coeff and zn must have the same length')
    
    X = coeff[-1, ..., np.newaxis] * (freqs - zn[-2])
    # X has shape (M1, M2, ..., nfreqs)
    
    for i in range(ncoeff - 1):
        idx = ncoeff - i - 1
        X = coeff[idx, ..., np.newaxis] * (freqs - zn[idx - 1]) / (1.0 + X)
    X = coeff[0, ..., np.newaxis] / (1.0 + X)
    return X

def thiele_ndarray(fn, zn):
    """Iterative Thiele algorithm to compute coefficients of Pade approximant

    Parameters
    ----------
    fn : array_like, shape (N1, N2, ..., nfit), complex
        Function values at the points zn
    zn : array_like, shape(nfit,), complex
        Points in the complex plane used to compute fn

    Returns
    -------
    array_like, shape(nfit, N1, N2, ...), complex
        Coefficients of Pade approximant
    """
    nfit = len(zn)
    fnt = np.moveaxis(fn, -1, 0)
    # fnt has shape (nfit, N1, N2, ...)
    
    # No need to allocate coeffs since g = coeffs at the end.
    # coeffs = np.zeros(fnt.shape, dtype=fnt.dtype)
    
    g = fnt.copy()
    # g has shape (nfit, N1, N2, ...)
    for i in range(1, nfit):
        # At this stage, coeffs[i-1] is already computed.
        # coeffs[i-1] = g[i-1]

        # We have to write (zn-zn[i-1] * g.T).T
        # to multiply by (zn-zn[i-1]) along the first axis of g.
        g[i:] = (g[i-1] - g[i:]) / ((zn[i:] - zn[i-1]) * g[i:].T).T

    return g


def thiele_theile_asidfjopwher_just_do_the_analytic_continuation_okay(fnvals_at_zn, zn, new_pts):
    """For when you are frustrated

    Parameters
    ----------
    fnvals_at_zn, : array_like, shape (N1, N2, ..., nw), complex
        function values at the points zn
    zn : array_like, shape (nw,), complex
        points where you evaluated the function
    new_pts : array_like, shape (nw_new,), complex
        points where you want the function to be evaluated
    
    Returns
    -------
    array_like, shape (N1, N2, ..., nw_new), complex
        (approximated) function values at the points new_pts
    """
    
    coeffs = thiele_ndarray(fnvals_at_zn, zn)
    return pade_thiele_ndarray(new_pts, zn, coeffs)