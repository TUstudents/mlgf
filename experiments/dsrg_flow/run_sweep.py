#!/usr/bin/env python3
"""Experiment 0: actual IP-EOM-DSRG flow-parameter sweep on the public NiuPy BeH2 test.

This is an end-to-end execution test of the published software stack. The
same small basis is also solved by full CI (PySCF) so the s-dependence can be
scored against an exact-in-basis charged spectrum rather than against another
approximate method.

The public NiuPy s=0.5 regression is a hard precondition: no other flow
parameter is evaluated unless that regression first passes.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

EV = 27.211386245988
REGRESSION_S = 0.50
S_VALUES = [0.0, 0.01, 0.05, 0.10, 0.25, 0.35, 0.50, 0.75, 1.00, 1.25]
# Exact expectations for NiuPy's public EOM_DSRG.kernel() BeH2 regression.
TEST_IPS = np.array([11.0712299826, 12.9096937411, 17.2339756217, 17.3971330810])
TEST_SPECS = np.array([1.97180714, 1.95473561, 0.0, 0.00213609])


def input_text(s: float) -> str:
    return f'''import niupy
import psi4
import forte
import forte.utils
import numpy as np

x = 1.000
molecule = psi4.geometry(f"""
Be 0.0   0.0             0.0
H  {{x}}   {{2.54-0.46*x}}   0.0
H  {{x}}  -{{2.54-0.46*x}}   0.0
symmetry c2v
units bohr
""")

set {{
  basis sto-6g
}}

set forte{{
  mcscf_reference         true
  active_space_solver     genci
  correlation_solver      mrdsrg
  corr_level              ldsrg2
  restricted_docc         [2,0,0,0]
  active                  [1,0,0,1]
  dsrg_s                  {s:.8f}
  e_convergence           10
  mcscf_g_convergence     8
  mcscf_e_convergence     12
  full_hbar               true
  full_mbar               false
  fourpdc                 mk
  relax_ref               once
  semi_canonical          true
}}

E = energy('forte')
with open('neutral_energy_ha.txt', 'w') as f:
    f.write(f"{{float(E):.16f}}\\n")
'''


def driver_text(s: float) -> str:
    return f'''import json
from pathlib import Path
import numpy as np
import niupy

eom = niupy.EOM_DSRG(
    opt_einsum=True,
    nroots=10,
    basis_per_root=20,
    collapse_per_root=2,
    max_cycle=200,
    tol_s=1e-10,
    tol_semi=1e-10,
    method_type="ip",
)
eom.kernel()
result = {{
    "s": {s!r},
    "evals_eV": np.asarray(eom.evals, dtype=float).tolist(),
    "spectroscopic_factors": np.asarray(eom.spec_info, dtype=float).tolist(),
    "spin": [str(x) for x in eom.spin],
    "symmetry": [str(x) for x in eom.symmetry],
}}
p = Path('neutral_energy_ha.txt')
if p.exists():
    result["neutral_energy_Ha"] = float(p.read_text().strip())
Path('result.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
'''


def run_case(s: float, workdir: Path) -> dict:
    tag = f"s_{s:.4f}".replace('.', 'p')
    case = workdir / tag
    case.mkdir(parents=True, exist_ok=True)
    (case / 'input.dat').write_text(input_text(s))
    (case / 'run_niupy.py').write_text(driver_text(s))

    with (case / 'psi4.stdout').open('w') as out, (case / 'psi4.stderr').open('w') as err:
        cp = subprocess.run(['psi4', 'input.dat', 'psi4.out'], cwd=case, stdout=out, stderr=err)
    if cp.returncode != 0:
        raise RuntimeError(f'Psi4/Forte failed for s={s}; see {case}/psi4.*')

    with (case / 'niupy.stdout').open('w') as out, (case / 'niupy.stderr').open('w') as err:
        cp = subprocess.run([sys.executable, 'run_niupy.py'], cwd=case, stdout=out, stderr=err)
    if cp.returncode != 0:
        raise RuntimeError(f'NiuPy failed for s={s}; see {case}/niupy.*')
    return json.loads((case / 'result.json').read_text())


def exact_fci_reference(outdir: Path, nroots: int = 24) -> pd.DataFrame:
    from pyscf import ao2mo, fci, gto, scf

    x = 1.0
    mol = gto.M(
        atom=[
            ('Be', (0.0, 0.0, 0.0)),
            ('H', (x, 2.54 - 0.46*x, 0.0)),
            ('H', (x, -(2.54 - 0.46*x), 0.0)),
        ],
        basis='sto-6g',
        unit='Bohr',
        charge=0,
        spin=0,
        verbose=3,
    )
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    C = mf.mo_coeff
    norb = C.shape[1]
    hcore = C.T @ mf.get_hcore() @ C
    eri = ao2mo.kernel(mol, C, compact=False).reshape((norb,) * 4)
    ecore = mol.energy_nuc()

    solver_n = fci.direct_spin1.FCI(mol)
    e0, ci0 = solver_n.kernel(hcore, eri, norb, (3, 3), ecore=ecore)

    solver_c = fci.direct_spin1.FCI(mol)
    solver_c.nroots = nroots
    ec, cic = solver_c.kernel(hcore, eri, norb, (2, 3), ecore=ecore, nroots=nroots)
    ec = np.atleast_1d(ec)
    if not isinstance(cic, (list, tuple)):
        cic = [cic]

    # Closed-shell singlet: alpha removal gives the M_s=-1/2 doublet component.
    # Multiplying by two yields the spin-summed removal strength.
    annih = [fci.addons.des_a(ci0, norb, (3, 3), p) for p in range(norb)]
    rows = []
    for k, (ek, cik) in enumerate(zip(ec, cic), start=1):
        alpha_weight = sum(abs(np.vdot(cik, v)) ** 2 for v in annih)
        spin_summed_weight = 2.0 * float(alpha_weight)
        try:
            ss, mult = solver_c.spin_square(cik, norb, (2, 3))
            s2 = float(ss)
            multiplicity = float(mult)
        except Exception:
            s2 = np.nan
            multiplicity = np.nan
        rows.append({
            'exact_root': k,
            'ip_eV': float((ek - e0) * EV),
            'dyson_weight_spin_summed': spin_summed_weight,
            'S2': s2,
            'multiplicity': multiplicity,
        })

    df = pd.DataFrame(rows).sort_values('ip_eV').reset_index(drop=True)
    df.to_csv(outdir / 'fci_reference.csv', index=False)
    meta = {
        'geometry': 'BeH2 public NiuPy IP test, x=1.0 bohr',
        'basis': 'STO-6G',
        'norb': int(norb),
        'neutral_fci_total_Ha': float(e0),
        'rhf_total_Ha': float(mf.e_tot),
        'dyson_weight_convention': '2 * alpha-removal strength for closed-shell singlet',
    }
    (outdir / 'fci_metadata.json').write_text(json.dumps(meta, indent=2))
    return df


def append_result(raw_rows: list[dict], s: float, d: dict) -> tuple[np.ndarray, np.ndarray]:
    e = np.asarray(d['evals_eV'], dtype=float).reshape(-1)
    p = np.asarray(d['spectroscopic_factors'], dtype=float).reshape(-1)
    if e.size != p.size:
        raise ValueError(f'eigenvalue/spec-factor length mismatch at s={s}: {e.size} != {p.size}')
    for k, (ek, pk) in enumerate(zip(e, p), start=1):
        raw_rows.append({
            's': s,
            'root': k,
            'ip_eV': float(ek),
            'spectroscopic_factor': float(pk),
            'spin': d['spin'][k - 1] if k - 1 < len(d['spin']) else '',
            'symmetry': d['symmetry'][k - 1] if k - 1 < len(d['symmetry']) else '',
            'neutral_energy_Ha': d.get('neutral_energy_Ha', np.nan),
        })
    return e, p


def validate_regression(e: np.ndarray, p: np.ndarray) -> dict:
    result = {
        'target_s': REGRESSION_S,
        'expected_ips_eV': TEST_IPS.tolist(),
        'expected_specs': TEST_SPECS.tolist(),
        'computed_ips_eV': e[:4].tolist(),
        'computed_specs': p[:4].tolist(),
        'ncomputed': int(min(e.size, p.size)),
        'ip_abs_tolerance_eV': 1e-8,
        'spec_abs_tolerance': 1e-6,
    }
    if e.size < 4 or p.size < 4:
        result['max_ip_deviation_eV'] = None
        result['max_spec_deviation'] = None
        result['passed'] = False
        result['reason'] = 'fewer than four regression roots returned'
        return result
    result['max_ip_deviation_eV'] = float(np.max(np.abs(e[:4] - TEST_IPS)))
    result['max_spec_deviation'] = float(np.max(np.abs(p[:4] - TEST_SPECS)))
    result['passed'] = bool(
        result['max_ip_deviation_eV'] < 1e-8
        and result['max_spec_deviation'] < 1e-6
    )
    return result


def match_principal(raw: pd.DataFrame, ref: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Principal photoemission roots are matched using energy + transition weight.
    # This avoids forcing dark/shake-up states into an order-based correspondence.
    refp = ref[ref.dyson_weight_spin_summed > 0.05].copy()
    all_matches = []
    summary = []
    for s, g in raw.groupby('s', sort=True):
        gp = g[g.spectroscopic_factor > 0.05].copy()
        if gp.empty or refp.empty:
            summary.append({'s': s, 'nmatch': 0, 'MAE_eV': np.nan, 'MSE_eV': np.nan, 'MAX_eV': np.nan})
            continue
        A = gp[['ip_eV', 'spectroscopic_factor']].to_numpy()
        B = refp[['ip_eV', 'dyson_weight_spin_summed']].to_numpy()
        cost = np.abs(A[:, None, 0] - B[None, :, 0]) + 0.25 * np.abs(A[:, None, 1] - B[None, :, 1])
        ii, jj = linear_sum_assignment(cost)
        pairs = []
        for i, j in zip(ii, jj):
            rg = gp.iloc[i]
            rr = refp.iloc[j]
            if rg.ip_eV > 35.0 or abs(rg.ip_eV - rr.ip_eV) > 5.0:
                continue
            err = float(rg.ip_eV - rr.ip_eV)
            pairs.append(err)
            all_matches.append({
                's': s,
                'dsrg_root': int(rg.root),
                'exact_root': int(rr.exact_root),
                'dsrg_ip_eV': float(rg.ip_eV),
                'exact_ip_eV': float(rr.ip_eV),
                'signed_error_eV': err,
                'abs_error_eV': abs(err),
                'dsrg_spec': float(rg.spectroscopic_factor),
                'exact_dyson_weight': float(rr.dyson_weight_spin_summed),
            })
        if pairs:
            a = np.asarray(pairs)
            summary.append({
                's': s,
                'nmatch': len(a),
                'MAE_eV': float(np.mean(np.abs(a))),
                'MSE_eV': float(np.mean(a)),
                'MAX_eV': float(np.max(np.abs(a))),
            })
        else:
            summary.append({'s': s, 'nmatch': 0, 'MAE_eV': np.nan, 'MSE_eV': np.nan, 'MAX_eV': np.nan})
    return pd.DataFrame(all_matches), pd.DataFrame(summary)


def write_raw_and_failures(outdir: Path, raw_rows: list[dict], failures: list[dict]) -> pd.DataFrame:
    raw = pd.DataFrame(raw_rows)
    raw.to_csv(outdir / 'sweep_raw.csv', index=False)
    pd.DataFrame(failures, columns=['s', 'error']).to_csv(outdir / 'failures.csv', index=False)
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workdir', type=Path, required=True)
    ap.add_argument('--outdir', type=Path, required=True)
    args = ap.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    args.outdir.mkdir(parents=True, exist_ok=True)

    ref = exact_fci_reference(args.outdir)
    raw_rows: list[dict] = []
    failures: list[dict] = []

    # Hard acceptance gate: reproduce the public NiuPy s=0.5 test first.
    print(f'=== regression gate s={REGRESSION_S:.4f} ===', flush=True)
    try:
        d = run_case(REGRESSION_S, args.workdir)
        e, p = append_result(raw_rows, REGRESSION_S, d)
        validation = validate_regression(e, p)
    except Exception as exc:
        failures.append({'s': REGRESSION_S, 'error': repr(exc)})
        validation = {
            'target_s': REGRESSION_S,
            'expected_ips_eV': TEST_IPS.tolist(),
            'expected_specs': TEST_SPECS.tolist(),
            'passed': False,
            'reason': repr(exc),
        }

    validation['failures'] = list(failures)
    write_raw_and_failures(args.outdir, raw_rows, failures)
    (args.outdir / 'validation.json').write_text(json.dumps(validation, indent=2))
    print(json.dumps(validation, indent=2))
    if not validation['passed']:
        raise SystemExit('Published NiuPy s=0.5 regression check did not pass; sweep aborted.')

    # Only validated software reaches the flow-parameter map.
    for s in S_VALUES:
        if abs(s - REGRESSION_S) < 1e-12:
            continue
        print(f'=== s={s:.4f} ===', flush=True)
        try:
            d = run_case(s, args.workdir)
            append_result(raw_rows, s, d)
        except Exception as exc:
            failures.append({'s': s, 'error': repr(exc)})
            print(f'FAILED s={s}: {exc}', file=sys.stderr, flush=True)

    raw = write_raw_and_failures(args.outdir, raw_rows, failures)
    matches, summary = match_principal(raw, ref) if not raw.empty else (pd.DataFrame(), pd.DataFrame())
    matches.to_csv(args.outdir / 'principal_matches.csv', index=False)
    summary.to_csv(args.outdir / 'summary.csv', index=False)

    validation['failures'] = failures
    validation['sweep_complete'] = len(failures) == 0
    (args.outdir / 'validation.json').write_text(json.dumps(validation, indent=2))

    print('\n=== exact FCI reference ===')
    print(ref.head(12).to_string(index=False))
    print('\n=== s sweep summary ===')
    print(summary.to_string(index=False) if not summary.empty else 'no successful cases')
    print('\n=== validation ===')
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
