# DSRG Experiment 0

One-off execution branch for the first go/no-go test of the adaptive intermediate-Hamiltonian proposal.

The workflow reproduces the public NiuPy BeH2/IP-EOM-DSRG test environment using the same upstream build recipe (Psi4 master, Forte `eom-dsrg`, Wick&d PR 20, NiuPy), validates the published `s=0.5` regression values, sweeps `s = 0...1.25`, and independently computes full-CI ionization references in the same BeH2/STO-6G basis with PySCF.

Outputs include raw IPs/spectroscopic factors, FCI reference poles and Dyson weights, matched principal-root errors, and MAE/MSE/MAX versus `s`.

This is a parent-theory/cancellation map. It does **not** yet implement the action-only `U^† H U v` high-rank residual or buffer-excluded re-flow, and it must not be interpreted as testing the final repair layer.
