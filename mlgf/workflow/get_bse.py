from mlgf.data import Dataset, Moldatum
from pyscf import scf, lib, dft
import joblib
import numpy as np
import argparse
from PIL import Image
import json
import os
import pandas as pd

from mlgf.workflow.get_ml_info import get_bse_singlets

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='get_bse.py')
    # must have key jobs with a list of dictionaries, each with key model_file, output_file, mlf_chkfile
    parser.add_argument('--json_spec', required=True, help='head directory where all molecules subdirectories have checkpoints from generate.py')
    # by default True

    args = parser.parse_args()
    json_spec = args.json_spec

    defaults = {'bse_nroot' : 10}

    assert('.json' in json_spec)

    with open(json_spec) as f:
        spec = json.load(f)
    
    bse_active_space = spec.get('bse_active_space', None)
    bse_nroot = spec.get('bse_nroot', defaults['bse_nroot'])
    validation_files = spec['validation_files']
    validation_keys = spec['validation_keys']
    skip_true_files = spec.get('skip_true_files', [])
    for validation_file in validation_files:
        for validation_key in validation_keys:
            validation = lib.chkfile.load(validation_file, validation_key)
            mlf_chkfile = validation['mlf_chkfile'].decode('utf-8')
            nmo = len(validation['mo_energy'])
            nocc = validation['nocc']

            qpe_ml = validation['qpe_ml']
            exci_s_ml, dipole_ml, oscillator_strength_ml, bse_active_space_ml = get_bse_singlets(mlf_chkfile, qpe_ml, nmo, nocc, nroot = bse_nroot, bse_active_space = bse_active_space, xc = 'pbe0')
            validation[f'exci_s_ml_nroot{bse_nroot}'] = exci_s_ml
            validation[f'dipole_ml_nroot{bse_nroot}'] = dipole_ml
            validation[f'oscillator_strength_ml_nroot{bse_nroot}'] = oscillator_strength_ml
            
            if validation_file in skip_true_files:
                lib.chkfile.save(validation_file, validation_key, validation)
                continue
            
            try:

                qpe_true = validation['qpe_true']
                exci_s_true, dipole_true, oscillator_strength_true, _ = get_bse_singlets(mlf_chkfile, qpe_true, nmo, nocc, nroot = bse_nroot, bse_active_space = bse_active_space_ml, xc = 'pbe0')

                validation[f'exci_s_true_nroot{bse_nroot}'] = exci_s_true
                validation[f'dipole_true_nroot{bse_nroot}'] = dipole_true
                validation[f'oscillator_strength_true_nroot{bse_nroot}'] = oscillator_strength_true
            except KeyError as e:
                print(validation_file, validation_key, ' has keyerror: ', str(e))
            
            lib.chkfile.save(validation_file, validation_key, validation)
        
 