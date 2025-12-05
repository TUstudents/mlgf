
import json
from mlgf.model.gnn_orchestrator import GraphOrchestrator, MoldatumDsetBatcher

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlgf.data import Dataset, Moldatum

import os
import numpy as np
import argparse
import joblib
import time
import warnings
import psutil
import json
import copy
import gc
import shutil

def load_mol_batcher(dset_store_dir, train_files, batch_size, seed = 42, dynamical_ftr_freqs = None, dset_purge_keys = [], core_projection_file_path = None):
    mol_batcher =  MoldatumDsetBatcher(train_files, batch_size = batch_size, dynamical_ftr_freqs = dynamical_ftr_freqs, seed = seed, core_projection_file_path = core_projection_file_path)
    mol_batcher.to_disk(dset_store_dir, mpi_rank = rank, mpi_size = size, dset_purge_keys = dset_purge_keys)
    return mol_batcher

def prepare_gnn_mpi(gnn_orch_file, train_files, dset_store_dir, torch_data_root, ensemble_n, model_alias, 
        feat_list_ii, feat_list_ij, target, dynamical_ftr_freqs = None, exclude_core = False, edge_cutoff = None, edge_cutoff_features = None,
        dset_batch_size = 10, nstatic_ij = 6, ndynamical_ij = 24, baseline_train_files = [], cat_feat_list_ii = [], cat_feat_list_ij = [], 
        ncat_ii_list = [], ncat_ij_list = [], model_kwargs = {}, loss_kwargs = {}, frontier_mo = [0, 0]
        , in_memory_data = False, seed = 42, dset_purge_keys = [], core_projection_file_path = None):
    """Prepares GraphDataset and GraphOrchestrator for subsequent training of MBGF-Net
    """
    mol_batcher = load_mol_batcher(dset_store_dir, train_files, dset_batch_size, seed = seed, dynamical_ftr_freqs = dynamical_ftr_freqs, dset_purge_keys = dset_purge_keys, core_projection_file_path = core_projection_file_path)
    comm.Barrier()
    gnn_orch = GraphOrchestrator(feat_list_ii, feat_list_ij, target, torch_data_root, in_memory_data = in_memory_data,
    cat_feature_list_ii = cat_feat_list_ii, cat_feature_list_ij = cat_feat_list_ij, ncat_ii_list = ncat_ii_list, ncat_ij_list = ncat_ij_list,
    exclude_core = exclude_core, 
    ensemble_n = ensemble_n, model_alias = model_alias, frontier_mo = frontier_mo, model_kwargs = model_kwargs, loss_kwargs = loss_kwargs,
    edge_cutoff = edge_cutoff, edge_cutoff_features = edge_cutoff_features, core_projection_file_path = core_projection_file_path)
    comm.Barrier()

    gnn_orch = gnn_orch.fit_transformers_mpi(mol_batcher, nstatic_ij, ndynamical_ij, rank, size, comm)
    comm.Barrier()
    gnn_orch = gnn_orch.load_torch_dataset_mpi(mol_batcher, torch_data_root, rank, size, comm)
    comm.Barrier()
    if rank == 0:
        joblib.dump(gnn_orch, gnn_orch_file)
        shutil.rmtree(dset_store_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='prepare_gnn_data.py')
    parser.add_argument('--json_spec', required=True, help='json file that has keys: gnn_orch_files (path string), train_files (list of lists same length as gnn_orch_files)')
    parser.add_argument('--no_new_folders', action = 'store_true', help='Do not go through all the jobs and make output folders from eacb output_file if they do not exist')

    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    comm = MPI.COMM_WORLD

    args = parser.parse_args()
    json_spec = args.json_spec
    no_new_folders = args.no_new_folders
    assert('.json' in json_spec)
     
    with open(json_spec) as f:
        job = json.load(f)
    
    # what joblib file to save the gnn_orchestrator to
    gnn_orch_file = job['gnn_orch_file']

    # numeric features and self-energy target
    feat_list_ii = job.get('feat_list_ii')
    feat_list_ij = job.get('feat_list_ij')

    # which freq points to use for the dynamical features (imag and maybe real)
    dynamical_ftr_freqs_imag = job.get('dynamical_ftr_freqs_imag', [1e-3, 0.1, 0.2, 0.5, 1.0, 2.0])
    dynamical_ftr_freqs_real = job.get('dynamical_ftr_freqs_real', [])
    dynamical_ftr_freqs = np.concatenate((np.array(dynamical_ftr_freqs_imag)*1j, np.array(dynamical_ftr_freqs_real)))
    target = job.get('target')
    
    # the .chk files use for training
    train_files = job['train_files']
    baseline_train_files = job.get('baseline_train_files', [])

    # Whether core orbitals (and associated edges) are discarded
    exclude_core = job.get('exclude_core', False)
    
    # Which categorical features are used (as on-hot encoder), and how many categories each has
    cat_feat_list_ii = job.get('cat_feat_list_ii', [])
    cat_feat_list_ij = job.get('cat_feat_list_ij', [])
    ncat_ii_list = job.get('ncat_ii_list', [])
    ncat_ij_list = job.get('ncat_ij_list', [])

    # These control the model architecture
    model_kwargs = job.get('model_kwargs', {})
    model_alias = job.get('model_alias')
    ensemble_n = job.get('ensemble_n', 1)

    # These control the loss function and model training
    loss_kwargs = job.get('loss_kwargs', {})
    frontier_mo = job.get('frontier_mo', [0, 0])
    torch_precision = job.get('torch_precision', 'float64')

    # These control the edge feature processing
    ndynamical_ij = job.get('ndynamical_ij', 24)
    nstatic_ij = job.get('nstatic_ij', 6)
    edge_cutoff = job.get('edge_cutoff', None)
    edge_cutoff_features = job.get('edge_cutoff_features', None)

    # These control how/where the database is batched and saved
    dset_batch_size = max(len(train_files) // size, 1)
    dset_store_dir = job['dset_store_dir']
    batcher_seed = job.get('batcher_seed', 42)
    torch_data_root = job.get('torch_data_root', None)
    in_memory_data = job.get('in_memory_data', False)
    purge_keys = ['C_ao_iao', 'C_ao_saiao', 'C_iao_saiao', 'coeff', 'conf_name', 'conf_num', 'dm_gw', 'dm_hf', 
    'fock', 'fock_iao', 'freqs', 'hcore', 'mol', 'ovlp', 'sigmaI', 'success', 'vj', 'vk', 'vk_hf', 'vxc', 'wts']
    core_projection_file_path = job.get('core_projection_file_path', None)
    print(core_projection_file_path)
    # Now prepare all the data for training a pytorch model (or ensemble of models)
    if rank == 0 and not no_new_folders:
        outdir = '/'.join(gnn_orch_file.split('/')[:-1])
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        elif not os.path.isdir(outdir):
            raise FileExistsError(f'{outdir} exists but is not a directory')
            comm.Abort()
        
        if not os.path.exists(dset_store_dir):
            os.makedirs(dset_store_dir)
            
    comm.Barrier()
    print('gnn_orch_file: ', gnn_orch_file)
    print('dset_store_dir: ', dset_store_dir)
    print('torch_data_root: ', torch_data_root)
    prepare_gnn_mpi(gnn_orch_file, train_files, dset_store_dir, torch_data_root, ensemble_n, model_alias, 
    feat_list_ii, feat_list_ij, target, dynamical_ftr_freqs = dynamical_ftr_freqs, exclude_core = exclude_core, edge_cutoff = edge_cutoff, edge_cutoff_features = edge_cutoff_features,
    dset_batch_size = dset_batch_size, seed = batcher_seed, nstatic_ij = nstatic_ij, ndynamical_ij = ndynamical_ij, baseline_train_files = baseline_train_files, 
    cat_feat_list_ii = cat_feat_list_ii, cat_feat_list_ij = cat_feat_list_ij, ncat_ii_list = ncat_ii_list, ncat_ij_list = ncat_ij_list, 
    model_kwargs = model_kwargs, loss_kwargs = loss_kwargs, frontier_mo = frontier_mo, in_memory_data = in_memory_data,
    dset_purge_keys = purge_keys, core_projection_file_path = core_projection_file_path)

    comm.Barrier()
    MPI.Finalize()




    
