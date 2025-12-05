import torch
import torch_geometric
from torch_geometric.data import Dataset as torchDataset
from torch_geometric.data import Data as torchData

from sklearn.preprocessing import StandardScaler
from mlgf.model.preprocessing import LogAugmenter, batched_update_standardscaler

from mlgf.data import Dataset, Moldatum
from mlgf.model.pytorch.data import unravel_rank2
from mlgf.lib.helpers import sigma_lo_mo, get_sigma_fit, get_sigma_from_ml, get_pade18
from mlgf.model.pytorch.helpers import one_hot_encode_category, predict_wrapper_graph_ensemble, get_graph_ensemble_mean, get_graph_ensemble_uncertainty
from mlgf.workflow.data_integrity import check_integrity

import os
import pandas as pd
import numpy as np
import argparse
import joblib
import time
import warnings
import psutil
import random
import gc
import shutil
import sys

class MoldatumDsetBatcher:
    """controller for batching the feature preparation on MPI procs
    """    

    def __init__(self, fnames, batch_size = 1, seed = 42, dynamical_ftr_freqs = None, core_projection_file_path = None, check_data_integrity = True):            

        if dynamical_ftr_freqs is None:
            self.dynamical_ftr_freqs = 1j*np.array([1e-3, 0.1, 0.2, 0.5, 1.0, 2.0])
        else:
            self.dynamical_ftr_freqs = dynamical_ftr_freqs

        self.check_data_integrity = check_data_integrity

        self.fnames = fnames[:]
        self.seed = seed
        if not self.seed is None:
            np.random.seed(seed)
            self.fnames = np.random.permutation(self.fnames)

        self.file_batches = [self.fnames[i:i+batch_size] for i in range(0, len(self.fnames), batch_size)]
        self.n_batches = len(self.file_batches)
        self.batch_size = batch_size
        self.dset_files = None
        self.dset_store_dir = None
        self.indexer = np.arange(self.n_batches*self.batch_size).reshape(self.n_batches,self.batch_size)
        self.core_projection_file_path = core_projection_file_path # /home/scv22/project/mlgf/examples/silicon_saiao
    def __len__(self):
        return self.n_batches
    
    def __getitem__(self, idx):
        if self.dset_files is None:
            dset = Dataset.from_files(self.file_batches[idx], data_format='chk', load_data = True, core_projection_file_path = self.core_projection_file_path)
            dset = dset.prep_dyn_features(self.dynamical_ftr_freqs, ftr_suffix = '', add_ef = True)
            return dset
        else:
            try:
                dset = joblib.load(self.dset_files[idx])
                return dset
            except FileNotFoundError:
                print(f'Dset file {self.dset_files[idx]} not found! Preparing new dset from file_batches[{idx}]...')
                dset = Dataset.from_files(self.file_batches[idx], data_format='chk', load_data = True, core_projection_file_path = self.core_projection_file_path)
                dset = dset.prep_dyn_features(self.dynamical_ftr_freqs, ftr_suffix = '', add_ef = True)
                return dset
    
    def to_disk(self, dset_store_dir, mpi_rank = 1, mpi_size = 1, dset_purge_keys = []):
        """write dset objects from each MPI proc to disk

        Args:
            dset_store_dir (string): where to write dsets as joblib
            mpi_rank (int, optional):  Defaults to 1.
            mpi_size (int, optional):  Defaults to 1.
            dset_purge_keys (list, optional): features to purge when saving dsets, i.e. large unused matrices. Defaults to [].
        """        

        self.dset_store_dir = dset_store_dir
        self.dset_files = [f'{self.dset_store_dir}/dset_{idx}.joblib' for idx in range(self.n_batches)]
        

        batch_idx_queue = np.arange(self.n_batches)
        batch_idx_queue = batch_idx_queue[batch_idx_queue % mpi_size == mpi_rank]

        if self.check_data_integrity:
            print(f'-----Checking integrity of data on rank {mpi_rank}-----')
            for idx in batch_idx_queue:
                files = self.file_batches[idx]
                for f in files:
                    check_integrity(f)

        for idx in batch_idx_queue:
            dset = Dataset.from_files(self.file_batches[idx], data_format='chk', load_data = True, purge_keys = dset_purge_keys, core_projection_file_path = self.core_projection_file_path)
            dset = dset.prep_dyn_features(self.dynamical_ftr_freqs, ftr_suffix = '', add_ef = True)
            joblib.dump(dset, self.dset_files[idx])
    
    def get_index_global(self, batch_num, file_num):
        return self.indexer[batch_num, file_num]

class GraphDataset(torchDataset):
    """A collection of DFT graph objects

    Args:
        torchDataset (Dataset from torch_geometric): inherits most functionality from torch_geometric
    """    
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, in_memory = False, indices_ = None, rank = None):
        
        self.rank = rank
        super().__init__(root, transform, pre_transform, pre_filter)
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.in_memory = in_memory
        
        if self.in_memory:
            self.data = self.load_in_memory()
        
        else:
            self.data = None
        
        # print('pytorch dset raw_dir: ', self.raw_dir)
        if indices_ is None:
            length = len(os.listdir(f'{self.raw_dir}'))
            self.indices_ = list(range(length))
        else:
            self.indices_ = indices_
        if not rank is None:
            self.processed_dir = f'{root}/processed_{rank}'
        
    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return []

    @property
    def processed_dir(self):
        if not self.rank is None:
            self.processed_dir = f'{self.root}/processed_{self.rank}'
        else:
            self.processed_dir = f'{self.root}/processed'

    @processed_dir.setter
    def processed_dir(self, rank):
        return f'{self.root}/processed_{rank}'
    
    def len(self):
        return len(self.indices_)

    def download(self):
        pass

    def load_in_memory(self):
        return [torch.load(os.path.join(self.raw_dir, f'data_{idx}.pt'))  for idx in range(self.len())]
    
    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.in_memory:
            return self.data[idx]
        else:
            return torch.load(os.path.join(self.raw_dir, 
                                 f'data_{self.indices_[idx]}.pt'))   

class Graph(torchData):
    """A DFT graph object

    Args:
        torchData (Data from torch_geometric): inherits most functionality from torch_geometric
    """    
    def __init__(self, x = None, edge_attr=None, edge_index=None):
        super().__init__(x = x, edge_attr=edge_attr, edge_index=edge_index)
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """

"""
GraphOrchestrator is the central object for training MBGF-Net and predicting with it
a wrapper for predicting the self energy with GNN ensembles
parameters are stored in model_ii_states or model_ij_states as lists of OrderedDictionaries
for GNN ensemble of n networks, the self-energy can be predicted in predict_full_sigma() method by taking the mean of n predictions
for GNN ensemble of n networks, the self-energy uncertainty can be estimated in uncertainty_full_sigma() method by taking a sample standard error of the mean with n-1 degrees of freedom
"""
class GraphOrchestrator:

    def __init__(self, feature_list_ii, feature_list_ij, target, torch_data_root, in_memory_data = True,
        cat_feature_list_ii = [], cat_feature_list_ij = [],
        ncat_ii_list = [], ncat_ij_list = [], 
        transformer_xii = None, scale_y = None, transformer_xij = None,
        exclude_core = False, coulomb_screen_tol = None, coulomb_screen_basis = 'saiao', 
        ensemble_n = 1, frontier_mo = [0, 0], model_alias = 'MultiModal', model_kwargs = {}, loss_kwargs = {}, 
        edge_cutoff = None, edge_cutoff_features = None, core_projection_file_path = None):
        
        self.loss_kwargs = loss_kwargs
        self.torch_data_root = torch_data_root
        
        default_transformer_xii, default_transformer_xij = StandardScaler(), StandardScaler()
        self.edge_cutoff = edge_cutoff
        self.edge_cutoff_features = edge_cutoff_features
        self.core_projection_file_path = core_projection_file_path

        self.feature_list_ii = feature_list_ii
        self.feature_list_ij = feature_list_ij
        self.feat_list_iijj = list(set(['_'.join(f.split('_')[:2]) for f in feature_list_ij if 'iijj' in f]))
        self.model_alias = model_alias

        assert(len(cat_feature_list_ii) == len(ncat_ii_list))
        assert(len(cat_feature_list_ij) == len(ncat_ij_list))

        self.cat_feature_list_ii = cat_feature_list_ii
        self.cat_feature_list_ij = cat_feature_list_ij
        self.ncat_ii_list = ncat_ii_list
        self.ncat_ij_list = ncat_ij_list
        self.ncat_ii = sum(ncat_ii_list)
        self.ncat_ij = sum(ncat_ij_list)
        self.model_kwargs = model_kwargs
        self.model_kwargs['ncat_ii'] = self.ncat_ii
        self.model_kwargs['ncat_ij'] = self.ncat_ij

        model_precision, loss_precision = self.model_kwargs.get('precision', 'double'), self.loss_kwargs.get('precision', 'double')
        if  model_precision != loss_precision:
            warnings.warn(f'Precision of model ({model_precision}) and precision of loss function ({loss_precision}) differ, modifying loss precision to be the model precision')
            self.loss_kwargs['precision'] = model_precision

        self.target = target

        self.exclude_core = exclude_core # only relevant for training purposes, not prediction
        self.coulomb_screen_tol = coulomb_screen_tol
        self.coulomb_screen_basis = coulomb_screen_basis

        if transformer_xii is None: 
            self.transformer_xii = default_transformer_xii
        else:
            self.transformer_xii = transformer_xii

        if transformer_xij is None: 
            self.transformer_xij = default_transformer_xij
        else:
            self.transformer_xij = transformer_xij

        self.dyn_imag_freq_points = []
        self.ensemble_n = ensemble_n
        # how many frontier MOs to use in loss function outside of homo/lumo, a 2 list or tuple like (a, b)
        # where the frontiers are considered as HOMO-a to LUMO+b
        self.frontier_mo = frontier_mo 
        self.is_directed = False
        self.in_memory_data = in_memory_data
        # self.loss_kwargs['energy_scale'] = self.scale_y
        
    def get_ii_cat(self, moldatum, exclude_core = False):
        """get diagonal categorical features

        Args:
            moldatum (Moldatum)
            exclude_core (bool, optional): whether to exclude core orbitals. Defaults to False.

        Returns:
            numpy.int : the binary data
        """        
        for i, ftr in enumerate(self.cat_feature_list_ii):
            if not 'cluster' in ftr:
                xc = moldatum.get_diag_features([ftr], exclude_core = exclude_core)
            else:
                xc = moldatum.get_diag_features(self.transformer_xii.cluster_feature_names, exclude_core = exclude_core)
                xc = self.transformer_xii.categorize(xc)
            
            if i == 0:
                xc_tot = xc
            else:
                xc_tot = np.hstack((xc_tot, xc))

        return xc_tot

    # def get_ij_cat(self, moldatum, exclude_core = False):
    #     xcats = []
    #     for i, ftr in enumerate(self.cat_feature_list_ij):
    #         if ftr == 'feature_sign':
    #             continue
    #         else:
    #             xc = moldatum.get_offdiag_features([ftr], exclude_core = exclude_core)
    #             xcats.append(xc)
    #     if len(xcats) == 0:
    #         return None
    #     return np.hstack(xcats)
    
    def convert_precision(self, tensor):
        if self.precision == 'double':
            return tensor.double()
        if self.precision == 'float':
            return tensor.float()
        if self.precision == 'half':
            return tensor.half()
        
        return tensor.double()
    
    def moldatum_to_graph(self, moldatum, features_ii, features_ij, transformer_ii, transformer_ij, concat_1hot = True, extras = True):
        """Important function for creating the DFT graphs from moldatum objects

        Args:
            moldatum (Moldatum)
            features_ii (list of string): feautres on nodes
            features_ij (list of string): features on edges
            transformer_ii (sklearn-like scaler object): transformer for nodes
            transformer_ij (sklearn-like scaler object): transformer for edges
            concat_1hot (bool, optional): concatenate binary features. Defaults to True.
            extras (bool, optional): add graph-level info as attributes (e.g. C_saiao_mo rotation, nmo, nocc). Defaults to True.

        Returns:
            _type_: _description_
        """        
        norbs = len(moldatum['mo_occ'])
        iu = np.array(np.triu_indices(norbs, 1)) # for the assignment of full_sigma
        
        x = moldatum.get_diag_features(features_ii)
        edges = moldatum.get_offdiag_features(features_ij)
        
        toremove_edges, toremove = np.empty(edges.shape[0], dtype = 'bool'), np.empty(x.shape[0], dtype = 'bool')
        toremove_edges[:], toremove[:] = False, False

        if self.exclude_core:
            toremove = np.array([i in moldatum['inds_core'] for i in range(x.shape[0])])

        if len(self.cat_feature_list_ii) != 0:
            cat_ii = self.get_ii_cat(moldatum)
            toremove = toremove | np.any(cat_ii == -1, axis = 1)
            cat_ii = cat_ii[~toremove,:]
            x = x[~toremove,:]
            # remove edges if either end node is removed
            toremove_edges = toremove_edges | toremove[iu[0,:]] | toremove[iu[1,:]]
        if type(transformer_ii) == float:
            x = x * transformer_ii
        else:
            x = transformer_ii.transform(x)

        if concat_1hot:
            xc_1hot = one_hot_encode_category(cat_ii, self.ncat_ii_list).numpy()
            x = np.hstack((x, xc_1hot))
        
        # for the graph where core nodes may be removed
        edge_index = np.array(np.triu_indices(x.shape[0], 1))
        to_remove_graph_edge = np.empty(edge_index.shape[1], dtype = 'bool')
        to_remove_graph_edge[:] = False
        
        if not self.edge_cutoff is None and not self.edge_cutoff_features is None:
            toremove_edges_new = np.max(np.abs(moldatum.get_offdiag_features(self.edge_cutoff_features)), axis = 1) < self.edge_cutoff
            toremove_edges = toremove_edges | toremove_edges_new

            to_remove_graph_edge_new = np.max(np.abs(moldatum.get_offdiag_features(self.edge_cutoff_features, exclude_core = self.exclude_core)), axis = 1) < self.edge_cutoff
            to_remove_graph_edge = to_remove_graph_edge | to_remove_graph_edge_new

        iu = iu[:,~toremove_edges]    
        edges = edges[~toremove_edges,:]
        edge_index = edge_index[:,~to_remove_graph_edge]
        
        if type(transformer_ij) == float:
            edges = edges * transformer_ij

        else:
            edges = transformer_ij.transform(edges)

        # make graph undirected, don't need this for multimodal where there is no explicit graph messagepassing
        if not self.is_directed:
            edges = np.vstack((edges, edges))
            edge_index = np.hstack((edge_index, np.array([edge_index[1,:], edge_index[0,:]]))) 
            iu = np.hstack((iu, np.array([iu[1,:], iu[0,:]]))) 

        x_tensor = self.convert_precision(torch.from_numpy(x))
        edge_attr_tensor = self.convert_precision(torch.from_numpy(edges))

        graph = Graph(x = x_tensor, edge_index = torch.from_numpy(edge_index), edge_attr = edge_attr_tensor)
        graph.nodes_plus_edges = x_tensor.size(0) + edge_attr_tensor.size(0)
            
        graph.nmo = len(moldatum['mo_energy'])
        graph.mo_energy = torch.from_numpy(moldatum['mo_energy'])
        graph.node_indices_nonzero = torch.from_numpy(np.arange(norbs)[~toremove])
        graph.edge_indices_nonzero = torch.from_numpy(iu).T
        graph.undirected = not self.is_directed
        graph.fname = getattr(moldatum, 'fname', None)

        graph.ef = moldatum['ef']
        try:
            graph.nomega = len(moldatum['omega_fit'])
            graph.iomega = self.convert_precision(torch.from_numpy(moldatum['omega_fit'].imag))
        except KeyError:
            omega, _ = get_pade18()
            omega = graph.ef + 1j*omega
            graph.nomega = len(omega)
            graph.iomega = self.convert_precision(torch.from_numpy(omega.imag))
        
        graph.num_elements_inv = torch.tensor(np.array([1.0/graph.nodes_plus_edges]))
        graph.n_edges = edge_attr_tensor.size(0)
        graph.n_nodes = x_tensor.size(0)

        if extras:
            graph.homo_ind = moldatum['nocc'] - 1
            graph.lumo_ind = moldatum['nocc']
            # graph.frontiers = list(range(graph.homo_ind-self.frontier_mo[0], graph.lumo_ind+self.frontier_mo[1]+1))
            # if self.loss_kwargs.get('frontier_weight', 0) > 0:
            C_lo_mo = torch.from_numpy(moldatum['C_saiao_mo'].copy())
            C_lo_mo = unravel_rank2(C_lo_mo)
            graph.C_lo_mo = self.convert_precision(C_lo_mo)
        return graph

    def load_dset(self, batcher, nstatic_ij, ndynamical_ij):
        """batched fitting of StandardScaling of ii and ij features for large datasets

        Args:
            batcher (MoldatumDsetBatcher): _description_
            nstatic_ij (int): number of ij static features
            ndynamical_ij (int): number of ij dyn features

        Returns:
            self
        """       
       
        self.train_files = batcher.fnames
        self.mol_batcher = batcher
        if batcher.dynamical_ftr_freqs is None:
            batcher.dynamical_ftr_freqs = 1j*np.array([1e-3, 0.1, 0.2, 0.5, 1.0, 2.0])
        self.dynamical_ftr_freqs = batcher.dynamical_ftr_freqs
         
        for i in range(len(self.mol_batcher)):

            dset = self.mol_batcher[i]
            xii = dset.get_diag_features(self.feature_list_ii, exclude_core = self.exclude_core)
            xij = dset.get_offdiag_features(self.feature_list_ij, exclude_core = self.exclude_core)

            if not self.edge_cutoff_features is None and not self.edge_cutoff is None:
                cluster_features = dset.get_offdiag_features(self.edge_cutoff_features, exclude_core = self.exclude_core)
                toremove = np.max(np.abs(cluster_features), axis = 1) < self.edge_cutoff
                xij = xij[~toremove,:]

            if i == 0:
                self.transformer_xii = StandardScaler()
                self.transformer_xij =  LogAugmenter(nstatic_ij, ndynamical_ij)
                # self.transformer_xij =  StandardScaler()

                self.transformer_xii.fit(xii)
                self.transformer_xij.fit(xij)

            else:
                new_transformer_ii = StandardScaler()
                new_transformer_ii.fit(xii)
                self.transformer_xii = batched_update_standardscaler(new_transformer_ii, self.transformer_xii)
                new_transformer_ij = LogAugmenter(nstatic_ij, ndynamical_ij)
                new_transformer_ij.fit(xij)
                self.transformer_xij = LogAugmenter.from_two_augmenters(new_transformer_ij, self.transformer_xij)
        
        if os.path.exists(self.torch_data_root):
            shutil.rmtree(self.torch_data_root)
            
        os.makedirs(f'{self.torch_data_root}/raw')

        self.precision = self.model_kwargs.get('precision', 'double')
        
        k = 0
        for i in range(len(self.mol_batcher)):
            dset = self.mol_batcher[i]
            for j, mol in enumerate(dset):
                gx = self.moldatum_to_graph(mol, self.feature_list_ii, self.feature_list_ij, self.transformer_xii, self.transformer_xij, concat_1hot = True)
                gy = self.moldatum_to_graph(mol, [self.target], [self.target], 1.0, 1.0, concat_1hot = False, extras = False)
                gx.sigma_ii = gy.x
                gx.sigma_ij = gy.edge_attr
                torch.save(gx, f'{self.torch_data_root}/raw/data_{k}.pt')
                k += 1

        self.data = GraphDataset(f'{self.torch_data_root}', in_memory = self.in_memory_data)
        self.data_baseline = []
        return self

    def fit_transformers_mpi(self, batcher, nstatic_ij, ndynamical_ij, rank, size, comm):
        """MPI batched fitting of StandardScaling of ii and ij features for large datasets

        Args:
            batcher (MoldatumDsetBatcher): _description_
            nstatic_ij (int): number of ij static features
            ndynamical_ij (int): number of ij dyn features
            rank (int): for MPI
            size (int): for MPI
            comm (MPI.comm): for MPI

        Returns:
            self
        """        
        self.mol_batcher = batcher
        batch_idx_queue = np.arange(batcher.n_batches)
        batch_idx_queue = batch_idx_queue[batch_idx_queue % size == rank]
        for i, idx in enumerate(batch_idx_queue):

            dset = self.mol_batcher[idx]
            xii = dset.get_diag_features(self.feature_list_ii, exclude_core = self.exclude_core)
            xij = dset.get_offdiag_features(self.feature_list_ij, exclude_core = self.exclude_core)

            if not self.edge_cutoff_features is None and not self.edge_cutoff is None:
                cluster_features = dset.get_offdiag_features(self.edge_cutoff_features, exclude_core = self.exclude_core)
                toremove = np.max(np.abs(cluster_features), axis = 1) < self.edge_cutoff
                xij = xij[~toremove,:]

            if i == 0:
                self.transformer_xii = StandardScaler()
                self.transformer_xij =  LogAugmenter(nstatic_ij, ndynamical_ij)
                # self.transformer_xij =  StandardScaler()

                self.transformer_xii.fit(xii)
                self.transformer_xij.fit(xij)
                
            else:
                new_transformer_ii = StandardScaler()
                new_transformer_ii.fit(xii)
                self.transformer_xii = batched_update_standardscaler(new_transformer_ii, self.transformer_xii)
                new_transformer_ij = LogAugmenter(nstatic_ij, ndynamical_ij)
                new_transformer_ij.fit(xij)
                self.transformer_xij = LogAugmenter.from_two_augmenters(new_transformer_ij, self.transformer_xij)
                
                # new_transformer_ij = StandardScaler()
                # new_transformer_ij.fit(xij)
                # self.transformer_xij = batched_update_standardscaler(new_transformer_ij, self.transformer_xij)
        
        comm.Barrier()
        transformers_xii_gather = comm.gather(self.transformer_xii)
        transformers_xij_gather = comm.gather(self.transformer_xij)
        # print(transformers_xij_gather)
        
        if rank == 0:
            transformers_xii_gather = [t for t in transformers_xii_gather if hasattr(t, 'n_features_in_')]
            for i in range(len(transformers_xii_gather)):
                if i == 0:
                    new_transformer_ii = transformers_xii_gather[i]
                else:
                    new_transformer_ii = batched_update_standardscaler(new_transformer_ii, transformers_xii_gather[i])

            transformers_xij_gather = [t for t in transformers_xij_gather if hasattr(t, 'n_pos')]
            for i in range(len(transformers_xij_gather)):
                if i == 0:
                    new_transformer_ij = transformers_xij_gather[i]
                else:
                    new_transformer_ij = LogAugmenter.from_two_augmenters(new_transformer_ij, transformers_xij_gather[i])   

            transformers_xii_gather = new_transformer_ii
            transformers_xij_gather = new_transformer_ij         

        comm.Barrier()
        transformers_xii_gather = comm.bcast(transformers_xii_gather, root=0)
        transformers_xij_gather = comm.bcast(transformers_xij_gather, root=0)
        comm.Barrier()
        self.transformer_xii = transformers_xii_gather
        self.transformer_xij = transformers_xij_gather
        return self

    def load_torch_dataset_mpi(self, batcher, torch_data_root, rank, size, comm):
        """batched creation of torch GraphDataset and underlying data files

        Args:
            batcher (MoldatumDsetBatcher): _description_
            torch_data_root (str): where to save GraphData objects to disk
            rank (int): for MPI
            size (int): for MPI
            comm (MPI.comm): for MPI

        Returns:
            self
        """  
        self.train_files = batcher.fnames
        self.torch_data_root = torch_data_root
        self.precision = self.model_kwargs.get('precision', 'double')
        self.dynamical_ftr_freqs = batcher.dynamical_ftr_freqs.copy()
        comm.Barrier()

        if rank == 0:
            if not os.path.exists(self.torch_data_root):
                os.makedirs(f'{self.torch_data_root}/raw')
            else:
                warnings.warn(f'The supplied torch_data_root exists, removing existing tree!')
                shutil.rmtree(self.torch_data_root)
                os.makedirs(f'{self.torch_data_root}/raw')
        
        comm.Barrier()

        batch_idx_queue = np.arange(batcher.n_batches)
        batch_idx_queue = batch_idx_queue[batch_idx_queue % size == rank]
        for i in batch_idx_queue:
            dset = batcher[i]
            for j, mol in enumerate(dset):
                idx = batcher.get_index_global(i, j)
                gx = self.moldatum_to_graph(mol, self.feature_list_ii, self.feature_list_ij, self.transformer_xii, self.transformer_xij, concat_1hot = True)
                gy = self.moldatum_to_graph(mol, [self.target], [self.target], 1.0, 1.0, concat_1hot = False, extras = False)
                gx.sigma_ii = gy.x
                gx.sigma_ij = gy.edge_attr
                torch.save(gx, f'{self.torch_data_root}/raw/data_{idx}.pt')
                
        comm.Barrier()
        if rank ==0:
            self.data = GraphDataset(f'{self.torch_data_root}', in_memory = self.in_memory_data)
        return self

    def mount_predict_dset(self, mlf_chkfile):  
        """prepare DFT features for prediction

        Args:
            mlf_chkfile (str): chkfile with DFT calculation
        """        
        file_list = [mlf_chkfile]
        self.pdset = Dataset.from_files(file_list, data_format='chk', core_projection_file_path = getattr(self, 'core_projection_file_path', None))
        self.pdset.prep_dyn_features(self.dynamical_ftr_freqs, ftr_suffix = '' , add_ef = True)    

    def predict_full_sigma(self, mlf_chkfile, remount_mlf_chkfile = True, exclude_core = False): # 
        """self-energy prediction

        Args:
            mlf_chkfile (string): chkfile with DFT calculation to compute self-energy from
            remount_mlf_chkfile (bool, optional): remount chkfile into the pgraph stored in the object. Defaults to True.
        Returns:
            np.complex64: predicted self-energy tensor in SAIAO basis (nmo x nmo x nomega)
        """    
        if remount_mlf_chkfile:
            self.mount_predict_dset(mlf_chkfile) #
        self.pgraph = self.moldatum_to_graph(self.pdset[0], self.feature_list_ii, self.feature_list_ij, self.transformer_xii, self.transformer_xij, concat_1hot = True)
        
        sigmas =  predict_wrapper_graph_ensemble(self.pgraph, self.model_states, self.model_alias, **self.model_kwargs)
        sigma = get_graph_ensemble_mean(sigmas)
        nw = sigma.shape[-1]//2
        return sigma[:, :, :nw] + 1j*sigma[:, :, nw:]

    def uncertainty_full_sigma(self, mlf_chkfile, remount_mlf_chkfile = True):
        """self-energy uncertainty

        Args:
            mlf_chkfile (string): chkfile with DFT calculation to compute self-energy from
            remount_mlf_chkfile (bool, optional): remount chkfile into the pgraph stored in the object. Defaults to True.
        Returns:
            np.complex64: estimated self-energy uncertainty tensor in SAIAO basis (nmo x nmo x nomega)
        """    
        if self.ensemble_n == 1:
            return None
        
        if remount_mlf_chkfile:
            self.mount_predict_dset(mlf_chkfile)

        self.pgraph = self.moldatum_to_graph(self.pdset[0], self.feature_list_ii, self.feature_list_ij, self.transformer_xii, self.transformer_xij, concat_1hot = True)
        
        sigma = predict_wrapper_graph_ensemble(self.pgraph, self.model_states, self.model_alias, **self.model_kwargs)
        sigma = get_graph_ensemble_uncertainty(sigma)
        nw = sigma.shape[-1]//2
        return sigma[:, :, :nw] + 1j*sigma[:, :, nw:]
    
    def uncertainty_training_example(self, index, device = 'cpu'):
        """self-energy uncertainty for the training data (faster since loaded )
        Args:
            mlf_chkfile (string): chkfile with DFT calculation to compute self-energy from
            remount_mlf_chkfile (bool, optional): remount chkfile into the pgraph stored in the object. Defaults to True.
        Returns:
            np.complex64: estimated self-energy uncertainty tensor in SAIAO basis (nmo x nmo x nomega)
        """    
        if self.ensemble_n == 1:
            return None
        graph = self.data[index].to(device)
        sigma = predict_wrapper_graph_ensemble(graph, self.model_states, self.model_alias, **self.model_kwargs, device = device)
        sigma = get_graph_ensemble_uncertainty(sigma)
        nw = sigma.shape[-1]//2
        return sigma[:, :, :nw] + 1j*sigma[:, :, nw:]

    def dump(self, filename):
        """pickile the object

        Args:
            filename (string): joblib file to pickle to
        """        
        
        if hasattr(self, 'model'):
            self.model_state = self.model.state_dict()
            delattr(self, 'model')

        if hasattr(self, 'models'):
            self.model_states = [m.state_dict() for m in self.models]
            delattr(self, 'models')

        joblib.dump(self, filename)
