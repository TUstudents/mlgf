import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


from torch.func import stack_module_state
from torch.func import functional_call
from torch import vmap

# torch.set_default_dtype(torch.float64)

import os
import numpy as np
import argparse
import joblib
import time
import warnings
import psutil
import json
import copy
import random
import gc

def construct_symmetric_tensor(tensor1, tensor2, graph):
    """Reconstruct the full rank3 self-energy tensor from node self-energy and edge_self-energy

    Args:
        tensor1 (torch.float64): the node self-energy
        tensor2 (torch.float64): the edge self-energy
        graph (gnn_orchestrator.GraphData object): the graph object that stores the indices of nodes and edges that have not been removed from the DFT graph

    Returns:
        torch.float64: cat[sigma(iw).real, sigma(iw).imag]
    """    
    edge_indices, node_indices = graph.edge_indices_nonzero, graph.node_indices_nonzero
    # Initialize the rank-3 tensor with zeros
    nmo = graph.nmo
    nw = tensor1.size(-1)
    result_tensor = torch.zeros(nmo, nmo, nw, dtype=tensor1.dtype, device=tensor1.device)
    
    # Fill the diagonal with tensor1 entries
    result_tensor[node_indices, node_indices, :] = tensor1

    # Fill the upper and lower triangle with tensor2 entries (graph is undirected so we don't have to flip stuff, though it could save computation)
    result_tensor[edge_indices[:,0], edge_indices[:,1], :] = tensor2

    return result_tensor

def unravel_rank2(arr):
    """unravels a matrix for data processing in GraphDataset (e.g. an orbital rotation matrix)

    Args:
        arr (torch.float64): 2d matrix, Norb x Norb

    Returns:
        torch.float64: 1d vector, (Norb * Norb) x 1
    """    
    # Unravel the N x N array into an N^2 x 1 array
    unraveled_array = arr.flatten()

    return unraveled_array

def reconstruct_rank2(unraveled_array, nmo):
    """inverts unravel_rank2, i.e. get back 2d matrix from 1d

    Args:
        unraveled_array (torch.float64): 1d
        nmo (int): number of nmo

    Returns:
        torch.float64: 2d, Norb x Norb
    """    
    # Reconstruct the N^2 x 1 array into an N x N array
    reconstructed_array = unraveled_array.view(nmo, nmo)

    return reconstructed_array