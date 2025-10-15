import numpy as np
from tenpy.linalg import np_conserved as npc
from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS
import logging
from tenpy.models.mixed_xk import MixedXKModel
logging.basicConfig(level=logging.INFO)
import pickle
import sys
from tenpy.tools.misc import to_array
from tenpy.tools import hdf5_io

import h5py

def load_data_h5(fname):
    with h5py.File(fname, 'r') as f:
        data = hdf5_io.load_from_hdf5(f)
        return data