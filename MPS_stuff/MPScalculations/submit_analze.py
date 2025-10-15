"""Example how to create a `config` for a job array and submit it using cluster_jobs.py."""

#import cluster_jobs # type: ignore
import copy
import numpy as np  # only needed if you use np below
import ANALYZE_MONO
import MIXEDxk_Hofstadter
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

config = {
    'jobname': 'an_HF',
    'task': {
        'type': 'PythonFunctionCall',
        'module': 'analyze_mono',
        'function': 'run_analyze'
    },
    'task_parameters': [],  # list of dict containing the **kwargs given to the `function`
    'requirements_slurm': {  # passed on to SLURM
        'time': '0-00:35:00',  # d-hh:mm:ss
        'mem': '4800M',
        'partition': 'cpu',
        'cpus-per-task': 4,
        'qos': 'normal',
        'nodes': 1,  # number of nodes
    },
    'options': {
        # you can add extra variables for the script_template in cluster_templates/* here
    }
}



loc = '/home/t30/all/ge54yin/Documents/MasterThesisFQHinTMDs/MPS_stuff/MPScalculations/'
locLoad = '/home/t30/all/ge54yin/Documents/MasterThesisFQHinTMDs/MPS_stuff/MPScalculations/'
#locLoad = '/tuph/t30/bigspace/ge92ted/dopedFQH/DMRGHofstadtBL/old_8/'
t2 = -0.25
chi = 256//2
Lx, Ly = 2, 6
#N = Lx*Ly - 2
N_list = [2, 4, 6]
#N = 2*Lx*Ly//6 + 1

#phis = np.linspace(0, 1, 21)
phis = np.array([0])

#Ks = np.arange(Ly)
Ks  = np.array([3])
#Vins = np.array([4, 7, 8])
#Vins = np.linspace(4, 10, 13)sca
#Vs = np.arange(0.2, 0.8, 0.2)
V = 0.5000
for N in N_list:    
    for phi in phis:
    #for kk in np.array([0, 4]):#range(6):
        for kk in Ks:
            fname = f'data_xk_Hofst_pi_2_chi{chi:d}_Lx{Lx:d}_Ly{Ly:d}_V{V:.2f}_t-1.0_tp{t2:.2f}_K{kk:d}_N{N:d}.h5'
            params = {'loc': loc, 'locLoad': locLoad, 'fname':fname, 'target': 2}
            kwargs = {'params': params, 'OnlyEntspec': False}
            config['task_parameters'].append(copy.deepcopy(kwargs))
            ANALYZE_MONO.run_analyze(kwargs)


# cluster_jobs.TaskArray(**config).run_local(task_ids=[2, 3], parallel=2) # run selected tasks
#cluster_jobs.JobConfig(**config).submit()  # run all tasks locally by creating a bash job script
#cluster_jobs.SlurmJob(**config).submit()  # submit to SLURM
# cluster_jobs.SGEJob(**config).submit()  # submit to SGE
