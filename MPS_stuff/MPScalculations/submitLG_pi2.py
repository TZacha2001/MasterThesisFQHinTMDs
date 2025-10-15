"""Example how to create a `config` for a job array and submit it using cluster_jobs.py."""

import cluster_jobs # type: ignore
import copy
import numpy as np  # only needed if you use np below

config = {
    'jobname': 'pi_2_mono',
    'task': {
        'type': 'PythonFunctionCall',
        'module': 'mixedxk_Hofstadter',
        'function': 'run_simulation_Hofst_LG_pi_2'
    },
    'task_parameters': [],  # list of dict containing the **kwargs given to the `function`
    'requirements_slurm': {  # passed on to SLURM
        'time': '0-46:02:00',  # d-hh:mm:ss
        'mem': '45G',
        'partition': 'cpu',
        'cpus-per-task': 12,
        'qos': 'normal',
        'nodes': 1,  # number of nodes
    },
    'options': {
        # you can add extra variables for the script_template in cluster_templates/* here
    }
}


Lx = 2
Ly = 9
N = 1 * Lx * Ly // 3
tps = np.array([-0.25])
#Vs = np.arange(0.2, 2, 0.05)
Vs = np.array([0.6, 2])
for V in Vs:
    for t2 in tps:
        #for Ktot in range(Ly):#np.array([0, 1, 2, 3, 4, 5]): 
        for Ktot in np.array([0, 3, 6]):
            kwargs = {}
            kwargs['V1'] = V
            kwargs['t'] = -1
            kwargs['t2'] = t2
            kwargs['Delta'] = 0
            kwargs['Lx'] = Lx
            kwargs['Ly'] = Ly
            kwargs['chi_max'] = 1024*4
            kwargs['Nsec'] = N # fix total particle number
            kwargs['ktot'] = Ktot # fix total momentum sector

            config['task_parameters'].append(copy.deepcopy(kwargs))



# cluster_jobs.TaskArray(**config).run_local(task_ids=[2, 3], parallel=2) # run selected tasks
#cluster_jobs.JobConfig(**config).submit()  # run all tasks locally by creating a bash job script
cluster_jobs.SlurmJob(**config).submit()  # submit to SLURM
# cluster_jobs.SGEJob(**config).submit()  # submit to SGE
