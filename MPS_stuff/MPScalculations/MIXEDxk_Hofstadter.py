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



def get_initial_state(ktot, N, Lx, Ly):
    array = np.concatenate((np.ones(N), np.zeros(Ly*Lx - N)))
    MATCH = False
    jj = 0
    while not MATCH: 
        # Shuffle the array to randomize the positions of ones and zeros
        np.random.shuffle(array)
        karr = 0
        for ii, entry in enumerate(array):
            karr += entry * (ii)%Ly
        jj += 1
        if karr%Ly == ktot or jj>1e10:
            MATCH = True

    return array

def get_initial_state_2siteUC(ktot, N, Lx, Ly):
    array = np.concatenate((np.ones(N), np.zeros(Ly*Lx - N)))
    # Shuffle the array to randomize the positions of ones and zeros
    MATCH = False
    while not MATCH: 
        np.random.shuffle(array)
        karr = 0
        for ii, entry in enumerate(array):
            karr += entry * (ii//2)%Ly
        if karr%Ly == ktot:
            MATCH = True
            
    return array





def load_dict(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)
    

##################################################################################################
# Checkerboard model in mixed x-k space
##################################################################################################

class MixedHofstadterLandauGauge_Pi_2(MixedXKModel):
    """Example: Haldane model in x-k-Basis.

    Spinless fermions, no extra orbitals (`N_orb` = 1 for up and down), on a triangular lattice,
    nearest-neighbor hopping (`t`) + onsite interactions (`U`)
    """
    def init_lattice(self, model_params):
        N_orb = 1  # for A (l=0) and B (l=1) sublattice
        model_params['N_rings'] = model_params['Lx']
        chinfo = npc.ChargeInfo([1], ["Charge"])
        charges = [[1]] # has shape (N_orb, chinfo.qnumber)
        return MixedXKModel.init_lattice(self, model_params, N_orb, chinfo, charges)

    def init_terms(self, model_params):
        # Read out parameters
        t = model_params.get('t', -1, 'real_or_array')
        t2 = model_params.get('t2', 0, 'real_or_array')
        V1 = model_params.get('V1', 1, 'real_or_array')
        phi_ext = model_params.get('phi_ext', 0., 'real')

        xk_lat = self.lat
        Ly = xk_lat.Ly
        Lx = xk_lat.N_rings
        N_orb = xk_lat.N_orb
        assert N_orb == 1

        # hopping
        intra_hopping = np.zeros((Lx, Ly, N_orb, Ly, N_orb), dtype=complex)
        inter_hopping = np.zeros((Lx, Ly, N_orb, Ly, N_orb), dtype=complex)
        inter_hopping_x2 = np.zeros((Lx, Ly, N_orb, Ly, N_orb), dtype=complex)
        for k in range(Ly):
            for xx in range(Lx):
                phiX = self.phi_x(xx)
                PhiX = self.Phi_x(xx)
                kval = k + phi_ext

                print(phiX, xx)
                #intracell
                intra_hopping[xx, k, 0, k, 0] += t*np.exp(2j*np.pi*kval/Ly)*np.exp(1j*phiX[2])
                intra_hopping[xx, k, 0, k, 0] += np.conj(intra_hopping[xx, k, 0, k, 0])


                #intercell
                inter_hopping[xx, k, 0, k, 0] += t*(np.exp(1j*phiX[0]) + np.exp(1j*phiX[1])*np.exp(2j*np.pi*kval/Ly))
                inter_hopping[xx, k, 0, k, 0] += t2*(np.exp(1j*PhiX[1])*np.exp(4j*np.pi*kval/Ly) + np.exp(1j*PhiX[2])*np.exp(-2j*np.pi*kval/Ly))
                inter_hopping_x2[xx, k, 0, k, 0] += t2*np.exp(1j*PhiX[0])*np.exp(2j*np.pi*kval/Ly)


        self.add_intra_ring_hopping(intra_hopping)
        self.add_inter_ring_hopping(inter_hopping, dx=1)
        self.add_inter_ring_hopping(inter_hopping_x2, dx=2)


       # interaction terms
        intra_int = np.zeros((Ly, N_orb, Ly, N_orb, Ly, N_orb, Ly, N_orb), dtype=complex)
        inter_int = np.zeros((Ly, N_orb, Ly, N_orb, Ly, N_orb, Ly, N_orb), dtype=complex)

        for k1 in range(Ly): 
            for k2 in range(Ly):
                for k3 in range(Ly):
                    k4 = (k1+k3-k2)%Ly
                    intra_int[k1, 0, k2, 0, k3, 0, k4, 0] += V1/Ly * (np.exp(-2j*np.pi*(k3-k4)/Ly))
                    inter_int[k1, 0, k2, 0, k3, 0, k4, 0] += V1/Ly * (1 + np.exp(-2j*np.pi*(k3-k4)/Ly))

        self.add_intra_ring_interaction(intra_int)
        self.add_inter_ring_interaction(inter_int, dx=1)


    def phi_x(self, x):
        xmod = x%2

        if xmod == 0:
            phi = np.array([np.pi/2, np.pi, 0])
        elif xmod == 1:
            phi = np.array([-np.pi/2, np.pi, np.pi])
        
        return phi
    
    def Phi_x(self, x):
        xmod = x%2

        if xmod == 0:
            phi = np.array([np.pi, -np.pi/2, 0])
        elif xmod == 1:
            phi = np.array([0, np.pi/2, 0])

        
        return phi


    
##################################################################################################
# define DMRG run
##################################################################################################



def simulate_Hofst_LG_pi_2(model_params, dmrg_params, psi=None, modl=None):
    print('started to simulate')
    print(model_params)
    Ly =  model_params.get('Ly')
    Lx = model_params.get('Lx')
    ktot = model_params.get('ktot')
    Nsec = model_params.get('Nsec')
    print(Nsec)
    if modl == None:
        modl = MixedHofstadterLandauGauge_Pi_2(model_params); print('created model')
    else:
        print('loaded model')

    state_str = (get_initial_state(ktot, Nsec, Lx, Ly)).astype(int)
    # Example usage


    print('initial state', state_str)
    if psi ==None:
        psi = MPS.from_product_state( modl.lat.mps_sites(), state_str, bc= modl.lat.bc_MPS)  # initial state
    else:
        psi = psi
    assert psi.get_total_charge()[1] == ktot%Ly and psi.get_total_charge()[0] == Nsec
    print('momentum of initial state:', psi.get_total_charge()[1], ', particle number of initial state:',  psi.get_total_charge()[0])

    #print('charges of initial state:', psi.get_total_charge())

    print('starting drmg run')
    # run dmrg on finite cylinder
    info = dmrg.run(psi, modl, dmrg_params)

    # print energies
    print("E=", info['E'])
    print('max bond dimension=', max(psi.chi))




        # save data
    V1 = model_params.get('V1')
    t = model_params.get('t')
    tp = model_params.get('t2')
    chimax = str((dmrg_params['trunc_params'])['chi_max'])
    savepackage = {'psi': psi, 'dmrg_params': dmrg_params, 'model_params': model_params, 'info': info['E'], 'model': modl}
    loc = '/home/t30/all/ge54yin/Documents/MasterThesisFQHinTMDs/MPS_stuff/MPScalculations/'

    save_str = ('data_xk_Hofst_pi_2'+'_chi'+ chimax +'_'
                +f'Lx{Lx:d}_Ly{Ly:d}_' 
                + f'V{V1:.2f}_t{t:.1f}_tp{tp:.2f}_K{ktot:d}_N{Nsec:d}' 
                + '.h5')
    with h5py.File(loc+save_str, 'w') as f:
        hdf5_io.save_to_hdf5(f, savepackage)






def run_simulation_Hofst_LG_pi_2(**kwargs):
    print('started simualtion')
    chi_max= kwargs['chi_max']

    model_params = {
    "bc_MPS": "infinite",
    "bc": ['periodic', 'periodic']}

    for key, value in kwargs.items():
        model_params[key] = value

    dmrg_params = {'trunc_params': {"chi_max": chi_max},'mixer': True,
    'chi_list': {0: chi_max,}
    }
    
    simulate_Hofst_LG_pi_2(model_params,dmrg_params)






def simulate_fluxthread_pi2(model_params, dmrg_params, psi=None, modl=None):
    print('started to simulate')
    print(model_params)
    Ly =  model_params.get('Ly')
    Lx = model_params.get('Lx')
    ktot = model_params.get('ktot')
    Nsec = model_params.get('Nsec')
    print(Nsec)
    if modl == None:
        modl = MixedHofstadterLandauGauge_Pi_2(model_params); print('created model')
    else:
        print('loaded model')

    
    # Example usage


    
    if psi ==None:
        state_str = (get_initial_state(ktot, Nsec, Lx, Ly)).astype(int)
        # Example usage
        #if Lx==3 and Ly==5 and Nsec==5 and ktot==0:
        #    state_str = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0])
        print('initial state', state_str)
        psi = MPS.from_product_state( modl.lat.mps_sites(), state_str, bc= modl.lat.bc_MPS)  # initial state
        dmrg_params['mixer'] = True
    else:
        psi = psi
        print('loaded psi')
        dmrg_params['mixer'] = None
    assert psi.get_total_charge()[1] == ktot%Ly and psi.get_total_charge()[0] == Nsec
    print('momentum of initial state:', psi.get_total_charge()[1], ', particle number of initial state:',  psi.get_total_charge()[0])

    #print('charges of initial state:', psi.get_total_charge())

    print('starting drmg run')
    # run dmrg on finite cylinder
    info = dmrg.run(psi, modl, dmrg_params)

    # print energies
    print("E=", info['E'])
    print('max bond dimension=', max(psi.chi))

    return psi, info


def charge_pumping_pi2(**kwargs):
    print('started simualtion')
    V1 = kwargs['V1']
    Lx, Ly = kwargs['Lx'], kwargs['Ly']
    chi_max= kwargs['chi_max']
    ktot, Nsec = kwargs['ktot'], kwargs['Nsec']
    phi_exts = kwargs['phi_exts']
    saveAll = kwargs.get('saveAll', False)

    model_params = {
        "bc_MPS": "infinite",
        "bc": ['periodic', 'periodic'],
        }
    
    for key, value in kwargs.items():
        model_params[key] = value
    del model_params['phi_exts']

    QLs = []
    infos = []

    print(f'phi_ext = {phi_exts[0]:.2f}')
    model_params['phi_ext'] = phi_exts[0]
    dmrg_params = {'trunc_params': {"chi_max": chi_max},
    'chi_list': {0: chi_max,}, 'max_sweeps': 250,
    }

    psi, info = simulate_fluxthread_pi2(model_params, dmrg_params)
    QL = psi.average_charge(bond=0)
    infos.append(info['E'])
    print(QL)
    QLs.append(QL[0])

    dmrg_params = {
    'trunc_params': {"chi_max": chi_max},
    'mixer': None,
    'mixer_params': {
        'amplitude': 0,
    },
    'max_sweeps': 100,
    }


    # save data
    V1 = model_params.get('V1')
    t = model_params.get('t')
    tp = model_params.get('t2')
    savepackage = {'QLs': QLs, 'dmrg_params': dmrg_params, 'model_params': model_params, 'info': infos}
    loc = '/home/t30/all/ge54yin/Documents/MasterThesisFQHinTMDs/MPS_stuff/MPScalculations/'
    

    save_str = ('data_xk_Hofst_pi2_cpump'+'_chi'+ str(chi_max) +'_'
                +f'Lx{Lx:d}_Ly{Ly:d}_' 
                + f'V{V1:.2f}_t{t:.1f}_tp{tp:.2f}_K{ktot:d}_N{Nsec:d}' 
                + f'flux{min(phi_exts):.1f}-{max(phi_exts):.1f}-{len(phi_exts)}'
                + '.h5')



    for pp, phi_ext in enumerate(phi_exts[1:]):
        model_params['phi_ext'] = phi_ext
        psi, info = simulate_fluxthread_pi2(model_params, dmrg_params, psi=psi)

        if saveAll:
            savepackage = {'psi': psi, 'dmrg_params': dmrg_params, 'model_params': model_params, 'info': infos}
            with h5py.File(loc+'psi_'+save_str, 'w') as f:
                hdf5_io.save_to_hdf5(f, savepackage)

        QL = psi.average_charge(bond=0)
        print(QL)
        QLs.append(QL[0])
        infos.append(info['E'])

        savepackage = {'QLs': QLs, 'dmrg_params': dmrg_params, 'model_params': model_params, 'info': infos}
        with h5py.File(loc+save_str, 'w') as f:
            hdf5_io.save_to_hdf5(f, savepackage)

    QLs = np.array(QLs)
    #return QLs
    savepackage = {'QLs': QLs, 'dmrg_params': dmrg_params, 'model_params': model_params, 'info': infos}

    with h5py.File(loc+save_str, 'w') as f:
        hdf5_io.save_to_hdf5(f, savepackage)





##################################################################################################
# run simulation
##################################################################################################

if __name__ == "__main__":
    kwargs = {}
    kwargs['V1'] = 0.5
    kwargs['D'] = 0
    kwargs['Lx'] = 2
    kwargs['Ly'] = 6
    kwargs['chi_max'] = 256//2
    kwargs['Nsec'] = 6 # fix total particle number
    kwargs['t2'] = -0.25
    for kk in np.array([3]):
        kwargs['ktot'] = kk # fix total momentum sector

        run_simulation_Hofst_LG_pi_2(**kwargs)