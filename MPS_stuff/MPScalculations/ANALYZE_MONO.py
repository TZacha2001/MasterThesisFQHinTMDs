import numpy as np
import h5py
from tenpy.tools import hdf5_io
from tenpy.tools import hdf5_io

import h5py

# Function to darken a color

def load_data_h5(fname):
    with h5py.File(fname, 'r') as f:
        data = hdf5_io.load_from_hdf5(f)
        return data
    


def extract(data, target=1):
    psi = data['psi']
    model_params = data['model_params']
    Lx, Ly = model_params['Lx'], model_params['Ly']
    info = data['info']

    corr_length = []
    for kk in range(Ly):
        clen2 = psi.correlation_length(charge_sector=[2, kk], target=target)
        clen1 = psi.correlation_length(charge_sector=[1, kk], target=target)
        clen0 = psi.correlation_length(charge_sector=[0, kk], target=target)
        #clen0_sm2 = psi.correlation_length(charge_sector=[0, -2, kk], target=target)
        corr_length.append([clen0, clen1, clen2])

    
    # Other quantities
    energy = info
    maxS = np.max(np.array(psi.entanglement_entropy()))
    ent_spect = psi.entanglement_spectrum(by_charge=True)[0]
    # print(ent_spect)
    # Return a dictionary with all relevant information
    return {

        'energy': energy,
        'maxS': maxS,
        'corr_length': corr_length,
        'ent_spect': ent_spect  # This might be complex; adapt as needed
    } 



def save_extr(loc, filname, locLoad, target=1):

    data_dict = load_data_h5(locLoad+filname)
    model_params = data_dict['model_params']
    Lx, Ly = data_dict['model_params']['Lx'], data_dict['model_params']['Ly']
    extr = extract(data_dict, target=target)

    print('\n')
    print('energy:', extr['energy'])
    print(extr['corr_length'])



            # save data
    Lx, Ly = model_params['Lx'], model_params['Ly']
    V1 = model_params.get('V1')
    t = model_params.get('t')
    D = model_params.get('D', 0)
    Nsec, ktot = model_params['Nsec'], model_params['ktot']


    tp = model_params['t2']
    save_str = ('extract_xk_Hofst'+'_chi'+ str( data_dict['dmrg_params']['trunc_params']['chi_max'])
                +'_'+f'Lx{Lx:d}_Ly{Ly:d}_'
                + f'V{V1:.2f}_t{t:.2f}_D{D:.2f}_tp{tp:.2f}'
                +f'_N{Nsec:d}_K{ktot:d}' + '.h5')
    with h5py.File(loc+save_str, 'w') as f:
        hdf5_io.save_to_hdf5(f, extr)


# WRONG
def save_entspec(loc, filname, locLoad):
    data_dict = load_data_h5(locLoad+filname)
    print('loaded data')
    model_params = data_dict['model_params']
    psi = data_dict['psi']
    Lx, Ly = data_dict['model_params']['Lx'], data_dict['model_params']['Ly']
    ent_spect = psi.entanglement_spectrum(by_charge=True)[0]

    # save data
    Vin = model_params.get('Vin')
    Vout = model_params.get('Vout')
    U = model_params.get('U')
    t = model_params.get('t')
    tp = model_params.get('t2')
    fluxes = model_params.get('phi_ext', (0, 0))
    chimax = str((data_dict['dmrg_params']['trunc_params'])['chi_max'])
    Nsec, ktot, Ssec = model_params['Nsec'], model_params['ktot'], model_params['Ssec']

    energy = data_dict['info']

    save_str = ('entspec_xk_Hofst_BL_pi_2'+'_chi'+ chimax +'_'
                + f'Lx{Lx:d}_Ly{Ly:d}_' 
                + f'V{Vin:.2f}_Vo{Vout:.2f}_U{U:.2f}_t{t:.1f}_tp{tp:.2f}'
                + f'_K{ktot:d}_N{Nsec:d}_S{Ssec:d}' 
                + f'fluxes{fluxes[0]:.2f}_{fluxes[1]:.2f}'
                + '.h5')
    
    savepackage = {'E': energy, 'entspec': ent_spect}

    with h5py.File(loc+save_str, 'w') as f:
        hdf5_io.save_to_hdf5(f, savepackage)


def run_analyze(kwargs):
    params = kwargs.get('params')
    OnlyEntspec = kwargs.get('OnlyEntspec', False)

    loc = params.get('loc')
    locLoad = params.get('locLoad')
    fname = params.get('fname')
    target = params.get('target', 1)
    print('loc = ', loc)
    #try:
    if OnlyEntspec:
        save_entspec(loc, fname, locLoad)
    else:
        save_extr(loc, fname, locLoad, target=target)
    #except:
        #print('failed for file = ', fname)


