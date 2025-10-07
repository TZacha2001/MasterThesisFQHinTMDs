import numpy as np
from numba import jit
from numba import prange
from matplotlib import pyplot as pl
import time


#define basis vectors
a1p = np.array([3./2, np.sqrt(3)/2])
a2p = np.array([-1./2, np.sqrt(3)/2])


#reciprocal basis vectors (small)
g1p = (2*np.pi/3) * np.array([3/2, np.sqrt(3)/2])
g2p = (2*np.pi/3) * np.array([-3/2, 3*np.sqrt(3)/2])


#reciprocal basis vectors (big)
h1 = (2*np.pi/3) * np.array([3,np.sqrt(3)])
h2 = (2*np.pi/3) * np.array([-3,np.sqrt(3)])


def Rot(theta):
    rot = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    return rot

# NN interaction
def xi_q(q):
    xi=0
    a0 = np.array([1,0])
    for nn in range(6):
        an = Rot(nn * np.pi/3)@a0
        xi += np.exp(-1j*np.inner(q, an))
    return xi


def create_bz(N0):
    dg1 = g1p/N0
    dg2 = g2p/N0
    bz = np.full((N0**2, 2), None, dtype=float)

    for mm in np.arange(N0):
        for nn in np.arange(N0):
            bz[mm+N0*nn] = dg1 * (mm) + dg2 * (nn)
    return bz


def get_mom_grid_mat(L, G1=g1p, G2=g2p):    
    dg1 = G1/L
    dg2 = G2/L
    momGrid = np.full((L, L, 2), None, dtype=float)

    for l1 in range(L):
        for l2 in range(L):
            momGrid[l1, l2] = dg1*(l1) + dg2*(l2)

    return momGrid
    


# pauli matrices
sigX = np.array([[0,1], [1,0]])
sigY = np.array([[0,-1j], [1j,0]])
sigZ = np.array([[1,0], [0,-1]])

sigs = np.full((2,2,3), None, dtype=complex)
sigs[:,:,0] = sigX
sigs[:,:,1] = sigY
sigs[:,:,2] = sigZ

# free dispersion (gapless for theta=pi/6, gapped otherwise)
def h_k_new(k, theta=0):
    k1 = np.inner(k, a1p)
    k2 = np.inner(k, a2p)
    k12 = np.inner(k, a1p+a2p)
    

    hx = -np.cos(k1+theta)-np.cos(k2-theta)+np.sin(k12-theta)+np.sin(theta)
    hy = np.sin(k1+theta)+np.sin(k2-theta)+np.cos(k12-theta)+np.cos(theta)
    hz = -2* np.cos(k2+theta)

    h_vec = np.array([hx, hy, hz])
    return h_vec, np.sqrt(hx**2+hy**2+hz**2)

def ham_k(k, theta=0):
    h_vec, norms = h_k_new(k, theta)
    return sigX*h_vec[0] + sigY*h_vec[1] + sigZ*h_vec[2]

# determine indices of kvec in momentum grid
@jit(nopython=True, fastmath=True)
def get_index(kvec, L):
    n1 = (kvec[0]*a1p[0] + kvec[1]*a1p[1])/(2*np.pi/L)
    n1 = int(np.round(n1))

    n2 = (kvec[0]*a2p[0] + kvec[1]*a2p[1])/(2*np.pi/L)
    n2 = int(np.round(n2))
    return n1%L, n2%L

#################################################################################
# Function to compute spin spin response (sublattice matrix)
#################################################################################

@jit(nopython=True, fastmath=True)
def get_chi_ab_jit(qs, kGrid, omgs, n_mu, full_evecs, full_bands, P_mat, A_mat, B_mat, eta=5e-3):
    '''
    calculate spin-spin response
    qs: external momenta (not to sum over) with shape (qlen, 2)
    kGrid: momenta to sum over with shape (L, L, 2)
    omgs: energies with shape (wlen)
    n_mu: array with shape (L, L, 2, 2) (last to indices correspond to spin and band index) 
    full_evecs: with shape (L, L, 2, 2) contains eigenvectors
    full_bands: with shape (L, L, 2, 2) contains eigenvalues

    '''
    qlen = len(qs[:, 0])
    wlen = len(omgs)
    L = len(kGrid[:,0,0])

    for kx in prange(L):
        for ky in prange(L):
            for qq in prange(qlen):
                kqx, kqy = get_index(qs[qq]+kGrid[kx, ky], L)
                for ww in prange(wlen):
                    for alph in prange(2): # spin index
                        for bet in prange(2): # spin index
                            for lam1 in prange(2): # band index
                                for lam2 in prange(2): # band index
                                    P_mat[ww, qq, kx, ky, alph, bet, lam1, lam2] = (n_mu[kqx, kqy, alph, lam1]-n_mu[kx, ky, bet, lam2]
                                            )/(omgs[ww] + 1.j*eta + full_bands[kqx, kqy, lam1]-full_bands[kx, ky, lam2]) 

                A_mat[qq, kx, ky] = full_evecs[kqx, kqy, :, :]
            B_mat[kx, ky] = full_evecs[kx, ky, : , :]

def get_chi_ab_APM(A_mat, B_mat, P_mat):
    N_uc = len(A_mat[0,:,0,0,0])
    chi = -1./(2*N_uc**2) * np.einsum('wqklstij, qklai, qklbi, klaj, klbj, stm, tsn->abmnqw', P_mat, np.conj(A_mat), A_mat, B_mat, np.conj(B_mat), sigs, sigs, optimize=True)
    return chi

# sum over sublattices to get total chi (which is gauge invariant)
r_a = np.array([np.array([0,0]), np.array([1./2, np.sqrt(3)/2])])
def get_fullChi(chi_AB, qs):
    q_len = len(qs[:,0])
    expAB = np.full((2,2, q_len), None, dtype=complex)
    for aa in range(2):
        for bb in range(2):
            expAB[aa, bb, :] = np.exp(-1.j*np.inner(qs, r_a[aa]-r_a[bb]))

    return np.einsum('abq, abstqw->qw', expAB, chi_AB)


def run_calculation(N_uc, wmax, wlen, eta=1e-1, U_RPA=0, J_RPA=0, thet=0):
    bz_1 = create_bz(N_uc) # this contains the ''external'' momenta (not to sum over)
    kGrid = get_mom_grid_mat(N_uc, g1p, g2p) # momenta to sum over

    full_bands0 = np.full((N_uc, N_uc, 2), None, dtype=float)
    full_evecs0 = np.full((N_uc, N_uc, 2, 2), None, dtype=complex)
    n_mu = np.full((N_uc, N_uc, 2, 2), 0, dtype=int)

    for nn in np.arange(N_uc):
        for mm in np.arange(N_uc):
            full_bands0[nn, mm, :], full_evecs0[nn, mm, :] = np.linalg.eigh(ham_k(kGrid[nn, mm], theta=thet))
            n_mu[nn, mm, :, 0] = np.array([1,1]) # fill lower band

    wTs = np.linspace(0, wmax, wlen)
    qlen = len(bz_1[:, 0])

    t0 = time.time()
    A_mat = np.full((qlen, N_uc, N_uc, 2, 2), None ,dtype=complex) # (q, kx, ky, a, lam)
    B_mat = np.full((N_uc, N_uc, 2, 2), None ,dtype=complex) # (k, a, lam)
    P_mat = np.full((wlen, qlen, N_uc, N_uc, 2, 2, 2, 2), None, dtype=complex) # (w, q, kx, ky, alpha, beta, lam1 lam2)

    get_chi_ab_jit(bz_1, kGrid, wTs, n_mu, full_evecs0, full_bands0, P_mat, A_mat, B_mat, eta)
    print(time.time()-t0)
    chiTest = get_chi_ab_APM(A_mat, B_mat, P_mat)
    print(time.time()-t0)

    xi = xi_q(bz_1)
    wsQ, xi = np.meshgrid(wTs, xi)
    print(xi.shape)

    chitot = get_fullChi(chiTest, bz_1)
    chi_RPA = chitot/(1 - (U_RPA - J_RPA*xi)*chitot)

    return chi_RPA


if __name__ == "__main__":
    thet = 0 #np.pi/6 # for gapless Dirac spin liquid choose theta=np.pi/6
    uRPA = 0# interaction strength in RPA
    J_RPA = 0.35#0.3 # NN interactions from Heisenberg spin-coupling
    Nuc = 12
    w_steps = 100
    chi_RPA = run_calculation(Nuc, 7.5, w_steps, eta=1e-1, U_RPA=uRPA, J_RPA=J_RPA, thet=thet)
    ks = np.arange(Nuc**2)
    ws = np.linspace(0, 7.5, w_steps)

    ks, ws = np.meshgrid(ks, ws)

    fig, ax = pl.subplots(1,1, figsize=(15, 2))

    ###########
    # define dynamical spin struc factor
    ###########
    Z = np.transpose(np.imag(chi_RPA))
    ######
    # log scale
    #aa=ax.pcolormesh(ks, ws, Z, cmap='Blues', norm=colors.LogNorm(vmin=1e-2, vmax=100), shading='gouraud')

    # linear scale
    pmesh=ax.pcolormesh(ks, ws, Z, cmap='Blues', vmin=None, vmax=10)#, shading='gouraud')
    fig.colorbar(pmesh, ax=ax, orientation='vertical')
    
    ax.set_ylabel('$\\omega/t$')
    ax.set_xlabel(f'momentum index')
    ######
    ax.xaxis.set_tick_params(direction='in')
    ax.yaxis.set_tick_params(direction='in')

    pl.show()

