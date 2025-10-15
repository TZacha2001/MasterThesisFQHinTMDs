import h5py

with h5py.File("data_xk_Hofst_pi_2_chi128_Lx2_Ly6_V0.50_t-1.0_tp-0.25_K3_N8.h5", "r") as f:
    print(list(f.keys()))
    dataset = f["psi"]
    data = dataset
    print(data)