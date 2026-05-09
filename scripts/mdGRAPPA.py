import sigpy as sp 
import h5py
from pygrappa import grappa
import numpy as np
import cupy as cp

import psutil
import os
import tracemalloc
import gc

def memory_usage():
    """Track current memory usage"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 ** 2  # MB

def track_memory(msg=""):
    """Helper to print memory usage"""
    print(f"{msg}: {memory_usage():.2f} MB")
    
try: 
    # Check if there is at least one GPU available
    if cp.cuda.runtime.getDeviceCount() > 0:
        device = sp.Device(0)
        print(f'GPU detected: {cp.cuda.runtime.getDeviceProperties(0)["name"].decode()}')
    else:
        device = sp.Device(-1)
except (ImportError, Exception):
    # If cupy isn't installed or fails, fall back to CPU
    device = sp.Device(-1)
xp = device.xp

print(f"Using device: {device} (Backend: {xp.__name__})")





#Volume recon with slice pyGRAPPA
#Volume recon with mdpyGRAPPA
import h5py
from pygrappa import mdgrappa
import numpy as np
import cupy as cp
slice_idx=128
DATA_DIR= "/home/vault/iwbi/iwbi112h/CEST_Data/"
infile_k='CEST_kdat_3D_R65_C32.h5'
infile_r='refs_3D.h5'
Rep=10
img_shape=(65,32,72,180,slice_idx)

# inter_kspace = np.empty(img_shape,dtype=np.complex64)
inter_kspace = []
with device:
    
    with h5py.File(DATA_DIR + infile_k,'r') as f:
        print('keys :',f.keys() )
        # track_memory('ref_temp')
        kdat_temp=f['kdat_01'][0,0,0,0,0,0,0,Rep,...]
        kdat_temp=np.squeeze(kdat_temp)
    print('kdat_temp',np.shape(kdat_temp))
        
    with h5py.File(DATA_DIR + infile_r,'r') as f:
        # print(f.keys())
        ref_temp=f['refs'][:]
    
    print('ref 3D shape:',ref_temp.shape)
        
    track_memory('after loading ')     
    kdat_temp=np.permute_dims(kdat_temp,(2,0,1,3))
    print('kdat_temp',np.shape(kdat_temp))

    for r in range(1): # central # of partitions
        # inter_kspace[r,...,0]=(grappa(kdat_temp[r,...], ref_temp, kernel_size = (5,5), coil_axis=0))
        kdat_temp=(mdgrappa(kdat_temp, ref_temp, kernel_size = (5,5,5), coil_axis=0))
        xp.get_default_memory_pool().free_all_blocks()
        # track_memory(f'r:{r}')
    kdat_temp=sp.ifft(kdat_temp,axes=[-1,-2])   
    kdat_temp=sp.rss(kdat_temp,axes=0)
    # inter_kspace.append(kdat_temp)
    print(np.shape(kdat_temp))
    # plotw(kdat_temp)
    # chunks=(1,kdat_temp.shape[1],kdat_temp.shape[2])
    # if RO_idx==0:
    #     with h5py.File(DATA_DIR+'CEST_recons_65.h5','w') as f:
    #         f.create_dataset(f'CEST_recon_RO_idx_{RO_idx}',data=kdat_temp,chunks=chunks)
    # else:
    #     with h5py.File(DATA_DIR+'CEST_recons_65.h5','a') as f:
    #         f.create_dataset(f'CEST_recon_RO_idx_{RO_idx}',data=kdat_temp,chunks=chunks)
    # del  ref_temp, kdat_temp
        
# print('inter_kspace shape:', np.shape(inter_kspace))
with h5py.File(DATA_DIR+f'CEST_mdGRAPPA_Rep{Rep}.h5','w') as f:
    f.create_dataset(f'CEST_recon_Rep_{Rep}',data=kdat_temp)

# chunks=(1,kdat_temp.shape[1],kdat_temp.shape[2])
# if RO_idx==0:
#     with h5py.File(DATA_DIR+'CEST_GRAPPA_recons_65.h5','w') as f:
#         f.create_dataset(f'CEST_recon_RO_idx_{RO_idx}',data=kdat_temp,chunks=chunks)
# else:
#     with h5py.File(DATA_DIR+'CEST_recons_65.h5','a') as f:
#         f.create_dataset(f'CEST_recon_RO_idx_{RO_idx}',data=kdat_temp,chunks=chunks)
# del  ref_temp, kdat_temp
        
# print('inter_kspace shape:', np.shape(inter_kspace))
