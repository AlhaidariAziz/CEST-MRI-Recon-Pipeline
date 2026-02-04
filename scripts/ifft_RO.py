# from 3D to 2D | IFFT over RO/FE | i.e. decouble Kx slices 


# Notes: 
    # the original k-space data ( [Reps], [Ch] , [PE2] , [PE1] , [FE] : (70, 32, 72, 180, 224) ) was devided into Four sections [kdat01... kdat04] for easier handling and to avoid memory crashes, If your kernel still krashes, then you need to either use memmap method or even further devide and conquore. 
    # Kdat01 is the first section ((0:15, 32, 72, 180, 224) ))
    # Kdat02 is the first section ((15:30, 32, 72, 180, 224) ))
    # Kdat03 is the first section ((30:50, 32, 72, 180, 224) ))
    # Kdat04 is the first section ((50:72, 32, 72, 180, 224) ))

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

#import files
import numpy as np
import sigpy as sp
import h5py

track_memory()

# DATA_DIR= 'c:\\Users\\abalh\\miniconda3\\LLR\\CEST_Data\\Prep_CEST_Data\\'
# DATA_DIR= "/home/hpc/iwbi/iwbi112h/CEST_Data/"
DATA_DIR= "/home/vault/iwbi/iwbi112h/CEST_Data/"
infile_k='CEST_kdat_3D_R65_C32.h5'
# infile_ref='refs_3D.h5'

print (File_path := DATA_DIR + infile_k)

data_keys=['kdat_01','kdat_02','kdat_03','kdat_04']
    
try: 
    import cupy as cp
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


for data in data_keys:
    print(f'ifft {data} ...')

    with h5py.File(DATA_DIR + infile_k, 'r') as f:
        print('keys :',f.keys())
        kdat = np.squeeze(f[data][:])
    # f.close()
    
    # f = h5py.File(DATA_DIR + infile_ref, 'r')
    # ref = f['refs'][:]
    # f.close()
    
    
    print(f'{data} shape:',kdat.shape)
    # print('Reference shape:',ref.shape)
    
    with device:
        # kdat01_2D=np.empty(kdat01.shape,dtype=np.complex64)
        gc.collect()
        if device == sp.Device(0):  # GPU
            xp.get_default_memory_pool().free_all_blocks()
            xp.get_default_pinned_memory_pool().free_all_blocks()
    
        for r in range(kdat.shape[0]-1,-1,-1):
            track_memory(f"Memory before looping over Reps: ")
            
            for c in range(kdat.shape[1]-1,-1,-1):
                kdat[r,c,:,:,:]=(sp.ifft(kdat[r,c,:,:,:],axes=[-1])) #uncomment to apply IFFT
            
            gc.collect()
            
    
    print(f'{data} 2D shape:', kdat.shape)
    print('Created/Ammended File: ' + DATA_DIR  + 'CEST_kdat_2D_R65_C32.h5')

    chunks = (1, kdat.shape[1], kdat.shape[2], kdat.shape[3], 1) # chunking the data based on RO dimension for lighter loading during Recon
    
    if data == data_keys[0]:
        with h5py.File(DATA_DIR  + 'CEST_kdat_2D_R65_C32.h5','w') as f: # to append on existed file
            f.create_dataset(data,data=kdat,chunks=chunks)
            f.flush()
    else:
       with h5py.File(DATA_DIR  + 'CEST_kdat_2D_R65_C32.h5','a') as f: # to append on existed file
            f.create_dataset(data,data=kdat,chunks=chunks)
            f.flush() 
           
    print(f'ifft {data} ... done')
    del kdat