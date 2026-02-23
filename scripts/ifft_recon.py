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
DATA_DIR= "/home/vault/iwbi/iwbi112h/CEST_DATA/"
infile_k='kdat_3D_R34_C44.h5'
# infile_ref='refs_3D.h5'
# infile_ref='refs_3D.h5'

print (File_path := DATA_DIR + infile_k)

    
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

prew=False #set True to load prewhitening matrix and apply it to kdat and ACS data  
#loading prewhitening Matrix 
if prew:
    print('prewhitening is True')
    with h5py.File(DATA_DIR + infile_k, 'r') as f:
        # print('keys :',f.keys())
        W = f['W'][:]
else:
    print('prewhitening is False')

data_keys = ['kdat_01','kdat_02'] #set this based on your knowledge of kdata keys within the h5 files

# raise SystemExit
for data in data_keys:
    # break
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

        #Pre_Whiten the noise
        if prew:
            # W shape is (32, 32) -> [New_Channel, Old_Channel]
            # kspace shape is -> (Reps,PE1, PE2, Cha, RO)

            # 'ij' refers to the indices of W (i=New_Channel, j=Old_Channel)
            # 'ajbcd' refers to the indices of kspace:
            # a=Reps, j=Cha (must match W), b=PE1, c=PE2, d=RO

            kdat = np.einsum('ij,abcjd->abcid', W, kdat)
            # print(kdat01.shape)
        try:
            if device == sp.Device(0):  # GPU
                xp.get_default_memory_pool().free_all_blocks()
                xp.get_default_pinned_memory_pool().free_all_blocks()
        except:
            pass
        for r in range(kdat.shape[-5]-1,-1,-1):
            track_memory(f"Memory before looping over Reps: ")
            
            for c in range(kdat.shape[-4]-1,-1,-1):
                kdat[r,c,:,:,:]=(sp.ifft(kdat[r,c,:,:],axes=[-1,-2,-3])) #uncomment to apply IFFT
            
            gc.collect()
            
    
    kdat = sp.rss(kdat[:], axes=-4) #root mean square over channels    
    print(f'{data} shape after rms :', kdat.shape)


    chunks = (1, kdat.shape[-3], kdat.shape[-2], 1) # chunking the data at Reps/RO dimensions for faster loading during Recon

    file_name= 'ifft_recon_prew.h5' if prew else 'ifft_recon.h5'

    if data == data_keys[0]:
        with h5py.File(DATA_DIR  + file_name,'w') as f: # to append on existed file
            f.create_dataset(data,data=kdat,chunks=chunks)
            f.flush()
    else:
       with h5py.File(DATA_DIR  + file_name,'a') as f: # to append on existed file
            f.create_dataset(data,data=kdat,chunks=chunks)
            f.flush() 
           
    print('Created/Ammended File: ' + DATA_DIR  + file_name)
    print(f'ifft {data} ... done')
    del kdat

