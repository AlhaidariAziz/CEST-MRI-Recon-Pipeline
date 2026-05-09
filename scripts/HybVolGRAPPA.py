import sigpy as sp 
import h5py
from pygrappa import grappa
import numpy as np
import cupy as cp
import time
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

# slice_idx=128
DATA_DIR= "/home/vault/iwbi/iwbi112h/CEST_Data/"
infile_k='CEST_kdat_2D_R65_C32.h5'
RO=224
RO_start=15 #start and end RO positin that cover FOV |expermintal
RO_end=205

Rep_idx=8
kernel_size=(8,8)
# img_shape=(65,32,72,180,slice_idx)
kernel_str = "x".join(map(str, kernel_size))

start_time = time.perf_counter()

with device:
    for RO_idx in range(RO_start,RO_end):
        
        with h5py.File(DATA_DIR + infile_k,'r') as f:
            print('keys :',f.keys() )
            ref_temp=f['ref_2D'][...,RO_idx]
            # # track_memory('ref_temp')
            # kdat_temp01=np.squeeze(f['kdat_01'][...,RO_idx])
            # # track_memory('kdat_temp01')
            # kdat_temp02=np.squeeze(f['kdat_02'][...,RO_idx])
            # # track_memory('kdat_temp02')


            ##include all Repetitions 
            # kdat_temp=np.concatenate((f['kdat_01'][...,RO_idx],   
            #                            f['kdat_02'][...,RO_idx],
            #                            f['kdat_03'][...,RO_idx],
            #                            f['kdat_04'][...,RO_idx]),
            #                             axis =0)

            #for single repietition
            kdat_temp=f['kdat_01'][Rep_idx,...,RO_idx]   
            kdat_temp=kdat_temp[None,...]
                                 
            
            # # track_memory('kdat_temp01 con')
            # del kdat_temp02
            # # track_memory('del temp02')
            # kdat_temp03=np.squeeze(f['kdat_03'][...,RO_idx])
            # # track_memory('temp 03')
            # kdat_temp04=np.squeeze(f['kdat_04'][...,RO_idx])
            # # track_memory('temp 04')
            # kdat_temp02=np.concatenate((kdat_temp03,kdat_temp04), axis =0)
            # # track_memory('temp 02 con')
            # del kdat_temp03, kdat_temp04
            # # track_memory('del temp03 , temp04')
            # kdat_temp=np.concatenate((kdat_temp01,kdat_temp02), axis =0)
            # # track_memory('temp')
            # del kdat_temp01, kdat_temp02
            # track_memory('del temp01, temp02 con')
            print('ref 2D shape:',ref_temp.shape)
            print('kdat shape:',kdat_temp.shape)
            
        track_memory(f'RO {RO_idx} :')     
        kdat_temp=np.permute_dims(kdat_temp,(0,3,1,2))
        print('con kdat_temp shape:',kdat_temp.shape)
        print('ref_temp shape:',ref_temp.shape)
        for r in range(kdat_temp.shape[0]): # central # of partitions
            # inter_kspace[r,...,0]=(grappa(kdat_temp[r,...], ref_temp, kernel_size = (5,5), coil_axis=0))
            kdat_temp[r,...]=(grappa(kdat_temp[r,...], ref_temp, kernel_size = kernel_size, coil_axis=0))
            xp.get_default_memory_pool().free_all_blocks()
            # track_memory(f'r:{r}')
        kdat_temp=sp.ifft(kdat_temp,axes=[-1,-2])   
        kdat_temp=sp.rss(kdat_temp,axes=1)
        # inter_kspace.append(kdat_temp)
        
        chunks=(1,kdat_temp.shape[1],kdat_temp.shape[2])
        if RO_idx==RO_start:
            with h5py.File(DATA_DIR+f'CEST_GRAPPA_recons_Rep_idx_{Rep_idx}_ker_{kernel_str}.h5','w') as f:
                f.create_dataset(f'CEST_recon_RO_idx_{RO_idx}',data=kdat_temp,chunks=chunks)
        else:
            with h5py.File(DATA_DIR+f'CEST_GRAPPA_recons_Rep_idx_{Rep_idx}_ker_{kernel_str}.h5','a') as f:
                f.create_dataset(f'CEST_recon_RO_idx_{RO_idx}',data=kdat_temp,chunks=chunks)
        del  ref_temp, kdat_temp
        
# print('inter_kspace shape:', np.shape(inter_kspace))

end_time = time.perf_counter()
duration = (end_time - start_time)/60

print(f"Reconstruction completed in {duration:.2f} Minutes")
