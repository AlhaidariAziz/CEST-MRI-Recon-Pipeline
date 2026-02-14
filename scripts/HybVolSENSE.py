import sigpy as sp 
import h5py
import numpy as np
import cupy as cp
import time
import psutil
import os
import tracemalloc
import gc
from pathlib import Path
from sigpy.mri import app
import argparse

cwd=Path.cwd().parent

DATA_DIR= "/home/vault/iwbi/iwbi112h/CEST_Data/"
infile_k='CEST_kdat_2D_R65_C32.h5'
# mps_file=cwd/'maps/mps_c_0.97.h5'
mps_file=cwd/'maps/mps_c_0.97.h5'
RO=224 #number of readout positions at each we have a hyber slice
RO_start=15 #start and end RO positin that cover FOV || 80
RO_end=205 #||160
#To do: add RO in h5 file dataset

print('Current directory:',cwd)
print('mps file:',mps_file)

parser = argparse.ArgumentParser(description='run SENSE reconstruction.')

parser.add_argument('--data',
                    default=DATA_DIR+infile_k,
                    help='raw dat file')

parser.add_argument('--r', type=float, default=1e-2,
                    help=' LLR regularization constant    [default: 1e-8]')

parser.add_argument('--i', type=int , default=20,
                    help=' Max iterations    [default: 20]')


start_time = time.perf_counter()

args = parser.parse_args()

print('>  lamda(r): ', args.r)
print('>  Max iteration (i): ', args.i)


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

# mps_=np.load('/home/hpc/iwbi/iwbi112h/LLR codebooks/mps_s.npy', allow_pickle=True)

# img_shape=(65,32,72,180,slice_idx)

# inter_kspace = np.empty(img_shape,dtype=np.complex64)
# recons_LLR = []
Rep=8
with device:
    for RO_idx in range(RO_start,RO_end):
    # for RO_idx in range(RO):
        
        with h5py.File(DATA_DIR + infile_k,'r') as f:
           
            # kdat_temp=np.concatenate((f['kdat_01'][...,RO_idx],  # all 65 Cest Reps
            #                 f['kdat_02'][...,RO_idx],
            #                 f['kdat_03'][...,RO_idx],
            #                 f['kdat_04'][...,RO_idx]),
            #                 axis =0)

            # kdat_temp=np.concatenate((f['kdat_01'][...,RO_idx],  # first 30 Cest Reps
            #                 f['kdat_02'][...,RO_idx]),                
            #                 axis =0)
            
            # kdat_temp=f['kdat_01'][...,RO_idx]  # first 15 Cest Reps
            kdat_temp=f['kdat_01'][Rep,...,RO_idx][None,...]  # single Cest Rep
            # print('con kdat_temp shape before permution:',kdat_temp.shape)
            
        # track_memory(f'RO {RO_idx} :')   
        print('RO:',RO_idx)
        kdat_temp=np.permute_dims(kdat_temp,(0,3,1,2))
        
        with h5py.File(mps_file,'r') as f:
            mps=f['mps'][...,RO_idx]
        # mps=mps_[...,RO_idx]
                
        # print('con kdat_temp shape:',kdat_temp.shape)
        # print('mps shape:',mps.shape)
        
            
        for r in range(kdat_temp.shape[0]): #loop over CEST Repititions
            
            #Wlet for rep slices
            # The reconstions value are directly replaced by kdat_temp at 2nd dim Ch=0
            kdat_temp[r,0,...]=sp.mri.app.SenseRecon(kdat_temp[r,...], mps, lamda=args.r, max_iter=args.i, device=device, show_pbar=False).run().get()
        # kdat_temp=np.squeeze(np.delete(kdat_temp,slice(1,None),axis=1)) #just deleteting c>0
        
        try:
            xp.get_default_memory_pool().free_all_blocks()
        except AttributeError:
            # We are likely using NumPy/CPU, so no pool to clear
            pass
        
        # print('recon kdat_temp :',kdat_temp.shape)
        recon = kdat_temp[:,0,...].get() if hasattr(kdat_temp, 'get') else kdat_temp[:,0,...]
        # print('recon :',recon.shape)
        del kdat_temp, mps
      
        recon = np.squeeze(recon)
        chunks=((1,)+ recon.shape[1::]) #(1,72,180)

        if RO_idx==RO_start:
            with h5py.File(DATA_DIR+f'CEST_SENSE_recons_{Rep}_r_{args.r}_i_{args.i}_mapc_{str(mps_file).split("_")[-1].rsplit(".",1)[0]}.h5','w') as f:
                f.create_dataset(f'CEST_recon_RO_idx_{RO_idx}',data=recon,chunks=chunks)
        else:
            with h5py.File(DATA_DIR+f'CEST_SENSE_recons_{Rep}_r_{args.r}_i_{args.i}_mapc_{str(mps_file).split("_")[-1].rsplit(".",1)[0]}.h5','a') as f:
                f.create_dataset(f'CEST_recon_RO_idx_{RO_idx}',data=recon,chunks=chunks)
        del recon

end_time = time.perf_counter()
duration = (end_time - start_time)/60

print(f"Reconstruction completed in {duration:.2f} Minutes")