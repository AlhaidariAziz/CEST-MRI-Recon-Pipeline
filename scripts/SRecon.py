import sigpy as sp
from sigpy.mri import app
import h5py
import torch
import numpy as np
import cupy as cp
import os
import psutil
import os
import tracemalloc
import gc
from pathlib import Path 

slice_idx=128
cwd=Path.cwd().parent

print('Current directory:',cwd)

DATA_DIR=cwd/'kdat/'
print('Data directory:',DATA_DIR)

infile='WM_2DF_00_30_s128.h5'

f = h5py.File(DATA_DIR / infile, 'r')
kdat_temp01 = f['kdat01_00_15_s128'][:]
kdat_temp02 = f['kdat02_15_30_s128'][:]
mps = f['mps'][:]
f.close()


#Join two datasets at the CEST repetition axis
kdat=np.concatenate((kdat_temp01,kdat_temp02), axis =-1).T
del kdat_temp01,kdat_temp02
print('kdat shape[CEST Rep],[coils],[PE2],[PE1]:',kdat.shape)


print('mps shape [coils],[PE2],[PE1],[RO] :',mps.shape)


def memory_usage():
    """Track current memory usage"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 ** 2  # MB

def track_memory(msg=""):
    """Helper to print memory usage"""
    print(f"{msg}: {memory_usage():.2f} MB")



# High Dimentional Recon dimensions setup ( Ntime, Necho, Ncoil, Nz, Ny, Nx)

kdat= kdat[:, np.newaxis,:,np.newaxis,...]

mps_ = mps[:, np.newaxis, ...,slice_idx] # reform original mps to align with hybird slice at RO position (i.e. here 128)
print('kdat shape [Reps,Necho,Ncoils,Nx,PE2,PE1] :', np.shape(kdat))
print('mps_ shape [coils,nx,PE2,PE1]:', np.shape(mps_))

del mps
    
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


# #HDLLR recon
print('LLR...')
kdat=sp.to_device(kdat,device=device)
mps_=sp.to_device(mps_,device=device)



recons_LLR = app.HighDimensionalRecon(kdat,
                                    mps=mps_,
                                    phase_sms=None, #changed from sms_phase_redu to None for testing
                                    combine_echo=False, 
                                    phase_echo=None, #changed from np.conj(shot_phase_redu) to None for testing
                                    # regu='TIK', #changed from LLR to TIK
                                    regu='LLR', #changed from LLR to TIK
                                    blk_shape=( 1 ,6,6 ),
                                    blk_strides=( 1 ,1 , 1),
                                    # solver='ConjugateGradient',
                                    solver='ADMM',
                                    lamda=1e-6,
                                    rho=5e-2,
                                    max_iter=20,      #max iter changed from 15 to 2 for testing
                                    show_pbar=True, verbose=False,
                                    device=device).run()
# recons.append(r.get().squeeze())
# del(r)

    
# track_memory(f"after kx_slice:")

# plotw(np.squeeze(recons_LLR))

gc.collect()
if device == sp.Device(0):  # GPU
    torch.cuda.empty_cache()

recons_LLR=np.squeeze(recons_LLR)

recons_LLR = recons_LLR.get() if hasattr(recons_LLR, 'get') else recons_LLR

os.makedirs(cwd / 'recons/', exist_ok=True)

with h5py.File(cwd /'recons/recons_LLR.h5','w') as f:
    f.create_dataset('recons_LLR',data=recons_LLR)
    
del recons_LLR

print('Done')

##CS Recons

print('CS...')
recons_L1W=[]

for r in range(kdat.shape[0]): #loop over CEST Repititions
    
    #Wlet for rep slices
    recons_L1W.append(sp.mri.app.L1WaveletRecon(kdat[r,0,...], mps_, lamda=1e-8, max_iter=20, device=device, show_pbar=False).run())

recons_L1W = recons_L1W.get() if hasattr(recons_L1W, 'get') else recons_L1W
                      
recons_L1W=np.squeeze(recons_L1W)
# plotw(np.squeeze(recons_L1W))



with h5py.File(cwd / 'recons/recons_L1W.h5','w') as f:
    f.create_dataset('recons_L1W',data=recons_L1W)
    
print('Done')
print('you can find recons in', cwd/'recons')
    

