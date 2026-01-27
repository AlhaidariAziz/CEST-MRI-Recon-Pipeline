import sigpy as sp
from sigpy.mri import app
import h5py
import torch
import numpy as np
import cupy as cp
import os
import psutil
import tracemalloc
import gc
from pathlib import Path 

slice_idx=128
cwd=Path.cwd().parent

print('Current directory:',cwd)

# DATA_DIR=cwd/'kdat/'
# print('Data directory:',DATA_DIR)
DATA_DIR= "/home/vault/iwbi/iwbi112h/CEST_Data/"

infile_k='CEST_kdat_3D_R65_C32.h5'
infile_ref='refs_3D.h5'


print (File_path := DATA_DIR + infile_k)

# imaging echo (kdat)
f = h5py.File(DATA_DIR + infile_k, 'r')
kdat01 = f['kdat_01'][:,:,:,:,:,:,:,0:2,...]

f.close()

f = h5py.File(DATA_DIR + infile_ref, 'r')
ref = f['refs'][:]
f.close()

print('kdat shape:',kdat01.shape) # kdat shape: (1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 72, 1, 1, 180, 32, 224)
print('Reference shape:',ref.shape) # Reference shape: (32, 36, 48, 128)

#  dimensions reorder ( [Ch] , [Part] , [PE] , [FE] )
kdat01=kdat01.squeeze()
kdat01=np.permute_dims(kdat01,(0,3,1,2,4))

print (' kdat shape [Reps], [Ch] , [PE2] , [PE1] , [FE] :', kdat01.shape)  #kdat shape [Reps], [Ch] , [PE2] , [PE1] , [FE] : (2, 32, 72, 180, 224)
print (' Ref  shape [Ch] , [PE2] , [PE1] , [FE] :',ref.shape) # Ref  shape [Ch] , [PE2] , [PE1] , [FE] : (32, 36, 48, 128)


# # kdat_copy=kdat.copy()

kshape=kdat01.shape[-4::]
print('kshape  [Ch] , [PE2] , [PE1] , [FE] :',kshape) #kshape  [Ch] , [PE2] , [PE1] , [FE] : (32, 72, 180, 224)
del kdat01


#Ref zero filling 

us=True
if us:
    ref_us=ref[...,0::2]
else:
    ref_us=ref
ref_zf = sp.resize(ref_us, kshape)
print('ref shape:', ref.shape)
print('ref_us shape:', ref_us.shape)
print('ref_zf shape:', ref_zf.shape)

# ref shape: (32, 36, 48, 128)
# ref_us shape: (32, 36, 48, 64)
# ref_zf shape: (32, 72, 180, 224)

del ref
del ref_us


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



# from 3D to 2D | IFFT over FE | i.e. decouble Kx slices 

with device:
    ref_2D=sp.ifft(ref_zf,axes=[-1]) #uncomment to apply IFFT
    gc.collect()
   
    if device == sp.Device(0):  # GPU
        torch.cuda.empty_cache()
    del ref_zf

    print('ref_2D shape:', ref_2D.shape) #ref_2D shape: (32, 72, 180, 224)


print(' mps estimation ...')
# #  slice mps for 2D Fourier 3D volume 

cp.get_default_memory_pool().free_all_blocks()

ACS=sp.to_device(ref_2D, device=device)

mps=[]
for kx_idx in range(kshape[-1]):
    mps.append(app.EspiritCalib(ACS[...,kx_idx],
                        crop=0.8,
                        device=device,
                        calib_width=36,
                        show_pbar=False).run())


mps=[x.get() if hasattr(x, 'get') else x for x in mps]
mps=np.permute_dims(mps,(1,2,3,0))


print('mps shape:', np.shape(mps)) #mps shape: (32, 72, 180, 224)

# np.save('mps_s',mps)


chunks=(ref_2D.shape[0:3]+(1,)) 
del ref_2D

with h5py.File(cwd / 'maps/mps.h5','w') as f:
    f.create_dataset('mps',data=mps,chunks=chunks)
    
print('Done')
print('you can find mps.h5 in', cwd/'maps')
    

