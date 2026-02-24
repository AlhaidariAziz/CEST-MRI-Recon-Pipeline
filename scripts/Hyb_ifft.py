# from 3D to 2D | IFFT over RO/FE or Partitions/PE2 | i.e. decouble Kx or Kz slices 


# Notes: 
    # the original k-space data ( [Reps], [Ch] , [PE2] , [PE1] , [FE]  ) are devided into several sections [kdat01, kdat02, ...] for easier handling and to avoid memory crashes, If your kernel still krashes, then you need to either use memmap method or even further devide and conquore. 


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
import argparse
track_memory()

parser=argparse.ArgumentParser(description='Ifft over Readout or PE2 dimension ')
parser.add_argument('--dim', type=int, required=True, choices=[-1,-3],help='flag for ifft dim, enter either -1 for RO or -3 for PE2')

args=parser.parse_args()

# DATA_DIR= 'c:\\Users\\abalh\\miniconda3\\LLR\\CEST_Data\\Prep_CEST_Data\\'
# DATA_DIR= "/home/hpc/iwbi/iwbi112h/CEST_Data/"
DATA_DIR= "/home/vault/iwbi/iwbi112h/CEST_DATA/" 
# infile_k='CEST_kdat_3D_R65_C32.h5' #WM
# infile_ref='refs_3D.h5'#WM
infile_k='kdat_3D_R34_C44_1shot.h5' #WM
infile_ref='ACS_3D_C44_1shot.h5'#WM
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

with h5py.File(DATA_DIR + infile_k, 'r') as f:
    # print(f.keys())
    data_keys=list(f.keys())
    f.close()

print('data keys',data_keys)
# data_keys=['kdat_01','kdat_02','kdat_03','kdat_04']

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
            # kspace shape is -> (Reps, Cha, PE2, PE1, RO)

            # 'ij' refers to the indices of W (i=New_Channel, j=Old_Channel)
            # 'ajbcd' refers to the indices of kspace:
            # a=Reps, j=Cha (must match W), b=PE1, c=PE2, d=RO

            kdat = np.einsum('ij,ajbcd->aibcd', W, kdat)
            # print(kdat01.shape)
        try:
            if device == sp.Device(0):  # GPU
                xp.get_default_memory_pool().free_all_blocks()
                xp.get_default_pinned_memory_pool().free_all_blocks()
        except:
            pass
        for r in range(kdat.shape[-5]-1,-1,-1):
            track_memory(f"Memory before looping over Reps: ")
            
            for c in range(kdat.shape[-2]-1,-1,-1):
                kdat[r,:,:,c,:]=(sp.ifft(kdat[r,:,:,c,:],axes=[args.dim])) #uncomment to apply IFFT
            
            gc.collect()
            
    
    print(f'{data} 2D shape:', kdat.shape)

    chunks = (1, kdat.shape[1], 1, kdat.shape[3], 1) # chunking the data based on RO dimension for lighter loading during Recon
    # kshape = kdat.shape [-4::] 
    file_name= 'kdat_2D_R34_C44_1shot_prew.h5' if prew else 'kdat_2D_R34_C44_1shot.h5'

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

# #Ifft RO for Ref data
# print('ref_3D to ref_2D.....')
# # kshape=(32, 72, 180, 224)
# # chunks=(1,32, 72, 180, 1)
# f = h5py.File(DATA_DIR + infile_ref, 'r')
# ref = f['refs'][:]
# # ref = f['refs'][...,32:96] #experementing with highly truncated lines_ Set ReadOut width ROW below 64
# f.close()

# print (' Ref  shape [Ch] , [PE2] , [PE1] , [FE] :',ref.shape) # Ref  shape [Ch] , [PE2] , [PE1] , [FE] : (32, 36, 48, 128)

# print('kshape  [Ch] , [PE2] , [PE1] , [FE] :',kshape) #kshape  [Ch] , [PE2] , [PE1] , [FE] : (32, 72, 180, 224)

# #Ref zero filling 

# us=False
# if us:
#     ref_us=ref[...,0::2] # remove over sampling
# else:
#     ref_us=ref

# # print('ref_us shape:',ref_us.shape)
# # raise SystemExit
# #Pre_Whiten the noise

# if prew:
#     # W shape is (32, 32) -> [New_Channel, Old_Channel]
#     # kspace shape is -> ( Cha, PE1, PE2, RO)

#     # 'ij' refers to the indices of W (i=New_Channel, j=Old_Channel)
#     # 'ajbcd' refers to the indices of kspace:
#     # a=Reps, j=Cha (must match W), b=PE1, c=PE2, d=RO

#     ref_us = np.einsum('ij,jbcd->ibcd', W, ref_us)
# ref_zf = sp.resize(ref_us, kshape)
# print('ref shape:', ref.shape)
# print('ref_us shape:', ref_us.shape)
# print('ref_zf shape:', ref_zf.shape)

# # ref shape: (32, 36, 48, 128)
# # ref_us shape: (32, 36, 48, 64)
# # ref_zf shape: (32, 72, 180, 224)

# del ref
# del ref_us

# file_name= 'CEST_kdat_2D_R65_C32_prew.h5' if prew else 'CEST_kdat_2D_R65_C32.h5'

# with device:
#     gc.collect()
#     ref_2D=sp.ifft(ref_zf,axes=[-1]) #apply ifft on RO >> HYBIRD SPACE: (kz,ky,x)

#     with h5py.File(DATA_DIR  + file_name ,'a') as f: # to append on existed file
#         if 'ref_2D' in f.keys():
#             del f['ref_2D']   
#         f.create_dataset('ref_2D',data=ref_2D,chunks=chunks[-4::])
            
# print('Created/Ammended ref_2D File: ' + file_name)
           