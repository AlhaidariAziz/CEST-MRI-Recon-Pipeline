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
import time
import argparse 


parser = argparse.ArgumentParser(description='run spiritCalib for 3D data in ksapace or Hybird space')

parser.add_argument('--ref_data',
                    default=None,
                    help='ACS h5 file')

parser.add_argument('--img_data',
                    default=None,
                    help='kdat h5 file')

parser.add_argument('--space',
                    default=None,
                    help='Hyb for Hybird or ksp for kspace')
parser.add_argument('--v',
                    type=bool,
                    default=False,
                    help='Verbose mode')
parser.add_argument('--prew',
                    type=bool,
                    default=False,
                    help='Apply prewhitening flags')

args = parser.parse_args()

if args.space is None:
     raise SystemExit(  "Please specify the --space argument: "
                        "'Hyb' to apply EspiritCalib on 3D Hybird k-space Slice by slice using 2D Kernel, or "
                        "'ksp' to apply EspiritCalib directly on 3D k-space using 3D Kernel." )

start_time = time.perf_counter()

cwd=Path.cwd().parent
DATA_DIR= "/home/vault/iwbi/iwbi112h/CEST_DATA/"
# DATA_DIR=cwd/'kdat/'

print('Current directory:',cwd)
if args.ref_data is None:
    # print('Data directory:',DATA_DIR)

    # infile_k='CEST_kdat_3D_R65_C32.h5' # WM
    infile_k='kdat_3D_R34_C44_1shot.h5' 
    # infile_k='kdat_3D_R34_C44_3shot_us_CAIP_4.h5'
    # infile_ref='refs_3D.h5'

    # infile_k='kdat_2D_kdat_3D_R34_C44_1shot.h5'
    # infile_k='kdat_2D_kdat_3D_R34_C44_3shot.h5'
    # infile_ref='kdat_3D_R34_C44_3shot_us_CAIP_4.h5'
    infile_ref='ACS_3D_C44_1shot.h5'

    # infile_k='kdat_3D_R34_C44_3shot_us_Cart_3.h5'
    # infile_ref='kdat_3D_R34_C44_3shot_us_Cart_3.h5'

else:
    infile_k=args.img_data
    infile_ref=args.ref_data

print (File_path := DATA_DIR + infile_k)


def memory_usage():
    """Track current memory usage"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 ** 2  # MB

def track_memory(msg=""):
    """Helper to print memory usage"""
    print(f"{msg}: {memory_usage():.2f} MB")
    
prew=args.prew

# imaging echo (kdat)
f = h5py.File(DATA_DIR + infile_k, 'r')
kdat01 = f['kdat_01'][0,...]
#loading prewhitening Matrix 
if prew:
    print('prewhitening is True')
    W = f['W'][:]
else:
    print('prewhitening is False')
f.close()

f = h5py.File(DATA_DIR + infile_ref, 'r')
ref = f['refs'][:]
# ref = f['refs'][...,32:96] #experementing with highly truncated lines_ Set ReadOut width ROW below 64
f.close()

print('kdat raw shape:',kdat01.shape) # kdat shape: (1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 72, 1, 1, 180, 32, 224)
print('Reference shape:',ref.shape) # Reference shape: (32, 36, 48, 128)

#  dimensions reorder ( [Ch] , [Part] , [PE] , [FE] )
kdat01=kdat01.squeeze()
# kdat01=np.permute_dims(kdat01,(0,3,1,2,4))

print (' kdat shape [Reps], [Ch] , [PE2] , [PE1] , [FE] :', kdat01.shape)  #kdat shape [Reps], [Ch] , [PE2] , [PE1] , [FE] : (2, 32, 72, 180, 224)
print (' Ref  shape [Ch] , [PE2] , [PE1] , [FE] :',ref.shape) # Ref  shape [Ch] , [PE2] , [PE1] , [FE] : (32, 36, 48, 128)


# # kdat_copy=kdat.copy()

kshape=kdat01.shape[-4::]
print('kshape  [Ch] , [PE2] , [PE1] , [FE] :',kshape) #kshape  [Ch] , [PE2] , [PE1] , [FE] : (32, 72, 180, 224)
del kdat01


#Ref zero filling 

us=False
if us:
    ref_us=ref[...,0::2] # remove over sampling
else:
    ref_us=ref

#apply Prewhitening on ACS
if prew:
    # W shape is (32, 32) -> [New_Channel, Old_Channel]
    # kspace shape is -> ( Cha, PE1, PE2, RO)

    # 'ij' refers to the indices of W (i=New_Channel, j=Old_Channel)
    # 'ajbcd' refers to the indices of kspace:
    # a=Reps, j=Cha (must match W), b=PE1, c=PE2, d=RO

    ref_us = np.einsum('ij,jbcd->ibcd', W, ref_us)

ref_zf = sp.resize(ref_us, kshape)
print('ref shape:', ref.shape)
print('ref_us shape:', ref_us.shape)
print('ref_zf shape:', ref_zf.shape)

# ref shape: (32, 36, 48, 128)
# ref_us shape: (32, 36, 48, 64)
# ref_zf shape: (32, 72, 180, 224)

del ref
del ref_us

print ('Ref Data Type:',ref_zf.dtype)
print ('Ref Data Type:',type(ref_zf))



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


if args.space=='Hyb':
    print ('mode: Hyb')

    # from 3D to 2D | IFFT over FE | i.e. decouble Kx slices 

    with device:
        ref_2D=sp.ifft(ref_zf,axes=[-1]) #apply ifft on RO >> HYBIRD SPACE: (kz,ky,x)
        gc.collect()

        try:
            if device == sp.Device(0):  # GPU
                torch.cuda.empty_cache()
        except (ImportError, Exception):
        # If cupy isn't installed or fails, fall back to CPU
            pass
        del ref_zf

        print('ref_2D shape:', ref_2D.shape) #ref_2D shape: (32, 72, 180, 224)


    print(' mps estimation ...')
    # #  slice mps for 2D Fourier 3D volume 
    try:
        if device == sp.Device(0):  # GPU
            cp.get_default_memory_pool().free_all_blocks()
    except:
        pass

    ref_2D=sp.to_device(ref_2D, device=device)

    c=0.97# Crop threshold for EspiritCalib
    w=36 # ACS region size for EspiritCalib
    kw=6 #kernel_width

    print(' Ecalib Threshold :', c)
    print(' kernel width :', kw)
    print(' calib region width :',w)

    mps=[]
    for kx_idx in range(kshape[-1]):
        mps.append(app.EspiritCalib(ref_2D[...,kx_idx],
                            crop=c,
                            device=device,
                            calib_width=w,
                            kernel_width=kw,
                            show_pbar=args.v).run())

    chunks=(ref_2D.shape[0:3]+(1,)) 
    del ref_2D

    mps=[x.get() if hasattr(x, 'get') else x for x in mps]
    mps=np.permute_dims(mps,(1,2,3,0))


    print('mps shape:', np.shape(mps)) #mps shape: (32, 72, 180, 224)



if args.space=='ksp':
    print ('mode: ksp')
    with device:
        try:
            if device == sp.Device(0):  # GPU
                torch.cuda.empty_cache()
        except (ImportError, Exception):
        # If cupy isn't installed or fails, fall back to CPU
            pass
        

    print('ref_3D shape:', ref_zf.shape) #ref_3D shape: (32, 72, 180, 224)


    print(' mps estimation ...')
    
    try:
        if device == sp.Device(0):  # GPU
            cp.get_default_memory_pool().free_all_blocks()
    except:
        pass

    ref_zf=sp.to_device(ref_zf, device=device)

    c=0# Crop threshold for EspiritCalib
    w=24 # ACS region size for EspiritCalib
    kw=6 #kernel width 
    t=0.05 #eigen values threshold for calibration matrix in Espirit
    print(' Ecalib corp :', c)
    print(' Ecalib Threshold :', t)
    print(' kernel width :', kw)
    print(' calib region width :',w)
    # raise SystemExit
    
    ref_zf=app.EspiritCalib(ref_zf,
                            crop=c,
                            device=device,
                            calib_width=w,
                            show_pbar=args.v,
                            thresh=t,
                            kernel_width=kw).run()


    mps=[ref_zf.get() if hasattr(ref_zf, 'get') else ref_zf]
    mps=np.squeeze(mps)
    chunks=(ref_zf.shape[0], 1, ref_zf.shape[2], 1) 
    del ref_zf
    
    # mps=np.permute_dims(mps,(1,2,3,0))


    print('mps shape:', np.shape(mps)) #mps shape: (32, 72, 180, 224)

    # np.save('mps_s',mps)

# name_str =  "_".join(infile_k.rsplit('.', 1)[0].split('_')[-4:])
name_str =  infile_k.replace('kdat_','').replace('.h5','')

file_name = f'maps/mps_c_{c}_t{t}_w_{w}_kw_{kw}_sp_{args.space}_'+name_str+'_prew.h5' if prew is True else f'maps/mps_c_{c}_t{t}_w_{w}_kw_{kw}_sp_{args.space}_'+name_str+'.h5'
with h5py.File(cwd / file_name,'w') as f:
    f.create_dataset('mps',data=mps,chunks=chunks)
    
print('Done')

end_time = time.perf_counter()
duration = (end_time - start_time)/60
print ('Duration:',duration)
print('you can find mps.h5 in', cwd/'maps')
print(f'saved as:{file_name}')
    

