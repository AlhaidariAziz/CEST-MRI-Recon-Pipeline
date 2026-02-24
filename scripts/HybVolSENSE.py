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
DATA_DIR= "/home/vault/iwbi/iwbi112h/CEST_DATA/"


parser = argparse.ArgumentParser(description='run SENSE reconstruction.')

# parser.add_argument('--data',
#                     default=DATA_DIR,
#                     help='raw dat file')

parser.add_argument('--r', type=float, default=1e-2,
                    help=' LLR regularization constant    [default: 1e-2]')

parser.add_argument('--Hybdim', required=True, choices=['RO','PE2'],help='select hyb decoubled slices dim')

parser.add_argument('--i', type=int , default=20,
                    help=' Max iterations    [default: 20]')

parser.add_argument('--prew', type=bool , default=False,
                    help=' prewhitening')


start_time = time.perf_counter()

args = parser.parse_args()

print('>  lamda(r): ', args.r)
print('>  Max iteration (i): ', args.i)
print('>  prewhitened: ', args.prew)



prew=args.prew
if prew:
    infile_k='kdat_2D_R34_C44_1shot_prew.h5'
    mps_file=cwd/'maps/mps_c_0.8_t0.02_w_24_kw_6_sp_ksp_prew.h5'
else:    
    infile_k='kdat_2D_kdat_3D_R34_C44_1shot.h5'
    mps_file=cwd/'maps/mps_c_0.8_t0.02_w_24_kw_6_sp_ksp_3D_R34_C44_1shot.h5'
    # infile_k='kdat_2D_R34_C44_3shot_us_Cart_3.h5'
    # mps_file=cwd/'maps/mps_c_0.8_t0.02_w_24_kw_6_sp_ksp.h5'


# mps_file=cwd/'maps/mps_c_0.97.h5'
# mps_file=cwd/'maps/mps_c_0.97.h5'
#To do: add RO in h5 file dataset

print('Current directory:',cwd)
print('mps file:',mps_file)

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

Part_start=0
Parts=24
Reps=34
Rep=16

if args.Hybdim=='PE2':

    with device:
        for Part_idx in range(Parts):
        # for RO_idx in range(RO):
            
            with h5py.File(DATA_DIR+infile_k,'r') as f:
            
                # kdat_temp=np.concatenate((f['kdat_01'][...,RO_idx],  # all 65 Cest Reps
                #                 f['kdat_02'][...,RO_idx],
                #                 f['kdat_03'][...,RO_idx],
                #                 f['kdat_04'][...,RO_idx]),
                #                 axis =0)

                # kdat_temp=np.concatenate((f['kdat_01'][...,RO_idx],  # first 30 Cest Reps
                #                 f['kdat_02'][...,RO_idx]),                
                #                 axis =0)
                
                # kdat_temp=f['kdat_01'][...,RO_idx]  # first 15 Cest Reps
                kdat_temp=f['kdat_01'][Rep,:,Part_idx,...][None,...]   # axial slices #[1,coils,PE1,RO] @ certain kz/PE2/partition
                # print('con kdat_temp shape before permution:',kdat_temp.shape)
                
            # track_memory(f'RO {RO_idx} :')   
            print('Part:',Part_idx)
            
            with h5py.File(mps_file,'r') as f:
                mps=f['mps'][:,Part_idx,...] #[coils,PE1,RO] @  certain kz/PE2/partition Part_idx 
                    
            # print('con kdat_temp shape:',kdat_temp.shape)
            # print('mps shape:',mps.shape)
            
                
            for r in range(kdat_temp.shape[0]): #loop over CEST Repititions
                
                #Wlet for rep slices
                # The reconstions value are directly replaced by kdat_temp at 2nd dim Ch=0
                print('kdat_temp shape:',kdat_temp.shape)
                print('mps shape:',mps.shape)
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
            print('recon shape before squeeze:',recon.shape)
            recon = np.squeeze(recon)
            print('recon shape after squeeze:',recon.shape)

            chunks=(1, recon.shape[-1]) #(1,72,180)

            if prew:
                idx=[0, 2 ,-2,-1] #expected string indices in mps_file
            else:
                idx=[0, 2 , -1] 
            
        
            mps_pars='_'.join([mps_file.name.split('_')[i] for i in idx]).rsplit('.',1)[0]
            name_str =  "_".join(infile_k.rsplit('.', 1)[0].split('_')[-6:])

            file_name=f'CEST_SENSE_recons_{Rep}_r_{args.r}_i_{args.i}_'+name_str+'_'+mps_pars+'.h5'
                

            if Part_idx==Part_start:
                with h5py.File(DATA_DIR+file_name,'w') as f:
                    f.create_dataset(f'CEST_recon_Part_idx_{Part_idx}',data=recon,chunks=chunks)
            else:
                with h5py.File(DATA_DIR+file_name,'a') as f:
                    f.create_dataset(f'CEST_recon_Part_idx_{Part_idx}',data=recon,chunks=chunks)
            del recon

RO=114 #number of readout positions at each we have a hyber slice
RO_start=0 #start and end RO positin that cover FOV 
RO_end=113 #||160

if args.Hybdim=='RO':
    with device:
        for RO_idx in range(RO):
            
            with h5py.File(DATA_DIR+infile_k,'r') as f:
            
                # kdat_temp=np.concatenate((f['kdat_01'][...,RO_idx],  # all 65 Cest Reps
                #                 f['kdat_02'][...,RO_idx],
                #                 f['kdat_03'][...,RO_idx],
                #                 f['kdat_04'][...,RO_idx]),
                #                 axis =0)

                # kdat_temp=np.concatenate((f['kdat_01'][...,RO_idx],  # first 30 Cest Reps
                #                 f['kdat_02'][...,RO_idx]),                
                #                 axis =0)
                
                # kdat_temp=f['kdat_01'][...,RO_idx]  # first 15 Cest Reps
                # kdat_temp=f['kdat_01'][Rep,...,RO_idx][None,...]  # Sagital slices at a certain RO position for a single rep 
                kdat_temp=f['kdat_01'][...,RO_idx][:]  # Sagital slices at a certain RO position for all reps 
                # print('con kdat_temp shape before permution:',kdat_temp.shape)
                
            # track_memory(f'RO {RO_idx} :')   
            print('RO:',RO_idx)
            # kdat_temp=np.permute_dims(kdat_temp,(0,3,1,2))
            
            with h5py.File(mps_file,'r') as f:
                mps=f['mps'][...,RO_idx]
                    
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
            # chunks=((1,)+ recon.shape[1::]) #(1,72,180)
            chunks=(1, recon.shape[-2],recon.shape[-1]) #(Reps,PE2,PE1)

            if prew:
                idx=[0, 2 ,-2,-1] #expected string indices in mps_file
            else:
                idx=[0, 2 , -1] 
            
        
            mps_pars='_'.join([mps_file.name.split('_')[i] for i in idx]).rsplit('.',1)[0]
            name_str =  "_".join(infile_k.rsplit('.', 1)[0].split('_')[-6:])

            # file_name=f'CEST_SENSE_recons_{Rep}_r_{args.r}_i_{args.i}_'+name_str+'_'+mps_pars+'.h5' #for certain Rep
            file_name=f'CEST_SENSE_recons_Reps_0_{Reps}_r_{args.r}_i_{args.i}_'+name_str+'_'+mps_pars+'.h5' # for all Reps
                

            if RO_idx==RO_start:
                with h5py.File(DATA_DIR+file_name,'w') as f:
                    f.create_dataset(f'CEST_recon_RO_idx_{RO_idx}',data=recon,chunks=chunks)
            else:
                with h5py.File(DATA_DIR+file_name,'a') as f:
                    f.create_dataset(f'CEST_recon_RO_idx_{RO_idx}',data=recon,chunks=chunks)
            del recon

end_time = time.perf_counter()
duration = (end_time - start_time)/60

print(f"Reconstruction completed in {duration:.2f} Minutes")

print(f'Created file at : {file_name}' )