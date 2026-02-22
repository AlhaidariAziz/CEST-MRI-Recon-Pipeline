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
from pathlib import Path
from sigpy.mri import app
import argparse

#To do: add RO in h5 file dataset


parser = argparse.ArgumentParser(description='run LLR reconstruction.')

# parser.add_argument('--data',
#                     default=DATA_DIR+infile_k,
#                     help='raw dat file')

parser.add_argument('--v',
                    type=bool,
                    default=False,
                    help='verbose mode')

parser.add_argument('--blk_shape',type=int, nargs=3, default=[1,6,6],  ################## added for easier comparison purposes
                    help='JETS LLR block_shape as 3 integers, e.g., --blk_shape 1 6 6')

parser.add_argument('--blk_strides',type=int, nargs=3, default=[1,1,1],  ################## added for easier comparison purposes
                    help='JETS LLR blk_strides as 3 integers, e.g., --blk_strides 1 1 1')

parser.add_argument('--r', type=float, default=1e-6,
                    help=' LLR regularization constant    [default: 1e-6]')

parser.add_argument('--i', type=int , default=20,
                    help=' Max iterations    [default: 20]')

parser.add_argument('--prew', type=bool , default=False,
                    help=' prewhitening')

# parser.add_argument('--coil_batch_size', type=int, default=None,
#                     help=' No. of coils to process simultaniously instead of all at once   [default: None]')



start_time = time.perf_counter()

args = parser.parse_args()

args.blk_shape = tuple(args.blk_shape)
args.blk_strides = tuple(args.blk_strides)

blk_shape_str = "x".join(map(str, args.blk_shape))
blk_strides_str = "x".join(map(str, args.blk_strides))
print('> blk_shape: ', args.blk_shape)
print('> blk_strides: ', args.blk_strides)
print('>  lamda(r): ', args.r)
print('>  Max iteration (i): ', args.i)
print('> prewhitening is set to: ', args.prew)

cwd=Path.cwd().parent
print('Current directory:',cwd)

DATA_DIR= "/home/vault/iwbi/iwbi112h/CEST_Data/"


prew=args.prew
if prew:
    infile_k='CEST_kdat_2D_R65_C32_prew.h5'
    mps_file=cwd/'maps/mps_c_0.8_w_36_kw_6_ROW_128_sp_ksp_prew.h5'
else:    
    infile_k='CEST_kdat_2D_R65_C32.h5'
    mps_file=cwd/'maps/mps_c_0.8_w_36_kw_6_ROW_128_sp_ksp.h5'


# infile_k='CEST_kdat_2D_R65_C32.h5'
# mps_file=cwd/'maps/mps_c_0.97.h5' # mps with threshold 0.97
# mps_file=cwd/'maps/mps_c_0.97_w_36_kw_6_ROW_128_sp_ksp.h5' # mps with threshold 0.97 
# mps_file=cwd/'maps/mps_c_0.80.h5'
# mps_file=cwd/'maps/mps_c_0.6.h5'
RO=224 #number of readout positions at each we have a hyber slice
StRO_idx=15 #start  RO positin for FOV (experimental 80|15)
EnRO_idx=205 # end RO position (experimental 160|205)
StRep_idx=1
EnRep_idx=10
MdRO_idx=StRO_idx + (EnRO_idx-StRO_idx)//2
print(f'RO start:{StRO_idx}, RO end :{EnRO_idx}')

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





#Volume recon with LLR



# img_shape=(65,32,72,180,slice_idx)



with device:
    # with h5py.File(DATA_DIR + infile_k,'r') as f: #Load the data once
        # print('Loading Data ... ')
        # kdat01=f['kdat_01'][0:9,...]

    for RO_idx in range(StRO_idx,EnRO_idx):
    # for RO_idx in range(RO):
        
        with h5py.File(DATA_DIR + infile_k,'r') as f:
            # print('keys :',f.keys() )
            # kdat_temp01=np.squeeze(f['kdat_01'][...,RO_idx])
            # # track_memory('kdat_temp01')
            # kdat_temp02=np.squeeze(f['kdat_02'][...,RO_idx])
            # # track_memory('kdat_temp02')
            # kdat_temp01=np.concatenate((kdat_temp01,kdat_temp02), axis =0)
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
            # # track_memory('del temp01, temp02 con')
            # kdat_temp=np.concatenate((f['kdat_01'][...,RO_idx],
            #                 f['kdat_02'][...,RO_idx],
            #                 f['kdat_03'][...,RO_idx],
            #                 f['kdat_04'][...,RO_idx]),
            #                 axis =0)

            # kdat_temp=np.concatenate((f['kdat_01'][...,RO_idx],
            #                 f['kdat_02'][...,RO_idx]),                
            #                 axis =0)
            
            kdat_temp=f['kdat_01'][StRep_idx:EnRep_idx,...,RO_idx]
        # print('con kdat_temp shape before permution:',kdat_temp.shape)
        # print('kdat shape:',kdat_temp.shape)
        # track_memory(f'RO {RO_idx} :')   
        print('RO:',RO_idx)
        kdat_temp=np.permute_dims(kdat_temp,(0,3,1,2))
        
        with h5py.File(mps_file,'r') as f:
            mps=f['mps'][...,RO_idx]
                
        # print('con kdat_temp shape:',kdat_temp.shape)
        # print('mps shape:',mps.shape)
        
        # HDRecon dimensions setup
        # add a Necho =1 dimension at index 1 and add a single partition Nz=1 at index 3 to align with HDRecon expectation
        #( Ntime, Necho, Ncoil, Nz, Ny, Nx)
        kdat_temp= kdat_temp[:, np.newaxis,:,np.newaxis,...]
        mps = mps[:, np.newaxis, ...]
        # print('kdat_temp HD shape:', np.shape(kdat_temp))
        # print('mps HD shape:', np.shape(mps))
        if RO_idx == MdRO_idx:
            v=True
        else:
            v=args.v

        recon = app.HighDimensionalRecon(kdat_temp,
                                        mps=mps,
                                        phase_sms=None, #changed from sms_phase_redu to None for testing
                                        combine_echo=False, 
                                        phase_echo=None, #changed from np.conj(shot_phase_redu) to None for testing
                                        # regu='TIK', #changed from LLR to TIK
                                        regu='LLR', #changed from LLR to TIK
                                        blk_shape=args.blk_shape,
                                        blk_strides=args.blk_strides,
                                        # solver='ConjugateGradient',
                                        solver='ADMM',
                                        lamda=args.r,
                                        rho=5e-2,
                                        max_iter=args.i,      #max iter changed from 15 to 2 for testing
                                        show_pbar=True, verbose=v,
                                        device=device).run()
        
        try:
            xp.get_default_memory_pool().free_all_blocks()
        except AttributeError:
            # We are likely using NumPy/CPU, so no pool to clear
            pass
        
        del kdat_temp, mps
        recon = recon.get() if hasattr(recon, 'get') else recon
        recon=np.squeeze(recon)
        chunks=((1,)+ recon.shape[1::]) #(1,72,180)
        
        if prew:
            idx=[0, 2 ,-2,-1] #selected strings "_".splitted indices in mps_file
        else:
            idx=[0, 2 , -1]

        mps_pars='_'.join([mps_file.name.split('_')[i] for i in idx]).rsplit('.',1)[0]
        # file_name=f'CEST_SENSE_recons_{Rep}_r_{args.r}_i_{args.i}_'+mps_pars+'.h5'
        file_name=f'CEST_LLR_recons_Reps_{StRep_idx}_{EnRep_idx}_RO_{StRO_idx}_{EnRO_idx}_{blk_shape_str}_r_{args.r}_i_{args.i}'+mps_pars+'.h5'
        
        if RO_idx==StRO_idx:
            with h5py.File(DATA_DIR + file_name,'w') as f:
                f.create_dataset(f'CEST_recon_RO_idx_{RO_idx}',data=recon,chunks=chunks)
        else:
            with h5py.File(DATA_DIR + file_name,'a') as f:
                f.create_dataset(f'CEST_recon_RO_idx_{RO_idx}',data=recon,chunks=chunks)
        del recon
        # exit()
end_time = time.perf_counter()
duration = (end_time - start_time)/60

print(f"Reconstruction completed in {duration:.2f} Minutes")