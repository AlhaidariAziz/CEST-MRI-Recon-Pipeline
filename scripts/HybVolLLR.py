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

parser.add_argument('--file',  default=None,help='file within default directory defined in the code script')

parser.add_argument('--v',
                    type=bool,
                    default=False,
                    help='verbose mode')

parser.add_argument('--blk_shape',type=int, nargs=3, default=[1,5,5],  ################## added for easier comparison purposes
                    help='JETS LLR block_shape as 3 integers, default: --blk_shape 1 5 5')

parser.add_argument('--blk_strides',type=int, nargs=3, default=[1,1,1],  ################## added for easier comparison purposes
                    help='JETS LLR blk_strides as 3 integers, default: --blk_strides 1 1 1')

parser.add_argument('--r', type=float, default=1e-2,
                    help=' LLR regularization constant    [default: 1e-2]')

parser.add_argument('--i', type=int , default=20,
                    help=' Max iterations    [default: 20]')

parser.add_argument('--splits', type=int , default=2,
                    help=' Into how many splits to divide the data for LLR reconstruction, e.g., 10. The code will reconstruct each split separately and combine the results at the end. This is useful to reduce memory usage when reconstructing large datasets with LLR.')

parser.add_argument('--prew', type=bool , default=False,
                    help=' prewhitening')

# parser.add_argument('--coil_batch_size', type=int, default=None,
#                     help=' No. of coils to process simultaniously instead of all at once   [default: None]')



start_time = time.perf_counter()

args = parser.parse_args()

if args.splits > 2 or args.splits <=0:
    raise SystemExit( "currently only support 1 or 2 splits. Please specify the --splits argument accordingly." )


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

DATA_DIR= "/home/vault/iwbi/iwbi112h/CEST_DATA/"

if args.file is None:
    prew=args.prew
    if prew:
        infile_k='CEST_kdat_2D_R65_C32_prew.h5'
        mps_file=cwd/'maps/mps_c_0.8_w_36_kw_6_ROW_128_sp_ksp_prew.h5'
    else:    
        # infile_k='CEST_kdat_2D_R65_C32.h5' #WM

        infile_k='kdat_2D_kdat_3D_R34_C44_1shot.h5'

else:
    infile_k=args.file
    


# mps_file=cwd/'maps/mps_c_0.8_w_36_kw_6_ROW_128_sp_ksp.h5' #WM
# mps_file=cwd/'maps/mps_c_0.8_t0.025_w_24_kw_6_sp_ksp_3D_R34_C44_1shot.h5'
# mps_file=cwd/'maps/mps_c_0.8_t0.05_w_24_kw_6_sp_ksp_3D_R34_C44_1shot.h5'
# mps_file=cwd/'maps/mps_c_0_t0.05_w_24_kw_6_sp_ksp_3D_R34_C44_3shot.h5'
# mps_file=cwd/'maps/mps_c_0_t0.05_w_24_kw_6_sp_ksp_3D_R34_C44_1shot.h5' #best mps from acs 
mps_file=cwd/'maps/mps_c_0_t0.05_w_24_kw_6_sp_ksp_3D_R34_C44_1shot_chopped.h5' #best mps from chopped acs 
RO=114 #number of readout positions at each we have a hyber slice
StRO_idx=0 #start  RO position for FOV (experimental 80|15)
EnRO_idx=114 # end RO position (experimental 160|205)
# StRep_idx=0
# EnRep_idx=34
Reps=34
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




##################
## output file name strings
#
# if prew:
#     idx=[0, 2 ,-2,-1] #selected strings "_".splitted indices in mps_file
# else:
#     idx=[0, 2 , -1]

mps_pars = "_".join(mps_file.name.rsplit('.',1)[0].split('_')[0:10])
# name_str =  "_".join(infile_k.rsplit('.', 1)[0].split('_')[-6:])
name_str =  infile_k.replace('kdat_2D_','').replace('.h5','')


#Volume recon with LLR

#splits
reps_per_split= Reps//args.splits
print(f'Numberf splits: {args.splits}, Reps per split: {reps_per_split}') 


with device:

    for s in range(args.splits):
        print(f'Processing split {s+1}/{args.splits} ...')
        StRep_idx = s*reps_per_split
        EnRep_idx = (s+1)*reps_per_split if s < args.splits - 1 else Reps 
        print(f'  Reps for this split: {StRep_idx} to {EnRep_idx}')

        for RO_idx in range(StRO_idx,EnRO_idx):
        # for RO_idx in range(RO):
            
            with h5py.File(DATA_DIR + infile_k,'r') as f:
 
                if args.splits == 1:
                    kdat_temp=np.concatenate((f['kdat_01'][...,RO_idx], # includ all Reps in LLR
                                              f['kdat_02'][...,RO_idx]),                
                                              axis =0)
                                    
                if args.splits == 2:
                    if s == 0:
                        kdat_temp=f['kdat_01'][:,...,RO_idx]
                    if s == 1:
                        kdat_temp=f['kdat_02'][:,...,RO_idx]
                    

            # print('con kdat_temp shape before permution:',kdat_temp.shape)
            # print('kdat shape:',kdat_temp.shape)
            # track_memory(f'RO {RO_idx} :')   
            print('RO:',RO_idx)
            
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
            if RO_idx == MdRO_idx: #setting verbose to ture
                v=True
            else:
                v=args.v

            recon = app.HighDimensionalRecon(kdat_temp,
                                            mps=mps,
                                            phase_sms=None, 
                                            combine_echo=False, 
                                            phase_echo=None, #changed from np.conj(shot_phase_redu) to None for testing
                                            # regu='TIK', 
                                            regu='LLR',
                                            blk_shape=args.blk_shape,
                                            blk_strides=args.blk_strides,
                                            # solver='ConjugateGradient',
                                            solver='ADMM',
                                            lamda=args.r,
                                            rho=5e-2,
                                            max_iter=args.i,      
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
            chunks=(1, *recon.shape[1::]) #(reps,72,180)
            

            file_name=f'CEST_LLR_recons_splits_{s+1}of{args.splits}_Reps_{StRep_idx}_{EnRep_idx}_RO_{StRO_idx}_{EnRO_idx}_{blk_shape_str}_r_{args.r}_i_{args.i}'+'_'+name_str+'_'+mps_pars+'.h5'

            if RO_idx==StRO_idx:
                with h5py.File(DATA_DIR + file_name,'w') as f:
                    f.create_dataset(f'CEST_recon_RO_idx_{RO_idx}',data=recon,chunks=chunks)
            else:
                with h5py.File(DATA_DIR + file_name,'a') as f:
                    f.create_dataset(f'CEST_recon_RO_idx_{RO_idx}',data=recon,chunks=chunks)
            del recon
            # exit()
        print(f'Created file at : {file_name}' )
end_time = time.perf_counter()
duration = (end_time - start_time)/60

print(f"Reconstruction completed in {duration:.2f} Minutes")