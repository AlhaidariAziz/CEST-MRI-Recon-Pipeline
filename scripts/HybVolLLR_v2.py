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

# [MPI] Try to import mpi4py
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False

# ... (parser arguments unchanged) ...

parser = argparse.ArgumentParser(description='run LLR reconstruction.')
parser.add_argument('--data', default=None, help='kdata h5 file full path')
parser.add_argument('--mps', default=None, help='mps h5 file name')
parser.add_argument('--v', type=bool, default=False, help='verbose mode')
parser.add_argument('--blk_shape', type=int, nargs=3, default=[1,5,5], help='JETS LLR block_shape as 3 integers')
parser.add_argument('--blk_strides', type=int, nargs=3, default=[1,1,1], help='JETS LLR blk_strides as 3 integers')
parser.add_argument('--r', type=float, default=1e-2, help='LLR regularization constant')
parser.add_argument('--i', type=int, default=20, help='Max iterations')
parser.add_argument('--splits', type=int, default=2, help='Into how many splits to divide the data')
parser.add_argument('--prew', type=bool, default=False, help='prewhitening')

start_time = time.perf_counter()

args = parser.parse_args()

# [MPI] Initialize MPI
if MPI_AVAILABLE:
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()
else:
    mpi_rank = 0
    mpi_size = 1

args.blk_shape = tuple(args.blk_shape)
args.blk_strides = tuple(args.blk_strides)

blk_shape_str = "x".join(map(str, args.blk_shape))
blk_strides_str = "x".join(map(str, args.blk_strides))

# [MPI] Only rank 0 prints most messages
if mpi_rank == 0:
    print('> blk_shape: ', args.blk_shape)
    print('> blk_strides: ', args.blk_strides)
    print('> lamda(r): ', args.r)
    print('> Max iteration (i): ', args.i)
    print('> prewhitening is set to: ', args.prew)
    print('> DATA: ', args.data)

cwd = Path.cwd().parent
if mpi_rank == 0:
    print('Current directory:', cwd)


if args.data is not None:
    DATA_DIR= args.data.rsplit('/',1)[0] + '/'
    infile_k=args.data.rsplit('/',1)[1]


else:
    prew=args.prew
    if prew:
        infile_k='CEST_kdat_2D_R65_C32_prew.h5'
    else: 
        # DATA_DIR= "/home/vault/iwbi/iwbi112h/CEST_Data/"
        # infile_k='CEST_kdat_2D_R65_C32.h5'    
        # mps_file=cwd/'maps/mps_c_0.8_t0.025_w_24_kw_6_sp_ksp_3D_R34_C44_1shot.h5'
        # mps_file=cwd/'maps/mps_c_0.8_t0.05_w_24_kw_6_sp_ksp_3D_R34_C44_1shot.h5'
        # mps_file=cwd/'maps/mps_c_0_t0.05_w_24_kw_6_sp_ksp_3D_R34_C44_3shot.h5'
        # mps_file=cwd/'maps/mps_c_0_t0.05_w_24_kw_6_sp_ksp_3D_R34_C44_1shot.h5' #best mps from acs 
        # mps_file=cwd/'maps/mps_c_0_t0.05_w_24_kw_6_sp_ksp_3D_R34_C44_1shot_chopped.h5' #best mps from chopped acs 
        raise SystemExit("Pls specify the data file full path with --data argument ")


if args.mps is not None:
    mps_file=  str(cwd) + '/maps/'  + args.mps
else:
    if prew:
        mps_file=cwd/'maps/mps_c_0.8_w_36_kw_6_ROW_128_sp_ksp_prew.h5'
    else:
        # mps_file=cwd/'maps/mps_c_0.97.h5'
        # mps_file=cwd/'maps/mps_c_0.8.h5'
        # mps_file=cwd/'maps/mps_c_0.8.h5'
        raise SystemExit("Pls specify the maps file name with --mps argument")



RO = 120
StRO_idx = 0
EnRO_idx = RO
Reps = 26
MdRO_idx = (StRO_idx + (EnRO_idx - StRO_idx)) // 2
if mpi_rank == 0:
    print(f'RO start:{StRO_idx}, RO end :{EnRO_idx}')

# [MPI] Determine which RO indices this rank processes
all_ro_indices = list(range(StRO_idx, EnRO_idx))
my_ro_indices = all_ro_indices[mpi_rank::mpi_size]   

if mpi_rank == 0:
    print(f"MPI size: {mpi_size}, each rank processes {len(my_ro_indices)} RO slices")

# [MPI] Adjust device selection: force CPU when using multiple MPI processes
if mpi_size > 1:
    # Force CPU to avoid GPU memory conflicts
    device = sp.Device(-1)
    xp = np
    if mpi_rank == 0:
        print("Using CPU because MPI size > 1")
else:
    # Original GPU/CPU detection for single process
    try:
        if cp.cuda.runtime.getDeviceCount() > 0:
            device = sp.Device(0)
            if mpi_rank == 0:
                print(f'GPU detected: {cp.cuda.runtime.getDeviceProperties(0)["name"].decode()}')
        else:
            device = sp.Device(-1)
    except (ImportError, Exception):
        device = sp.Device(-1)
    xp = device.xp
    if mpi_rank == 0:
        print(f"Using device: {device} (Backend: {xp.__name__})")

# ... (output name string generation unchanged) ...
if hasattr(args.mps, 'name'):
    mps_pars = "_".join(args.mps.name.rsplit('.',1)[0].split('_')[0::])
else:
    mps_pars = "_".join(args.mps.rsplit('.',1)[0].split('_')[0::])
name_str = infile_k.replace('kdat_2D_','').replace('.h5','')

reps_per_split = Reps // args.splits
if mpi_rank == 0:
    print(f'Number of splits: {args.splits}, Reps per split: {reps_per_split}')

# [MPI] Each rank writes its own output file
if mpi_size > 1:
    file_suffix = f"_rank{mpi_rank:03d}"
else:
    file_suffix = ""

# [MPI] Main reconstruction loop – only over my_ro_indices
with device:
    for s in range(args.splits):
        if mpi_rank == 0:
            print(f'Processing split {s+1}/{args.splits} ...')
        StRep_idx = s * reps_per_split
        EnRep_idx = (s+1)*reps_per_split if s < args.splits-1 else Reps
        if mpi_rank == 0:
            print(f'  Reps for this split: {StRep_idx} to {EnRep_idx}')

        for idx, RO_idx in enumerate(my_ro_indices):
            print(f'Rank {mpi_rank}: RO {RO_idx} ({idx+1}/{len(my_ro_indices)})')

            # Load k-space data (same as original)
            with h5py.File(DATA_DIR + infile_k, 'r') as f:
                if args.splits == 1:
                    kdat_temp = np.concatenate((f['kdat_01'][...,RO_idx],
                                                f['kdat_02'][...,RO_idx]), axis=0)
                if args.splits == 2:
                    if s == 0:
                        kdat_temp = f['kdat_01'][:,...,RO_idx]
                    if s == 1:
                        kdat_temp = f['kdat_02'][:,...,RO_idx]

            with h5py.File(mps_file, 'r') as f:
                mps = f['mps'][...,RO_idx]

            # Reshape for HDRecon
            kdat_temp = kdat_temp[:, np.newaxis, :, np.newaxis, ...]
            mps = mps[:, np.newaxis, ...]

            # Verbose only for middle RO and only on rank 0
            if RO_idx == MdRO_idx and mpi_rank == 0:
                v = True
            else:
                v = args.v

            recon = app.HighDimensionalRecon(kdat_temp,
                                            mps=mps,
                                            phase_sms=None,
                                            combine_echo=False,
                                            phase_echo=None,
                                            regu='LLR',
                                            blk_shape=args.blk_shape,
                                            blk_strides=args.blk_strides,
                                            solver='ADMM',
                                            lamda=args.r,
                                            rho=5e-2,
                                            max_iter=args.i,
                                            show_pbar=True,
                                            verbose=v,
                                            device=device).run()

            if mpi_size == 1 and hasattr(xp, 'get_default_memory_pool'):
                try:
                    xp.get_default_memory_pool().free_all_blocks()
                except AttributeError:
                    pass

            del kdat_temp, mps
            recon = recon.get() if hasattr(recon, 'get') else recon
            recon = np.squeeze(recon)
            chunks = (1, *recon.shape[1::])

            # [MPI] Modified filename: includes rank suffix when using MPI
            file_name = (f'CEST_LLR_recons_splits_{s+1}of{args.splits}_Reps_{StRep_idx}_{EnRep_idx}_RO_{StRO_idx}_{EnRO_idx}_{blk_shape_str}_r_{args.r}_i_{args.i}'
                         + '_' + name_str + '_' + mps_pars + file_suffix + '.h5')

            # Write each rank's results to its own file
            if RO_idx == my_ro_indices[0]:
                with h5py.File(DATA_DIR + '/Ranks/' + file_name, 'w') as f:
                    f.create_dataset(f'CEST_recon_RO_idx_{RO_idx}', data=recon, chunks=chunks)
            else:
                with h5py.File(DATA_DIR + '/Ranks/' + file_name, 'a') as f:
                    f.create_dataset(f'CEST_recon_RO_idx_{RO_idx}', data=recon, chunks=chunks)
            del recon

        if mpi_rank == 0:
            print(f'Created file : {file_name} at {DATA_DIR}')

# [MPI] Wait for all ranks to finish writing (only if MPI is used)
if MPI_AVAILABLE and mpi_size > 1:
    comm.Barrier()

end_time = time.perf_counter()
duration = (end_time - start_time) / 60

# [MPI] Only rank 0 prints total time
if mpi_rank == 0:
    print(f"Reconstruction completed in {duration:.2f} Minutes")
    if mpi_size > 1:
        print(f"Results are saved in {mpi_size} separate files with suffix _rank*.h5")
        print("You can combine them later using h5py or the provided combine script.")