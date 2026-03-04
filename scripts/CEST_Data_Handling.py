"""
#To do: update and clean the code
        singlton Chunking at Rep and RO dimension
        

This script converts .dat to .h5 files


Adapted from Author: Zhengguo Tan <zhengguo.tan@gmail.com>


"""

import argparse
import h5py
import os
import pathlib
import torch
import twixtools
import argparse 
import numpy as np
import sigpy as sp

from sigpy.mri import app, epi, sms, cc

parser=argparse.ArgumentParser(description='Extract data from Twix file to h5py')
parser.add_argument('--data', default=None, help='data file path')
parser.add_argument('--acs', type=bool, default=False, help='let the code know if it for a separate ACS data')
parser.add_argument('--mode', type=str,required=True,choices=['map', 'read'], help='let the code know if it for a separate ACS data')

args=parser.parse_args()


# print(np.__file__)
# print(torch.__file__)
# print(sp.__file__)
if args.data is None:
    # DIR = os.path.dirname(os.path.realpath(__file__))
    DIR='/home/vault/iwbi/iwbi112h/CEST_DATA/'

    #2D Grappa single offset (WM)
    # infile= 'meas_MID00454_FID01852_100_wip_Snap_Cai_4_3_1_1_2.dat'

    # 2D Grappa 65 offsets (WM)
    # infile='meas_MID00455_FID01853_100_wip_Snap_Cai_4_3_1_1_2_1p00.dat'
    # 1D Grappa single offset (WASABI)
    # infile='meas_MID00018_FID13340_gre_cest_100_p145_ice_001.dat'

    # 1D Grappa 35 offset  (WASABI)
    # infile='meas_MID00022_FID13344_gre_cest_WASABI_32.dat'


    # CEST_34 offsets (Simon)
    infile ='meas_MID00070_FID33988_CEST_3shot.dat'  #3shot
    # infile ='meas_MID00071_FID33989_CEST_1shot.dat'  #1shot
    # infile = 'acs_1shot.dat' # ACS

    file_dir=DIR+infile
    out_dir = DIR
else:
    file_dir=args.data
    out_dir=np.__file__



if torch.cuda.is_available():
    device = sp.Device(0)
else:
    device = sp.cpu_device

print('> device: ', device)

# %% prepare output string
print('>>> .dat: ', file_dir)


print('> output path: ', out_dir)
# make a new directory if not exist
# pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

# %% read in twix data
twixobj = twixtools.read_twix(file_dir)

img_meas = twixobj[-1]

# %% use hdr



REMOVE_OS = False  #over-sampling
REMOVE_OS_Ref = False  #over-sampling for reference scan
# if REMOVE_OS is True:
#     os = 2
# else:
#     os = 1

if args.mode == 'map':

    mapped = twixtools.map_twix(img_meas)

    # kdat twix
    kdat_twix = mapped['image']

    kdat_twix.flags['regrid'] = True
    kdat_twix.flags['remove_os'] = REMOVE_OS
    kdat_twix.flags['zf_missing_lines'] = True 
    kdat_twix.flags['average']['Seg'] = False
    kdat_twix.flags['skip_empty_lead'] = False

    Accel_PE1 = twix['hdr']['MeasYaps']['sPat'] ['lAccelFactPE'] 
    Accel_3D = twix['hdr']['MeasYaps']['sPat'] ['lAccelFact3D']
    Accel_total = twix['hdr']['MeasYaps']['sPat'] ['dTotalAccelFact']


    # N_Accel_PE1 = 2 #dummy to perform some operations

    # calculate the total accelaration factor (excluding partial fourier and eliptical sampling if any)

    print('> Accel_PE1: ', Accel_PE1)
    print('> Accel_3D: ', Accel_3D)
    print('> Total Acceleration: ', Accel_total)

    N_Part= mapped['hdr']['Config']['NPar']
    N_Lin= mapped['hdr']['Config']['NLinMeas']


    if Accel_total > 1:
        refs_twix = mapped['refscan']
        refs_twix.flags['regrid'] = True
        refs_twix.flags['remove_os'] = REMOVE_OS_Ref
        refs_twix.flags['skip_empty_lead'] = True
        refs_twix.flags['average']['Seg'] = False

    # %% data shape
    print('> unsorted kdat shape: ', kdat_twix.shape)

    central_part = kdat_twix.kspace_center_par # central slice in 3D
    central_Lin = kdat_twix.kspace_center_lin # central slice in 3D

    print('> central_part: ', central_part)
    print('> central_Lin: ', central_Lin)

    Reps=kdat_twix.shape[-7]
    # central_part_ref = 18
    N_y = kdat_twix.shape[-3]     # Lin
    N_x = kdat_twix.shape[-1]     # Col
    N_coil = kdat_twix.shape[-2]  # Cha
    print('> N_Offs: ', Reps)
    # print('> N_Offs_ref: ', N_Offs_ref)
    # %% read out data
    # kdat = kdat_twix[:]

    # pcor = pcor_twix[:]
    # print('> pcor shape: ', pcor.shape)


    #uncomment below for ref
    if Accel_total > 1:
        refs = refs_twix[:]
        print('> refs shape: ', refs.shape)

    print('> N_Partitions ', int(N_Part))

    kdat_twix = np.squeeze(kdat_twix[:])
    print('> unsorted squeezed kdat shape: ', kdat_twix.shape)


    
    if args.acs is False:
        kdat_twix = np.transpose(kdat_twix, (-5,-2,-4,-3,-1)) #choose the order that suituble for your application. 
        file_name = f'/kdat_3D_R{int(Reps)}_C{int(N_coil)}_'+ infile.split('_')[-1].rsplit('.',1)[0]+'.h5'
    else:
        kdat_twix = np.transpose(kdat_twix, (-2,-4,-3,-1)) #choose the order that suituble for your ACS. 
        file_name = f'/ACS_3D_C{int(N_coil)}_'+ infile.split('_')[-1].rsplit('.',1)[0]+'.h5'

    print('> sorted kdat shape: ', kdat_twix.shape)
    # raise SystemExit('Stop here for now')


    with h5py.File(out_dir + file_name, 'w') as f:
        # split CEST Data into 2 parts for easier handling, it can be split into more parts if needed based on Reps size. 
        if args.acs is False:
            dset=f.create_dataset('kdat_01', data = kdat_twix[0:17,...])
            dset.attrs['Accel_PE1'] = Accel_PE1
            dset.attrs['Accel_3D'] = Accel_3D
            dset.attrs['Accel_total'] = Accel_total
            dset.attrs['central_Lin'] = central_Lin
            dset.attrs['central_Part'] = central_part
            f.create_dataset('kdat_02', data = kdat_twix[17::,...])
            if Accel_total > 1:
                f.create_dataset('ref', data = refs)
            f.close()
        else:
            dset=f.create_dataset('refs', data = kdat_twix[:])
            dset.attrs['Accel_PE1'] = Accel_PE1
            dset.attrs['Accel_3D'] = Accel_3D
            dset.attrs['Accel_total'] = Accel_total
            dset.attrs['central_Lin'] = central_Lin
            dset.attrs['central_Part'] = central_part
            f.close()
elif args.mode == 'read':

    image_mdbs = []
    for mdb in img_meas['mdb']:
        if mdb.is_image_scan():
            image_mdbs.append(mdb)

    n_line = 1 + max([mdb.cLin for mdb in image_mdbs])
    n_part = 1 + max([mdb.cPar for mdb in image_mdbs])
    n_channel, n_column = image_mdbs[0].data.shape
    # if args.acs is False:
    n_reps = 1 + max([mdb.cEco for mdb in image_mdbs])

    if args.acs is False:
        kdat_twix = np.zeros([n_reps, n_part, n_line, n_channel, n_column], dtype=np.complex64)
        for mdb in image_mdbs:
            kdat_twix[mdb.cEco, mdb.cPar, mdb.cLin] += mdb.data
        #original shape ((Reps:34, PE2:24, PE1:96, Coils:44, 114))
        kdat_twix = np.transpose(kdat_twix, (-5,-2,-4,-3,-1)) #_choose the order that suituble for your application. 
        file_name = f'/kdat_3D_R{int(n_reps)}_C{int(n_channel)}_'+ infile.split('_')[-1].rsplit('.',1)[0]+'.h5'

    else:
        kdat_twix = np.zeros([n_part, n_line, n_channel, n_column], dtype=np.complex64)
        for mdb in image_mdbs:
            kdat_twix[mdb.cPar, mdb.cLin] += mdb.data
        #original shape (( PE2:24, PE1:96, Coils:44, 114))
        kdat_twix = np.transpose(kdat_twix, (-2,-4,-3,-1)) #choose the order that suituble for your ACS. 
        file_name = f'/ACS_3D_C{int(n_channel)}_'+ infile.split('_')[-1].rsplit('.',1)[0]+'.h5'


        
    
    with h5py.File(out_dir + file_name, 'w') as f:
        # split CEST Data into 2 parts for easier handling, it can be split into more parts if needed based on Reps size. 
        if args.acs is False:
            dset=f.create_dataset('kdat_01', data = kdat_twix[0:17,...])
            
            f.create_dataset('kdat_02', data = kdat_twix[17::,...])
            f.close()
        else:
            dset=f.create_dataset('refs', data = kdat_twix[:])
            f.close()
            
print ('Done, saved  : ', out_dir + file_name)
# %%
