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

import numpy as np
import sigpy as sp

from sigpy.mri import app, epi, sms, cc


# print(np.__file__)
# print(torch.__file__)
# print(sp.__file__)

# DIR = os.path.dirname(os.path.realpath(__file__))
DIR='/home/vault/iwbi/iwbi112h/CEST_Data'
#2D Grappa single offset
# infile= 'meas_MID00454_FID01852_100_wip_Snap_Cai_4_3_1_1_2.dat'

# 2D Grappa 65 offsets
infile='meas_MID00455_FID01853_100_wip_Snap_Cai_4_3_1_1_2_1p00.dat'
outprefstr = '/home/vault/iwbi/iwbi112h/CEST_Data'
# 1D Grappa single offset 
# infile='meas_MID00018_FID13340_gre_cest_100_p145_ice_001.dat'

# 1D Grappa 35 offset 
# infile='meas_MID00022_FID13344_gre_cest_WASABI_32.dat'

# %% argument parser
parser = argparse.ArgumentParser(description='prepare data and store output k-space data separately.')

parser.add_argument('--data',
                    default= infile,
                    help='raw dat file.')

args = parser.parse_args()


if torch.cuda.is_available():
    device = sp.Device(0)
else:
    device = sp.cpu_device

print('> device: ', device)

# %% prepare output string
print('>>> dat: ', args.data)

instr = DIR + '/' + args.data
# outprefstr = instr.split('.dat')[0]

print('> output prefix: ', outprefstr)
# make a new directory if not exist
# pathlib.Path(outprefstr).mkdir(parents=True, exist_ok=True)

# %% read in twix data
twixobj = twixtools.read_twix(instr)

twix = twixobj[-1]

# %% use hdr



REMOVE_OS = True  #over-sampling
REMOVE_OS_Ref = True  #over-sampling for reference scan
if REMOVE_OS is True:
    os = 2
else:
    os = 1


mapped = twixtools.map_twix(twix)


# kdat twix
kdat_twix = mapped['image']

kdat_twix.flags['regrid'] = True
kdat_twix.flags['remove_os'] = REMOVE_OS
kdat_twix.flags['zf_missing_lines'] = True
kdat_twix.flags['average']['Seg'] = False
kdat_twix.flags['skip_empty_lead'] = False

N_Accel_PE1 = 2 #dummy to perform some operations

N_Part= mapped['hdr']['Config']['NPar']
N_Lin= mapped['hdr']['Config']['NLinMeas']


if N_Accel_PE1 > 1:
    refs_twix = mapped['refscan']
    refs_twix.flags['regrid'] = True
    refs_twix.flags['remove_os'] = REMOVE_OS_Ref
    refs_twix.flags['skip_empty_lead'] = True
    refs_twix.flags['average']['Seg'] = False

# %% data shape
N_Offs = kdat_twix.shape[-9]  # Repetition with different Saturation Offsets 
central_part = kdat_twix.kspace_center_par # central slice in 3D
central_Lin = kdat_twix.kspace_center_lin # central slice in 3D


offset_size=kdat_twix.shape[-9]
# central_part_ref = 18
N_y = kdat_twix.shape[-3]     # Lin
N_x = kdat_twix.shape[-1]     # Col
N_coil = kdat_twix.shape[-2]  # Cha
print('> N_Offs: ', N_Offs)
# print('> N_Offs_ref: ', N_Offs_ref)
# %% read out data
# kdat = kdat_twix[:]
print('> unsorted kdat shape: ', kdat_twix.shape)

# pcor = pcor_twix[:]
# print('> pcor shape: ', pcor.shape)


#uncomment below for ref
if N_Accel_PE1 > 1:
    refs = refs_twix[:]
    print('> refs shape: ', refs.shape)




print('> N_Partitions ', int(N_Part))
print('> Number of Repetions ', int(offset_size))




f = h5py.File(outprefstr + f'/kdat_3D_R65_C32' + '.h5', 'w')
f.create_dataset('kdat_01', data = kdat_twix[:,:,:,:,:,:,:,0:15,...])
f.create_dataset('kdat_02', data = kdat_twix[:,:,:,:,:,:,:,15:30,...])
f.create_dataset('kdat_03', data = kdat_twix[:,:,:,:,:,:,:,30:45,...])
f.create_dataset('kdat_04', data = kdat_twix[:,:,:,:,:,:,:,45::,...])
if N_Accel_PE1 > 1:
    f.create_dataset('ref', data = refs)
# f.create_dataset('kdat_02', data = kdat_twix[:,:,:,:,:,:,:,30::,...])
# f.create_dataset('Partitions', data=N_Part)
# f.create_dataset('Accel_PE1', data=N_Accel_PE1)
# f.create_dataset('Accel_PE2', data=N_Accel_PE2)
f.create_dataset('central_Lin', data=central_Lin)
# f.create_dataset('central_Lin_ref', data=central_Lin_ref)
f.create_dataset('central_Part', data=central_part)
# f.create_dataset('central_Part_ref', data=central_part_ref)
f.create_dataset('N_Offs', data=N_Offs)
# f.create_dataset('N_Offs_ref', data=N_Offs_ref)
# f.create_dataset('N_virt', data=N_virt)
f.close()

print ('Done')
# %%
