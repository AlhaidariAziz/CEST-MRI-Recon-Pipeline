
"""
This script converts .dat to .h5 files


Adapted from Author: Zhengguo Tan <zhengguo.tan@gmail.com>

CEST_Adapted by: Abdulaziz Alhaidari <ab.alhaidari@yahoo.com>

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

DIR = os.path.dirname(os.path.realpath(__file__))

#2D Grappa single offset
# infile= 'meas_MID00454_FID01852_100_wip_Snap_Cai_4_3_1_1_2.dat'

# 2D Grappa 65 offsets
infile='HD_Melon\\meas_MID00455_FID01853_100_wip_Snap_Cai_4_3_1_1_2_1p00.dat'
outprefstr = 'D:miniconda3\\LLR\\CEST_Data\\HD_Melon'
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



REMOVE_OS = False  #over-sampling
REMOVE_OS_Ref = False  #over-sampling for reference scan
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
# Config=mapped [-1]['hdr']['Config']
# N_Accel_PE1 = Config.lAccelFactPE
N_Accel_PE1 = 2 #dummy to perform some operations
# N_Accel_PE1 = mapped['hdr']['Config']['lAccelFactPE']
# N_Accel_PE2 = Config.lAccelFact3D
# N_Accel_PE2 = mapped['hdr']['Config']['lAccelFact3D']
# N_Part= Config.NPar
N_Part= mapped['hdr']['Config']['NPar']
# N_Lin= Config.NLinMeas
N_Lin= mapped['hdr']['Config']['NLinMeas']


# print('> Partitions ' + str(N_Part) +\
#       ', PE1 Acceleration ' + str(N_Accel_PE1) +\
#       ', PE2 Acceleration ' + str(N_Accel_PE2))

# %% phase-correction and reference scan data

# phase-correction
# # pcor_twix = mapped[-1]['phasecorr']
# pcor_twix.flags['regrid'] = True
# pcor_twix.flags['remove_os'] = REMOVE_OS
# pcor_twix.flags['skip_empty_lead'] = True
# pcor_twix.flags['average']['Seg'] = False

# # refscan
# if N_Accel_PE1 > 1:
#     refs_twix = mapped['refscan']
#     refs_twix.flags['regrid'] = True
#     refs_twix.flags['remove_os'] = REMOVE_OS_Ref
#     refs_twix.flags['skip_empty_lead'] = True
#     refs_twix.flags['average']['Seg'] = False

# %% data shape
N_Offs = kdat_twix.shape[-9]  # Repetition with different Saturation Offsets 
central_part = kdat_twix.kspace_center_par # central slice in 3D
central_Lin = kdat_twix.kspace_center_lin # central slice in 3D

# N_Offs_ref = refs_twix.shape[-9] 
# central_part_ref = refs_twix.kspace_center_par  # central slice in 3D for reference scan
# central_Lin_ref = refs_twix.kspace_center_lin  # central slice in 3D for reference scan

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
# if N_Accel_PE1 > 1:
#     refs = refs_twix[:]
#     print('> refs shape: ', refs.shape)


# # %% coil sensitivity maps
# if N_Accel_PE1 > 1:

#     if False:
#         N_virt = 14
#         refs_cc,S = cc.scc(refs, P=N_virt, coil_dim=-2, device=device)
#         # refs_cc, S = cc.cc_huang(refs, P=N_virt, coil_dim=-2, device=device)
#         # coil_compressor= mr.app.CoilCompression(refs, num_coils=N_virt)
#         # refs_cc = coil_compressor.run()

#         print("Original shape:", refs.shape)
#         print("Compressed shape:", refs_cc.shape)
#     else:
#         N_virt = N_coil
#         refs_cc = refs.copy().squeeze()

#     # N_y= 153 # number of image lines in image data    
#     # zero-fill refs
#     refs_cc_shape = list(refs_cc.shape)
#     # refs_zf = sp.resize(refs_cc, [N_Part,N_y, N_virt, N_x])
#     print('> refs_cc shape: ', refs_cc.shape)
#     # print('> refs_zf shape: ', refs_zf.shape)
#     # reshape
#     refs_prep = np.squeeze(refs_cc)
#     # refs_prep = np.squeeze(refs_zf)
#     refs_prep = np.swapaxes(refs_prep, -2, -3)
#     refs_prep = np.swapaxes(refs_prep, -3, -4)
#     print('> refs_prep shape [Rep, ch , Part , PE, RO ]: ', refs_prep.shape)
#     # refs_prep = refs_prep[:, [central_part_ref], :, :] #select central 3D_slice
#     # print('> refs_prep shape for single 3D slice [ ch , Part , PE, RO ]: ', refs_prep.shape)

#     f = h5py.File(outprefstr + '/refs_3D.h5', 'w')
#     f.create_dataset('refs', data=refs_prep)
#     f.close()


    # print('> estimate coil sensitivity maps: ')

    # mps = []

    
    # for s in range(N_slices):
    #     # s= central_part_ref # only one ref slice in 3D mode: The central slice 
    #     print('  ' + str(s).zfill(3))

    #     c = app.EspiritCalib(refs_prep[:, s, :, :],
    #                          crop=0.,
    #                          device=device, show_pbar=False).run()
    #     mps.append(sp.to_device(c))

    # mps = np.array(mps)
    # mps = np.swapaxes(mps, 0, 1)
    # # coil sensitivity maps are acquired in the single-slice mode
    # # mps_reord = sms.reorder_slices_mb1(mps, N_slices)
    # # print('> mps shape: ', mps.shape)
    # print('> mps shape [for central 3D ref slice]: ', mps.shape)
    # # no need to reorder mps in the single-slice mode
    # # print('> mps_reord shape: ', mps_reord.shape)
    # f = h5py.File(outprefstr + '/coils.h5', 'w')
    # f.create_dataset('coil', data=mps)
    # f.close()


# %% extract useful kdat




print('> N_Partitions ', int(N_Part))
print('> Number of Repetions ', int(offset_size))


# for i in range(offset_size):


#     print('  ' + str(i).zfill(3) + ' ' )

#     # correcting kdat
#     k = kdat_twix[..., [i], :, :, :, :, :, :, :, :].squeeze()
#     # p = pcor[..., [sid], :, :, :, :]
#     print('k shape before CC:',k.shape)
#     k_cc,S= cc.scc(k, P=N_virt, coil_dim=-2, device=device)
#     print('k shape after CC:',k_cc.shape)



#     f = h5py.File(outprefstr + f'/kdat_1D_p{str(i).zfill(2)}' + '.h5', 'w')
#     f.create_dataset('kdat_cc', data=k_cc)
#     f.create_dataset('Partitions', data=N_Part)
#     # f.create_dataset('Accel_PE1', data=N_Accel_PE1)
#     # f.create_dataset('Accel_PE2', data=N_Accel_PE2)
#     f.create_dataset('central_Lin', data=central_Lin)
#     f.create_dataset('central_Lin_ref', data=central_Lin_ref)
#     f.create_dataset('central_Part', data=central_part)
#     f.create_dataset('central_Part_ref', data=central_part_ref)
#     f.create_dataset('N_Offs', data=N_Offs)
#     f.create_dataset('N_Offs_ref', data=N_Offs_ref)
#     f.create_dataset('N_virt', data=N_virt)
#     f.close()
#     print('  ' + str(i).zfill(3) + ' ' )

# correcting kdat
k = kdat_twix[:].squeeze()
# p = pcor[..., [sid], :, :, :, :]
print('k shape before CC:',k.shape)
k_cc,S= cc.scc(k, P=N_virt, coil_dim=-2, device=device)
print('k shape after CC:',k_cc.shape)



f = h5py.File(outprefstr + f'/kdat_3D' + '.h5', 'w')
f.create_dataset('kdat_cc', data=k_cc)
f.create_dataset('Partitions', data=N_Part)
# f.create_dataset('Accel_PE1', data=N_Accel_PE1)
# f.create_dataset('Accel_PE2', data=N_Accel_PE2)
f.create_dataset('central_Lin', data=central_Lin)
f.create_dataset('central_Lin_ref', data=central_Lin_ref)
f.create_dataset('central_Part', data=central_part)
f.create_dataset('central_Part_ref', data=central_part_ref)
f.create_dataset('N_Offs', data=N_Offs)
f.create_dataset('N_Offs_ref', data=N_Offs_ref)
f.create_dataset('N_virt', data=N_virt)
f.close()

print ('Done')