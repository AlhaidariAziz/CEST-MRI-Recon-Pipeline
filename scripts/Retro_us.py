import numpy as np 
import h5py
import argparse

parser = argparse.ArgumentParser(description='Retrospective Undersampling pattern')
parser.add_argument('--type', required=True,choices=('Cart','CAIP'), help='Type of undersampling pattern')
parser.add_argument('--R', type=int, required=True, help='undersampling factor, for 2D define RPE2 as well RPE1=R//RPE2')
parser.add_argument('--RPE2', type=int, default=1, help='undersampling factor in PE2 direction')
parser.add_argument('--yshift', type=bool,default=False, help='apply yshift=1 across offsets')
parser.add_argument('--sym_yshift', type=bool,default=False, help='apply yshift=1 across offsets (symmetrical around PPM=0)')
parser.add_argument('--dc_accel', type=int, default=None,
                    help='target acceleration after adding diamond dense center')
args = parser.parse_args()

if args.type == 'CAIP' and args.RPE2 == 1 :
    raise SystemExit( "Please specify the --RPE2 > 1 argument for CAIP undersampling pattern" )

R=args.R ; Rz=args.RPE2 ;  Ry=R//Rz

print('us accelaration:',R)
print('us type:',args.type)
print('us PE1 accelaration:',Ry)
print('us PE2 accelaration:',Rz)
print('yshift:',args.yshift)
print('sym_yshift:',args.sym_yshift)
print('dense center deaccelration:',args.dc_accel)


DIR='/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/'
# infile='kdat_3D_R34_C44_3shot.h5'  #3shot
# infile='kdat_3D_R34_C44_3shot_chopped.h5'  #3shot
infile='kdat_3D_R34_C52_3Shot.h5'  #3shot


with h5py.File(DIR+infile,'r') as f:
    kdat=np.concatenate(([f[k][:] for k in f.keys()]),axis=0)
    f.close()

kshape=kdat.shape
print('kdat shape',kshape)    

#extract central region for Calibration
w=24
kz_ll=kshape[-3]//2-w//2
kz_ul=kshape[-3]//2+w//2
ky_ll=kshape[-2]//2-w//2
ky_ul=kshape[-2]//2+w//2
kx_ll=kshape[-1]//2-w//2
kx_ul=kshape[-1]//2+w//2
refs=kdat[ 0, :, kz_ll:kz_ul, ky_ll:ky_ul, kx_ll:kx_ul]

if args.sym_yshift:
    if args.dc_accel is None:
        name_str= infile.rsplit('.',1)[0]+f'_us_{args.type}_{Ry}x{Rz}_sym_yshift_{args.sym_yshift}.h5'
    else:
        name_str= infile.rsplit('.',1)[0]+f'_us_{args.type}_{Ry}x{Rz}_sym_yshift_{args.sym_yshift}_dc_{args.dc_accel}.h5'

else:    
    name_str= infile.rsplit('.',1)[0]+f'_us_{args.type}_{Ry}x{Rz}_yshift_{args.yshift}.h5'


with h5py.File(DIR + name_str,'w') as f:
    f.create_dataset('refs',data=refs)
    f['refs'].attrs['kshape']=kshape
    f.close()
del refs

#us



if args.type == 'Cart':   

    mask=np.zeros(kdat.shape[-3:-1],dtype=int)
    mask[::Rz,::Ry]=1

    mask=np.repeat(mask[None,None,...,None],kdat.shape[-5],axis=-5)
    mask=np.repeat(mask,kdat.shape[-4],axis=-4)
    mask=np.repeat(mask,kdat.shape[-1],axis=-1)

    if args.yshift is True:
        print('applying yshift...')
        for r in range(kdat.shape[0]):
            mask[r,...]=np.roll(mask[r,...],r,axis=-2)
        kdat*=mask
        
    else:
        kdat*=mask

elif args.type == 'CAIP': ##
    #to do 
    #add CAIP Shifts other than 1
    
    ky=kdat.shape[-2]
    kz=kdat.shape[-3]
    Bys=int(np.ceil(ky/R)) #sampling blocks in ky , round up when Bys is non integer 
    SBys=Rz #  ky indices per sampling block || Rz is the maximum shiftings per CAIP Block, that is the shifting the sample along kz, assuming CAIP delta is 1


    mask=np.zeros(kdat.shape[-3:-1],dtype=int)

    mask=mask.T
    #Caip with delta=1
    for By in range(Bys):
        for SBy in range(SBys):
            SBy_ky_idx=min((By*R)+((SBy+1)*Ry)-1,ky-1)
            if (By*R)+((SBy+1)*Ry)-1 > ky-1:
                # print("Hit last ky index, Exiting loop.")
                break  
            mask[SBy_ky_idx,SBy::Rz]=1
    print(f"R={R},Ry={Ry},Rz={Rz}")
    print(f"Mask undersampling = {np.size(mask)/np.count_nonzero(mask)}")
    mask=mask.T
    mask=np.repeat(mask[None,None,...,None],kdat.shape[-5],axis=-5)
    mask=np.repeat(mask,kdat.shape[-4],axis=-4)
    mask=np.repeat(mask,kdat.shape[-1],axis=-1)

    if args.yshift is True:
        print('applying yshift...')
        for r in range(kdat.shape[0]):
            mask[r,...]=np.roll(mask[r,...],r,axis=-2)
        kdat*=mask

    elif args.sym_yshift is True:
        print('applying symmetric yshift...')

        zero_idx = kdat.shape[0] // 2

        for r in range(kdat.shape[0]):
            if r == 0:
                ys = 0          # S0
            else:
                ys = abs(r - zero_idx)

            mask[r, ...] = np.roll(mask[r, ...], ys, axis=-2)

        if args.dc_accel is not None:
            print('applying diamond dense center...')

            R_base = R
            R_target = args.dc_accel

            if R_target >= R_base:
                raise ValueError('--dc_accel must be smaller than the base acceleration R')

            kz = mask.shape[-3]
            ky = mask.shape[-2]

            kz_c = kz // 2
            ky_c = ky // 2

            center_fraction = (1 / R_target - 1 / R_base) / (1 - 1 / R_base)

            q = np.sqrt(center_fraction / 2)

            kz_radius = max(1, int(np.round(q * kz)))
            ky_radius = max(1, int(np.round(q * ky)))

            zz, yy = np.ogrid[:kz, :ky]

            diamond = (
                np.abs(zz - kz_c) / kz_radius
                + np.abs(yy - ky_c) / ky_radius
            ) <= 1

            mask[..., diamond, :] = 1

            current_accel = mask.size / np.count_nonzero(mask)
            print(f'Dense center target acceleration: {R_target}')
            print(f'Actual mask acceleration: {current_accel:.2f}')
            print(f'Diamond radii: kz={kz_radius}, ky={ky_radius}')


        kdat *= mask
        
    else:
        kdat*=mask



chunks=(1,kshape[1],1,kshape[3],1)
with h5py.File(DIR + name_str,'a') as f:
    f.create_dataset('kdat_01',data=kdat[0:17,...],chunks=chunks)
    f.create_dataset('kdat_02',data=kdat[17::,...],chunks=chunks)
    
del kdat


print('Created File: ' + DIR + name_str)
 