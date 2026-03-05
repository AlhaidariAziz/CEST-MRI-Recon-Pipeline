import numpy as np 
import h5py
import argparse

parser = argparse.ArgumentParser(description='Retrospective Undersampling pattern')
parser.add_argument('--type', required=True,choices=('Cart','CAIP'), help='Type of undersampling pattern')
parser.add_argument('--R', type=int, required=True, help='undersampling factor')
parser.add_argument('--RPE1', type=int, help='undersampling factor in PE1 direction')

# parser.add_argument('--shift_PE1P', type=int, default=0, help='through plane shift')
# parser.add_argument('--shift_PE1R', type=int, default=0 , help='through repetition shifts')

args = parser.parse_args()

if args.type == 'CAIP' and args.RPE1 is None:
    raise SystemExit( "Please specify the --RPE1 argument for CAIP undersampling pattern, e.g., --RPE1 4." )


DIR='/home/vault/iwbi/iwbi112h/CEST_DATA/'
infile='kdat_3D_R34_C44_3shot.h5'  #3shot


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

with h5py.File(DIR+infile.rsplit('.',1)[0]+f'_us_{args.type}_{args.R}.h5','w') as f:
    f.create_dataset('refs',data=refs)
    f['refs'].attrs['kshape']=kshape
    f.close()
del refs

#us
if args.type == 'Cart':
    for i in range(args.R-1):
        kdat[:,:,:,i+1::args.R,:]=0
    # kdat[:,:,:,1::3,:]=0
    # kdat[:,:,:,2::3,:]=0

elif args.type == 'CAIP': ##CAIP with constant shift of 1
    #to do 
    #add CAIP Shift
    Ry=args.RPE1;R=args.R;Rz=R//Ry
    ky=kdat.shape[-2]
    kz=kdat.shape[-3]
    Bys=ky//R #sampling blocks in ky
    SBys=R//Rz #sampling sub-blocks in ky

    mask=np.zeros(kdat.shape[-3:-1],dtype=int)

    mask=mask.T
    #Caip with delta=1
    for By in range(Bys):
    # for By in range(1):
        for SBy in range(Rz):
        # for SBy in range(1):
            mask[(By*R)+(SBy+1)*Ry-1,SBy::Rz]=1
            # m3[0:2,0::2]=0
    mask=mask.T

    kdat*=mask[None,None,...,None]



   
chunks=(1,kshape[1],1,kshape[3],1)
with h5py.File(DIR+infile.rsplit('.',1)[0]+f'_us_{args.type}_{args.R}.h5','a') as f:
    f.create_dataset('kdat_01',data=kdat[0:17,...],chunks=chunks)
    f.create_dataset('kdat_02',data=kdat[17::,...],chunks=chunks)
    
del kdat


print('Created File: ' + DIR+infile.rsplit('.',1)[0]+f'_us_{args.type}_{args.R}.h5')
 