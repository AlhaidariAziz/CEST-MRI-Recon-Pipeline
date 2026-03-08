import numpy as np 
import h5py
import argparse

parser = argparse.ArgumentParser(description='Retrospective Undersampling pattern')
parser.add_argument('--type', required=True,choices=('Cart','CAIP'), help='Type of undersampling pattern')
parser.add_argument('--R', type=int, required=True, help='undersampling factor, for 2D define RPE2 as well RPE1=R//RPE2')
parser.add_argument('--RPE2', type=int, default=1, help='undersampling factor in PE2 direction')
parser.add_argument('--yshift', type=bool,default=False, help='apply yshift=1 across Repeated measurment')

args = parser.parse_args()

if args.type == 'CAIP' and args.RPE2 == 1 :
    raise SystemExit( "Please specify the --RPE2 > 1 argument for CAIP undersampling pattern" )

R=args.R ; Rz=args.RPE2 ;  Ry=R//Rz

print('us accelaration:',R)
print('us type:',args.type)
print('us PE1 accelaration:',Ry)
print('us PE2 accelaration:',Rz)
print('yshift:',args.yshift)


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
    #add CAIP Shift
    
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



chunks=(1,kshape[1],1,kshape[3],1)
with h5py.File(DIR + name_str,'a') as f:
    f.create_dataset('kdat_01',data=kdat[0:17,...],chunks=chunks)
    f.create_dataset('kdat_02',data=kdat[17::,...],chunks=chunks)
    
del kdat


print('Created File: ' + DIR + name_str)
 