#!/bin/bash
#SBATCH --job-name=mps_ksp_07062026_w36
#SBATCH --partition=long256
##SBATCH --partition=work
##SBATCH --partition=broadwell512
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=64
#BATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=ab.alhaidari@yahoo.com

echo "Running on $(hostname)"
echo "Start time: $(date)"


unset SLURM_EXPORT_ENV
unset SLURM_EXPORT_ENV


# module load python/3.12-conda
conda init bash
eval "$(conda shell.bash hook)"
conda activate LLRv3
# module load cuda/12.4.1

DATA_DIR='/home/hpc/iwbi/iwbi112h/LLR codebooks/lab_llr_alhaidari/scripts/'

mkdir $TMPDIR/$SLURM_JOB_ID


free -h 
python --version

python -c"print('gona Work! Hoooa')"
##Original Work

# python "${DATA_DIR}mps.py" --merged_refs --space ksp --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_APT_C6/kdat_3D_R26_C58_C6.h5" --w 36 
# python "${DATA_DIR}mps.py" --merged_refs --space ksp --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_APT_C16/kdat_3D_R26_C58_C16.h5" --w 36 
python "${DATA_DIR}mps.py"  --space ksp --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_3D_R34_C52_3Shot.h5" --ref_data "ACS_3D_C52.h5" --w 36
# python "${DATA_DIR}mps.py" --merged_refs=True --space ksp --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_APT_C16/kdat_3D_R26_C58_CAIP16.h5" 
# python "${DATA_DIR}mps.py" --space ksp --prew True 
# python "${DATA_DIR}ifft_RO.py"  

python -c"print('Worked! Hoooa')"
rm -r $TMPDIR/$SLURM_JOB_ID

echo "End time: $(date)"

