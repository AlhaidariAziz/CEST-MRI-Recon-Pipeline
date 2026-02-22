#!/bin/bash
#SBATCH --job-name=mps_ksp_14022026
#SBATCH --partition=work
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
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

python "${DATA_DIR}mps.py" --space ksp --prew True 
# python "${DATA_DIR}ifft_RO.py"  

python -c"print('Worked! Hoooa')"
rm -r $TMPDIR/$SLURM_JOB_ID

echo "End time: $(date)"

