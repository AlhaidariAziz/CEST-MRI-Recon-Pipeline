#!/bin/bash -l

#SBATCH --export=NONE

#SBATCH --ntasks=2
#SBATCH --gres=gpu:a100:1 # a100 GPU
##SBATCH --gres=gpu:1 # just any GPU
##SBATCH --gres=gpu:a100:1 -C a100_80 # supposed to be for a100 (80GB) GPU
#SBATCH -p a100
##SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
##SBATCH --job-name=HybVolLLR_0402
#SBATCH --job-name=ifft_RO_1502
#SBATCH --mail-type=END
#SBATCH --mail-user=ab.alhaidari@yahoo.com

unset SLURM_EXPORT_ENV

module unload python
# module load cuda/12.4.1
module load python/3.12-conda
# conda init bash
eval "$(conda shell.bash hook)"
conda activate LLRv3
module load cuda/12.4.1

DATA_DIR='/home/hpc/iwbi/iwbi112h/LLR codebooks/lab_llr_alhaidari/scripts/'

mkdir $TMPDIR/$SLURM_JOB_ID


free -h 

echo "Running on $(hostname)"
echo "Start time: $(date)"

##Original Work

# python "${DATA_DIR}mps.py" --space ksp # space = Hyb || ksp
# python "${DATA_DIR}HybVolLLR.py" --blk_shape 1 1 1 --i 30
# python "${DATA_DIR}HybVolLLR.py" --blk_shape 1 2 2 --i 30
# python "${DATA_DIR}HybVolLLR.py" --blk_shape 1 6 6 --i 50
# python "${DATA_DIR}HybVolLLR.py" --blk_shape 1 1 1 --i 30 --r 1e-2
# python "${DATA_DIR}HybVolLLR.py" --blk_shape 1 2 2 --i 30 --r 1e-2
# python "${DATA_DIR}HybVolLLR.py" --blk_shape 1 4 4 --i 30 --r 1e-2
# python "${DATA_DIR}HybVolLLR.py" --blk_shape 1 5 5 --i 30 --r 1e-2
# python "${DATA_DIR}HybVolLLR.py" --blk_shape 1 6 6 --i 30 --r 1e-2
# python "${DATA_DIR}HybVolLLR.py" --blk_shape 1 8 8 --i 30 --r 1e-2
# python "${DATA_DIR}HybVolLLR.py" --blk_shape 1 10 10 --i 30 --r 1e-2
# python "${DATA_DIR}HybVolLLR.py" --blk_shape 1 12 12 --i 30 --r 1e-2
# python "${DATA_DIR}HybVolGRAPPA.py" 
# python "${DATA_DIR}mdGRAPPA.py"
python "${DATA_DIR}ifft_RO.py"

rm -r $TMPDIR/$SLURM_JOB_ID




echo "End time: $(date)"
